"""Port of `compute_observed_full_statistic_v2` / `compute_null_full_statistics`
(low_level_full_test.cpp): the O(n_trt)-per-resample score-type test statistic
that lets sceptre evaluate B resampled treatment assignments without
refitting a GLM for each one.

  top         = sum_{i in trt} a[i]
  lower_left  = sum_{i in trt} w[i]
  lower_right = sum_k ( sum_{i in trt} D[k, i] )^2
  z = top / sqrt(lower_left - lower_right)

All indices here are 0-based (unlike the R/C++ source, which is 1-based).

`compute_null_full_statistics` computes this for every one of the B
(ragged-length) synthetic index sets. A direct port would loop over B in
Python, but at real scale (B up to ~5,498, called once per (gene, target)
pair, ~34,886 pairs) that loop dominates runtime -- this instead flattens
the B ragged arrays into one long array plus a per-element "which resample"
label, and computes all B sums at once via `np.bincount` (segment sums),
which does the O(B)-scale work in C rather than a Python loop (the segment
sums themselves use `np.add.reduceat`; see `_segment_sums`). The
flattening (`flatten_synthetic_idxs`) only depends on the target's
synthetic draws, not the gene, so callers that reuse the same
`synthetic_idxs` across many genes (see pipeline/discovery.py) should
flatten once and reuse the result via `compute_null_full_statistics_flat`.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from ..crt.permutations import draws_for_target


def compute_observed_full_statistic(
    a: np.ndarray, w: np.ndarray, D: np.ndarray, trt_idxs: np.ndarray
) -> float:
    top = a[trt_idxs].sum()
    lower_left = w[trt_idxs].sum()
    lower_right = np.square(D[:, trt_idxs].sum(axis=1)).sum()
    return float(top / np.sqrt(lower_left - lower_right))


def flatten_synthetic_idxs(synthetic_idxs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (flat_idxs, lengths, B): flat_idxs is the concatenation of all B
    ragged index arrays, and lengths[j] is how many entries resample j
    contributed. Segments are therefore *contiguous* in flat_idxs, which is
    what lets the statistic use `np.add.reduceat` (see
    `compute_null_full_statistics_flat`)."""
    B = len(synthetic_idxs)
    lengths = np.fromiter((len(idxs) for idxs in synthetic_idxs), dtype=np.int64, count=B)
    if lengths.sum() == 0:
        return np.empty(0, dtype=np.int64), lengths, B
    flat_idxs = np.concatenate(synthetic_idxs)
    return flat_idxs, lengths, B


def _segment_sums(values: np.ndarray, lengths: np.ndarray, B: int) -> np.ndarray:
    """Contiguous segment sums along the last axis, for 1-D or 2-D `values`.

    `np.add.reduceat` does all rows in one call, which matters because the
    2-D case is the dominant cost of this module (measured: 61.7 ms for one
    `np.bincount` per row of D versus 20.8 ms for a single `reduceat`, on a
    real-scale 5,498-draw batch).

    Empty segments need care: `reduceat` given a repeated offset returns the
    *element at that offset* rather than 0, so zero-length resamples would
    silently contribute a stray value. Such a draw is possible in principle
    (probability ~exp(-n_trt) per resample, so negligible for real targets
    but not impossible for tiny ones), so they are reduced among the nonempty
    segments and written back as exact zeros.
    """
    nonempty = lengths > 0
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    out = np.zeros(values.shape[:-1] + (B,), dtype=float)
    if not nonempty.any():
        return out
    out[..., nonempty] = np.add.reduceat(values, offsets[nonempty], axis=-1)
    return out


def draws_to_matrix(synthetic_idxs: list[np.ndarray], n_cells: int) -> sparse.csr_matrix:
    """The B resample index sets as a `(B, n_cells)` 0/1 CSR matrix.

    Row j holds resample j's treated cells, so this is precisely
    `(flat_idxs, lengths)` relabelled: the concatenation is the `indices`
    array and the cumulative lengths are the `indptr`. Nothing is copied that
    `flatten_synthetic_idxs` did not already copy.

    Built **once per target** and reused across every gene paired with it,
    where flattening used to be redone once per *pair*.
    """
    B = len(synthetic_idxs)
    lengths = np.fromiter((len(idxs) for idxs in synthetic_idxs), dtype=np.int64, count=B)
    indptr = np.zeros(B + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    indices = (
        np.concatenate(synthetic_idxs).astype(np.int64, copy=False)
        if B and lengths.sum()
        else np.empty(0, dtype=np.int64)
    )
    if indices.size and (indices.min() < 0 or indices.max() >= n_cells):
        # Fancy indexing would have wrapped a negative index to a cell at the
        # other end of the array and returned a plausible-looking number. A
        # malformed draw is a bug in whatever produced it, so it is refused
        # here rather than silently absorbed -- exactly this masked an
        # off-by-one in the R ground-truth dumper, which turned every 0 into
        # a -1 and went unnoticed because `a[-1]` is valid Python.
        raise ValueError(
            f"resample cell indices out of range for n_cells={n_cells}: "
            f"[{indices.min()}, {indices.max()}]"
        )
    return sparse.csr_matrix(
        (np.ones(indices.size), indices, indptr), shape=(B, n_cells), copy=False
    )


class StagedDraws:
    """A target's resamples, materialized one stage at a time.

    The staged test asks for `[0, B1)` for every pair, and for `[B1, B1+B2)`
    and beyond only when a pair escalates. Building the whole draw matrix up
    front therefore does work that is usually thrown away: on a real
    permutation run, **33,621 of 34,886 pairs stopped at stage 1** and 8
    reached stage 3, yet every target had all 30,497 rows materialized.

    Slices are memoized, so the cost is paid once per target per stage
    actually reached rather than once per pair -- a target's draws are shared
    by every gene paired with it.

    Subclasses differ only in where the indices come from. The CRT holds its
    own per-target draws; permutations hold a reference into one array shared
    by every target, which is what makes them cheap to keep unmaterialized.
    """

    __slots__ = ("n_cells", "n_draws", "_cache")

    def __init__(self, n_cells: int, n_draws: int):
        self.n_cells = n_cells
        self.n_draws = n_draws
        self._cache: dict[tuple[int, int], sparse.csr_matrix] = {}

    def _index_arrays(self, lo: int, hi: int) -> list[np.ndarray]:
        raise NotImplementedError

    def slice(self, lo: int, hi: int) -> sparse.csr_matrix:
        lo, hi = max(0, int(lo)), min(int(hi), self.n_draws)
        if hi <= lo:
            return sparse.csr_matrix((0, self.n_cells))
        key = (lo, hi)
        hit = self._cache.get(key)
        if hit is None:
            hit = draws_to_matrix(self._index_arrays(lo, hi), self.n_cells)
            self._cache[key] = hit
        return hit


class ListDraws(StagedDraws):
    """CRT draws: a list of ragged index arrays, one per resample.

    Lazy like the permutation source, though the saving is smaller. The CRT
    sampler produces every draw up front, so the index arrays exist whatever
    happens and only the CSR construction can be deferred -- but that is
    still most of the cost, because the stages beyond the first are rarely
    reached.

    Measured on 6,000 real pairs, fresh process per run, two runs each:

        eager   305.3 s, 305.3 s   peak 5.78 GB, 5.83 GB
        lazy    275.7 s, 274.2 s   peak 5.79 GB, 5.88 GB

    1.11x faster at indistinguishable memory. An earlier reading claimed
    deferral cost 2.2 GB here; that was an artefact of comparing two
    configurations inside one process, where `ru_maxrss` reports a
    process-lifetime high-water mark and whichever ran second inherited the
    first's peak. Measure variants in separate processes.
    """

    __slots__ = ("_idxs",)

    def __init__(self, synthetic_idxs: list[np.ndarray], n_cells: int):
        super().__init__(n_cells, len(synthetic_idxs))
        self._idxs = synthetic_idxs

    def _index_arrays(self, lo: int, hi: int) -> list[np.ndarray]:
        return self._idxs[lo:hi]


class PermutationSliceDraws(StagedDraws):
    """Permutation draws: a prefix of each row of one shared array.

    Holds a *reference* to the shared permutations rather than a copy, so a
    target costs nothing until a stage is actually reached. That is the whole
    point of the mechanism -- every target reads the same draws -- and it was
    being thrown away by materializing per target.

    **Deliberately not memoized**, unlike the base class. Stage 1 no longer
    reaches here at all -- `PermutationPrefixSums` serves it from one
    per-gene scan -- so what remains is the escalation stages, 1,257 and 8
    pairs of 34,886 on day0. Caching those buys a rebuild for the occasional
    second gene that escalates on the same target, and costs a matrix that
    never goes away: with targets no longer processed in chunks there is no
    point at which a chunk's caches are dropped, and ~1,000 escalating
    targets holding a B2 slice each would be tens of GB. Rebuilding is the
    cheaper mistake to make.
    """

    __slots__ = ("_perms", "_n_trt")

    def __init__(self, perms: np.ndarray, n_trt: int, n_cells: int):
        super().__init__(n_cells, perms.shape[0])
        self._perms = perms
        self._n_trt = n_trt

    def slice(self, lo: int, hi: int) -> sparse.csr_matrix:
        lo, hi = max(0, int(lo)), min(int(hi), self.n_draws)
        if hi <= lo:
            return sparse.csr_matrix((0, self.n_cells))
        return draws_to_matrix(self._index_arrays(lo, hi), self.n_cells)

    def _index_arrays(self, lo: int, hi: int) -> list[np.ndarray]:
        # Delegates rather than repeating the one-line slice, because the
        # duplicate was real: `draws_for_target` had no caller in the package
        # while the tests exercised only it, so the tested prefix and the
        # executed prefix were different code. They were kept in step by
        # memory, and when the sort was dropped both had to be edited for the
        # analysis to change at all.
        return draws_for_target(self._perms[lo:hi], self._n_trt)


def as_staged_draws(draws, n_cells: int) -> StagedDraws:
    """Accept a `StagedDraws`, a CSR matrix, or a list of index arrays.

    The list and matrix forms are kept because they are the natural things to
    write by hand in a test; the analysis always passes a `StagedDraws`.
    """
    if isinstance(draws, StagedDraws):
        return draws
    if sparse.issparse(draws):
        csr = draws.tocsr()
        return ListDraws(
            [csr.indices[csr.indptr[i] : csr.indptr[i + 1]] for i in range(csr.shape[0])],
            csr.shape[1],
        )
    return ListDraws(list(draws), n_cells)


def stack_pieces(a: np.ndarray, w: np.ndarray, D: np.ndarray) -> np.ndarray:
    """`a`, `w` and `D`'s rows as one `(n_cells, p + 2)` dense array.

    All three need the same per-resample segment sums, so stacking them lets a
    single matmul produce all of them. Built once per *gene* and reused across
    that gene's targets.
    """
    return np.column_stack([a, w, D.T])


def compute_null_statistics_from_draws(stacked: np.ndarray, draws: sparse.csr_matrix) -> np.ndarray:
    """The null statistics, as one sparse-dense matmul.

    `draws @ stacked` computes every resample's segment sums of `a`, `w` and
    each row of `D` at once, because a 0/1 row of `draws` dotted with a column
    is exactly that resample's sum over its treated cells.

    This replaces gathering `D[:, flat_idxs]` and reducing it. The gather was
    the single hottest operation in the package -- it materializes a
    `(p, sum(lengths))` temporary, 158 MB at real dataset scale, per pair --
    and the
    matmul needs no temporary at all. **Measured 42.7 ms -> 11.2 ms**, 3.8x,
    on a real-scale call.

    Empty resamples need no special handling here: an empty CSR row sums to
    zero on its own, where `np.add.reduceat` would have returned the element
    at the repeated offset and needed correcting.

    Not bit-identical to the gather: a sparse matmul accumulates in a
    different order, so results differ by ~1e-13 absolute on the statistic.
    That is far below the Monte Carlo noise the p-values already carry.
    """
    if draws.shape[0] == 0:
        return np.empty(0)
    if draws.nnz == 0:
        return np.full(draws.shape[0], np.nan)
    return statistics_from_segment_sums(draws @ stacked)  # (B, p + 2)


def statistics_from_segment_sums(sums: np.ndarray) -> np.ndarray:
    """The null statistics, given each resample's segment sums of `stacked`.

    Split out because there are two ways to obtain those sums and only one
    way to turn them into statistics. The CRT reaches them through a sparse
    matmul against a per-target draw matrix; permutations reach them by
    reading one column out of a per-gene prefix-sum array
    (`PermutationPrefixSums`). Keeping the arithmetic in one place means the
    two routes cannot drift.
    """
    top = sums[:, 0]
    lower_left = sums[:, 1]
    d_rows = sums[:, 2:]
    lower_right = np.einsum("bk,bk->b", d_rows, d_rows)
    with np.errstate(invalid="ignore"):
        return top / np.sqrt(lower_left - lower_right)


def compute_null_full_statistics_flat(
    a: np.ndarray,
    w: np.ndarray,
    D: np.ndarray,
    flat_idxs: np.ndarray,
    lengths: np.ndarray,
    B: int,
) -> np.ndarray:
    """Vectorized core: see module docstring. `(flat_idxs, lengths, B)` is the
    output of `flatten_synthetic_idxs`, which callers looping over many genes
    for the same target's synthetic draws should compute once and reuse."""
    if flat_idxs.size == 0:
        return np.full(B, np.nan)
    top = _segment_sums(a[flat_idxs], lengths, B)
    lower_left = _segment_sums(w[flat_idxs], lengths, B)
    # per-row (covariate-dimension) segment sums of D, then sum of squares over rows
    D_row_sums = _segment_sums(D[:, flat_idxs], lengths, B)
    lower_right = np.sum(D_row_sums**2, axis=0)
    return top / np.sqrt(lower_left - lower_right)


def compute_null_full_statistics(
    a: np.ndarray, w: np.ndarray, D: np.ndarray, synthetic_idxs: list[np.ndarray]
) -> np.ndarray:
    """synthetic_idxs: a list of B (possibly ragged, i.e. differently-sized)
    0-based integer index arrays -- one synthetic treated-cell set per resample.
    This targets sceptre's `use_all_cells=True` case (the only one the
    complement-control-group CRT path uses): each resample's own set size is
    used directly, rather than a fixed external n_trt.

    Convenience wrapper around `compute_null_full_statistics_flat` for callers
    with a single synthetic_idxs list to consume once; prefer flattening once
    and calling `compute_null_full_statistics_flat` directly when the same
    synthetic_idxs will be reused across many (a, w, D) triples."""
    flat_idxs, lengths, B = flatten_synthetic_idxs(synthetic_idxs)
    return compute_null_full_statistics_flat(a, w, D, flat_idxs, lengths, B)


class PermutationPrefixSums:
    """One gene's segment sums along the *shared* permutation rows.

    Every target reads the same permutation rows and differs only in how far
    along each row it reads: a target with `n_trt` cells takes `row[:n_trt]`.
    So the segment sums a target needs are a **prefix sum** of the gene's
    `stacked` values gathered along those rows, and the sums for *every*
    target are columns of one cumulative sum::

        G = stacked[perms[lo:hi, :m]]   # (B, m, p + 2)
        C = G.cumsum(axis=1)            # prefix sums along each row
        sums_for_n_trt = C[:, n_trt - 1] # (B, p + 2), a read

    This is what R's structure implies and pysceptre did not do. The CRT
    cannot do it -- its draws are genuinely per-target, with no shared
    ordering to accumulate along -- which is why the per-target matmul
    existed in the first place.

    The arithmetic is the reason to bother. A gene with `t` targets paid
    `t` matmuls of `B * n_trt * (p + 2)`; it now pays one gather-and-scan of
    `B * m * (p + 2)` and `t` reads. At day0's stage 1 (`B = 499`, `m = 692`,
    `p + 2 = 13`, median `n_trt = 557`) that is 4.5M operations once against
    3.6M per target.

    **Bounded, and it falls back rather than growing.** The array is
    `B * m * (p + 2) * 8` bytes, where `m` is the *largest* target, and that
    is per worker. On day0 (`m = 1,772`, `p + 2 = 13`) the stages are far
    apart: 92 MB at `B1 = 499`, 921 MB at `B2 = 4999`, 4.6 GB at
    `B3 = 24999`. Stages past the first are also rare -- 1,257 and 8 pairs of
    34,886 -- so they are not worth the memory. `statistics` returns `None`
    above `max_bytes` and the caller uses the per-target draw matrix for that
    stage, which is the path the CRT uses anyway.

    **`max_bytes` must be set from the real `m`, not a guess.** The first
    default here was 64 MB, chosen against an `m` of a few hundred, and on
    day0 it declined *every* stage-1 scan -- so this class never once ran on
    the dataset it was written for, and the tests missed it because their
    fixtures used `m = 120`, where the scan is 1.2 MB and the gate cannot
    fire. A profile caught it: `csr_matvecs` was still 58% of runtime after
    the change that was supposed to remove it. The default is now 256 MB,
    which admits stage 1 up to `m` of about 4,900 while still declining
    day0's stage 2 by a factor of 3.6.
    """

    __slots__ = ("_stacked", "_perms", "_max_bytes", "_cache")

    def __init__(self, stacked: np.ndarray, perms: np.ndarray, max_bytes: float = 256e6):
        self._stacked = stacked
        self._perms = perms
        self._max_bytes = max_bytes
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def _scan(self, lo: int, hi: int) -> np.ndarray | None:
        key = (lo, hi)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        rows = self._perms[lo:hi]
        if rows.size == 0:
            return None
        nbytes = rows.shape[0] * rows.shape[1] * self._stacked.shape[1] * self._stacked.itemsize
        if nbytes > self._max_bytes:
            return None
        scan = np.cumsum(self._stacked[rows], axis=1)
        self._cache[key] = scan
        return scan

    def statistics(self, lo: int, hi: int, n_trt: int) -> np.ndarray | None:
        """Null statistics for `[lo, hi)` for a target of `n_trt` cells.

        `None` means "too large to scan, use the draw matrix instead".
        """
        if n_trt <= 0:
            return None
        scan = self._scan(lo, hi)
        if scan is None:
            return None
        if n_trt > scan.shape[1]:
            raise ValueError(
                f"target needs {n_trt} cells but the shared permutation rows hold "
                f"only {scan.shape[1]}; the draws were built for a different target set"
            )
        return statistics_from_segment_sums(scan[:, n_trt - 1, :])
