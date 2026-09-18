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


def row_slice(draws: sparse.csr_matrix, lo: int, hi: int) -> sparse.csr_matrix:
    """Rows `[lo, hi)` of a CSR matrix, without copying its data.

    `draws[lo:hi]` would copy; the staged test takes three consecutive slices
    of the same draw matrix per pair, so the copies add up. `data` and
    `indices` are numpy views here and only the small `indptr` is rebuilt.
    """
    start, end = int(draws.indptr[lo]), int(draws.indptr[hi])
    return sparse.csr_matrix(
        (
            draws.data[start:end],
            draws.indices[start:end],
            draws.indptr[lo : hi + 1] - start,
        ),
        shape=(hi - lo, draws.shape[1]),
        copy=False,
    )


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
    sums = draws @ stacked  # (B, p + 2)
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
