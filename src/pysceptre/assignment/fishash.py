"""fishash: assign guides to cells with a one-sided Fisher test per (guide, cell).

Port of fishash's `fishash()`, its internal `fishash_internal()` and `impute_masked_counts()`
(jackkamm/fishash 0.99.5, commit 5eabd3c; MIT, see `THIRD_PARTY_LICENSES`). Not from sceptre.

Every nonzero count is tested with a one-sided Fisher exact test on its 2x2 table (this guide
or another, this cell or another), the cut is set by an FDR procedure (Guo and Sarkar's
block procedure by default), and for up to `refit` further passes the off-cell margins are
taken from an estimate of the noise alone. The counts are never transformed: the test itself
conditions on each cell's and each guide's totals. See `docs/design.md`, "gRNA assignment".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ._common import as_count_matrix, check_ids
from .hypergeom import _log_phyper_rounded

__all__ = [
    "FishashIteration",
    "FishashResult",
    "ImputedCounts",
    "assign_grnas_fishash",
    "impute_masked_counts",
]

_PADJ_METHODS = ("GS", "BY", "BH")
# fishash's Benjamini-Yekutieli constant uses this truncated Euler-Mascheroni constant.
_EULER_TRUNCATED = 0.57721


@dataclass(frozen=True)
class FishashIteration:
    """One pass of the test.

    Attributes:
        log_pval_cutoff: The pass's cut on the log p-value.
        n_significant: GS: the number of cells kept by the block-level BH step (`B`).
            BH and BY: the number of significant entries.
        n_assigned: Entries assigned in this pass.
        impute_n_iter: Inner iterations of the noise fit run before this pass; 0 for pass 1.
    """

    log_pval_cutoff: float
    n_significant: int
    n_assigned: int
    impute_n_iter: int


@dataclass(frozen=True)
class FishashResult:
    """What `assign_grnas_fishash` returns.

    Attributes:
        assigned: `(n_grnas, n_cells)` boolean CSR matrix, True where a guide is assigned.
        log_pval: `(n_grnas, n_cells)` CSR matrix of log p-values at every nonzero count.
        odds_ratio: Empirical odds ratio of each tested 2x2 table, same pattern.
        odds_ratio_regularized: The odds ratio with 1 added to both off-diagonal counts.
        demux_type: `(n_cells,)` "singlet", "doublet" or "unknown" (no guide assigned).
        assignment: `(n_cells,)` comma-joined ids of the assigned guides, "" if none, in
            fishash's order (gRNA order under GS; least to most significant under BH and BY).
        log_pval_cutoff: The final pass's cut on the log p-value.
        num_iter: Number of passes run.
        grna_ids: Row names.
        iterations: One `FishashIteration` per pass.
        background_guide_freqs: Guide frequencies of the noise fit the final pass used, or
            None when no refit pass ran.
        background_cell_sizes: Cell sizes of that fit, or None.
    """

    assigned: sparse.csr_matrix
    log_pval: sparse.csr_matrix
    odds_ratio: sparse.csr_matrix
    odds_ratio_regularized: sparse.csr_matrix
    demux_type: np.ndarray
    assignment: np.ndarray
    log_pval_cutoff: float
    num_iter: int
    grna_ids: tuple[str, ...]
    iterations: tuple[FishashIteration, ...]
    background_guide_freqs: np.ndarray | None
    background_cell_sizes: np.ndarray | None


@dataclass(frozen=True)
class ImputedCounts:
    """What `impute_masked_counts` returns.

    Attributes:
        counts: `(n_grnas, n_cells)` CSR matrix: the counts, with masked entries replaced by
            `guide_freqs[g] * cell_sizes[c]`.
        guide_freqs: `(n_grnas,)` noise frequency of each guide, summing to 1.
        cell_sizes: `(n_cells,)` noise size of each cell.
        n_iter: Alternating iterations run.
    """

    counts: sparse.csr_matrix
    guide_freqs: np.ndarray
    cell_sizes: np.ndarray
    n_iter: int


def _seqsum(x: np.ndarray) -> float:
    # Left to right, as R's sum() and Matrix's margins accumulate; numpy's sum is pairwise.
    return float(np.cumsum(x)[-1]) if x.size else 0.0


def _log(x: float) -> float:
    return math.log(x) if x > 0 else (-math.inf if x == 0 else math.nan)


def _p_adjust_bh(p: np.ndarray) -> np.ndarray:
    """R's `p.adjust(p, "BH")`: NaN kept in place, `n` counts the others, identity if n <= 1."""
    out = np.array(p, dtype=np.float64)
    ok = ~np.isnan(out)
    q = out[ok]
    n = q.size
    if n <= 1:
        return out
    i = np.arange(n, 0, -1, dtype=np.float64)
    o = np.argsort(-q, kind="stable")
    adjusted = np.minimum(1.0, np.minimum.accumulate((n / i) * q[o]))
    res = np.empty(n)
    res[o] = adjusted
    out[ok] = res
    return out


@dataclass
class _Entries:
    """The counts' nonzero entries in column-major order, and what never changes between passes."""

    data: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    indptr: np.ndarray
    n_grnas: int
    n_cells: int
    row_sums: np.ndarray
    col_sums: np.ndarray
    tot: float

    @classmethod
    def from_csc(cls, c: sparse.csc_matrix) -> _Entries:
        g, n = c.shape
        data = c.data
        rows = c.indices.astype(np.int64)
        cols = np.repeat(np.arange(n, dtype=np.int64), np.diff(c.indptr))
        return cls(
            data=data,
            rows=rows,
            cols=cols,
            indptr=c.indptr.astype(np.int64),
            n_grnas=g,
            n_cells=n,
            row_sums=np.bincount(rows, weights=data, minlength=g),
            col_sums=np.bincount(cols, weights=data, minlength=n),
            tot=_seqsum(data),
        )


def _impute_aligned(e: _Entries, data: np.ndarray, mask: np.ndarray, eps: float, max_iter: int):
    """`impute_masked_counts` on arrays aligned with `e`'s pattern, which must contain the mask."""
    g, n = e.n_grnas, e.n_cells
    d0 = np.where(mask, 0.0, data)
    rs0 = np.bincount(e.rows, weights=d0, minlength=g)
    cs0 = np.bincount(e.cols, weights=d0, minlength=n)
    skip_row = rs0 == 0
    skip_col = cs0 == 0
    mr = e.rows[mask]
    mc = e.cols[mask]
    cell_sizes = np.ones(n)
    guide_freqs = np.zeros(g)
    prev_freqs = prev_sizes = None
    n_iter = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        for it in range(1, max_iter + 1):
            n_iter = it
            guide_freqs = _seqsum(cell_sizes) - np.bincount(mr, weights=cell_sizes[mc], minlength=g)
            guide_freqs = rs0 / guide_freqs
            guide_freqs[skip_row] = 0.0
            guide_freqs = guide_freqs / _seqsum(guide_freqs)
            cell_sizes = 1.0 - np.bincount(mc, weights=guide_freqs[mr], minlength=n)
            cell_sizes = cs0 / cell_sizes
            cell_sizes[skip_col] = 0.0
            if it > 1:
                d_g = np.abs(np.log(guide_freqs) - np.log(prev_freqs))
                d_c = np.abs(np.log(cell_sizes) - np.log(prev_sizes))
                d_g = d_g[~np.isnan(d_g)]
                d_c = d_c[~np.isnan(d_c)]
                # R's max(..., na.rm = TRUE) of nothing is -Inf, which counts as converged.
                max_g = d_g.max() if d_g.size else -math.inf
                max_c = d_c.max() if d_c.size else -math.inf
                if max_g < eps and max_c < eps:
                    break
            prev_freqs = guide_freqs
            prev_sizes = cell_sizes
    values = d0.copy()
    values[mask] = guide_freqs[mr] * cell_sizes[mc]
    return values, guide_freqs, cell_sizes, n_iter


def impute_masked_counts(
    counts: sparse.spmatrix | np.ndarray,
    mask: sparse.spmatrix | np.ndarray,
    *,
    eps: float = 1e-4,
    max_iter: int = 10,
) -> ImputedCounts:
    """Replace masked entries of a count matrix by a rank-one Poisson fit to the rest.

    Port of fishash's `impute_masked_counts`. The unmasked counts are fitted as
    `guide_freqs[g] * cell_sizes[c]` by alternating closed-form updates, until both factors
    move less than `eps` on the log scale or `max_iter` iterations have run.

    Args:
        counts: `(n_grnas, n_cells)` integer counts, scipy.sparse or dense.
        mask: Same shape; nonzero (or True) marks an entry to replace. May mark zero counts.
        eps: Convergence tolerance on the largest change in log factor.
        max_iter: Maximum alternating iterations.

    Returns:
        An `ImputedCounts`.

    Raises:
        ValueError: shapes differ, or `counts` is not a count matrix.
    """
    c = as_count_matrix(counts, name="counts")
    if sparse.issparse(mask):
        m = sparse.csc_matrix(mask, dtype=bool, copy=True)
    else:
        m = sparse.csc_matrix(np.asarray(mask).astype(bool))
    if m.shape != c.shape:
        raise ValueError(f"mask has shape {m.shape} but counts has shape {c.shape}")
    m.eliminate_zeros()
    # Align counts and mask on the union of their patterns (both are non-negative, so the sum
    # cancels nothing); a stored zero in the union changes no sum.
    union = (c + m.astype(np.float64)).tocsc()
    union.sum_duplicates()
    union.sort_indices()
    rows = union.indices.astype(np.int64)
    cols = np.repeat(np.arange(c.shape[1], dtype=np.int64), np.diff(union.indptr))
    data = np.asarray(c.tocsr()[rows, cols]).ravel()
    mask_aligned = np.asarray(m.tocsr()[rows, cols]).ravel().astype(bool)
    e = _Entries(
        data=data,
        rows=rows,
        cols=cols,
        indptr=union.indptr.astype(np.int64),
        n_grnas=c.shape[0],
        n_cells=c.shape[1],
        row_sums=np.zeros(0),
        col_sums=np.zeros(0),
        tot=0.0,
    )
    values, gf, cs, n_iter = _impute_aligned(e, data, mask_aligned, eps, max_iter)
    out = sparse.csc_matrix((values, rows, union.indptr), shape=c.shape).tocsr()
    out.eliminate_zeros()
    return ImputedCounts(counts=out, guide_freqs=gf, cell_sizes=cs, n_iter=n_iter)


@dataclass
class _Pass:
    log_pval: np.ndarray
    odds_ratio: np.ndarray
    odds_ratio_regularized: np.ndarray
    assigned: np.ndarray
    cutoff: float
    n_significant: int
    order: np.ndarray | None  # BH/BY: the entries in R's final data-frame order
    nr: np.ndarray
    nb: np.ndarray


def _one_pass(
    e: _Entries,
    bg: np.ndarray,
    row_sums_bg: np.ndarray,
    col_sums_bg: np.ndarray,
    tot_bg: float,
    x: np.ndarray,
    k: np.ndarray,
    cache: _Pass | None,
    padj_cutoff: float,
    padj_method: str,
    min_count: float,
    min_frac: float,
    exclude_empty: bool,
) -> _Pass:
    """One call of fishash's `fishash_internal`, in R's order of operations."""
    data, rows, cols = e.data, e.rows, e.cols
    if exclude_empty:
        n_rows = int(np.count_nonzero(e.row_sums > 0))
        n_cols = int(np.count_nonzero(e.col_sums > 0))
    else:
        n_rows, n_cols = e.n_grnas, e.n_cells
    n_entries = float(n_rows) * float(n_cols)

    cs_e = e.col_sums[cols]
    rs_e = (row_sums_bg[rows] - bg) + data
    tot_e = (tot_bg - col_sums_bg[cols]) + cs_e
    nr = np.rint(rs_e)
    nb = np.rint(tot_e - rs_e)
    if cache is None:
        log_pval = _log_phyper_rounded(x, nr, nb, k, lower_tail=False)
    else:
        # phyper is a pure function of its rounded arguments, and x and k never change.
        log_pval = cache.log_pval.copy()
        redo = np.flatnonzero((nr != cache.nr) | (nb != cache.nb))
        if redo.size:
            log_pval[redo] = _log_phyper_rounded(
                x[redo], nr[redo], nb[redo], k[redo], lower_tail=False
            )
    if np.isnan(log_pval).any():
        raise ValueError(
            "a log p-value is NaN: the noise estimate outside some cell sums to less than one "
            "count, which fishash cannot test (R returns NA here)"
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        inner = ((tot_e - rs_e) - cs_e) + data
        odds_ratio = ((data * inner) / (rs_e - data)) / (cs_e - data)
        odds_ratio_regularized = ((data * inner) / ((rs_e - data) + 1.0)) / ((cs_e - data) + 1.0)

    order = None
    if padj_method in ("BH", "BY"):
        if padj_method == "BY":
            cm = (math.log(n_entries) + 1.0 / (2.0 * n_entries)) + _EULER_TRUNCATED
        else:
            cm = 1.0
        o = np.argsort(log_pval, kind="stable")
        rank = np.arange(1, o.size + 1, dtype=np.float64)
        padj = np.minimum(((n_entries * cm) * np.exp(log_pval[o])) / rank, 1.0)
        padj = np.minimum.accumulate(padj[::-1])
        n_significant = int(np.count_nonzero(padj <= padj_cutoff))
        cutoff = ((math.log(padj_cutoff) - math.log(cm)) - math.log(n_entries)) + _log(
            n_significant
        )
        if int(np.count_nonzero(log_pval <= cutoff)) != n_significant:
            raise RuntimeError(
                "fishash's check sum(log_pval <= cutoff) == n_signif failed (R's stopifnot)"
            )
        order = o[::-1]
    else:
        starts = e.indptr[:-1]
        nonempty = np.diff(e.indptr) > 0
        colmin = np.zeros(e.n_cells)
        if log_pval.size:
            colmin[nonempty] = np.minimum(np.minimum.reduceat(log_pval, starts[nonempty]), 0.0)
        block_p = np.minimum(np.exp(colmin) * float(n_rows), 1.0)
        block_padj = _p_adjust_bh(block_p)
        n_significant = int(np.count_nonzero(block_padj <= padj_cutoff))
        cutoff = (math.log(padj_cutoff) - _log(n_entries)) + _log(n_significant)

    assigned = log_pval <= cutoff
    if min_count > 0:
        assigned &= data >= min_count
    if min_frac > 0:
        recip = 1.0 / np.maximum(e.col_sums, 1.0)
        assigned &= (data * recip[cols]) >= min_frac

    return _Pass(
        log_pval=log_pval,
        odds_ratio=odds_ratio,
        odds_ratio_regularized=odds_ratio_regularized,
        assigned=assigned,
        cutoff=cutoff,
        n_significant=n_significant,
        order=order,
        nr=nr,
        nb=nb,
    )


def _background_margins(background, e: _Entries):
    """Margins of a user-supplied background over its own pattern, and its values at `e`."""
    if sparse.issparse(background):
        b = sparse.csc_matrix(background, dtype=np.float64, copy=True)
    else:
        b = sparse.csc_matrix(np.asarray(background, dtype=np.float64))
    if b.shape != (e.n_grnas, e.n_cells):
        raise ValueError(
            f"background has shape {b.shape} but grna_matrix has shape {(e.n_grnas, e.n_cells)}"
        )
    b.sum_duplicates()
    b.sort_indices()
    if not np.all(np.isfinite(b.data)) or np.any(b.data < 0):
        raise ValueError("background must be finite and non-negative")
    b_rows = b.indices.astype(np.int64)
    b_cols = np.repeat(np.arange(e.n_cells, dtype=np.int64), np.diff(b.indptr))
    row_sums = np.bincount(b_rows, weights=b.data, minlength=e.n_grnas)
    col_sums = np.bincount(b_cols, weights=b.data, minlength=e.n_cells)
    values = np.asarray(b.tocsr()[e.rows, e.cols]).ravel() if e.data.size else np.zeros(0)
    return values, row_sums, col_sums, _seqsum(b.data)


def _assignment_strings(e: _Entries, assigned: np.ndarray, order: np.ndarray | None, ids):
    idx = np.flatnonzero(assigned) if order is None else order[assigned[order]]
    out = np.full(e.n_cells, "", dtype=object)
    if idx.size == 0:
        return out
    cells = e.cols[idx]
    perm = np.argsort(cells, kind="stable")
    idx, cells = idx[perm], cells[perm]
    bounds = np.flatnonzero(np.diff(cells)) + 1
    for group in np.split(np.arange(idx.size), bounds):
        out[cells[group[0]]] = ",".join(ids[r] for r in e.rows[idx[group]])
    return out


def _entry_matrix(e: _Entries, values: np.ndarray) -> sparse.csr_matrix:
    return sparse.csc_matrix((values, e.rows, e.indptr), shape=(e.n_grnas, e.n_cells)).tocsr()


def assign_grnas_fishash(
    grna_matrix: sparse.spmatrix | np.ndarray,
    grna_ids: Sequence[str],
    *,
    padj_cutoff: float = 0.05,
    padj_method: str = "GS",
    min_count: float = 2,
    min_frac: float = 0.0,
    refit: int = 10,
    exclude_empty: bool = True,
    background: sparse.spmatrix | np.ndarray | None = None,
) -> FishashResult:
    """Assign guides to cells with fishash.

    Port of fishash's `fishash()` with the same defaults. For each nonzero count it runs a
    one-sided Fisher exact test of whether guide and cell co-occur more than their totals
    predict, and assigns the entries that pass an FDR cut and the count and fraction floors.
    With `refit > 0`, each later pass masks the previous pass's assignments, refits the noise
    as a rank-one Poisson matrix (`impute_masked_counts`), and takes the off-cell margins from
    that fit; the loop stops early once the assignments stop changing.

    Args:
        grna_matrix: `(n_grnas, n_cells)` raw integer UMI counts, scipy.sparse or dense.
        grna_ids: The `n_grnas` row names, distinct.
        padj_cutoff: FDR level, in (0, 1).
        padj_method: "GS" (Guo and Sarkar's procedure, which treats tests within a cell as
            dependent and cells as independent), "BH" or "BY".
        min_count: An entry needs at least this many UMIs to be assigned (when positive).
        min_frac: An entry needs at least this fraction of its cell's UMIs (when positive).
        refit: Maximum number of refit passes after the first.
        exclude_empty: Count only guides and cells with a nonzero total in the number of tests.
        background: Optional `(n_grnas, n_cells)` matrix of noise counts used for the
            off-cell margins of the first pass; only allowed with `refit = 0`.

    Returns:
        A `FishashResult`.

    Raises:
        ValueError: invalid arguments, a matrix with no nonzero count, a background given with
            `refit > 0` (R refuses it too), or a NaN p-value (where R would return NA).
        RuntimeError: fishash's own BH/BY consistency check fails.
    """
    if padj_method not in _PADJ_METHODS:
        raise ValueError(f"padj_method must be one of {_PADJ_METHODS}, got {padj_method!r}")
    if not 0.0 < padj_cutoff < 1.0:
        raise ValueError(f"padj_cutoff must be in (0, 1), got {padj_cutoff}")
    if int(refit) != refit or refit < 0:
        raise ValueError(f"refit must be a non-negative integer, got {refit}")
    refit = int(refit)
    if background is not None and refit > 0:
        raise ValueError("a background cannot be combined with refit > 0 (fishash refuses it too)")

    counts = as_count_matrix(grna_matrix)
    ids = check_ids(grna_ids, counts.shape[0], "grna_ids")
    if counts.nnz == 0:
        raise ValueError("grna_matrix has no nonzero counts")
    e = _Entries.from_csc(counts)
    x = np.floor((e.data - 1.0) + 1e-7)
    k = np.rint(e.col_sums[e.cols])

    iterations: list[FishashIteration] = []
    mask = prev = None
    cache = None
    gf = cs = None
    for i in range(1, refit + 2):
        impute_n_iter = 0
        if i == 1:
            if background is None:
                bg, rs_bg, cs_bg, tot_bg = e.data, e.row_sums, e.col_sums, e.tot
            else:
                bg, rs_bg, cs_bg, tot_bg = _background_margins(background, e)
        else:
            bg, gf, cs, impute_n_iter = _impute_aligned(e, e.data, mask, eps=1e-4, max_iter=10)
            rs_bg = np.bincount(e.rows, weights=bg, minlength=e.n_grnas)
            cs_bg = np.bincount(e.cols, weights=bg, minlength=e.n_cells)
            tot_bg = _seqsum(bg)
        p = _one_pass(
            e, bg, rs_bg, cs_bg, tot_bg, x, k, cache,
            padj_cutoff, padj_method, min_count, min_frac, exclude_empty,
        )  # fmt: skip
        cache = p
        iterations.append(
            FishashIteration(
                log_pval_cutoff=p.cutoff,
                n_significant=p.n_significant,
                n_assigned=int(np.count_nonzero(p.assigned)),
                impute_n_iter=impute_n_iter,
            )
        )
        if i > 3:
            mask = mask | p.assigned
        else:
            mask = p.assigned.copy()
        if i > 1 and np.array_equal(prev, p.assigned):
            break
        prev = p.assigned

    n_assigned = np.bincount(e.cols[p.assigned], minlength=e.n_cells)
    demux_type = np.where(
        n_assigned == 1, "singlet", np.where(n_assigned > 1, "doublet", "unknown")
    )
    hit = np.flatnonzero(p.assigned)
    assigned = sparse.csr_matrix(
        (np.ones(hit.size, dtype=bool), (e.rows[hit], e.cols[hit])), shape=counts.shape
    )
    return FishashResult(
        assigned=assigned,
        log_pval=_entry_matrix(e, p.log_pval),
        odds_ratio=_entry_matrix(e, p.odds_ratio),
        odds_ratio_regularized=_entry_matrix(e, p.odds_ratio_regularized),
        demux_type=demux_type.astype(object),
        assignment=_assignment_strings(e, p.assigned, p.order, ids),
        log_pval_cutoff=float(p.cutoff),
        num_iter=i,
        grna_ids=ids,
        iterations=tuple(iterations),
        background_guide_freqs=gf,
        background_cell_sizes=cs,
    )
