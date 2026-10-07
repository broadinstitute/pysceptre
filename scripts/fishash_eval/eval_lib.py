"""Shared library for the fishash evaluation: loading, scoring, normalizations and callers.

Orientation: every matrix taken or returned here is pysceptre's guides x cells,
`(n_guides, n_cells)`, the transpose of AnnData's cells x guides. Inputs may be any scipy
sparse format (or a dense array); each function says which format it returns.

Importable, with no command line. numpy and scipy are the only module-level dependencies:
pyarrow is imported by the two loaders and scikit-learn by `call_gmm`, so the scoring, the
normalizations and the quantile callers run without either.

Nothing here densifies the whole matrix. Where a result must equal what dense numpy code gives
(the lab's CMO procedure and the quantile callers), the same numpy calls run on C-ordered
dense blocks of whole rows or columns, about `BLOCK_ELEMENTS` elements each unless a chunk size
says otherwise, so memory does not grow with the matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

__all__ = [
    "BLOCK_ELEMENTS",
    "Calls",
    "Dataset",
    "call_gmm",
    "call_q95",
    "call_q_oracle",
    "cmo_clr_q95",
    "confusion",
    "load_assigned",
    "load_dataset",
    "mean_infections",
    "norm_clr_cell",
    "norm_clr_guide",
    "norm_depth",
    "norm_noise_size",
    "norm_raw",
]

# Elements in one dense block when no chunk size is given: 8 MiB of float64.
BLOCK_ELEMENTS = 1 << 20


@dataclass(frozen=True)
class Dataset:
    """One simulated dataset as `simulate.R` exported it. Orientation: guides x cells.

    Attributes:
        counts: `(n_guides, n_cells)` CSC float64 UMI counts; the stored entries are exactly the
            nonzero counts, row indices sorted.
        truth: `(n_guides, n_cells)` CSC bool ground truth, True where the cell carries the
            guide. A true entry can sit at a zero count.
        signal: `(n_guides, n_cells)` CSC float64, the signal part of the counts
            (`counts_signal`), stored zeros dropped.
        meta: The dataset's `meta.json`.
        grna_ids: The row names, `"feature_1"` to `"feature_<n_guides>"`.
    """

    counts: sparse.csc_matrix
    truth: sparse.csc_matrix
    signal: sparse.csc_matrix
    meta: dict
    grna_ids: list[str]


@dataclass(frozen=True)
class Calls:
    """What a caller returns. Orientation: guides x cells.

    Attributes:
        assigned: `(n_guides, n_cells)` CSR bool, True where the guide is called in the cell.
        thresholds: `(n_guides,)` float64 per-guide cut, a value being called when strictly
            above it; NaN for `call_gmm`, which has no cut on the value scale.
        fitted: `(n_guides,)` bool; False where `call_gmm` skipped a guide (and called nothing).
            All True for the quantile callers.
        converged: `call_gmm` only, `(n_guides,)` bool: scikit-learn's `converged_` for each
            fitted guide, False where none was fitted. None for the quantile callers.
    """

    assigned: sparse.csr_matrix
    thresholds: np.ndarray
    fitted: np.ndarray
    converged: np.ndarray | None = None


# ---------------------------------------------------------------------------------------------
# Loading


def _read_triplets(path: Path, shape: tuple[int, int], *, logical: bool) -> sparse.csc_matrix:
    """A `(guide, cell[, value])` parquet as a guides x cells CSC matrix.

    The file must hold 0-based indices in column-major order (by cell, then guide) without
    duplicates, which is how `eval_common.R`'s `write_triplets` writes it.
    """
    import pyarrow.parquet as pq

    names = ["guide", "cell"] if logical else ["guide", "cell", "value"]
    table = pq.read_table(path, columns=names)
    for name in names:
        if table.column(name).null_count:
            raise ValueError(f"{path}: column {name!r} has nulls")
    guide = table.column("guide").to_numpy().astype(np.int64)
    cell = table.column("cell").to_numpy().astype(np.int64)
    n_guides, n_cells = shape
    if guide.size and (
        guide.min() < 0 or guide.max() >= n_guides or cell.min() < 0 or cell.max() >= n_cells
    ):
        raise ValueError(f"{path}: an index is outside the shape {shape}")
    key = cell * n_guides + guide
    if np.any(np.diff(key) <= 0):
        raise ValueError(f"{path}: entries are not in column-major order, or repeat")
    indptr = np.zeros(n_cells + 1, dtype=np.int64)
    np.cumsum(np.bincount(cell, minlength=n_cells), out=indptr[1:])
    if logical:
        data = np.ones(guide.size, dtype=bool)
    else:
        data = table.column("value").to_numpy().astype(np.float64)
    m = sparse.csc_matrix((data, guide, indptr), shape=shape)
    if not logical:
        m.eliminate_zeros()
    return m


def load_dataset(dataset_dir) -> Dataset:
    """Load one exported dataset directory. Orientation: guides x cells.

    Reads `meta.json`, `counts.parquet`, `ground_truth.parquet` and `counts_signal.parquet`
    from `dataset_dir` (one `<sim_label>/` directory under `sims/<scenario>/`) with pyarrow,
    and checks the counts and truth entry numbers against `meta.json`.

    Args:
        dataset_dir: The dataset's directory.

    Returns:
        A `Dataset` whose matrices are `(n_guides, n_cells)` CSC.

    Raises:
        ValueError: a file is out of order, out of shape, or disagrees with `meta.json`.
    """
    d = Path(dataset_dir)
    meta = json.loads((d / "meta.json").read_text())
    shape = (int(meta["n_guides"]), int(meta["n_cells"]))
    counts = _read_triplets(d / "counts.parquet", shape, logical=False)
    truth = _read_triplets(d / "ground_truth.parquet", shape, logical=True)
    signal = _read_triplets(d / "counts_signal.parquet", shape, logical=False)
    if counts.nnz != int(meta["nnz_counts"]):
        raise ValueError(f"{d}: {counts.nnz} nonzero counts, meta.json says {meta['nnz_counts']}")
    if truth.nnz != int(meta["nnz_truth"]):
        raise ValueError(f"{d}: {truth.nnz} true entries, meta.json says {meta['nnz_truth']}")
    grna_ids = [f"feature_{i}" for i in range(1, shape[0] + 1)]
    return Dataset(counts=counts, truth=truth, signal=signal, meta=meta, grna_ids=grna_ids)


def load_assigned(run_dir, shape) -> sparse.csr_matrix:
    """Load a run's `assigned.parquet` as a guides x cells CSR bool matrix.

    Args:
        run_dir: The run's directory, holding `assigned.parquet` with 0-based `guide` and
            `cell` columns. Their order is not checked.
        shape: `(n_guides, n_cells)`.

    Returns:
        `(n_guides, n_cells)` CSR bool, True at every listed entry.

    Raises:
        ValueError: an index is outside `shape`, or an entry is listed twice.
    """
    import pyarrow.parquet as pq

    path = Path(run_dir) / "assigned.parquet"
    table = pq.read_table(path, columns=["guide", "cell"])
    guide = table.column("guide").to_numpy().astype(np.int64)
    cell = table.column("cell").to_numpy().astype(np.int64)
    n_guides, n_cells = (int(s) for s in shape)
    if guide.size and (
        guide.min() < 0 or guide.max() >= n_guides or cell.min() < 0 or cell.max() >= n_cells
    ):
        raise ValueError(f"{path}: an index is outside the shape {(n_guides, n_cells)}")
    m = _bool_csr(guide, cell, (n_guides, n_cells))
    if m.nnz != guide.size:
        raise ValueError(f"{path}: {guide.size - m.nnz} entries are listed more than once")
    return m


# ---------------------------------------------------------------------------------------------
# Scoring


def _pattern(m, name: str, *, positive: bool = False) -> sparse.csr_matrix:
    """Where `m` is nonzero (or > 0, with `positive`), as a canonical CSR bool matrix."""
    if sparse.issparse(m):
        c = sparse.csr_matrix(m, copy=True)
    else:
        c = sparse.csr_matrix(np.asarray(m))
    if c.ndim != 2:
        raise ValueError(f"{name} must be 2-D")
    c.sum_duplicates()
    c.data = c.data > 0 if positive else c.data != 0
    c.eliminate_zeros()
    return c


def _scores(tp: int, n_true: int, n_est: int, total: int) -> dict[str, float]:
    fn = n_true - tp
    fp = n_est - tp
    tn = total - tp - fn - fp
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    s = precision + recall
    f1 = 2.0 * precision * recall / s if s > 0 else float("nan")
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _nnz(m: sparse.csr_matrix) -> int:
    return int(np.count_nonzero(m.data))


def confusion(assigned, truth, counts) -> dict[str, dict[str, float]]:
    """Confusion counts and scores of `assigned` against `truth`, as the fishash paper scores them.

    Port of `get_confusion` in the fishash analysis repository's `bin/process_assignments.R`,
    on its two subsets of entries:

    - "full": all `n_guides * n_cells` entries;
    - "nonzero": the entries whose count is > 0. A true entry at a zero count, and an
      assignment at a zero count, are outside this subset.

    TP counts entries both assigned and true, FN = (true entries) - TP, FP = (assigned
    entries) - TP and TN = (entries in the subset) - TP - FN - FP, all exact integers computed
    on sparse patterns. Precision = TP / (TP + FP) and recall = TP / (TP + FN) are NaN on
    0 / 0, as R gives; F1 = 2PR / (P + R) is NaN when P or R is NaN or P + R = 0.

    Args:
        assigned: `(n_guides, n_cells)` assignments; a nonzero entry is an assignment.
        truth: `(n_guides, n_cells)` ground truth; a nonzero entry is true.
        counts: `(n_guides, n_cells)` UMI counts, which define the "nonzero" subset.

    Returns:
        `{"full": {...}, "nonzero": {...}}`, each with int "TP", "FP", "FN", "TN" and float
        "precision", "recall", "f1".

    Raises:
        ValueError: the shapes differ.
    """
    est = _pattern(assigned, "assigned")
    tru = _pattern(truth, "truth")
    c = _pattern(counts, "counts", positive=True)
    if not est.shape == tru.shape == c.shape:
        raise ValueError(
            f"shapes differ: assigned {est.shape}, truth {tru.shape}, counts {c.shape}"
        )
    hit = est.multiply(tru).tocsr()
    n_guides, n_cells = est.shape
    full = _scores(_nnz(hit), _nnz(tru), _nnz(est), int(n_guides) * int(n_cells))
    nonzero = _scores(
        _nnz(hit.multiply(c).tocsr()),
        _nnz(tru.multiply(c).tocsr()),
        _nnz(est.multiply(c).tocsr()),
        _nnz(c),
    )
    return {"full": full, "nonzero": nonzero}


def mean_infections(moi, hurdle: float = 0.1):
    """Mean number of guides per cell in fishash's simulator, at a given MOI.

    fishash's `simulate_guidebender2` sets `avg_infections <- moi/(1 - exp(-moi)) *
    (1 - hurdle_prob)`: a cell carries a zero-truncated Poisson(moi) number of guides, set to
    zero with probability `hurdle_prob`. Same operations, same order. The fishash paper's
    simulations use `hurdle_prob = 0.1`.

    Args:
        moi: Positive scalar or array.
        hurdle: The hurdle probability.

    Returns:
        A float for a scalar `moi`, else an array of its shape.
    """
    m = np.asarray(moi, dtype=np.float64)
    out = m / (1.0 - np.exp(-m)) * (1.0 - hurdle)
    return float(out) if out.ndim == 0 else out


# ---------------------------------------------------------------------------------------------
# Normalizations: guides x cells counts in, a CSR float64 matrix with the same pattern out.


def _canonical_csr(m, name: str) -> sparse.csr_matrix:
    """`m` as a canonical CSR float64 copy: duplicates summed, stored zeros dropped, sorted."""
    if sparse.issparse(m):
        c = sparse.csr_matrix(m, dtype=np.float64, copy=True)
    else:
        a = np.asarray(m)
        if a.ndim != 2:
            raise ValueError(f"{name} must be 2-D, got shape {a.shape}")
        c = sparse.csr_matrix(a.astype(np.float64))
    if c.ndim != 2:
        raise ValueError(f"{name} must be 2-D")
    c.sum_duplicates()
    c.eliminate_zeros()
    c.sort_indices()
    if not np.all(np.isfinite(c.data)):
        raise ValueError(f"{name} has non-finite values")
    return c


def _counts_csr(counts, name: str = "counts") -> sparse.csr_matrix:
    c = _canonical_csr(counts, name)
    if np.any(c.data < 0):
        raise ValueError(f"{name} has negative values")
    if c.shape[0] == 0 or c.shape[1] == 0:
        raise ValueError(f"{name} has shape {c.shape}; it needs at least one guide and one cell")
    return c


def _chunk(chunk: int | None, width: int) -> int:
    """Rows per dense block of `width` columns: `chunk`, or as many as `BLOCK_ELEMENTS` allows."""
    if chunk is None:
        return max(1, BLOCK_ELEMENTS // max(int(width), 1))
    if int(chunk) != chunk or chunk < 1:
        raise ValueError(f"a chunk size must be a positive integer, got {chunk!r}")
    return int(chunk)


def _clr_rows(m: sparse.csr_matrix, chunk_rows: int | None):
    """The lab's CLR along each row of the canonical CSR `m`, on dense blocks of whole rows.

    For row r of the `(n_rows, n_cols)` matrix: `gm_r = exp(sum_c log1p(m_rc) / n_cols)` and
    `log1p(m_rc / gm_r)`, by the verbatim numpy calls on C-ordered `(rows, n_cols)` blocks, so
    every value equals what those calls give on the whole dense matrix in C order.

    Returns:
        `(clr, gm, top)`: `clr` CSR with `m`'s pattern (an absent entry is exactly 0.0 in the
        dense result), `gm` `(n_rows,)`, and `top` `(n_rows,)`, the verbatim `argmax(axis=1)`.
    """
    n_rows, n_cols = m.shape
    step = _chunk(chunk_rows, n_cols)
    gm = np.empty(n_rows, dtype=np.float64)
    top = np.empty(n_rows, dtype=np.intp)
    clr_data = np.empty(m.nnz, dtype=np.float64)
    for r0 in range(0, n_rows, step):
        r1 = min(r0 + step, n_rows)
        _cmo = m[r0:r1].toarray(order="C")
        _log_counts = np.log1p(np.where(_cmo > 0, _cmo, 0))
        _gm = np.exp(np.sum(_log_counts, axis=1) / n_cols)
        clr = np.log1p(_cmo / _gm[:, None])
        gm[r0:r1] = _gm
        top[r0:r1] = np.argmax(clr, axis=1)
        lo, hi = m.indptr[r0], m.indptr[r1]
        local = np.repeat(np.arange(r1 - r0), np.diff(m.indptr[r0 : r1 + 1]))
        clr_data[lo:hi] = clr[local, m.indices[lo:hi]]
    out = sparse.csr_matrix((clr_data, m.indices.copy(), m.indptr.copy()), shape=m.shape)
    return out, gm, top


def norm_raw(counts) -> sparse.csr_matrix:
    """`log1p(x)`. Orientation: guides x cells in, `(n_guides, n_cells)` CSR float64 out.

    Args:
        counts: `(n_guides, n_cells)` non-negative counts.

    Returns:
        The transformed values on the counts' nonzero pattern; a zero count stays zero.
    """
    c = _counts_csr(counts)
    c.data = np.log1p(c.data)
    return c


def norm_clr_cell(counts, *, chunk_cells: int | None = None) -> sparse.csr_matrix:
    """The lab's per-cell CLR. Orientation: guides x cells in, `(n_guides, n_cells)` CSR out.

    For each cell c, `gm_c = exp(sum_g log1p(x_gc) / n_guides)` over all `n_guides` guides
    (zeros included), then `log1p(x_gc / gm_c)`. The values are those of `cmo_clr_q95`'s
    "clr", bit for bit: both come from the same dense row blocks of cells x all guides.

    Args:
        counts: `(n_guides, n_cells)` non-negative counts, cast to float64.
        chunk_cells: Cells per dense block; default from `BLOCK_ELEMENTS`.

    Returns:
        CSR float64 on the counts' nonzero pattern; a zero count stays zero.
    """
    c = _counts_csr(counts)
    clr, _, _ = _clr_rows(c.T.tocsr(), chunk_cells)
    return clr.T.tocsr()


def norm_clr_guide(counts, *, chunk_guides: int | None = None) -> sparse.csr_matrix:
    """The same CLR along the other margin, per guide. Orientation: guides x cells in and out.

    For each guide g, `gm_g = exp(sum_c log1p(x_gc) / n_cells)` over all `n_cells` cells
    (zeros included), then `log1p(x_gc / gm_g)`: the lab's CMO formula applied to the guides x
    cells matrix, computed on dense blocks of guides x all cells. Within a guide the transform
    is strictly increasing.

    Args:
        counts: `(n_guides, n_cells)` non-negative counts, cast to float64.
        chunk_guides: Guides per dense block; default from `BLOCK_ELEMENTS`.

    Returns:
        `(n_guides, n_cells)` CSR float64 on the counts' nonzero pattern.
    """
    c = _counts_csr(counts)
    clr, _, _ = _clr_rows(c, chunk_guides)
    return clr


def norm_depth(counts) -> sparse.csr_matrix:
    """Depth normalization, `log1p(x * median(N) / N_c)`. Orientation: guides x cells.

    `N_c` is cell c's total count over all guides and `median(N)` the median of the totals of
    all cells, empty cells included.

    Args:
        counts: `(n_guides, n_cells)` non-negative counts.

    Returns:
        `(n_guides, n_cells)` CSR float64 on the counts' nonzero pattern.
    """
    c = _counts_csr(counts)
    totals = np.bincount(c.indices, weights=c.data, minlength=c.shape[1])
    med = float(np.median(totals))
    c.data = np.log1p(c.data * med / totals[c.indices])
    return c


def norm_noise_size(counts, cell_sizes) -> sparse.csr_matrix:
    """`log1p(x / kappa_c)`, kappa from fishash's noise fit. Orientation: guides x cells.

    `cell_sizes` is `FishashResult.background_cell_sizes`: each cell's size in fishash's
    rank-one fit to the counts left after its calls. A cell whose size is not positive and
    finite (fishash sets 0 where every count of the cell was called) keeps `log1p(x)`.

    Args:
        counts: `(n_guides, n_cells)` non-negative counts.
        cell_sizes: `(n_cells,)` noise sizes.

    Returns:
        `(n_guides, n_cells)` CSR float64 on the counts' nonzero pattern.

    Raises:
        ValueError: `cell_sizes` is None or not of length `n_cells`.
    """
    c = _counts_csr(counts)
    if cell_sizes is None:
        raise ValueError("cell_sizes is None: fishash ran no refit pass")
    kappa = np.asarray(cell_sizes, dtype=np.float64)
    if kappa.shape != (c.shape[1],):
        raise ValueError(f"cell_sizes has shape {kappa.shape}, expected ({c.shape[1]},)")
    scale = np.where(np.isfinite(kappa) & (kappa > 0), kappa, 1.0)
    c.data = np.log1p(c.data / scale[c.indices])
    return c


# ---------------------------------------------------------------------------------------------
# The lab's CMO procedure and the callers


def _bool_csr(rows: np.ndarray, cols: np.ndarray, shape) -> sparse.csr_matrix:
    m = sparse.csr_matrix(
        (np.ones(rows.size, dtype=bool), (rows.astype(np.int64), cols.astype(np.int64))),
        shape=shape,
    )
    m.sum_duplicates()
    return m


def _threshold_calls(v: sparse.csr_matrix, q, chunk_guides: int | None):
    """Per-guide `np.quantile` over all cells and the strict `>` call, on dense guide blocks.

    `q` is a scalar, taken by one `np.quantile(block, q, axis=0)` per block exactly as the
    lab's threshold step does, or a `(n_guides,)` array, one quantile per guide. Each block is
    a C-ordered `(n_cells, guides)` slice with every cell, zeros included.

    Returns:
        `(thresholds (n_guides,), assigned (n_guides, n_cells) CSR bool)`.
    """
    n_guides, n_cells = v.shape
    if n_guides == 0 or n_cells == 0:
        raise ValueError(f"values has shape {v.shape}; it needs at least one guide and one cell")
    step = _chunk(chunk_guides, n_cells)
    per_guide = np.ndim(q) > 0
    thresholds = np.empty(n_guides, dtype=np.float64)
    hit_guides: list[np.ndarray] = []
    hit_cells: list[np.ndarray] = []
    for g0 in range(0, n_guides, step):
        g1 = min(g0 + step, n_guides)
        block = v[g0:g1].T.toarray(order="C")
        if per_guide:
            thr = np.array([np.quantile(block[:, j], q[g0 + j]) for j in range(g1 - g0)])
        else:
            thr = np.quantile(block, q, axis=0)
        positive = block > thr
        cells, guides = np.nonzero(positive)
        thresholds[g0:g1] = thr
        hit_guides.append(guides + g0)
        hit_cells.append(cells)
    assigned = _bool_csr(np.concatenate(hit_guides), np.concatenate(hit_cells), v.shape)
    return thresholds, assigned


def cmo_clr_q95(counts, *, chunk_cells: int | None = None, chunk_guides: int | None = None):
    """The lab's CMO demultiplexing (per-cell CLR, then a 95th-percentile cut per tag).

    Orientation: guides x cells in and out. The lab's code takes a dense cells x tags matrix
    and divides by its hard-coded 25 tags; here the divisor is `n_guides`. Its steps:

    1. `gm_c = exp(sum_g log1p(x_gc) / n_guides)` per cell and `clr = log1p(x / gm_c)`;
    2. per guide, `threshold = np.quantile(clr, 0.95)` over all cells, zeros included;
    3. `positive = clr > threshold` (strict), `n_pos` its count per cell, `top_idx` the
       per-cell `argmax` of `clr`, and status "Negative" (n_pos 0), "Singlet" (1) or
       "Doublet" (more).

    Steps 1 and the argmax run on C-ordered dense blocks of cells x all guides, steps 2 and 3
    on dense blocks of all cells x a few guides, with the lab's numpy calls, so every value,
    tie and comparison equals what the lab's code gives on `counts.T` as a C-ordered float64
    dense matrix (what `.toarray()` returns for a CSR AnnData X). Integer counts give the same
    floats as the lab's code on the integer matrix: numpy casts them to float64 exactly.

    Args:
        counts: `(n_guides, n_cells)` non-negative counts, cast to float64.
        chunk_cells: Cells per dense block in step 1; default from `BLOCK_ELEMENTS`.
        chunk_guides: Guides per dense block in steps 2 and 3; default from `BLOCK_ELEMENTS`.

    Returns:
        A dict with "clr" (`(n_guides, n_cells)` CSR float64 on the counts' pattern), "clr_gm"
        `(n_cells,)`, "thresholds" `(n_guides,)`, "positive" (`(n_guides, n_cells)` CSR bool),
        "n_pos" `(n_cells,)` int64, "top_idx" `(n_cells,)` guide index, and "status"
        `(n_cells,)` str.
    """
    c = _counts_csr(counts)
    clr_cells, gm, top = _clr_rows(c.T.tocsr(), chunk_cells)
    clr = clr_cells.T.tocsr()
    thresholds, positive = _threshold_calls(clr, 0.95, chunk_guides)
    n_pos = np.bincount(positive.indices, minlength=c.shape[1]).astype(np.int64)
    status = np.where(n_pos == 0, "Negative", np.where(n_pos == 1, "Singlet", "Doublet"))
    return {
        "clr": clr,
        "clr_gm": gm,
        "thresholds": thresholds,
        "positive": positive,
        "n_pos": n_pos,
        "top_idx": top,
        "status": status,
    }


def call_q95(values, q: float = 0.95, *, chunk_guides: int | None = None) -> Calls:
    """Call a guide in the cells strictly above its q-quantile. Orientation: guides x cells.

    The lab's threshold step on any normalized matrix: per guide, `np.quantile` at `q` over
    all `n_cells` values (zeros included, numpy's default linear interpolation), and a call
    where the value is strictly greater. Given `cmo_clr_q95`'s "clr" it returns that
    function's thresholds and positives exactly.

    Args:
        values: `(n_guides, n_cells)` normalized values (zero where the count is zero).
        q: The quantile, in [0, 1].
        chunk_guides: Guides per dense block; default from `BLOCK_ELEMENTS`.

    Returns:
        `Calls` with `(n_guides, n_cells)` CSR bool `assigned` and the thresholds.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    v = _canonical_csr(values, "values")
    thresholds, assigned = _threshold_calls(v, float(q), chunk_guides)
    return Calls(assigned=assigned, thresholds=thresholds, fitted=np.ones(v.shape[0], bool))


def call_q_oracle(values, truth, *, chunk_guides: int | None = None) -> Calls:
    """The quantile cut placed with the truth: a diagnostic ceiling. Orientation: guides x cells.

    Per guide g, `np.quantile` over all `n_cells` values at `1 - k_g / n_cells`, where `k_g`
    is the number of cells truly carrying g (zero counts included), and a call strictly above
    it. Without ties this calls the guide's `k_g` highest cells; a guide with no true cell gets
    its maximum as the cut and no call.

    Args:
        values: `(n_guides, n_cells)` normalized values.
        truth: `(n_guides, n_cells)` ground truth; a nonzero entry is true.
        chunk_guides: Guides per dense block; default from `BLOCK_ELEMENTS`.

    Returns:
        `Calls` with `(n_guides, n_cells)` CSR bool `assigned` and the thresholds.

    Raises:
        ValueError: the shapes differ.
    """
    v = _canonical_csr(values, "values")
    t = _pattern(truth, "truth")
    if t.shape != v.shape:
        raise ValueError(f"truth has shape {t.shape} but values has shape {v.shape}")
    n_true = np.diff(t.indptr)
    q = 1.0 - n_true / v.shape[1]
    thresholds, assigned = _threshold_calls(v, q, chunk_guides)
    return Calls(assigned=assigned, thresholds=thresholds, fitted=np.ones(v.shape[0], bool))


def call_gmm(values, *, nonzero: bool = True, random_state: int = 0) -> Calls:
    """Two-component Gaussian mixture per guide, crispat's EM model. Orientation: guides x cells.

    Per guide, scikit-learn's `GaussianMixture(n_components=2, n_init=10,
    covariance_type="tied", random_state=random_state)` is fitted to a `(n, 1)` float64 column
    in cell order: the guide's nonzero values when `nonzero` is True, else all `n_cells`
    values, zeros included. A cell is called when its posterior for the component with the
    higher mean is > 0.5. With `nonzero=True` a zero cell is never called.

    With `nonzero=False` and the values `log10(x + 1)`, this is crispat 0.9.10's `fit_em`
    (`crispat.gauss`) guide by guide, except that a guide with fewer than 2 nonzero values, or
    whose fitted values are all equal, is skipped and calls nothing, where crispat would still
    fit it. crispat's own `fit_em(nonzero=True)` cannot run: it hands scikit-learn a 1-D array.
    Fits run with one OpenMP thread, so the result does not depend on the core count.

    Args:
        values: `(n_guides, n_cells)` normalized values.
        nonzero: Fit and call the nonzero values only.
        random_state: Passed to `GaussianMixture`.

    Returns:
        `Calls` with `(n_guides, n_cells)` CSR bool `assigned`, NaN thresholds, and per-guide
        `fitted` and `converged`.
    """
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits

    v = _canonical_csr(values, "values")
    n_guides, n_cells = v.shape
    fitted = np.zeros(n_guides, dtype=bool)
    converged = np.zeros(n_guides, dtype=bool)
    hit_guides: list[np.ndarray] = []
    hit_cells: list[np.ndarray] = []
    with threadpool_limits(limits=1, user_api="openmp"):
        for g in range(n_guides):
            lo, hi = v.indptr[g], v.indptr[g + 1]
            cells = v.indices[lo:hi]
            vals = v.data[lo:hi]
            if np.count_nonzero(vals) < 2:
                continue
            if nonzero:
                x = np.ascontiguousarray(vals, dtype=np.float64).reshape(-1, 1)
                where = cells
            else:
                x = v[g].toarray(order="C").reshape(-1, 1)
                where = np.arange(n_cells)
            if x.min() == x.max():
                continue
            gmm = GaussianMixture(
                n_components=2, n_init=10, covariance_type="tied", random_state=random_state
            )
            gmm.fit(x)
            posterior = gmm.predict_proba(x)
            high = int(np.argmax(gmm.means_))
            called = where[posterior[:, high] > 0.5]
            fitted[g] = True
            converged[g] = bool(gmm.converged_)
            hit_guides.append(np.full(called.size, g, dtype=np.int64))
            hit_cells.append(called.astype(np.int64))
    if hit_guides:
        assigned = _bool_csr(np.concatenate(hit_guides), np.concatenate(hit_cells), v.shape)
    else:
        assigned = sparse.csr_matrix(v.shape, dtype=bool)
    return Calls(
        assigned=assigned,
        thresholds=np.full(n_guides, np.nan),
        fitted=fitted,
        converged=converged,
    )
