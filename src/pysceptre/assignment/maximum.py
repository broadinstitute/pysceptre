"""sceptre's maximum assignment: each cell gets the gRNA with the most UMIs in it (low MOI).

Port of sceptre 0.10.3's `assign_grnas(method = "maximum")`, with each cell's top gRNA and its
share of the cell's gRNA UMIs computed as sceptre's `import_data` computes them; see
`docs/design.md`, "Thresholding and maximum".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ._common import as_count_matrix, check_ids

__all__ = ["MaximumResult", "assign_grnas_maximum"]


@dataclass(frozen=True)
class MaximumResult:
    """What `assign_grnas_maximum` returns.

    Attributes:
        assigned: `(n_grnas, n_cells)` boolean CSR matrix with exactly one assignment per cell,
            its top gRNA. A cell with no gRNA UMIs is assigned the first gRNA, as in sceptre;
            the UMI rule flags it.
        cells_w_zero_or_twoplus_grnas: Ascending 0-based int64 indices of the cells sceptre
            flags, and its low-MOI QC removes: the top gRNA holds at most
            `umi_fraction_threshold` of the cell's gRNA UMIs, or the cell has fewer than
            `min_grna_n_umis_threshold` gRNA UMIs.
        max_grna: `(n_cells,)` int64 row of each cell's top gRNA, the first on a tie.
        max_grna_frac_umis: `(n_cells,)` the top gRNA's share of the cell's gRNA UMIs; NaN for a
            cell with none.
        grna_n_umis: `(n_cells,)` the cell's gRNA UMIs.
        grna_ids: Row names.
    """

    assigned: sparse.csr_matrix
    cells_w_zero_or_twoplus_grnas: np.ndarray
    max_grna: np.ndarray
    max_grna_frac_umis: np.ndarray
    grna_n_umis: np.ndarray
    grna_ids: tuple[str, ...]


def assign_grnas_maximum(
    grna_matrix: sparse.spmatrix | np.ndarray,
    grna_ids: Sequence[str],
    *,
    umi_fraction_threshold: float = 0.8,
    min_grna_n_umis_threshold: float = 5,
) -> MaximumResult:
    """Assign gRNAs to cells with sceptre's maximum method, which sceptre allows in low MOI only.

    Every cell is assigned the gRNA with the most UMIs in it, the first in row order on a tie.
    A cell is flagged in `cells_w_zero_or_twoplus_grnas` when that gRNA holds at most
    `umi_fraction_threshold` of the cell's gRNA UMIs, or when the cell has fewer than
    `min_grna_n_umis_threshold` gRNA UMIs. Drop the flagged cells before a low-MOI analysis,
    as sceptre's QC does.

    Args:
        grna_matrix: `(n_grnas, n_cells)` raw integer gRNA UMI counts, scipy.sparse or dense.
        grna_ids: The `n_grnas` row names, distinct.
        umi_fraction_threshold: In (0, 1), as sceptre requires.
        min_grna_n_umis_threshold: At least 0, as sceptre requires.

    Returns:
        A `MaximumResult`.

    Raises:
        ValueError: invalid counts or ids, a matrix with no gRNA, `umi_fraction_threshold`
            outside (0, 1), or a negative or non-finite `min_grna_n_umis_threshold`.
    """
    frac_cut = float(umi_fraction_threshold)
    if not 0.0 < frac_cut < 1.0:
        raise ValueError(
            f"umi_fraction_threshold must be in (0, 1), got {umi_fraction_threshold!r}"
        )
    umi_cut = float(min_grna_n_umis_threshold)
    if not np.isfinite(umi_cut) or umi_cut < 0:
        raise ValueError(
            f"min_grna_n_umis_threshold must be a finite number >= 0, got {min_grna_n_umis_threshold!r}"
        )
    counts = as_count_matrix(grna_matrix)
    n_grnas, n_cells = counts.shape
    ids = check_ids(grna_ids, n_grnas, "grna_ids")
    if n_grnas == 0:
        raise ValueError("grna_matrix has no gRNA")

    per_cell = np.diff(counts.indptr)
    col = np.repeat(np.arange(n_cells), per_cell)
    occupied = np.flatnonzero(per_cell > 0)
    cell_max = np.zeros(n_cells)
    grna_n_umis = np.zeros(n_cells)
    if occupied.size:
        starts = counts.indptr[occupied]
        cell_max[occupied] = np.maximum.reduceat(counts.data, starts)
        grna_n_umis[occupied] = np.add.reduceat(counts.data, starts)
    at_max = np.flatnonzero(counts.data == cell_max[col])
    top_cells, first = np.unique(col[at_max], return_index=True)
    max_grna = np.zeros(n_cells, dtype=np.int64)
    max_grna[top_cells] = counts.indices[at_max[first]]
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = cell_max / grna_n_umis
    flagged = np.flatnonzero((frac <= frac_cut) | (grna_n_umis < umi_cut)).astype(np.int64)
    assigned = sparse.csr_matrix(
        (np.ones(n_cells, dtype=bool), (max_grna, np.arange(n_cells))),
        shape=(n_grnas, n_cells),
    )
    return MaximumResult(
        assigned=assigned,
        cells_w_zero_or_twoplus_grnas=flagged,
        max_grna=max_grna,
        max_grna_frac_umis=frac,
        grna_n_umis=grna_n_umis,
        grna_ids=ids,
    )
