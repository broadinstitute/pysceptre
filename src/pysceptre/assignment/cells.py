"""From an assignment matrix to the cell sets the discovery engine takes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from ._common import check_ids

__all__ = ["cells_by_grna", "cells_by_target", "cells_w_zero_or_twoplus_grnas"]

_NON_TARGETING = "non-targeting"


def cells_by_grna(
    assigned: sparse.spmatrix | np.ndarray, grna_ids: Sequence[str]
) -> dict[str, np.ndarray]:
    """Each gRNA's assigned cells.

    Args:
        assigned: `(n_grnas, n_cells)` boolean matrix, scipy.sparse or dense, as returned in
            `FishashResult.assigned` or `MixtureResult.assigned`.
        grna_ids: The row names.

    Returns:
        `{grna_id: ascending 0-based int64 cell indices}`, in `grna_ids` order, an empty array
        for a gRNA assigned to no cell.
    """
    a = sparse.csr_matrix(assigned, dtype=bool)
    a.eliminate_zeros()
    a.sort_indices()
    ids = check_ids(grna_ids, a.shape[0], "grna_ids")
    return {g: a.indices[a.indptr[r] : a.indptr[r + 1]].astype(np.int64) for r, g in enumerate(ids)}


def cells_by_target(
    grna_cells: dict[str, np.ndarray], grna_target_data_frame: pd.DataFrame
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Group per-gRNA cell sets into the two dicts `run_discovery_analysis` takes.

    Args:
        grna_cells: `{grna_id: cell indices}`, as from `cells_by_grna`.
        grna_target_data_frame: Columns `grna_id` and `grna_target`; a target of
            "non-targeting" marks a non-targeting gRNA.

    Returns:
        `(grna_target_cells, ntc_grna_cells)`. The first maps each targeting target to the union
        of its gRNAs' cells (ascending), keys in order of first appearance in the data frame; the
        second maps each non-targeting gRNA to its own cells. A gRNA in the data frame but not in
        `grna_cells` contributes no cells.

    Raises:
        KeyError: a required column is missing.
    """
    missing = [c for c in ("grna_id", "grna_target") if c not in grna_target_data_frame.columns]
    if missing:
        raise KeyError(f"grna_target_data_frame is missing column(s) {missing}")
    design = grna_target_data_frame.loc[:, ["grna_id", "grna_target"]].astype(str)
    empty = np.zeros(0, dtype=np.int64)
    targeting: dict[str, list[np.ndarray]] = {}
    ntc: dict[str, np.ndarray] = {}
    for grna_id, target in design.itertuples(index=False):
        cells = np.asarray(grna_cells.get(grna_id, empty), dtype=np.int64)
        if target == _NON_TARGETING:
            ntc[grna_id] = np.unique(cells)
        else:
            targeting.setdefault(target, []).append(cells)
    grna_target_cells = {t: np.unique(np.concatenate(parts)) for t, parts in targeting.items()}
    return grna_target_cells, ntc


def cells_w_zero_or_twoplus_grnas(assigned: sparse.spmatrix | np.ndarray) -> np.ndarray:
    """The cells sceptre's low-MOI QC removes after a thresholding, mixture or fishash assignment.

    `assign_grnas_maximum` returns its own set, which follows other rules.

    Args:
        assigned: `(n_grnas, n_cells)` boolean matrix, scipy.sparse or dense.

    Returns:
        Ascending 0-based int64 indices of the cells assigned no gRNA, or two or more.
    """
    a = sparse.csc_matrix(assigned, dtype=bool)
    a.eliminate_zeros()
    return np.flatnonzero(np.diff(a.indptr) != 1).astype(np.int64)
