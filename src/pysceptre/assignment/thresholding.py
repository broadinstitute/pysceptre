"""sceptre's thresholding assignment: a gRNA goes to every cell where its count reaches a cut.

Port of sceptre 0.10.3's `assign_grnas(method = "thresholding")`; see `docs/design.md`,
"Thresholding and maximum".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ._common import as_count_matrix, check_ids

__all__ = ["ThresholdingResult", "assign_grnas_thresholding"]


@dataclass(frozen=True)
class ThresholdingResult:
    """What `assign_grnas_thresholding` returns.

    Attributes:
        assigned: `(n_grnas, n_cells)` boolean CSR matrix of assignments.
        threshold: The count a gRNA needed in a cell.
        grna_ids: Row names.
    """

    assigned: sparse.csr_matrix
    threshold: float
    grna_ids: tuple[str, ...]


def assign_grnas_thresholding(
    grna_matrix: sparse.spmatrix | np.ndarray,
    grna_ids: Sequence[str],
    *,
    threshold: float = 5,
) -> ThresholdingResult:
    """Assign gRNAs to cells with sceptre's thresholding method.

    A gRNA is assigned to every cell in which its UMI count is at least `threshold`.

    Args:
        grna_matrix: `(n_grnas, n_cells)` raw integer gRNA UMI counts, scipy.sparse or dense.
        grna_ids: The `n_grnas` row names, distinct.
        threshold: The count a gRNA needs in a cell; at least 1, as sceptre requires.

    Returns:
        A `ThresholdingResult`.

    Raises:
        ValueError: invalid counts or ids, or a `threshold` that is not a finite number >= 1.
    """
    t = float(threshold)
    if not np.isfinite(t) or t < 1:
        raise ValueError(f"threshold must be a finite number >= 1, got {threshold!r}")
    counts = as_count_matrix(grna_matrix).tocsr()
    counts.sort_indices()
    ids = check_ids(grna_ids, counts.shape[0], "grna_ids")
    assigned = sparse.csr_matrix(
        (counts.data >= t, counts.indices.copy(), counts.indptr.copy()),
        shape=counts.shape,
        dtype=bool,
    )
    assigned.eliminate_zeros()
    return ThresholdingResult(assigned=assigned, threshold=t, grna_ids=ids)
