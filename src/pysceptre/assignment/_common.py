"""Input checks shared by the assignment ports."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse


def as_count_matrix(m, *, name: str = "grna_matrix") -> sparse.csc_matrix:
    """`m` as a canonical CSC float64 matrix of UMI counts.

    Duplicate entries are summed, stored zeros dropped and row indices sorted, so the stored
    entries are exactly the nonzero counts in column-major order, the order R's
    `TsparseMatrix` gives them.

    Raises:
        ValueError: `m` is not 2-D, or holds a negative, non-finite or non-integer value.
    """
    if sparse.issparse(m):
        if m.ndim != 2:
            raise ValueError(f"{name} must be 2-D, got {m.ndim} dimensions")
        c = sparse.csc_matrix(m, dtype=np.float64, copy=True)
    else:
        a = np.asarray(m)
        if a.ndim != 2:
            raise ValueError(f"{name} must be 2-D, got shape {a.shape}")
        c = sparse.csc_matrix(a.astype(np.float64))
    c.sum_duplicates()
    c.eliminate_zeros()
    c.sort_indices()
    d = c.data
    if not np.all(np.isfinite(d)):
        raise ValueError(f"{name} has non-finite values")
    if np.any(d < 0):
        raise ValueError(f"{name} has negative values")
    if np.any(d != np.floor(d)):
        raise ValueError(f"{name} must hold integer UMI counts; it has non-integer values")
    return c


def check_ids(ids: Sequence[str], n: int, name: str) -> tuple[str, ...]:
    """`ids` as a tuple of `n` distinct strings.

    Raises:
        ValueError: wrong length or duplicates.
    """
    out = tuple(str(i) for i in ids)
    if len(out) != n:
        raise ValueError(f"{name} has {len(out)} entries but the matrix has {n} rows")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} has duplicate entries")
    return out
