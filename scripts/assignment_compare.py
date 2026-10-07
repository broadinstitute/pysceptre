"""Compare two calls of the same assignment entry by entry, and say why any entry differs.

Used by tests/validation/test_fishash_vs_r.py and scripts/fishash_eval/compare_py_r.py. An
entry assigned by exactly one of the two calls is a flip, classified as

    boundary   the reference log p-value lies within the comparison tolerance of the reference
               cutoff, so a last-bit difference in either could move the entry across it;
    defect     anything else -- a real disagreement to investigate.

compare_py_r.py refines the classes with pass-level evidence only it has (B-flip,
half-rounding). Orientation is pysceptre's: (n_grnas, n_cells).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

# A log p-value is compared to rel * max(1, |log p|). Counts of one get a looser band: their
# p-value is P(X >= 1) = 1 - P(X = 0), the case where a last-bit difference matters most.
REL_LOG_P = 1e-12
REL_LOG_P_COUNT1 = 1e-9


def log_p_tolerance(counts, log_p, *, rel=REL_LOG_P, rel_count1=REL_LOG_P_COUNT1) -> np.ndarray:
    """Absolute tolerance on each log p-value, from its count and magnitude."""
    counts = np.asarray(counts, dtype=np.float64)
    log_p = np.asarray(log_p, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        scale = np.maximum(1.0, np.abs(log_p))
    return np.where(counts == 1, rel_count1, rel) * scale


def flips(
    assigned_ref,
    assigned_new,
    *,
    counts=None,
    log_p_ref=None,
    cutoff_ref: float | None = None,
) -> pd.DataFrame:
    """Every entry assigned by exactly one of two calls, classified.

    Args:
        assigned_ref: `(n_grnas, n_cells)` sparse boolean, the reference call (R).
        assigned_new: The same shape, the call under test (Python).
        counts: Optional counts, to report each flip's count and choose its tolerance.
        log_p_ref: Optional reference log p-values (sparse, at the counts' entries).
        cutoff_ref: Optional reference cut on the log p-value.

    Returns:
        One row per flip: guide, cell, in_ref, in_new, count, log_p_ref, cutoff_ref, margin,
        tolerance, flip_class. Empty when the calls agree.
    """
    a = sparse.csr_matrix(assigned_ref, dtype=bool)
    b = sparse.csr_matrix(assigned_new, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} and {b.shape}")
    d = (a != b).tocoo()
    rows, cols = d.row.astype(np.int64), d.col.astype(np.int64)
    out = pd.DataFrame(
        {
            "guide": rows,
            "cell": cols,
            "in_ref": np.asarray(a[rows, cols]).ravel() if rows.size else np.zeros(0, bool),
            "in_new": np.asarray(b[rows, cols]).ravel() if rows.size else np.zeros(0, bool),
        }
    )
    count = (
        np.asarray(sparse.csr_matrix(counts)[rows, cols]).ravel()
        if counts is not None and rows.size
        else np.full(rows.size, np.nan)
    )
    lp = (
        np.asarray(sparse.csr_matrix(log_p_ref)[rows, cols]).ravel()
        if log_p_ref is not None and rows.size
        else np.full(rows.size, np.nan)
    )
    cut = np.nan if cutoff_ref is None else float(cutoff_ref)
    with np.errstate(invalid="ignore"):
        margin = np.abs(lp - cut)
    tol = log_p_tolerance(count, lp)
    out["count"] = count
    out["log_p_ref"] = lp
    out["cutoff_ref"] = cut
    out["margin"] = margin
    out["tolerance"] = tol
    out["flip_class"] = np.where(margin <= tol, "boundary", "defect")
    return out


def describe(table: pd.DataFrame, limit: int = 50) -> str:
    """A printable summary of a flip table, at most `limit` rows."""
    if table.empty:
        return "no flips"
    counts = table["flip_class"].value_counts().to_dict()
    return f"{len(table)} flips {counts}\n{table.head(limit).to_string(index=False)}"
