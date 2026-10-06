"""Rank of a design matrix, and which of its columns are redundant."""

from __future__ import annotations

import numpy as np


def redundant_columns(X: np.ndarray) -> tuple[int, list[int]]:
    """`(rank, redundant)` for a finite `(n, p)` design matrix.

    `redundant` lists, left to right, the columns that are linear combinations
    of the ones before them -- the columns R's `glm.fit` would report as NA.
    It is empty when `X` has full column rank.

    Rank comes from the singular values of `R` in a thin QR of `X`, with the
    relative tolerance `numpy.linalg.matrix_rank` uses. Never from `X.T @ X`,
    whose eigenvalues square the condition number. See docs/design.md, "A rank
    check that does not square the condition number".
    """
    n, p = X.shape
    eps = np.finfo(float).eps
    r = np.linalg.qr(np.asarray(X, dtype=float), mode="r")

    def _rank(block: np.ndarray) -> int:
        sv = np.linalg.svd(block, compute_uv=False)
        if sv.size == 0 or sv[0] <= 0:
            return 0
        return int(np.sum(sv > sv[0] * max(n, block.shape[1]) * eps))

    rank = _rank(r)
    if rank == p:
        return rank, []
    kept: list[int] = []
    for j in range(p):
        trial = kept + [j]
        if _rank(r[:, trial]) == len(trial):
            kept.append(j)
    return rank, [j for j in range(p) if j not in kept]
