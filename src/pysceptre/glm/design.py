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


def validate_design_matrix(covariate_matrix: np.ndarray, n_cells: int | None = None) -> None:
    """Refuse a design matrix the GLM cannot fit, and say which columns are at fault.

    Without this a rank-deficient design fails deep inside the batched weighted
    least squares as a bare `LinAlgError`, with no mention of covariates. The
    rank test is `glm.design.redundant_columns`.

    Raises:
        ValueError: not 2-D, empty, non-finite entries, a row count that
            disagrees with the data, or rank deficiency.
    """
    X = np.asarray(covariate_matrix)
    if X.ndim != 2:
        raise ValueError(f"covariate_matrix must be 2-D (n_cells, p), got shape {X.shape}")
    n, p = X.shape
    if n == 0 or p == 0:
        raise ValueError(f"covariate_matrix is empty, shape {X.shape}")
    if n_cells is not None and n != n_cells:
        raise ValueError(
            f"covariate_matrix has {n} rows but the response matrix has {n_cells} cells"
        )
    if p > n:
        raise ValueError(
            f"covariate_matrix has shape {X.shape}, more columns ({p}) than rows ({n}), so it "
            "cannot be full rank. It is expected as (n_cells, p); this looks transposed."
        )
    X = X.astype(float, copy=False)
    if not np.all(np.isfinite(X)):
        bad = np.flatnonzero(~np.all(np.isfinite(X), axis=0)).tolist()
        raise ValueError(f"covariate_matrix has non-finite values in column(s) {bad[:10]}")

    # Redundant columns are named left to right, so the first of a collinear set is kept --
    # R's convention, and the useful one, since column 0 is usually the intercept.
    rank, redundant = redundant_columns(X)
    if rank == 0:
        raise ValueError("covariate_matrix is all zeros")
    if not redundant:
        return
    raise ValueError(
        f"covariate_matrix is rank deficient: rank {rank} of {p} columns. "
        f"Column(s) {redundant} are linear combinations of the ones before them, so the "
        "weighted least squares inside the GLM fit is singular. This is usually an aliased "
        "contrast (a factor's full dummy set alongside an intercept), a duplicated column, or "
        "a covariate that is constant within another's levels. Drop the named column(s), or "
        "build the design with one reference level held out."
    )
