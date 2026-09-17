"""Port of sceptre's `compute_D_matrix` / `compute_precomputation_pieces`.

These are plain R-level functions (not compiled), so this is a direct,
low-risk port of visible source rather than a reconstruction.

Note on `D`: R's `eigen(Zt_wZ, symmetric = TRUE)` and numpy's `np.linalg.eigh`
may return eigenvalues/eigenvectors in different order and with different
signs (LAPACK routine differences, and R returns eigenvalues in *decreasing*
order while `eigh` returns them in *increasing* order). This does not affect
correctness: `D`'s rows only ever appear in the test statistic via
`sum_k (sum_{i in trt} D[k, i])^2`, which is invariant under row permutation
and sign flips of `D`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PrecomputationPieces:
    mu: np.ndarray
    w: np.ndarray
    a: np.ndarray
    D: np.ndarray


def compute_D_matrix(Zt_wZ: np.ndarray, wZ: np.ndarray) -> np.ndarray:
    """Zt_wZ: (p, p) symmetric. wZ: (n, p). Returns D: (p, n)."""
    eigvals, eigvecs = np.linalg.eigh(Zt_wZ)
    lambda_minus_half = 1.0 / np.sqrt(eigvals)
    D = (lambda_minus_half[:, None] * eigvecs.T) @ wZ.T
    return D


def compute_precomputation_pieces(
    expression_vector: np.ndarray,
    covariate_matrix: np.ndarray,
    fitted_coefs: np.ndarray,
    theta: float,
    full_test_stat: bool = True,
) -> PrecomputationPieces:
    mu = np.exp(covariate_matrix @ fitted_coefs)
    if not full_test_stat:
        raise NotImplementedError(
            "full_test_stat=False is the reduced-statistic branch used by sceptre's "
            "score-only code paths, which the complement+CRT discovery-analysis path "
            "targeted here never exercises; not implemented."
        )
    denom = 1 + mu / theta
    w = mu / denom
    a = (expression_vector - mu) / denom
    wZ = w[:, None] * covariate_matrix
    Zt_wZ = covariate_matrix.T @ wZ
    D = compute_D_matrix(Zt_wZ, wZ)
    return PrecomputationPieces(mu=mu, w=w, a=a, D=D)
