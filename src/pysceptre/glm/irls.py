"""Batched IRLS for Poisson (log link) and binomial (logit link) GLMs.

Ports the algorithm behind R's `stats::glm.fit`: iteratively reweighted
least squares on the working response, with deviance-based convergence
(`epsilon = 1e-8`, `maxit = 25`, matching R's defaults). Because both models
have concave, canonical-link log-likelihoods, IRLS converges to the unique
MLE regardless of implementation details -- exact line-for-line replication
of R's algorithm is not required for the fitted coefficients to match R's
`glm.fit` output to numerical precision.

Vectorized across a batch of response columns (k genes/targets) sharing one
design matrix X, since that's the actual batching opportunity in sceptre's
"complement" control-group mode (see pipeline/discovery.py): the p x p
weighted normal-equations system differs per column (because the IRLS
weights depend on that column's own fitted mu), but can be solved for all
k columns at once via a batched `numpy.linalg.solve`.

The per-iteration matmuls here are "thin" (p is a handful of covariates, so
every matmul has a tiny inner/output dimension even though n is large) and
called many times in a tight loop. Profiling showed multi-threaded OpenBLAS
is a net *loss* for this shape/call-pattern in this environment -- thread
coordination overhead per call dominates the actual (small) amount of work
each call does (measured: a 150-column batch over 100k cells went from
68.7s to 17.7s just by forcing BLAS to one thread). `threadpoolctl` scopes
that restriction to this module's own matmuls rather than clobbering BLAS
threading process-wide (which would also affect any other numpy/scipy work
happening in the same process outside pysceptre).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

_EPS = 1e-8
_MAXIT = 25
_MU_FLOOR = 1e-10


@dataclass
class GlmFitBatchResult:
    coefs: np.ndarray  # (k, p)
    fitted_values: np.ndarray  # (k, n)
    deviance: np.ndarray  # (k,)
    n_iter: np.ndarray  # (k,)
    converged: np.ndarray  # (k,) bool


def _as_2d(Y: np.ndarray) -> tuple[np.ndarray, bool]:
    """Promote a single response vector to a batch of one.

    Shapes here are batch-axis-first -- `(k, n)`, k responses of n
    observations -- matching numpy's own batched-linalg convention
    (`np.linalg.solve` takes `(k, p, p)`), keeping each response contiguous,
    and letting the normal equations be assembled with no transposes at all.
    """
    if Y.ndim == 1:
        return Y[None, :], True
    return Y, False


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    # 2 * sum(y*log(y/mu) - (y - mu)), with the convention y*log(y/mu) = 0 at y = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2.0 * np.sum(term - (y - mu), axis=-1)


def _binomial_deviance(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(y > 0, y * np.log(y / mu), 0.0)
        t2 = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
    return 2.0 * np.sum(t1 + t2, axis=-1)


def _batched_wls_solve(
    X: np.ndarray, X_outer_flat: np.ndarray, w: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Solve, for each of k columns, (X^T diag(w_k) X) beta_k = X^T diag(w_k) z_k.

    X: (n, p) shared design matrix. X_outer_flat: (n, p*p), the precomputed
    (and iteration-independent) per-row outer product X[n,:] outer X[n,:],
    flattened -- passed in rather than recomputed every IRLS iteration.
    w, z: (k, n) per-response IRLS weights and working response.
    Returns beta: (k, p).

    Deliberately expressed as two plain matmuls (BLAS-backed) rather than a
    per-iteration `einsum` building a (k, p, n) intermediate: for realistic
    problem sizes (k in the hundreds to thousands of genes/targets, n in the
    hundreds of thousands of cells, p a handful of covariates), materializing
    a (k, p, n) tensor every one of up to 25 IRLS iterations dominated
    runtime by orders of magnitude (profiled: ~36s for a 292-gene x
    100k-cell batch) -- p is small, so there's no need to ever allocate
    anything of size k*p*n.
    """
    p = X.shape[1]
    # Batch-axis-first means both products come out already batched, and the
    # reshape below is a view rather than a copy -- with (n, k) inputs the
    # same computation needs a transpose on each of A and b.
    A = (w @ X_outer_flat).reshape(-1, p, p)  # (k, n) @ (n, p*p) -> (k, p, p)
    b = (w * z) @ X  # (k, n) @ (n, p) -> (k, p)
    return np.linalg.solve(A, b[:, :, None])[:, :, 0]  # (k, p)


def _fit_batch(
    X: np.ndarray, Y: np.ndarray, family: str, *, eps: float = _EPS, maxit: int = _MAXIT
) -> GlmFitBatchResult:
    n, p = X.shape
    Y, was_1d = _as_2d(Y)
    k, n_y = Y.shape
    if n_y != n:
        raise ValueError(f"X has {n} rows but Y has {n_y} columns")

    if family == "poisson":
        mu = Y + 0.1
        deviance_fn = _poisson_deviance
    elif family == "binomial":
        mu = (Y + 0.5) / 2.0
        deviance_fn = _binomial_deviance
    else:
        raise ValueError(f"unsupported family: {family}")

    eta = np.log(mu) if family == "poisson" else np.log(mu / (1 - mu))
    dev_old = np.full(k, np.inf)
    converged = np.zeros(k, dtype=bool)
    n_iter = np.zeros(k, dtype=int)
    beta = np.zeros((k, p))
    X_outer_flat = (X[:, :, None] * X[:, None, :]).reshape(n, p * p)  # iteration-independent

    # `mu` is carried forward as loop state (rather than recomputed via exp(eta)
    # at the top of every iteration) and every elementwise op below is
    # restricted to the not-yet-converged columns -- at real dataset scale
    # (hundreds of genes/targets x hundreds of thousands of cells), each full
    # (k, n) elementwise pass costs real wall-clock time, and IRLS otherwise
    # redundantly recomputes exp(eta) == mu twice per iteration.
    for it in range(1, maxit + 1):
        active = ~converged
        n_active = int(np.sum(active))
        if n_active == 0:
            break
        all_active = n_active == k  # avoid a full-array fancy-index *copy* in
        # the (extremely common) case where every column is still active --
        # a boolean-indexing copy of the whole (k, n) array every iteration
        # was measured to cost more than the work it saves once most/all
        # columns converge together, which is the typical case here since
        # every gene/target shares the same design matrix.

        # Row slices: each response is contiguous, so this copy is sequential.
        Y_a = Y if all_active else Y[active]
        eta_a = eta if all_active else eta[active]
        mu_a = mu if all_active else mu[active]

        if family == "poisson":
            dmu_deta_a = mu_a
            var_a = mu_a
        else:  # binomial
            dmu_deta_a = mu_a * (1 - mu_a)
            var_a = dmu_deta_a

        w_a = (dmu_deta_a**2) / var_a
        z_a = eta_a + (Y_a - mu_a) / dmu_deta_a

        beta_a = _batched_wls_solve(X, X_outer_flat, w_a, z_a)
        eta_new_a = beta_a @ X.T  # (k_a, p) @ (p, n) -> (k_a, n)

        mu_new_a = (
            np.clip(np.exp(eta_new_a), _MU_FLOOR, None)
            if family == "poisson"
            else np.clip(1.0 / (1.0 + np.exp(-eta_new_a)), _MU_FLOOR, 1 - _MU_FLOOR)
        )
        dev_new_a = deviance_fn(Y_a, mu_new_a)

        dev_old_a = dev_old if all_active else dev_old[active]
        newly_converged_a = np.abs(dev_new_a - dev_old_a) / (np.abs(dev_new_a) + 0.1) < eps

        if all_active:
            beta[:, :] = beta_a
            eta[:, :] = eta_new_a
            mu[:, :] = mu_new_a
            dev_old[:] = dev_new_a
            n_iter[:] = it
            converged[:] = newly_converged_a
        else:
            beta[active] = beta_a
            eta[active] = eta_new_a
            mu[active] = mu_new_a
            dev_old[active] = dev_new_a
            n_iter[active] = it
            converged[active] = newly_converged_a

    mu_final = mu

    if was_1d:
        return GlmFitBatchResult(
            coefs=beta[0],
            fitted_values=mu_final[0],
            deviance=dev_old[0],
            n_iter=n_iter[0],
            converged=converged[0],
        )
    return GlmFitBatchResult(
        coefs=beta, fitted_values=mu_final, deviance=dev_old, n_iter=n_iter, converged=converged
    )


def fit_poisson_glm_batch(
    X: np.ndarray, Y: np.ndarray, *, eps: float = _EPS, maxit: int = _MAXIT
) -> GlmFitBatchResult:
    """X: (n, p) shared design matrix. Y: (n, k) or (n,) response column(s)."""
    with threadpool_limits(limits=1, user_api="blas"):
        return _fit_batch(X, Y, "poisson", eps=eps, maxit=maxit)


def fit_binomial_glm_batch(
    X: np.ndarray, Y: np.ndarray, *, eps: float = _EPS, maxit: int = _MAXIT
) -> GlmFitBatchResult:
    """X: (n, p) shared design matrix. Y: (n, k) or (n,) 0/1 indicator column(s)."""
    with threadpool_limits(limits=1, user_api="blas"):
        return _fit_batch(X, Y, "binomial", eps=eps, maxit=maxit)
