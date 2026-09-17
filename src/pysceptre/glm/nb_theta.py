"""Exact port of sceptre's `estimate_theta` (negative-binomial dispersion
estimation given a fixed, already-fitted mean `mu`).

Ported verbatim from `compute_nb_size_param_functions.cpp` at the pinned
upstream commit katsevich-lab/sceptre@e7866dd4bc158e4415588c4dde0a69da6ac93166
(verified against the source directly, not reconstructed). Three-stage
fallback: Newton-Raphson MLE on the NB profile likelihood (using digamma/
trigamma) -> method-of-moments Newton iteration matching residual variance
to the residual degrees of freedom -> the method-of-moments pilot estimate
itself, whichever succeeds first (in that preference order, MLE first).

Performance note: `_nb_score`/`_nb_info` evaluate `digamma`/`trigamma` at
`theta + y`, which depends only on the *value* of each count `y_i`, not on
`mu_i` or which cell it came from. Since `y` holds small non-negative
integer counts, real datasets have far fewer distinct values than cells
(profiled: this was ~90% of `perform_response_precomputation`'s total time
at real dataset scale, dominated by evaluating these special functions on
full length-n_cells arrays every Newton iteration). Deduplicating by unique
count value before calling digamma/trigamma, then weighting by how many
cells had that value, is mathematically exact (these are deterministic
scalar functions -- grouping identical inputs changes nothing about the
result, mod floating-point summation order), not an approximation.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, polygamma


def _trigamma(x: np.ndarray | float) -> np.ndarray | float:
    return polygamma(1, x)


def nb_theta_pilot_est(y: np.ndarray, mu: np.ndarray) -> float:
    n = y.size
    denom = np.sum((y / mu - 1) ** 2)
    return n / denom


def _nb_score(
    theta: float, mu: np.ndarray, y: np.ndarray, unique_y: np.ndarray, y_inverse: np.ndarray
) -> float:
    # Gather the deduplicated digamma values back to full per-cell shape (via
    # `y_inverse`) *before* summing, rather than summing the weighted unique
    # values directly -- this preserves the exact same floating-point
    # summation order as evaluating digamma(theta+y) directly (just without
    # redundantly calling digamma on repeated values), which matters because
    # Newton-Raphson on a near-degenerate profile likelihood (e.g. theta far
    # out in the near-Poisson regime) is numerically sensitive enough that a
    # merely mathematically-equivalent but differently-ordered sum can send
    # it down a different, divergent path.
    digamma_full = digamma(theta + unique_y)[y_inverse]
    mu_plus_theta = (
        mu + theta
    )  # computed once, reused below (IEEE754 addition is commutative/order-exact, so this is bit-identical to recomputing)
    return float(
        np.sum(
            digamma_full
            - digamma(theta)
            + np.log(theta)
            + 1
            - np.log(mu_plus_theta)
            - (y + theta) / mu_plus_theta
        )
    )


def _nb_info(
    theta: float, mu: np.ndarray, y: np.ndarray, unique_y: np.ndarray, y_inverse: np.ndarray
) -> float:
    trigamma_full = _trigamma(theta + unique_y)[y_inverse]
    mu_plus_theta = mu + theta
    return float(
        np.sum(
            _trigamma(theta)
            - trigamma_full
            - 1 / theta
            + 2 / mu_plus_theta
            - (y + theta) / mu_plus_theta**2
        )
    )


def nb_theta_mle(
    t0: float,
    y: np.ndarray,
    mu: np.ndarray,
    limit: int,
    eps: float,
    unique_y: np.ndarray,
    y_inverse: np.ndarray,
) -> tuple[float, bool]:
    # Mirrors C++'s `while (++it < limit && fabs(del) > eps)`: `it` is
    # incremented as part of the loop condition itself (before the body
    # runs), so the post-loop value of `it` is exactly `limit` iff the loop
    # was terminated by running out of iterations rather than by converging.
    it = 0
    delta = 1.0
    while True:
        it += 1
        if not (it < limit and abs(delta) > eps):
            break
        t0 = abs(t0)
        delta = _nb_score(t0, mu, y, unique_y, y_inverse) / _nb_info(t0, mu, y, unique_y, y_inverse)
        t0 += delta
    warning = (t0 < 0) or (it == limit) or not np.isfinite(t0)
    return t0, warning


def nb_theta_mm(
    t0: float, y: np.ndarray, mu: np.ndarray, dfr: float, limit: int, eps: float
) -> tuple[float, bool]:
    it = 0
    delta = 1.0
    while True:
        it += 1
        if not (it < limit and abs(delta) > eps):
            break
        t0 = abs(t0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            # Newton's method can transiently overshoot to a very large/inf t0
            # before the post-loop isfinite() check catches it -- same as the
            # ported C++, which computes in double precision without guarding
            # against intermediate overflow either.
            numer = np.sum((y - mu) ** 2 / (mu + mu**2 / t0)) - dfr
            denom = np.sum((y - mu) ** 2 / (mu + t0) ** 2)
            delta = numer / denom
        t0 -= delta
    warning = (t0 < 0) or (it == limit) or not np.isfinite(t0)
    return t0, warning


def estimate_theta(
    y: np.ndarray,
    mu: np.ndarray,
    dfr: float,
    limit: int = 50,
    eps: float = np.finfo(float).eps ** 0.25,
) -> tuple[float, int]:
    """Returns (theta_estimate, method) with method in {1: MLE, 2: MM, 3: pilot}."""
    unique_y, y_inverse = np.unique(y, return_inverse=True)
    t0 = nb_theta_pilot_est(y, mu)
    estimate = t0
    method = 3
    try:
        est, warn = nb_theta_mle(t0, y, mu, limit, eps, unique_y, y_inverse)
        estimate = est
        method = 1
        if warn:
            est, warn = nb_theta_mm(t0, y, mu, dfr, limit, eps)
            estimate = est
            method = 2
            if warn:
                estimate = t0
                method = 3
    except Exception:
        pass
    return estimate, method


def perform_response_precomputation(
    expressions: np.ndarray, covariate_matrix: np.ndarray
) -> tuple[np.ndarray, float]:
    """Port of `perform_response_precomputation`: Poisson IRLS fit for the mean,
    then NB dispersion (theta) estimated from the Poisson-fitted mu, clamped to
    [0.01, 1000]. Returns (fitted_coefs, theta)."""
    from .irls import fit_poisson_glm_batch

    fit = fit_poisson_glm_batch(covariate_matrix, expressions)
    theta_est, _method = estimate_theta(
        y=expressions,
        mu=fit.fitted_values,
        dfr=covariate_matrix.shape[0] - covariate_matrix.shape[1],
        limit=50,
        eps=np.finfo(float).eps ** 0.25,
    )
    theta = max(min(theta_est, 1000.0), 0.01)
    return fit.coefs, theta
