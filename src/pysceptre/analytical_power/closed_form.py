"""Analytical power helpers for homogeneous and covariate-adjusted designs.

The homogeneous helpers port PerturbPlan's `compute_distribution_teststat`, `compute_zero_prob`,
`compute_QC` and `rejection_computation` (`Katsevich-Lab/perturbplan`, MIT --
see `THIRD_PARTY_LICENSES`). Those are plain R-level functions, not the
package's `*_cpp` kernels, which serve its design/planning path; so this is a
direct port of visible source rather than a reconstruction, the same category
as `precompute/pieces.py`.

The homogeneous helpers are vectorised over pairs and hold no state. Why this
estimator, and the places where an obvious-looking simplification produces a
different one, are in `docs/design.md`, "Analytical per-pair power".
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom, norm

SIDES = ("left", "right", "both")


def var_nb(mean: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Negative-binomial variance for a given mean and NB size.

    `size` is theta, not the dispersion: pass `1 / dispersion` if that is
    what you hold.
    """
    return mean + mean**2 / size


def zero_prob(
    fold_change_mean: np.ndarray, expression_mean: np.ndarray, expression_size: np.ndarray
) -> np.ndarray:
    """P(count == 0) for an NB gene under a multiplicative fold change."""
    trt_mean = expression_mean * fold_change_mean
    return (expression_size / (trt_mean + expression_size)) ** expression_size


def test_stat_distribution(
    *,
    num_trt_cells: np.ndarray,
    num_cntrl_cells: np.ndarray,
    num_trt_cells_sq: np.ndarray,
    expression_mean: np.ndarray,
    expression_size: np.ndarray,
    fold_change_mean: np.ndarray,
    fold_change_sd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd of the score statistic under the alternative.

    `fold_change_mean` is a *multiplier*: a 15% knockdown is 0.85, not 0.15.

    `num_trt_cells` is the **sum** of per-gRNA cell counts for the target, and
    `num_trt_cells_sq` the sum of their squares -- not the size of the union
    of perturbed cells. See `docs/design.md`, "Analytical per-pair power".

    Returns `(mean, sd)`, both broadcast to the shape of the inputs.
    """
    num_test_cells = num_trt_cells + num_cntrl_cells
    trt_test_prop = num_trt_cells / num_test_cells
    cntrl_test_prop = 1.0 - trt_test_prop

    trt_expression_mean = expression_mean * fold_change_mean
    cntrl_expression_mean = expression_mean
    pooled_expression_mean = (
        trt_expression_mean * trt_test_prop + cntrl_expression_mean * cntrl_test_prop
    )

    # The score statistic's denominator: the pooled-variance scale the numerator is divided by.
    denominator_sq = var_nb(pooled_expression_mean, expression_size) * (
        1.0 / num_cntrl_cells + 1.0 / num_trt_cells
    )

    cntrl_var = var_nb(cntrl_expression_mean, expression_size) / num_cntrl_cells
    trt_var_within = (
        trt_expression_mean
        + cntrl_expression_mean**2 * (fold_change_sd**2 + fold_change_mean**2) / expression_size
    ) / num_trt_cells
    # The across-gRNA term, and the only place num_trt_cells_sq enters: gRNAs of one target differ
    # in effect, so a target's cells are not exchangeable.
    trt_var_across = (
        cntrl_expression_mean**2 * fold_change_sd**2 * num_trt_cells_sq / num_trt_cells**2
    )

    sd = np.sqrt((cntrl_var + trt_var_within + trt_var_across) / denominator_sq)
    mean = cntrl_expression_mean * (fold_change_mean - 1.0) / np.sqrt(denominator_sq)
    return mean, sd


def test_stat_distribution_covariates(
    *,
    treatment: np.ndarray,
    covariate_matrix: np.ndarray,
    null_mean: np.ndarray,
    alternative_mean: np.ndarray,
    expression_size: float,
) -> tuple[float, float]:
    """Mean and sd for one cell-level NB score comparison, with fixed effects.

    `treatment` is binary; `covariate_matrix` includes an intercept. All inputs
    refer to the same cells. Means are positive vectors or scalars broadcast
    over cells. `null_mean` is the restricted-null fit appropriate to the
    alternative, not automatically the unperturbed baseline. NB size may be
    positive infinity for the Poisson limit.

    The caller supplies both mean vectors. No nuisance fitting, shared-guide
    variance or resampling calibration is performed. See `docs/design.md`,
    "Covariate-aware score moments", for the frozen-fit approximation.
    """
    x = np.asarray(treatment, dtype=float)
    Z = np.asarray(covariate_matrix, dtype=float)
    if x.ndim != 1 or x.size == 0 or Z.ndim != 2 or Z.shape[0] != x.size or Z.shape[1] == 0:
        raise ValueError("treatment and covariate_matrix must describe the same nonempty cells")
    if not np.all((x == 0) | (x == 1)) or x.min() == x.max():
        raise ValueError("treatment must be binary with both treated and control cells")
    mu0 = np.broadcast_to(np.asarray(null_mean, dtype=float), x.shape)
    mu1 = np.broadcast_to(np.asarray(alternative_mean, dtype=float), x.shape)
    theta = float(expression_size)
    if (
        not np.all(np.isfinite(Z))
        or not np.all(np.isfinite(mu0))
        or not np.all(np.isfinite(mu1))
        or np.any(mu0 <= 0)
        or np.any(mu1 <= 0)
        or np.isnan(theta)
        or theta <= 0
    ):
        raise ValueError(
            "means must be finite and positive, covariates finite, and NB size positive"
        )

    d = 1.0 + mu0 / theta
    sqrt_w = np.sqrt(mu0 / d)
    design = sqrt_w[:, None] * Z
    scales = np.linalg.norm(design, axis=0)
    design = design[:, scales > 0] / scales[scales > 0]
    targets = np.column_stack((sqrt_w * x, sqrt_w))
    coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
    residuals = targets - design @ coefficients
    if np.linalg.norm(residuals[:, 1]) > 1e-10 * np.linalg.norm(sqrt_w):
        raise ValueError("covariate_matrix must include an intercept in its column span")
    weighted_r = residuals[:, 0]
    if np.linalg.norm(weighted_r) <= 1e-12 * np.linalg.norm(sqrt_w * x):
        raise ValueError("treatment is fully explained by the covariates")
    information = float(weighted_r @ weighted_r)
    r = weighted_r / sqrt_w
    mean = np.sum(r * (mu1 - mu0) / d) / np.sqrt(information)
    sd = np.sqrt(np.sum((r / d) ** 2 * var_nb(mu1, theta)) / information)
    return float(mean), float(sd)


def qc_failure_prob(
    *,
    fold_change_mean: np.ndarray,
    expression_mean: np.ndarray,
    expression_size: np.ndarray,
    num_trt_cells: np.ndarray,
    num_cntrl_cells: np.ndarray,
    n_nonzero_trt_thresh: int,
    n_nonzero_cntrl_thresh: int,
) -> np.ndarray:
    """P(the pair fails sceptre's pairwise nonzero-count QC).

    Both thresholds at 0 make this 0 everywhere, which is the setting for
    pairs that already passed QC in a real analysis.
    """
    cntrl_nonzero_prob = 1.0 - zero_prob(1.0, expression_mean, expression_size)
    trt_nonzero_prob = 1.0 - zero_prob(fold_change_mean, expression_mean, expression_size)
    # sf(k-1) is P(X >= k). At a threshold of 0 this is sf(-1) == 1, i.e. no discount.
    cntrl_kept = binom.sf(n_nonzero_cntrl_thresh - 1, num_cntrl_cells, cntrl_nonzero_prob)
    trt_kept = binom.sf(n_nonzero_trt_thresh - 1, num_trt_cells, trt_nonzero_prob)
    return 1.0 - cntrl_kept * trt_kept


def rejection_prob(mean: np.ndarray, sd: np.ndarray, *, side: str, cutoff: float) -> np.ndarray:
    """P(the test rejects), for a normal statistic with the given mean and sd.

    `cutoff` is the nominal p-value threshold the test is read at, and it is
    not interchangeable with a screen's alpha: the validated configuration is
    `side="left"` at `cutoff = alpha / 2`, sceptre's p-value being two-sided
    while a knockdown is a one-sided claim.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {list(SIDES)}, got {side!r}")
    if side == "left":
        return norm.cdf(norm.ppf(cutoff), loc=mean, scale=sd)
    if side == "right":
        return norm.sf(norm.ppf(1.0 - cutoff), loc=mean, scale=sd)
    return norm.sf(norm.ppf(1.0 - cutoff / 2.0), loc=mean, scale=sd) + norm.cdf(
        norm.ppf(cutoff / 2.0), loc=mean, scale=sd
    )
