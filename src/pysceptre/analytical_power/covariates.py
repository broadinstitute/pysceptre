"""Cell-level inputs and expected restricted-null fits for analytical power."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .closed_form import rejection_prob, test_stat_distribution_covariates

if TYPE_CHECKING:
    from .inputs import GeneFit


def _expected_null_mean(
    covariate_basis: np.ndarray, alternative_mean: np.ndarray, expression_size: float
) -> np.ndarray:
    """Fit expected NB nuisance scores to zero under the specified alternative."""
    Q, mu1, theta = covariate_basis, alternative_mean, expression_size
    normalizer = np.sum(mu1 / (1.0 + mu1 / theta))

    def loss_gradient(coefs):
        t = Q @ coefs - np.log(theta)
        loss = np.sum(theta * np.logaddexp(0, t) + mu1 * np.logaddexp(0, -t)) / normalizer
        gradient = Q.T @ (theta * expit(t) - mu1 * expit(-t)) / normalizer
        return loss, gradient

    def hessian(coefs):
        t = Q @ coefs - np.log(theta)
        weights = (theta + mu1) * expit(t) * expit(-t)
        return Q.T @ (weights[:, None] * Q) / normalizer

    initial = np.linalg.lstsq(Q, np.log(mu1), rcond=None)[0]
    fit = minimize(
        loss_gradient,
        initial,
        jac=True,
        hess=hessian,
        method="trust-exact",
        options={"gtol": 1e-10, "maxiter": 100},
    )
    coefs = fit.x
    # Refine a nearly converged solution when objective rounding stops the optimizer.
    for _ in range(3):
        _, gradient = loss_gradient(coefs)
        if np.max(np.abs(gradient)) <= 1e-10:
            break
        step = np.linalg.solve(hessian(coefs), gradient)
        if np.max(np.abs(Q @ step)) > 1e-4:
            break
        coefs = coefs - step
    _, gradient = loss_gradient(coefs)
    mu0 = np.exp(Q @ coefs)
    if (
        not np.all(np.isfinite(mu0))
        or np.any(mu0 <= 0)
        or not np.all(np.isfinite(gradient))
        or np.max(np.abs(gradient)) > 1e-10
    ):
        raise ValueError(f"expected restricted-null NB fit did not converge: {fit.message}")
    return mu0


def _compute_power_covariates(
    discovery_pairs: pd.DataFrame,
    covariate_matrix: np.ndarray,
    gene_fits: Mapping[str, GeneFit],
    grna_target_cells: Mapping[str, np.ndarray],
    *,
    fold_change_mean: float,
    cutoff: float,
    num_total_cells: int,
    side: str,
) -> pd.DataFrame:
    """Compute fixed-effect power from a common cell design and per-gene fits."""
    Z = np.asarray(covariate_matrix, dtype=float)
    if (
        Z.ndim != 2
        or min(Z.shape) == 0
        or Z.shape[0] != num_total_cells
        or not np.all(np.isfinite(Z))
    ):
        raise ValueError("covariate_matrix must be finite, nonempty and have num_total_cells rows")
    n = Z.shape[0]
    scales = np.linalg.norm(Z, axis=0)
    design = Z[:, scales > 0] / scales[scales > 0]
    U, singular_values, _ = np.linalg.svd(design, full_matrices=False)
    rank = np.sum(
        singular_values > max(design.shape) * np.finfo(float).eps * singular_values.max(initial=0)
    )
    Q = U[:, :rank] * np.sqrt(n)
    if np.linalg.norm(np.ones(n) - Q @ (Q.sum(axis=0) / n)) > 1e-10 * np.sqrt(n):
        raise ValueError("covariate_matrix must include an intercept in its column span")

    missing = [c for c in ("grna_target", "response_id") if c not in discovery_pairs.columns]
    if missing:
        raise KeyError(f"discovery_pairs is missing column(s): {missing}")
    out = discovery_pairs.loc[:, ["grna_target", "response_id"]].reset_index(drop=True)
    if out.isna().any().any():
        raise ValueError("discovery_pairs must have non-missing target and gene IDs")
    unknown_genes = sorted(set(out["response_id"]) - set(gene_fits))
    unknown_targets = sorted(set(out["grna_target"]) - set(grna_target_cells))
    if unknown_genes:
        raise KeyError(f"gene_fits has no entry for gene(s): {unknown_genes[:5]}")
    if unknown_targets:
        raise KeyError(f"grna_target_cells has no entry for target(s): {unknown_targets[:5]}")

    target_indices = {}
    for target in out["grna_target"].unique():
        idx = np.asarray(grna_target_cells[target])
        if idx.ndim != 1 or idx.dtype.kind not in "iu" or np.any(idx < 0) or np.any(idx >= n):
            raise ValueError(
                f"target {target!r} must have integer cell indices in [0, num_total_cells)"
            )
        idx = np.unique(idx)
        if idx.size == 0 or idx.size == n:
            raise ValueError(f"target {target!r} needs non-empty treated and complement groups")
        target_indices[target] = idx

    means, sds = np.empty(len(out)), np.empty(len(out))
    baseline_means, sizes = np.empty(len(out)), np.empty(len(out))
    for gene, rows in out.groupby("response_id", sort=False, observed=True).indices.items():
        fit = gene_fits[gene]
        coefs = np.asarray(fit.fitted_coefs, dtype=float)
        theta = float(fit.theta)
        if coefs.shape != (Z.shape[1],) or not np.all(np.isfinite(coefs)):
            raise ValueError(f"gene {gene!r} must have one finite coefficient per covariate column")
        if not np.isfinite(theta) or theta <= 0:
            raise ValueError(f"gene {gene!r} must have a finite positive NB size theta")
        with np.errstate(over="ignore", under="ignore"):
            baseline = np.exp(Z @ coefs)
        if not np.all(np.isfinite(baseline)) or np.any(baseline <= 0):
            raise ValueError(f"gene {gene!r} has non-finite or non-positive baseline fitted means")
        baseline_means[rows], sizes[rows] = baseline.mean(), theta
        for row in rows:
            x = np.zeros(n)
            x[target_indices[out.at[row, "grna_target"]]] = 1.0
            mu1 = baseline * np.where(x == 1, fold_change_mean, 1.0)
            if not np.all(np.isfinite(mu1)) or np.any(mu1 <= 0):
                raise ValueError("alternative cell means must be finite and positive")
            mu0 = _expected_null_mean(Q, mu1, theta)
            means[row], sds[row] = test_stat_distribution_covariates(
                treatment=x,
                covariate_matrix=Z,
                null_mean=mu0,
                alternative_mean=mu1,
                expression_size=theta,
            )

    out["num_trt_cells"] = [target_indices[t].size for t in out["grna_target"]]
    out["num_cntrl_cells"] = n - out["num_trt_cells"]
    out["expression_mean"], out["expression_size"] = baseline_means, sizes
    out["mean_test_stat"], out["sd_test_stat"] = means, sds
    out["qc_failure_prob"] = 0.0
    out["power"] = rejection_prob(means, sds, side=side, cutoff=cutoff)
    return out
