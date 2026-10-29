"""Covariate-aware analytical power checked against independent special cases."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pysceptre.analytical_power import compute_power
from pysceptre.analytical_power.covariates import _expected_null_mean


def _args(Z, x, coefs=None):
    if coefs is None:
        coefs = np.r_[np.log(2.0), np.zeros(Z.shape[1] - 1)]
    return dict(
        discovery_pairs=pd.DataFrame({"grna_target": ["t"], "response_id": ["g"]}),
        method="covariate",
        covariate_matrix=Z,
        gene_fits={"g": SimpleNamespace(fitted_coefs=coefs, theta=2.0)},
        grna_target_cells={"t": np.flatnonzero(x)},
        fold_change_mean=0.85,
        fold_change_sd=0.0,
        cutoff=0.05,
        num_total_cells=len(x),
    )


def _batches():
    batch = np.repeat([0, 1], 500)
    x = np.zeros(1000)
    x[np.r_[np.arange(50), 500 + np.arange(450)]] = 1
    Z = np.column_stack([np.ones(1000), batch])
    return Z, x, np.array([0.0, np.log(4.0)])


@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_covariate_power_reduces_to_homogeneous(side):
    x = np.r_[np.ones(300), np.zeros(700)]
    args = _args(np.ones((1000, 1)), x)
    actual = compute_power(**args, side=side)
    expected = compute_power(
        args["discovery_pairs"],
        pd.DataFrame({"grna_target": ["t"], "num_cells": [300]}),
        pd.DataFrame({"response_id": ["g"], "expression_mean": [2.0], "expression_size": [2.0]}),
        fold_change_mean=0.85,
        fold_change_sd=0.0,
        cutoff=0.05,
        num_total_cells=1000,
        side=side,
    )
    cols = ["mean_test_stat", "sd_test_stat", "power"]
    np.testing.assert_allclose(actual[cols], expected[cols], rtol=1e-9, atol=1e-11)
    assert "num_trt_cells_sq" not in actual.columns


@pytest.mark.parametrize("theta,fold_change", [(2.0, 0.85), (1e-10, 0.01)])
def test_covariate_power_matches_confounded_batch_oracle(theta, fold_change):
    Z, x, coefs = _batches()
    args = _args(Z, x, coefs)
    args["gene_fits"]["g"].theta = theta
    args["fold_change_mean"] = fold_change
    result = compute_power(**args).iloc[0]
    p = np.array([0.1, 0.9])
    mu_c = np.array([1.0, 4.0])
    mu_t = fold_change * mu_c
    mu0 = p * mu_t + (1 - p) * mu_c
    H = 500 * p * (1 - p)
    d = 1 + mu0 / theta
    information = np.sum(H * mu0 / d)
    U = np.sum(H * (mu_t - mu_c) / d)
    var_t = mu_t + mu_t**2 / theta
    var_c = mu_c + mu_c**2 / theta
    V = np.sum(H * ((1 - p) * var_t + p * var_c) / d**2)
    mean, sd = U / np.sqrt(information), np.sqrt(V / information)
    np.testing.assert_allclose(
        result[["mean_test_stat", "sd_test_stat", "power"]].to_numpy(dtype=float),
        [mean, sd, norm.cdf(norm.ppf(0.05), loc=mean, scale=sd)],
        rtol=1e-8,
        atol=1e-10,
    )
    assert result["expression_mean"] == pytest.approx(2.5)
    assert result["num_trt_cells"] == 500
    assert result["num_cntrl_cells"] == 500
    assert result["qc_failure_prob"] == 0


def test_covariate_null_has_nominal_power():
    args = _args(*_batches())
    args["fold_change_mean"] = 1.0
    result = compute_power(**args).iloc[0]
    np.testing.assert_allclose(
        result[["mean_test_stat", "sd_test_stat", "power"]].to_numpy(dtype=float),
        [0.0, 1.0, 0.05],
        atol=1e-9,
    )


def test_covariate_power_is_invariant_to_redundant_rescaled_columns():
    Z, x, coefs = _batches()
    expected = compute_power(**_args(Z, x, coefs))
    expanded_Z = np.column_stack([Z[:, 0] * 1e-6, Z[:, 1] * 1e6, Z[:, 1] * 2])
    expanded_coefs = np.array([coefs[0] / 1e-6, coefs[1] / 1e6, 0.0])
    actual = compute_power(**_args(expanded_Z, x, expanded_coefs))
    cols = ["expression_mean", "mean_test_stat", "sd_test_stat", "power"]
    np.testing.assert_allclose(actual[cols], expected[cols], rtol=1e-8, atol=1e-10)


def test_covariate_power_uses_union_indices():
    Z, x, coefs = _batches()
    args = _args(Z, x, coefs)
    expected = compute_power(**args)
    idx = args["grna_target_cells"]["t"]
    args["grna_target_cells"] = {"t": np.r_[idx, idx[:20]]}
    pd.testing.assert_frame_equal(compute_power(**args), expected)


def test_covariate_power_rejects_perfect_confounding():
    x = np.repeat([0.0, 1.0], 50)
    with pytest.raises(ValueError, match="fully explained|information|identif"):
        compute_power(**_args(np.column_stack([np.ones(100), x]), x))


@pytest.mark.parametrize("indices", [np.array([0.5, 1]), np.array([-1, 1]), np.array([0, 1000])])
def test_covariate_power_rejects_invalid_cell_indices(indices):
    args = _args(*_batches())
    args["grna_target_cells"] = {"t": indices}
    with pytest.raises(ValueError):
        compute_power(**args)


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_total_cells": 999},
        {"fold_change_sd": 0.1},
        {"n_nonzero_trt_thresh": 1},
        {"n_nonzero_cntrl_thresh": 1},
        {"method": "unknown"},
        {"method": "homogeneous"},
    ],
)
def test_covariate_power_rejects_unsupported_or_mixed_options(overrides):
    args = _args(*_batches())
    args.update(overrides)
    with pytest.raises(ValueError):
        compute_power(**args)


def test_covariate_power_accepts_empty_pairs():
    args = _args(*_batches())
    args["discovery_pairs"] = args["discovery_pairs"].iloc[:0]
    result = compute_power(**args)
    assert result.empty
    assert {"grna_target", "response_id", "power", "mean_test_stat", "sd_test_stat"} <= set(result)


def test_expected_null_solves_nb_score_for_continuous_covariates():
    z = np.linspace(-2, 2, 401)
    Z = np.column_stack([np.ones(len(z)), z])
    mu_baseline = np.exp(Z @ np.array([0.4, 0.6]))
    x = ((np.arange(len(z)) % 5 == 0) | (z > 1)).astype(float)
    mu_alt = mu_baseline * np.where(x == 1, 0.65, 1.0)
    theta = 1.7
    mu0 = _expected_null_mean(Z, mu_alt, theta)
    score = Z.T @ ((mu_alt - mu0) / (1 + mu0 / theta))
    np.testing.assert_allclose(score / len(z), 0, atol=1e-8)
    np.testing.assert_allclose(
        Z @ np.linalg.lstsq(Z, np.log(mu0), rcond=None)[0], np.log(mu0), atol=1e-10
    )
    assert np.max(np.abs(mu0 - mu_baseline)) > 0.1


def test_covariate_power_preserves_pair_order_across_genes_and_targets():
    Z, x, coefs = _batches()
    args = _args(Z, x, coefs)
    args["gene_fits"]["g2"] = SimpleNamespace(fitted_coefs=np.array([np.log(3.0), 0.0]), theta=4.0)
    args["grna_target_cells"]["t2"] = np.arange(0, 1000, 5)
    pairs = pd.DataFrame(
        {"grna_target": ["t2", "t", "t2", "t", "t2"], "response_id": ["g2", "g", "g", "g2", "g2"]}
    )
    args["discovery_pairs"] = pairs
    result = compute_power(**args)
    pd.testing.assert_frame_equal(result[["grna_target", "response_id"]], pairs)
    for row in result.itertuples(index=False):
        p = np.array([0.2, 0.2]) if row.grna_target == "t2" else np.array([0.1, 0.9])
        mu_c = np.array([3.0, 3.0]) if row.response_id == "g2" else np.array([1.0, 4.0])
        theta = 4.0 if row.response_id == "g2" else 2.0
        mu_t = 0.85 * mu_c
        mu0 = p * mu_t + (1 - p) * mu_c
        H, d = 500 * p * (1 - p), 1 + mu0 / theta
        information = np.sum(H * mu0 / d)
        mean = np.sum(H * (mu_t - mu_c) / d) / np.sqrt(information)
        variance = (
            np.sum(H * ((1 - p) * (mu_t + mu_t**2 / theta) + p * (mu_c + mu_c**2 / theta)) / d**2)
            / information
        )
        np.testing.assert_allclose(
            [row.mean_test_stat, row.sd_test_stat], [mean, np.sqrt(variance)], rtol=1e-8
        )
        assert row.num_trt_cells == (200 if row.grna_target == "t2" else 500)


@pytest.mark.parametrize("key", ["gene_fits", "grna_target_cells"])
def test_covariate_power_rejects_missing_lookup_keys(key):
    args = _args(*_batches())
    args[key] = {}
    with pytest.raises(KeyError):
        compute_power(**args)


@pytest.mark.parametrize("key", ["covariate_matrix", "gene_fits", "grna_target_cells"])
def test_covariate_power_requires_all_cell_inputs(key):
    args = _args(*_batches())
    args[key] = None
    with pytest.raises(ValueError, match="requires"):
        compute_power(**args)


def test_covariate_power_rejects_scalar_input_tables():
    args = _args(*_batches())
    args["baseline_expression_stats"] = pd.DataFrame()
    with pytest.raises(ValueError, match="omit"):
        compute_power(**args)
