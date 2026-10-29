"""Cell-level score moments checked against homogeneous and stratified formulas."""

import numpy as np
import pytest

from pysceptre.analytical_power import (
    test_stat_distribution as homogeneous_moments,
)
from pysceptre.analytical_power import (
    test_stat_distribution_covariates as covariate_moments,
)


@pytest.mark.parametrize("theta", [2.0, np.inf])
def test_homogeneous_reduction(theta):
    x = np.r_[np.ones(300), np.zeros(700)]
    mu1 = 2.0 * np.where(x == 1, 0.85, 1.0)
    got = covariate_moments(
        treatment=x,
        covariate_matrix=np.ones((1000, 1)),
        null_mean=mu1.mean(),
        alternative_mean=mu1,
        expression_size=theta,
    )
    expected = homogeneous_moments(
        num_trt_cells=300,
        num_cntrl_cells=700,
        num_trt_cells_sq=300**2,
        expression_mean=2.0,
        expression_size=theta,
        fold_change_mean=0.85,
        fold_change_sd=0.0,
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_null_with_covariate_dependent_assignment():
    batch = np.repeat([0.0, 1.0], 500)
    x = np.zeros(1000)
    x[np.r_[np.arange(50), 500 + np.arange(450)]] = 1
    mu = np.where(batch == 0, 1.0, 4.0)
    got = covariate_moments(
        treatment=x,
        covariate_matrix=np.column_stack([np.ones(1000), batch]),
        null_mean=mu,
        alternative_mean=mu,
        expression_size=2.0,
    )
    np.testing.assert_allclose(got, (0.0, 1.0), atol=1e-12)


def test_confounded_batches_match_stratum_moments():
    batch = np.repeat([0, 1], 500)
    x = np.zeros(1000)
    x[np.r_[np.arange(50), 500 + np.arange(450)]] = 1
    p = np.array([0.1, 0.9])
    mu_t, mu_c, theta = 1.7, 2.0, 2.0
    mu0 = p * mu_t + (1 - p) * mu_c
    got = covariate_moments(
        treatment=x,
        covariate_matrix=np.column_stack([np.ones(1000), batch]),
        null_mean=mu0[batch],
        alternative_mean=np.where(x == 1, mu_t, mu_c),
        expression_size=theta,
    )
    H = 500 * p * (1 - p)
    d = 1 + mu0 / theta
    information = np.sum(H * mu0 / d)
    U = np.sum(H * (mu_t - mu_c) / d)
    V = np.sum(H * ((1 - p) * (mu_t + mu_t**2 / theta) + p * (mu_c + mu_c**2 / theta)) / d**2)
    np.testing.assert_allclose(
        got, (U / np.sqrt(information), np.sqrt(V / information)), rtol=1e-12
    )


def test_fully_confounded_treatment_raises():
    x = np.repeat([0.0, 1.0], 50)
    with pytest.raises(ValueError, match="fully explained"):
        covariate_moments(
            treatment=x,
            covariate_matrix=np.column_stack([np.ones(100), x]),
            null_mean=2.0,
            alternative_mean=2.0,
            expression_size=2.0,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"treatment": []},
        {"treatment": [0, 0, 0, 0]},
        {"treatment": [0, 1, 0.5, 1]},
        {"covariate_matrix": np.ones((3, 1))},
        {"covariate_matrix": np.arange(4)[:, None]},
        {"null_mean": 0.0},
        {"alternative_mean": np.nan},
        {"expression_size": -1.0},
    ],
)
def test_invalid_inputs_raise(overrides):
    args = dict(
        treatment=np.array([0, 1, 0, 1]),
        covariate_matrix=np.ones((4, 1)),
        null_mean=2.0,
        alternative_mean=2.0,
        expression_size=2.0,
    )
    args.update(overrides)
    with pytest.raises(ValueError):
        covariate_moments(**args)
