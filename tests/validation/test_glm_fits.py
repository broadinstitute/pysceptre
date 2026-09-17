import numpy as np

from pysceptre.glm.irls import fit_binomial_glm_batch, fit_poisson_glm_batch
from pysceptre.glm.nb_theta import estimate_theta, perform_response_precomputation


def test_poisson_coefs_match_r_per_gene(ground_truth):
    X = np.array(ground_truth["X"])
    for gene in ground_truth["genes"]:
        y = np.array(gene["y"], dtype=float)
        fit = fit_poisson_glm_batch(X, y)
        np.testing.assert_allclose(fit.coefs, gene["fitted_coefs"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(fit.fitted_values, gene["mu"], rtol=1e-6, atol=1e-8)


def test_theta_matches_r_per_gene(ground_truth):
    X = np.array(ground_truth["X"])
    n, p = X.shape
    for gene in ground_truth["genes"]:
        y = np.array(gene["y"], dtype=float)
        fit = fit_poisson_glm_batch(X, y)
        theta_est, _method = estimate_theta(
            y=y, mu=fit.fitted_values, dfr=n - p, limit=50, eps=np.finfo(float).eps ** 0.25
        )
        theta = max(min(theta_est, 1000.0), 0.01)
        np.testing.assert_allclose(theta, gene["theta"], rtol=1e-4)


def test_perform_response_precomputation_end_to_end(ground_truth):
    X = np.array(ground_truth["X"])
    for gene in ground_truth["genes"]:
        y = np.array(gene["y"], dtype=float)
        coefs, theta = perform_response_precomputation(y, X)
        np.testing.assert_allclose(coefs, gene["fitted_coefs"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(theta, gene["theta"], rtol=1e-4)


def test_binomial_fitted_probabilities_match_r_per_target(ground_truth):
    X = np.array(ground_truth["X"])
    n = X.shape[0]
    for target in ground_truth["targets"]:
        indicator = np.zeros(n)
        trt_idxs_0based = np.array(target["trt_idxs_1based"]) - 1
        indicator[trt_idxs_0based] = 1.0
        fit = fit_binomial_glm_batch(X, indicator)
        np.testing.assert_allclose(
            fit.fitted_values, target["fitted_probabilities"], rtol=1e-6, atol=1e-8
        )


def test_standalone_nb_theta(ground_truth):
    st = ground_truth["standalone"]["nb_theta"]
    y = np.array(st["y"])
    mu = np.array(st["mu"])
    theta_est, method = estimate_theta(
        y=y, mu=mu, dfr=st["dfr"], limit=50, eps=np.finfo(float).eps ** 0.25
    )
    expected_theta, expected_method = st["theta_result"]
    np.testing.assert_allclose(theta_est, expected_theta, rtol=1e-4)
    assert method == int(expected_method)
