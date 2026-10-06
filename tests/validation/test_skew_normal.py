import numpy as np

from pysceptre.test_statistic.skew_normal import fit_and_evaluate_skew_normal, fit_skew_normal_funct


def test_fit_skew_normal_funct_matches_r(ground_truth):
    sn = ground_truth["standalone"]["skew_normal"]
    y = np.array(sn["input_null_stats"])
    fit = fit_skew_normal_funct(y)
    xi_r, omega_r, alpha_r, mean_r, sd_r = sn["fit"]
    np.testing.assert_allclose(fit.xi, xi_r, rtol=1e-6)
    np.testing.assert_allclose(fit.omega, omega_r, rtol=1e-6)
    np.testing.assert_allclose(fit.alpha, alpha_r, rtol=1e-6)
    np.testing.assert_allclose(fit.mean, mean_r, rtol=1e-6)
    np.testing.assert_allclose(fit.sd, sd_r, rtol=1e-6)


def test_fit_and_evaluate_skew_normal_matches_r(ground_truth):
    sn = ground_truth["standalone"]["skew_normal"]
    y = np.array(sn["input_null_stats"])
    z_test = sn["z_test"]
    result = fit_and_evaluate_skew_normal(z_test, y, side_code=0)
    xi_r, omega_r, alpha_r, p_r = sn["eval_result"]
    np.testing.assert_allclose(result.xi, xi_r, rtol=1e-6)
    np.testing.assert_allclose(result.omega, omega_r, rtol=1e-6)
    np.testing.assert_allclose(result.alpha, alpha_r, rtol=1e-6)
    np.testing.assert_allclose(result.p, p_r, rtol=1e-4)


def test_fit_and_evaluate_skew_normal_matches_r_on_real_signal_pair_resampling_dist(ground_truth):
    """Uses R's own stage-2 resampling_dist (its B2=4999 CRT null draws) for the
    real signal pair, which escalated to stage 2 and had its SN fit accepted --
    confirms the full escalation-relevant SN machinery against a real case, not
    just the synthetic standalone example."""
    sp = ground_truth["signal_pair"]
    assert sp["stage"] == 2
    null_stats = np.array(sp["resampling_dist"])
    result = fit_and_evaluate_skew_normal(sp["z_orig"], null_stats, side_code=0)
    xi_r, omega_r, alpha_r = sp["sn_params"]
    np.testing.assert_allclose(result.xi, xi_r, rtol=1e-6)
    np.testing.assert_allclose(result.omega, omega_r, rtol=1e-6)
    np.testing.assert_allclose(result.alpha, alpha_r, rtol=1e-6)
    assert result.used
    np.testing.assert_allclose(result.p, sp["p_value"], rtol=1e-3)


def test_a_degenerate_null_falls_back_instead_of_raising():
    """sceptre's C++ takes the square root of a negative variance or skewness term
    as NaN and rejects the fit; the port used to raise a math domain error and
    abort the run."""
    from pysceptre.test_statistic.skew_normal import fit_and_evaluate_skew_normal

    rng = np.random.default_rng(0)
    flat = 1238858.24 + rng.normal(0, 1e-9, 4999)
    skewed = -(rng.exponential(1, 4999) ** 3)
    for nulls in (flat, skewed):
        with np.errstate(all="ignore"):
            assert not fit_and_evaluate_skew_normal(5.0, nulls, 0).used
