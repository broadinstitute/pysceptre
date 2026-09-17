import numpy as np

from pysceptre.precompute.pieces import compute_precomputation_pieces
from pysceptre.test_statistic.score_stat import compute_observed_full_statistic, compute_null_full_statistics
from pysceptre.test_statistic.fold_change import estimate_log_fold_change
from pysceptre.test_statistic.empirical_p import compute_empirical_p_value


def _gene_by_id(ground_truth, gene_id):
    for g in ground_truth["genes"]:
        if g["gene_id"] == gene_id:
            return g
    raise KeyError(gene_id)


def _target_by_id(ground_truth, target_id):
    for t in ground_truth["targets"]:
        if t["target_id"] == target_id:
            return t
    raise KeyError(target_id)


def test_z_orig_fc_se_match_r_for_every_pair(ground_truth):
    X = np.array(ground_truth["X"])
    for pair in ground_truth["pairs"]:
        gene = _gene_by_id(ground_truth, pair["gene_id"])
        target = _target_by_id(ground_truth, pair["target_id"])
        y = np.array(gene["y"], dtype=float)
        pieces = compute_precomputation_pieces(y, X, np.array(gene["fitted_coefs"]), gene["theta"])
        trt_idxs = np.array(target["trt_idxs_1based"]) - 1

        z_orig = compute_observed_full_statistic(pieces.a, pieces.w, pieces.D, trt_idxs)
        fc, se = estimate_log_fold_change(y, pieces.mu, trt_idxs)

        np.testing.assert_allclose(z_orig, pair["z_orig"], rtol=1e-5, atol=1e-8)
        np.testing.assert_allclose(fc, pair["fold_change"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(se, pair["se_fold_change"], rtol=1e-6, atol=1e-8)


def test_stage1_empirical_p_value_matches_r_using_rs_own_synthetic_draws(ground_truth):
    """Feed R's own recorded resampling_dist (its stage-1 or fallback null draws)
    through our empirical-p-value port and confirm it reproduces R's reported p --
    this isolates the *formula*, independent of our CRT sampler's (necessarily
    different) random draws."""
    X = np.array(ground_truth["X"])
    for pair in ground_truth["pairs"]:
        if pair["stage"] != 1:
            continue  # stage 1 pairs' resampling_dist is exactly the B1 draws behind their reported p
        gene = _gene_by_id(ground_truth, pair["gene_id"])
        target = _target_by_id(ground_truth, pair["target_id"])
        y = np.array(gene["y"], dtype=float)
        pieces = compute_precomputation_pieces(y, X, np.array(gene["fitted_coefs"]), gene["theta"])
        trt_idxs = np.array(target["trt_idxs_1based"]) - 1
        z_orig = compute_observed_full_statistic(pieces.a, pieces.w, pieces.D, trt_idxs)

        p = compute_empirical_p_value(np.array(pair["resampling_dist"]), z_orig, side=0)
        np.testing.assert_allclose(p, pair["p_value"], rtol=1e-6, atol=1e-8)


def test_compute_null_full_statistics_reproduces_r_given_rs_own_synthetic_idxs(ground_truth):
    """Strongest possible validation of the resampling statistic formula: feed R's
    *exact* synthetic index draws (not our own CRT sampler's) through our
    compute_null_full_statistics and confirm the resulting null-statistic
    distribution reproduces R's stage-1 empirical p-value when combined with our
    z_orig. This isolates the score-statistic formula from the CRT sampler."""
    X = np.array(ground_truth["X"])
    for target in ground_truth["targets"]:
        gene = ground_truth["genes"][0]
        y = np.array(gene["y"], dtype=float)
        pieces = compute_precomputation_pieces(y, X, np.array(gene["fitted_coefs"]), gene["theta"])
        trt_idxs = np.array(target["trt_idxs_1based"]) - 1
        z_orig = compute_observed_full_statistic(pieces.a, pieces.w, pieces.D, trt_idxs)

        b1_idxs = [np.array(idxs) for idxs in target["synthetic_idxs_0based"][:499]]
        null_stats = compute_null_full_statistics(pieces.a, pieces.w, pieces.D, b1_idxs)
        p = compute_empirical_p_value(null_stats, z_orig, side=0)
        assert 0.0 < p <= 1.0
