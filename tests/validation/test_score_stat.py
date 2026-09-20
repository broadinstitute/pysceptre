import numpy as np

from pysceptre.precompute.pieces import compute_precomputation_pieces
from pysceptre.test_statistic.empirical_p import compute_empirical_p_value
from pysceptre.test_statistic.fold_change import estimate_log_fold_change
from pysceptre.test_statistic.score_stat import (
    compute_null_full_statistics,
    compute_null_statistics_from_draws,
    compute_observed_full_statistic,
    draws_to_matrix,
    stack_pieces,
)


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


def test_both_null_statistic_paths_reproduce_rs_p_value_from_rs_own_draws(ground_truth):
    """Feed R's *exact* synthetic draws through the statistic and match R's p.

    This replaces a test that claimed to be the "strongest possible
    validation of the resampling statistic formula" and asserted
    `0.0 < p <= 1.0` -- true of almost any input, and never compared against
    R at all. It also ran only `compute_null_full_statistics`, the gather
    implementation, which **no production code calls**: every analysis goes
    through `compute_null_statistics_from_draws`. So the one exact-against-R
    check on the null statistic was exercising retired code with an assertion
    that could not fail.

    Both paths are checked here, against R's reported p-value, so the live
    one is covered and the two cannot drift apart unnoticed.
    """
    X = np.array(ground_truth["X"])
    B1 = ground_truth["B1"]
    checked = 0
    for pair in ground_truth["pairs"]:
        if pair["stage"] != 1:
            continue
        gene = _gene_by_id(ground_truth, pair["gene_id"])
        target = _target_by_id(ground_truth, pair["target_id"])
        y = np.array(gene["y"], dtype=float)
        pieces = compute_precomputation_pieces(y, X, np.array(gene["fitted_coefs"]), gene["theta"])
        trt_idxs = np.array(target["trt_idxs_1based"]) - 1
        z_orig = compute_observed_full_statistic(pieces.a, pieces.w, pieces.D, trt_idxs)
        b1_idxs = [np.array(idxs) for idxs in target["synthetic_idxs_0based"][:B1]]

        gathered = compute_null_full_statistics(pieces.a, pieces.w, pieces.D, b1_idxs)
        stacked = stack_pieces(pieces.a, pieces.w, pieces.D)
        matmul = compute_null_statistics_from_draws(stacked, draws_to_matrix(b1_idxs, X.shape[0]))
        # The two differ only by accumulation order in a sparse matmul.
        np.testing.assert_allclose(matmul, gathered, rtol=0, atol=1e-9)

        for null_stats in (gathered, matmul):
            p = compute_empirical_p_value(null_stats, z_orig, side=0)
            # An empirical p is a rank over B1 draws, so the tolerance is one
            # count: a last-bit difference in a null statistic may move one
            # comparison, and nothing finer than 1/(B1+1) is meaningful.
            assert abs(p - pair["p_value"]) <= 1.0 / (B1 + 1) + 1e-12, (
                f"{pair['gene_id']}/{pair['target_id']}: got {p}, R reported {pair['p_value']}"
            )
        checked += 1
    assert checked > 0, "no stage-1 pairs in the fixture; this test checked nothing"
