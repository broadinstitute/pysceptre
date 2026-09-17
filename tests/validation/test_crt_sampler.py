import numpy as np
from scipy import stats

from pysceptre.crt.sampler import crt_index_sampler_fast, crt_index_sampler_naive
from pysceptre.precompute.pieces import compute_precomputation_pieces
from pysceptre.test_statistic.score_stat import compute_null_full_statistics


def test_fast_sampler_per_cell_inclusion_rate_matches_fitted_probabilities(ground_truth):
    """Distributional check (not bit-for-bit, see sampler.py docstring): over
    many draws, the fraction of synthetic sets that include cell j should be
    close to fitted_probabilities[j]."""
    rng = np.random.default_rng(0)
    target = ground_truth["targets"][0]
    p = np.array(target["fitted_probabilities"])
    B = 3000
    synthetic_idxs = crt_index_sampler_fast(p, B, rng)

    inclusion_count = np.zeros(p.size)
    for idxs in synthetic_idxs:
        inclusion_count[idxs] += 1
    inclusion_rate = inclusion_count / B

    # Monte Carlo error at B=3000 for p~0.1 is sd ~ sqrt(p(1-p)/B) ~ 0.0055; allow generous slack
    np.testing.assert_allclose(inclusion_rate, p, atol=0.03)


def test_fast_sampler_matches_naive_sampler_distributionally(ground_truth):
    """The 'fast' sampler (Binomial count + WOR placement) and the 'naive'
    sampler (independent per-cell Bernoulli draws) are two different algorithms
    for the *same* distribution -- cross-check via per-cell inclusion rate."""
    rng_fast = np.random.default_rng(1)
    rng_naive = np.random.default_rng(2)
    target = ground_truth["targets"][1]
    p = np.array(target["fitted_probabilities"])
    B = 3000

    fast_idxs = crt_index_sampler_fast(p, B, rng_fast)
    naive_idxs = crt_index_sampler_naive(p, B, rng_naive)

    def inclusion_rate(idxs_list):
        counts = np.zeros(p.size)
        for idxs in idxs_list:
            counts[idxs] += 1
        return counts / B

    rate_fast = inclusion_rate(fast_idxs)
    rate_naive = inclusion_rate(naive_idxs)
    np.testing.assert_allclose(rate_fast, rate_naive, atol=0.035)


def test_null_statistic_distribution_from_our_sampler_resembles_rs(ground_truth):
    """KS-test comparison: the distribution of null test statistics produced by
    our CRT sampler (feeding into the real score-statistic formula) should
    resemble R's recorded null-statistic distribution for the same
    (a, w, D, fitted_probabilities) -- not identical (different RNG draws),
    but statistically indistinguishable."""
    X = np.array(ground_truth["X"])
    rng = np.random.default_rng(42)

    gene = ground_truth["genes"][0]
    target = ground_truth["targets"][0]
    y = np.array(gene["y"], dtype=float)
    pieces = compute_precomputation_pieces(y, X, np.array(gene["fitted_coefs"]), gene["theta"])
    p = np.array(target["fitted_probabilities"])

    B = 499
    our_synthetic_idxs = crt_index_sampler_fast(p, B, rng)
    our_null_stats = compute_null_full_statistics(pieces.a, pieces.w, pieces.D, our_synthetic_idxs)

    r_synthetic_idxs = [np.array(idxs) for idxs in target["synthetic_idxs_0based"][:B]]
    r_null_stats = compute_null_full_statistics(pieces.a, pieces.w, pieces.D, r_synthetic_idxs)

    ks_stat, ks_pvalue = stats.ks_2samp(our_null_stats, r_null_stats)
    assert ks_pvalue > 0.01, f"null-statistic distributions differ (KS p={ks_pvalue}, stat={ks_stat})"
