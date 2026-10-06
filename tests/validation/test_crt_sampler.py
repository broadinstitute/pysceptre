import numpy as np
from scipy import stats

from pysceptre.crt.sampler import (
    crt_index_sampler_exact,
    crt_index_sampler_fast,
    crt_index_sampler_naive,
)
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
    assert ks_pvalue > 0.01, (
        f"null-statistic distributions differ (KS p={ks_pvalue}, stat={ks_stat})"
    )


# --- the exact sampler, for the NT-cells control group -------------------------------


def _inclusion_rate(idxs_list, n_cells):
    counts = np.zeros(n_cells)
    for idxs in idxs_list:
        counts[idxs] += 1
    return counts / len(idxs_list)


def test_exact_sampler_includes_each_cell_at_its_probability():
    """Against the NT cells a target is a large share of its combined cells, so
    probabilities run well past what the fast sampler's approximation assumes."""
    p = np.array([0.0, 1e-10, 0.002, 0.05, 0.25, 0.5, 0.9, 1.0])
    B = 20000
    rate = _inclusion_rate(crt_index_sampler_exact(p, B, np.random.default_rng(0)), p.size)
    se = np.sqrt(p * (1 - p) / B)
    assert np.all(np.abs(rate - p) <= 4 * se + 1e-12)
    assert rate[0] == 0.0 and rate[-1] == 1.0


def test_exact_sampler_never_lists_a_cell_twice_in_one_draw():
    p = np.full(300, 0.4)
    for idxs in crt_index_sampler_exact(p, 5000, np.random.default_rng(1)):
        assert np.all(np.diff(idxs) > 0)


def test_fast_sampler_double_counts_cells_when_probabilities_are_large():
    """Why the NT-cells CRT does not use it: at p = 0.4 its with-replacement
    placement lists the same cell twice in one draw routinely. Kept as a test so
    the reason stays checkable rather than remembered."""
    p = np.full(300, 0.4)
    draws = crt_index_sampler_fast(p, 5000, np.random.default_rng(1))
    repeats = sum(idxs.size - np.unique(idxs).size for idxs in draws)
    assert repeats > 1000


def test_exact_sampler_draws_cells_independently():
    """Two cells' inclusions in the same draw are independent, as R's are."""
    p = np.array([0.3, 0.3, 0.6])
    B = 40000
    draws = crt_index_sampler_exact(p, B, np.random.default_rng(2))
    both = np.mean([(0 in d) and (1 in d) for d in draws])
    assert abs(both - 0.09) < 4 * np.sqrt(0.09 * 0.91 / B)


def test_exact_sampler_matches_the_naive_reference():
    a = np.random.default_rng(5).normal(size=1500)
    p = np.random.default_rng(6).uniform(0.05, 0.45, size=1500)
    exact = [a[d].sum() for d in crt_index_sampler_exact(p, 4000, np.random.default_rng(7))]
    naive = [a[d].sum() for d in crt_index_sampler_naive(p, 4000, np.random.default_rng(8))]
    assert stats.ks_2samp(exact, naive).pvalue > 0.01


def test_exact_sampler_handles_tiny_draw_counts():
    for B in (1, 2, 7):
        draws = crt_index_sampler_exact(np.array([1.0, 0.5, 0.0]), B, np.random.default_rng(3))
        assert len(draws) == B
        assert all(0 in d and 2 not in d for d in draws)
    assert crt_index_sampler_exact(np.array([0.5]), 0, np.random.default_rng(3)) == []


def test_the_sampler_is_chosen_by_the_targets_share_of_the_cells():
    """Above 0.2% of the cells the fast sampler's repeats stop being negligible.
    The choice reads only the counts, so chunking cannot flip it."""
    from pysceptre.crt.sampler import crt_index_sampler

    p = np.full(10_000, 0.002)
    for n_trt, expected in ((20, crt_index_sampler_fast), (21, crt_index_sampler_exact)):
        got = crt_index_sampler(p, 50, np.random.default_rng(9), n_trt)
        want = expected(p, 50, np.random.default_rng(9))
        assert all(np.array_equal(a, b) for a, b in zip(got, want, strict=True))
