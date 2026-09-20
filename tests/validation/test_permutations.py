"""Permutation resampling, sceptre's alternative to the CRT for high-MOI.

Two things need testing here and they pull in different directions.

The **sampler** must be correct: uniformly random subsets without
replacement, and -- less obviously -- each row must be a random *ordering*,
because every target takes a prefix of it. A sampler that returned sorted
rows would look fine on inspection and be silently wrong, since the prefix
of a sorted subset is biased toward low cell indices.

The **mechanism** must behave as documented, including where it is worse
than the CRT. Permutations cannot be invariant to a change of pair list: the
shared draws are sized by the largest target, so adding a bigger one moves
every result. That is asserted rather than merely written down, so the
documentation cannot quietly drift away from the behaviour.
"""

from __future__ import annotations

import warnings
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from pysceptre.crt.permutations import draws_for_target, permutation_draws
from pysceptre.pipeline import discovery
from pysceptre.pipeline.api import _resampling_budget, run_discovery_analysis


def test_rows_hold_distinct_cells():
    draws = permutation_draws(60, 8, 500, np.random.default_rng(0))
    assert draws.shape == (500, 8)
    assert all(len(set(row.tolist())) == 8 for row in draws)
    assert draws.min() >= 0 and draws.max() < 60


def test_rows_are_orderings_not_sorted_sets():
    """The property every target's prefix depends on.

    If rows came back sorted, `row[:n_trt]` would be the n_trt *smallest*
    cell indices rather than a random subset -- a severe bias that nothing
    downstream would catch.
    """
    draws = permutation_draws(500, 20, 400, np.random.default_rng(0))
    ascending = (np.diff(draws, axis=1) > 0).all(axis=1)
    assert ascending.mean() < 0.05, "rows look sorted; prefixes would be biased"


def test_prefixes_are_uniform_over_cells():
    """A prefix must cover cells uniformly, not favour any part of the range."""
    n_cells, m, b = 40, 10, 4000
    draws = permutation_draws(n_cells, m, b, np.random.default_rng(1))
    first_two = draws[:, :2].ravel()
    counts = np.bincount(first_two, minlength=n_cells)
    expected = first_two.size / n_cells
    # Chi-square-ish sanity: no cell should be wildly over- or under-drawn.
    assert counts.min() > 0.7 * expected
    assert counts.max() < 1.3 * expected


def test_draws_are_reproducible_and_seed_dependent():
    a = permutation_draws(100, 6, 50, np.random.default_rng(3))
    b = permutation_draws(100, 6, 50, np.random.default_rng(3))
    c = permutation_draws(100, 6, 50, np.random.default_rng(4))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_sampler_rejects_impossible_and_degenerate_requests():
    with pytest.raises(ValueError, match="more cells than the dataset has"):
        permutation_draws(10, 11, 5, np.random.default_rng(0))
    assert permutation_draws(10, 0, 5, np.random.default_rng(0)).shape == (5, 0)
    assert permutation_draws(10, 3, 0, np.random.default_rng(0)).shape == (0, 3)


def test_target_prefix_is_the_right_size_and_left_unsorted():
    """R reads `curr_vect[j]` for `j < n_trt` and sorts nothing; so do we.

    Sorting would be 1.3M `np.sort` calls on a real run (~5% of runtime) to
    fix the order a sum is taken in. The prefix is already a uniformly random
    subset, so the sort buys nothing mathematically.
    """
    draws = permutation_draws(100, 10, 20, np.random.default_rng(0))
    got = draws_for_target(draws, 4)
    assert len(got) == 20
    assert all(len(x) == 4 for x in got)
    assert all(len(set(x.tolist())) == 4 for x in got)
    # Each prefix is exactly the head of its row, untouched.
    assert all(np.array_equal(x, row[:4]) for x, row in zip(got, draws, strict=True))
    assert not all(np.array_equal(x, np.sort(x)) for x in got), "prefixes look sorted"


def test_unsorted_indices_do_not_change_the_statistic_beyond_rounding():
    """The price of dropping the sort, asserted so it cannot grow unnoticed.

    scipy accepts unsorted CSR indices and sums a row's nonzeros in stored
    order, so the result differs only by floating-point associativity.
    """
    from scipy import sparse

    rng = np.random.default_rng(0)
    n_cells, b, n_trt, p = 500, 200, 40, 6
    perms = permutation_draws(n_cells, n_trt, b, rng)
    X = rng.normal(size=(n_cells, p))

    def csr(rows):
        ind = np.concatenate(rows)
        ptr = np.arange(len(rows) + 1, dtype=np.int64) * n_trt
        return sparse.csr_matrix((np.ones(ind.size), ind, ptr), shape=(len(rows), n_cells))

    unsorted = csr(draws_for_target(perms, n_trt)) @ X
    ordered = csr([np.sort(r) for r in draws_for_target(perms, n_trt)]) @ X
    assert np.abs(unsorted - ordered).max() < 1e-10
    assert np.allclose(unsorted, ordered, rtol=0, atol=1e-10)


def test_target_larger_than_the_shared_draws_is_refused():
    draws = permutation_draws(100, 5, 10, np.random.default_rng(0))
    with pytest.raises(ValueError, match="sized by the largest target"):
        draws_for_target(draws, 6)


def test_b3_matches_rs_rule_per_mechanism():
    """R: `B3 <- if (resampling_mechanism == "permutations") 24999L else 0L`."""
    assert _resampling_budget("skew_normal", 0, 100, 0.1, "crt") == (4999, 0)
    assert _resampling_budget("skew_normal", 0, 100, 0.1, "permutations") == (4999, 24999)


def _screen(n_genes=4, n_cells=1500, n_targets=5, seed=0, sizes=None):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n_cells), rng.normal(size=(n_cells, 2))])
    theta, mu = 5.0, 20.0
    Y = rng.negative_binomial(theta, theta / (theta + mu), size=(n_genes, n_cells)).astype(float)
    genes = [f"g{i}" for i in range(n_genes)]
    sizes = sizes or [120] * n_targets
    targets = {
        f"t{j}": np.sort(rng.choice(n_cells, sizes[j], replace=False)) for j in range(n_targets)
    }
    return Y, genes, X, targets


def _run(Y, genes, X, targets, target_sel=None, **kwargs):
    sel = target_sel or list(targets)
    pairs = pd.DataFrame(
        [(g, t) for g in genes for t in sel], columns=["response_id", "grna_target"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_discovery_analysis(
            response_matrix=Y,
            gene_ids=genes,
            covariate_matrix=X,
            grna_target_cells={t: targets[t] for t in sel},
            pairs=pairs,
            seed=0,
            chunk_memory_gb=8.0,
            **kwargs,
        ).set_index(["response_id", "grna_target"])


def test_permutations_run_end_to_end_and_give_usable_p_values():
    Y, genes, X, targets = _screen()
    out = _run(Y, genes, X, targets, resampling_mechanism="permutations")
    assert len(out) == len(genes) * len(targets)
    assert out.p_value.between(0, 1).all()
    assert out.p_value.notna().all()


def test_permutations_are_reproducible_for_a_fixed_pair_list():
    """Not composition-invariant is not the same as not deterministic."""
    Y, genes, X, targets = _screen()
    a = _run(Y, genes, X, targets, resampling_mechanism="permutations")
    b = _run(Y, genes, X, targets, resampling_mechanism="permutations")
    np.testing.assert_array_equal(a.p_value.to_numpy(), b.p_value.to_numpy())


def test_permutations_do_not_depend_on_chunking_or_workers():
    """The draws are made once for the whole run, not per chunk."""
    Y, genes, X, targets = _screen()
    base = _run(Y, genes, X, targets, resampling_mechanism="permutations")
    for kwargs in ({"target_chunk_size": 2}, {"n_jobs": 3}):
        other = _run(Y, genes, X, targets, resampling_mechanism="permutations", **kwargs)
        np.testing.assert_array_equal(base.p_value.to_numpy(), other.p_value.to_numpy())


def test_adding_a_larger_target_moves_every_permutation_result():
    """The documented limitation, asserted so it cannot drift.

    M is the largest target's cell count, so a bigger target resizes the
    shared draws and every result changes. This is the property the CRT has
    and permutations cannot.
    """
    Y, genes, X, targets = _screen(n_targets=5, sizes=[120, 120, 120, 120, 400])
    small = [f"t{j}" for j in range(4)]  # all 120 cells
    base = _run(Y, genes, X, targets, target_sel=small, resampling_mechanism="permutations")
    bigger = _run(
        Y, genes, X, targets, target_sel=[*small, "t4"], resampling_mechanism="permutations"
    )
    common = base.index.intersection(bigger.index)
    assert len(common) == len(base)
    changed = base.loc[common, "p_value"].to_numpy() != bigger.loc[common, "p_value"].to_numpy()
    assert changed.any(), "adding a larger target should have moved the shared draws"


def test_the_crt_is_unaffected_by_the_same_change():
    """The contrast that justifies keeping the CRT the default."""
    Y, genes, X, targets = _screen(n_targets=5, sizes=[120, 120, 120, 120, 400])
    small = [f"t{j}" for j in range(4)]
    base = _run(Y, genes, X, targets, target_sel=small)
    bigger = _run(Y, genes, X, targets, target_sel=[*small, "t4"])
    common = base.index.intersection(bigger.index)
    np.testing.assert_array_equal(
        base.loc[common, "p_value"].to_numpy(), bigger.loc[common, "p_value"].to_numpy()
    )


def test_unknown_mechanism_is_rejected():
    Y, genes, X, targets = _screen()
    with pytest.raises(ValueError, match="resampling_mechanism must be one of"):
        _run(Y, genes, X, targets, resampling_mechanism="bootstrap")


def test_no_logistic_fit_is_performed_on_the_permutation_path():
    """The gRNA logistic fit exists only to draw from, so permutations skip it.

    R makes the same split -- `perform_grna_precomputation` is reached from
    `crt_glm_factored_out` and `discovery_ntcells_crt` but never from
    `perm_test_glm_factored_out` -- and running it anyway was the single
    largest term in a permutation profile, at 36% of runtime, for a result
    with no reader. Asserted on the call itself because nothing in the
    output would reveal the waste.
    """
    Y, genes, X, targets = _screen()
    calls = []
    real = discovery.fit_binomial_glm_batch

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    with mock.patch.object(discovery, "fit_binomial_glm_batch", counting):
        _run(Y, genes, X, targets, resampling_mechanism="permutations")
        assert calls == [], "a logistic fit was performed for permutations"
        _run(Y, genes, X, targets)
        assert calls, "the CRT still needs the fit"


def test_targets_carry_no_fitted_probabilities_under_permutations():
    """The field is `None` rather than stale, so a future reader fails loudly."""
    Y, genes, X, targets = _screen()
    perms = permutation_draws(
        X.shape[0], max(len(v) for v in targets.values()), 20, np.random.default_rng(0)
    )
    got = discovery.fit_all_targets(targets, X, B1=10, B2=10, B3=0, seed=0, permutations=perms)
    assert all(p.fitted_probabilities is None for p in got.values())

    crt = discovery.fit_all_targets(targets, X, B1=10, B2=10, B3=0, seed=0)
    assert all(p.fitted_probabilities is not None for p in crt.values())


def test_prefix_sums_match_the_per_target_matmul_bit_for_bit():
    """The whole point: two routes to the same segment sums.

    Bit-identity is available only because the prefixes are no longer sorted
    -- CSR stored order is then the permutation row order, which is also the
    order `cumsum` accumulates in. Sorting made the two orders differ and
    this would have been a ~1e-13 agreement instead.
    """
    from pysceptre.test_statistic.score_stat import PermutationPrefixSums, draws_to_matrix

    rng = np.random.default_rng(0)
    n_cells, b, m, k = 3000, 200, 120, 13
    perms = permutation_draws(n_cells, m, b, rng)
    stacked = rng.normal(size=(n_cells, k))
    scan = PermutationPrefixSums(stacked, perms)._scan(0, b)
    for n_trt in (1, 7, 60, 119, 120):
        via_matmul = draws_to_matrix(draws_for_target(perms, n_trt), n_cells) @ stacked
        np.testing.assert_array_equal(scan[:, n_trt - 1, :], via_matmul)


def test_prefix_sums_decline_a_stage_too_large_to_scan():
    """`None` means "use the draw matrix", which is how B2 and B3 stay bounded."""
    from pysceptre.test_statistic.score_stat import PermutationPrefixSums

    rng = np.random.default_rng(0)
    perms = permutation_draws(1000, 50, 100, rng)
    stacked = rng.normal(size=(1000, 13))
    assert PermutationPrefixSums(stacked, perms, max_bytes=1e9).statistics(0, 100, 10) is not None
    assert PermutationPrefixSums(stacked, perms, max_bytes=1.0).statistics(0, 100, 10) is None


def test_the_default_budget_admits_a_real_stage_one_scan():
    """The gate must fire in the *useful* direction at real dimensions.

    The first default was 64 MB, picked against an `m` of a few hundred. On
    day0 `m` is 1,772 and stage 1 needs 92 MB, so every scan was declined and
    the class never ran on real data -- while every test passed, because they
    all used `m = 120` where the scan is 1.2 MB. So this asserts at day0's
    shape, and that the escalation stages are still refused.
    """
    from pysceptre.test_statistic.score_stat import PermutationPrefixSums

    n_cells, m, k = 40_000, 1_772, 13
    B1, B2 = 499, 4999
    rng = np.random.default_rng(0)
    stacked = rng.normal(size=(n_cells, k))

    perms = permutation_draws(n_cells, m, B1, rng)
    got = PermutationPrefixSums(stacked, perms).statistics(0, B1, 558)
    assert got is not None, "stage 1 at day0's dimensions must use the scan"
    assert got.shape == (B1,)

    # B2 at the same `m` is 921 MB and must still fall back. The rows need
    # only the right shape: the budget is checked before anything is
    # gathered, so drawing real permutations here would cost seconds to
    # exercise a branch that never looks at them.
    wide = np.zeros((B2, m), dtype=np.int64)
    assert PermutationPrefixSums(stacked, wide).statistics(0, B2, 558) is None


def test_prefix_sums_refuse_a_target_longer_than_the_shared_rows():
    from pysceptre.test_statistic.score_stat import PermutationPrefixSums

    rng = np.random.default_rng(0)
    perms = permutation_draws(1000, 20, 50, rng)
    pre = PermutationPrefixSums(rng.normal(size=(1000, 13)), perms)
    with pytest.raises(ValueError, match="different target set"):
        pre.statistics(0, 50, 21)


def test_the_prefix_route_and_the_matmul_route_agree_end_to_end():
    """Disable the fast route and the whole analysis must be unchanged.

    Asserted on a full run rather than on the sums alone, so that a mistake
    in *which* column a target reads -- an off-by-one in `n_trt` would be
    silent and plausible -- cannot pass.
    """
    from pysceptre.test_statistic.score_stat import PermutationPrefixSums

    Y, genes, X, targets = _screen()
    fast = _run(Y, genes, X, targets, resampling_mechanism="permutations")
    with mock.patch.object(PermutationPrefixSums, "statistics", return_value=None):
        slow = _run(Y, genes, X, targets, resampling_mechanism="permutations")
    np.testing.assert_array_equal(fast.p_value.to_numpy(), slow.p_value.to_numpy())
    np.testing.assert_array_equal(fast.z_orig.to_numpy(), slow.z_orig.to_numpy())
    np.testing.assert_array_equal(fast.stage.to_numpy(), slow.stage.to_numpy())


def test_the_crt_does_not_use_the_prefix_route():
    """It has no shared ordering to scan along, so the matmul must stay."""
    from pysceptre.test_statistic.score_stat import PermutationPrefixSums

    Y, genes, X, targets = _screen()
    with mock.patch.object(
        PermutationPrefixSums, "statistics", side_effect=AssertionError("CRT used prefix sums")
    ):
        _run(Y, genes, X, targets)
