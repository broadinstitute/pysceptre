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

import numpy as np
import pandas as pd
import pytest

from pysceptre.crt.permutations import draws_for_target, permutation_draws
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


def test_target_prefix_is_sorted_and_sized():
    draws = permutation_draws(100, 10, 20, np.random.default_rng(0))
    got = draws_for_target(draws, 4)
    assert len(got) == 20
    assert all(len(x) == 4 for x in got)
    # Sorted per draw, matching what the CRT sampler returns, so the CSR the
    # statistic consumes is canonical either way.
    assert all(np.array_equal(x, np.sort(x)) for x in got)
    assert all(len(set(x.tolist())) == 4 for x in got)


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
