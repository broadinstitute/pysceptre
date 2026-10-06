"""Law tests for `nested_permutation_draws`, the shared draws for the NT-cells control group.

The contract: for every `k` in `[m, M]`, the first `k` entries of each row are
a uniformly random `k`-subset of `{0, ..., n_control + k - 1}`, and the nested
chain of those sets has the law of sceptre's `hybrid_fisher_iwor_sampler`.

Tested here: shape, dtype, distinctness, every prefix's range and the argument
checks; uniform prefix sets for every `k`; the inclusion rate of each prefix's
newest element and of every element; the joint law of the whole chain, against
its closed form and against a forward port of R's sampler run on numpy
uniforms, with a chain that has the right prefix laws but is not R's as the
case that must fail; and that a target's prefix is exactly what
`PermutationSliceDraws` materializes. All of it is seeded, so every p-value
below is fixed.
"""

from __future__ import annotations

import functools
import itertools
import math

import numpy as np
import pytest
from scipy import stats

from pysceptre.crt.permutations import nested_permutation_draws
from pysceptre.test_statistic.score_stat import PermutationSliceDraws

# The level every seeded goodness-of-fit check below must clear.
ALPHA = 1e-4


def _forward_reference(
    n_control: int, m: int, M: int, b: int, rng: np.random.Generator
) -> np.ndarray:
    """sceptre's `hybrid_fisher_iwor_sampler(N, m, M, B)`, step for step, on numpy uniforms.

    Ported from sceptre 0.10.3's `src/generate_samples_functions.cpp`. Each
    draw consumes `M` uniforms in R's order: `m` for the Fisher-Yates part,
    then one per inductive step. Row `j` of the `(b, M)` result is R's vector
    `v` for draw `j`, so for `k >= m` its first `k` entries are R's `k`-subset.
    R never calls this with `m = 0`, and neither do the tests.
    """
    N = n_control
    rows = []
    for u in rng.random((b, M)).tolist():
        x = list(range(N + max(m, 1)))
        for i in range(m):
            pos = math.floor((N + m - i) * u[i])
            x[pos], x[N + m - i - 1] = x[N + m - i - 1], x[pos]
        v = x[N : N + m] + [0] * (M - m)
        x[N] = N + m
        for i in range(m + 1, M + 1):
            p = i / (N + i)
            pos = N if u[i - 1] > 1 - p else math.floor(u[i - 1] * N / (1 - p))
            v[i - 1] = x[pos]
            x[pos] = x[N]
            x[N] = N + i
        rows.append(v)
    return np.array(rows, dtype=np.int64).reshape(b, M)


def _set_codes(draws: np.ndarray, k: int) -> np.ndarray:
    """Each row's first `k` entries as a bitmask: the set, with its order dropped."""
    return np.bitwise_or.reduce(np.int64(1) << draws[:, :k], axis=1)


def _uniform_set_support(n_control: int, k: int) -> np.ndarray:
    """The bitmask of every `k`-subset of `{0, ..., n_control + k - 1}`, sorted."""
    combos = itertools.combinations(range(n_control + k), k)
    return np.array(sorted(sum(1 << e for e in c) for c in combos), dtype=np.int64)


def _chain_codes(draws: np.ndarray, n_control: int, m: int) -> np.ndarray:
    """Each row's chain `(S_m, ..., S_M)` as one integer.

    `S_m` is the bitmask of the first `m` entries, so their order is dropped,
    and step `i > m` is the element it adds, `row[i - 1]`. Together these
    determine every prefix set from `m` to `M` and nothing else.
    """
    radix = n_control + draws.shape[1]
    codes = _set_codes(draws, m)
    for t in range(m, draws.shape[1]):
        codes = codes * radix + draws[:, t]
    return codes


def _chain_law(n_control: int, m: int, M: int) -> tuple[np.ndarray, np.ndarray]:
    """Every chain `(S_m, ..., S_M)` and its probability, encoded as `_chain_codes` does.

    `S_m` has probability `1 / C(n_control + m, m)`. Step `i` adds the newest
    element `n_control + i - 1` with probability `i / (n_control + i)`, and
    each of the `n_control` older elements not yet chosen with probability
    `1 / (n_control + i)`. Returns the codes sorted, and their probabilities.
    """
    radix = n_control + M
    law: dict[int, float] = {}

    def grow(chosen: frozenset[int], code: int, prob: float, i: int) -> None:
        if i > M:
            law[code] = prob
            return
        newest = n_control + i - 1
        grow(chosen | {newest}, code * radix + newest, prob * i / (n_control + i), i + 1)
        for e in range(newest):
            if e not in chosen:
                grow(chosen | {e}, code * radix + e, prob / (n_control + i), i + 1)

    for first in itertools.combinations(range(n_control + m), m):
        start = 1 / math.comb(n_control + m, m)
        grow(frozenset(first), sum(1 << e for e in first), start, m + 1)
    items = sorted(law.items())
    return (
        np.array([c for c, _ in items], dtype=np.int64),
        np.array([p for _, p in items]),
    )


def _counts(codes: np.ndarray, support: np.ndarray) -> np.ndarray:
    """How often each code in the sorted `support` occurs; every code must be in it."""
    assert np.isin(codes, support).all(), "a draw fell outside the support"
    return np.bincount(np.searchsorted(support, codes), minlength=support.size)


# --- 1. structure, arguments, edge cases -------------------------------------


@pytest.mark.parametrize(
    ("n_control", "m", "M", "b"),
    [
        (3, 1, 4, 500),
        (20, 3, 12, 500),
        pytest.param(1, 1, 30, 300, id="one-control-cell"),
        pytest.param(5, 0, 4, 200, id="m-is-0"),
        pytest.param(7, 4, 4, 300, id="m-equals-M"),
        pytest.param(0, 2, 6, 200, id="no-control-cells"),
    ],
)
def test_rows_are_distinct_and_every_prefix_stays_in_its_universe(n_control, m, M, b):
    draws = nested_permutation_draws(n_control, m, M, b, np.random.default_rng(0))
    assert draws.shape == (b, M)
    assert draws.dtype == np.int64
    assert draws.min() >= 0
    assert (np.diff(np.sort(draws, axis=1), axis=1) > 0).all(), "a row repeats an element"
    running_max = np.maximum.accumulate(draws, axis=1)
    for k in range(max(m, 1), M + 1):
        assert running_max[:, k - 1].max() < n_control + k, f"prefix {k} out of range"


def test_without_control_cells_each_prefix_is_its_whole_universe():
    """With no control cells the only `k`-subset of `{0, ..., k - 1}` is all of it."""
    draws = nested_permutation_draws(0, 2, 7, 200, np.random.default_rng(0))
    for k in range(2, 8):
        assert (np.sort(draws[:, :k], axis=1) == np.arange(k)).all()


@pytest.mark.parametrize(
    ("n_control", "m", "M", "b"),
    [
        (5, 3, 2, 10),
        (5, 0, -1, 10),
        (5, -1, 2, 10),
        (5, 1, 2, -1),
        (-1, 1, 2, 10),
    ],
)
def test_rejects_invalid_arguments(n_control, m, M, b):
    with pytest.raises(ValueError):
        nested_permutation_draws(n_control, m, M, b, np.random.default_rng(0))


@pytest.mark.parametrize(("m", "M", "b"), [(2, 5, 0), (0, 0, 7), (0, 0, 0)])
def test_empty_requests_return_empty_arrays(m, M, b):
    draws = nested_permutation_draws(10, m, M, b, np.random.default_rng(0))
    assert draws.shape == (b, M)
    assert draws.dtype == np.int64


def test_reproducible_for_a_fixed_seed_and_seed_dependent():
    a = nested_permutation_draws(30, 2, 9, 200, np.random.default_rng(3))
    b = nested_permutation_draws(30, 2, 9, 200, np.random.default_rng(3))
    c = nested_permutation_draws(30, 2, 9, 200, np.random.default_rng(4))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# --- 2. each prefix set is uniform -------------------------------------------


@pytest.mark.parametrize(
    ("n_control", "m", "M", "seed"),
    [(3, 1, 4, 11), (2, 2, 5, 12), (2, 0, 3, 13)],
)
def test_every_prefix_set_is_uniform(n_control, m, M, seed):
    b = 60_000
    draws = nested_permutation_draws(n_control, m, M, b, np.random.default_rng(seed))
    for k in range(max(m, 1), M + 1):
        support = _uniform_set_support(n_control, k)
        assert support.size == math.comb(n_control + k, k)
        counts = _counts(_set_codes(draws, k), support)
        assert stats.chisquare(counts).pvalue > ALPHA, f"prefix {k}"


# --- 3. inclusion rates --------------------------------------------------------


@pytest.mark.parametrize(
    ("n_control", "m", "M", "seed"),
    [(3, 1, 4, 21), (20, 3, 12, 22), (50, 5, 40, 23)],
)
def test_newest_element_is_in_prefix_k_at_rate_k_over_n_control_plus_k(n_control, m, M, seed):
    b = 20_000
    draws = nested_permutation_draws(n_control, m, M, b, np.random.default_rng(seed))
    for k in range(m, M + 1):
        hits = int((draws[:, :k] == n_control + k - 1).any(axis=1).sum())
        result = stats.binomtest(hits, b, k / (n_control + k))
        assert result.pvalue > ALPHA, f"prefix {k}: {hits} of {b}"


def test_every_element_is_in_prefix_k_at_the_same_rate():
    n_control, m, M, b = 50, 5, 40, 20_000
    draws = nested_permutation_draws(n_control, m, M, b, np.random.default_rng(31))
    n_checks = sum(n_control + k for k in range(m, M + 1))
    # Two-sided normal bound, Bonferroni over every (k, element) checked.
    z_max = stats.norm.isf(ALPHA / (2 * n_checks))
    for k in range(m, M + 1):
        p = k / (n_control + k)
        counts = np.bincount(draws[:, :k].ravel(), minlength=n_control + k)
        assert counts.size == n_control + k
        z = (counts - b * p) / math.sqrt(b * p * (1 - p))
        assert np.abs(z).max() < z_max, f"prefix {k}"


# --- 4. joint law of the nested chain ----------------------------------------

CHAIN_CASES = [(3, 1, 4), (2, 2, 5)]
CHAIN_DRAWS = 60_000


@functools.cache
def _chain_draws(n_control: int, m: int, M: int) -> dict[str, np.ndarray]:
    """Chain codes from both samplers, on independent fixed seeds."""
    seed = 1000 * n_control + 100 * m + M
    nested = nested_permutation_draws(n_control, m, M, CHAIN_DRAWS, np.random.default_rng(seed))
    forward = _forward_reference(n_control, m, M, CHAIN_DRAWS, np.random.default_rng(seed + 1))
    return {
        "nested": _chain_codes(nested, n_control, m),
        "forward": _chain_codes(forward, n_control, m),
    }


@pytest.mark.parametrize(("n_control", "m", "M"), CHAIN_CASES)
def test_chain_law_closed_form_is_a_distribution(n_control, m, M):
    codes, probs = _chain_law(n_control, m, M)
    assert codes.size == math.comb(n_control + m, m) * (n_control + 1) ** (M - m)
    assert np.unique(codes).size == codes.size
    assert probs.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("sampler", ["nested", "forward"])
@pytest.mark.parametrize(("n_control", "m", "M"), CHAIN_CASES)
def test_chain_law_matches_the_closed_form(n_control, m, M, sampler):
    """Run on the forward port too, which is what lets the next test lean on it."""
    support, probs = _chain_law(n_control, m, M)
    counts = _counts(_chain_draws(n_control, m, M)[sampler], support)
    expected = probs * (counts.sum() / probs.sum())
    assert stats.chisquare(counts, expected).pvalue > ALPHA


@pytest.mark.parametrize(("n_control", "m", "M"), CHAIN_CASES)
def test_chain_law_matches_a_forward_port_of_r(n_control, m, M):
    draws = _chain_draws(n_control, m, M)
    support = np.union1d(draws["nested"], draws["forward"])
    table = np.vstack([_counts(draws["nested"], support), _counts(draws["forward"], support)])
    _, pvalue, _, _ = stats.chi2_contingency(table)
    assert pvalue > ALPHA


def test_chain_test_rejects_a_coupling_with_the_same_prefix_laws():
    """Uniform prefix sets are not enough: a different chain must fail the chain test.

    With `n_control = 2`, `m = 1`, `M = 2`, `S_1 = {a}` is uniform and `S_2` adds
    the newest element with R's probability 1/2, but otherwise takes
    `(a + 1) % 3` with probability 0.3 and `(a + 2) % 3` with 0.2 instead of
    0.25 each. Every pair is still reached with probability 1/6, so both prefix
    sets are uniform and only the chain differs from R's.
    """
    n_control, m, M, b = 2, 1, 2, 60_000
    rng = np.random.default_rng(51)
    first = rng.integers(0, 3, size=b)
    u = rng.random(b)
    second = np.where(u < 0.5, 3, np.where(u < 0.8, (first + 1) % 3, (first + 2) % 3))
    draws = np.column_stack([first, second]).astype(np.int64)
    for k in (1, 2):
        counts = _counts(_set_codes(draws, k), _uniform_set_support(n_control, k))
        assert stats.chisquare(counts).pvalue > ALPHA
    support, probs = _chain_law(n_control, m, M)
    counts = _counts(_chain_codes(draws, n_control, m), support)
    assert stats.chisquare(counts, probs * b).pvalue < 1e-12


# --- 5. what a target reads ----------------------------------------------------


def test_target_prefix_is_what_permutation_slice_draws_materializes():
    """A target with `k` treated cells has `n_control + k` combined positions and reads `row[:k]`."""
    n_control, m, M, b = 40, 2, 15, 300
    perms = nested_permutation_draws(n_control, m, M, b, np.random.default_rng(41))
    for k in range(m, M + 1):
        n_cells = n_control + k
        source = PermutationSliceDraws(perms, k, n_cells)
        mat = source.slice(0, b)
        assert mat.shape == (b, n_cells)
        assert (np.asarray(mat.sum(axis=1)).ravel() == k).all()
        expected = np.zeros((b, n_cells))
        np.put_along_axis(expected, perms[:, :k], 1.0, axis=1)
        assert np.array_equal(mat.toarray(), expected), f"prefix {k}"
        assert np.array_equal(source.slice(100, 250).toarray(), expected[100:250])
    with pytest.raises(ValueError):
        PermutationSliceDraws(perms, M + 1, n_control + M + 1).slice(0, b)
