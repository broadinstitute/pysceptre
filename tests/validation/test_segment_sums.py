"""Tests for the reduceat-based segment sums in test_statistic/score_stat.py.

The optimization replaces one `np.bincount` per row of D with a single
`np.add.reduceat`, which is only valid because `flatten_synthetic_idxs`
produces *contiguous* segments. The empty-segment case is the trap: reduceat
given a repeated offset returns the element at that offset rather than 0.
"""

import numpy as np

from pysceptre.test_statistic.score_stat import (
    _segment_sums,
    compute_null_full_statistics,
    flatten_synthetic_idxs,
)


def _reference_segment_sums(values, lengths, B):
    """Obvious, slow implementation: explicit per-segment slicing."""
    values = np.atleast_2d(values)
    out = np.zeros((values.shape[0], B))
    start = 0
    for j, n in enumerate(lengths):
        if n:
            out[:, j] = values[:, start : start + n].sum(axis=1)
        start += n
    return out


def test_matches_reference_on_ragged_segments():
    rng = np.random.default_rng(0)
    lengths = np.array([3, 1, 7, 2, 5])
    values = rng.normal(size=(4, lengths.sum()))
    np.testing.assert_allclose(
        _segment_sums(values, lengths, len(lengths)),
        _reference_segment_sums(values, lengths, len(lengths)),
    )


def test_empty_segment_yields_exact_zero_not_a_stray_element():
    """The reduceat trap. With lengths [3, 0, 2], a naive reduceat returns the
    element at the repeated offset (1.0 here) for the middle segment."""
    lengths = np.array([3, 0, 2])
    values = np.ones((2, 5))
    got = _segment_sums(values, lengths, 3)
    np.testing.assert_array_equal(got, np.array([[3.0, 0.0, 2.0], [3.0, 0.0, 2.0]]))

    # and the naive version really does get it wrong, so this test has teeth
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    naive = np.add.reduceat(values, offsets, axis=-1)
    assert naive[0][1] == 1.0


def test_all_segments_empty():
    lengths = np.array([0, 0, 0])
    got = _segment_sums(np.empty((2, 0)), lengths, 3)
    np.testing.assert_array_equal(got, np.zeros((2, 3)))


def test_leading_and_trailing_empty_segments():
    lengths = np.array([0, 2, 0, 3, 0])
    values = np.arange(1.0, 6.0)[None, :]  # [1,2] then [3,4,5]
    got = _segment_sums(values, lengths, 5)
    np.testing.assert_array_equal(got, np.array([[0.0, 3.0, 0.0, 12.0, 0.0]]))


def test_one_dimensional_input():
    lengths = np.array([2, 0, 3])
    got = _segment_sums(np.ones(5), lengths, 3)
    np.testing.assert_array_equal(got, np.array([2.0, 0.0, 3.0]))


def test_statistic_is_unchanged_by_the_optimization(ground_truth):
    """Guards the real invariant: the null statistics for the R-drawn signal
    pair must equal what the slow per-segment reference produces."""
    sp = ground_truth["signal_pair"]
    target = next(t for t in ground_truth["targets"] if t["target_id"] == sp["target_id"])
    a, w, D = np.array(sp["a"]), np.array(sp["w"]), np.array(sp["D"])
    synthetic = [np.array(i) for i in target["synthetic_idxs_0based"]][:400]

    got = compute_null_full_statistics(a, w, D, synthetic)

    flat, lengths, B = flatten_synthetic_idxs(synthetic)
    top = _reference_segment_sums(a[flat], lengths, B)[0]
    ll = _reference_segment_sums(w[flat], lengths, B)[0]
    lr = np.sum(_reference_segment_sums(D[:, flat], lengths, B) ** 2, axis=0)
    np.testing.assert_allclose(got, top / np.sqrt(ll - lr), rtol=1e-12)
