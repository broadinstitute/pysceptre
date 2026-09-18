"""Negative-control pair construction for the calibration check.

These test construction, not p-values: the statistical engine is the discovery
engine and is covered elsewhere. What is new here is which pairs get built and
which get filtered, and that is deterministic given a seed, so it can be
checked exactly rather than distributionally.

The group-count expectations encode R's `sample_combinations_v2`, which is C++
and cannot be read from an installed sceptre. The values were recovered by
calling it across the argument space under sceptre 0.10.3; see the
`pipeline/calibration.py` docstring. If sceptre changes that rule, these fail
and the port has to be re-derived -- which is the point of pinning them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from pysceptre.pipeline.calibration import (
    build_negative_control_pairs,
    group_cells,
    group_name,
    n_synthetic_groups,
    negative_control_pairs_from_names,
    nonzero_counts,
    sample_ntc_groups,
)

# (n_calibration_pairs, n_genes, pass_qc_rate, expected) -- observed by calling
# sceptre:::sample_combinations_v2 directly, with 1,499 NTC gRNAs and groups of
# 15, that configuration.
R_GROUP_COUNTS = [
    (33_135, 9_045, 0.967, 100),  # a real run: lands on the floor
    (500_000, 9_045, 0.967, 286),
    (33_135, 100, 0.967, 1_714),
    (33_135, 50, 0.5, 6_627),
    (33_135, 10, 0.9, 18_409),
    (271_350, 9_045, 1.0, 150),  # just above the floor
    (180_000, 9_045, 1.0, 100),  # just below it
    (7, 9_045, 0.1, 100),  # tiny request still gets the floor
    # day0, from an object R had already run run_calibration_check on: 625
    # groups over 292 genes for 34,886 pairs at p_hat = mean(pass_qc) = 0.9571.
    # Independent of the probes above -- different dataset, different gRNA
    # library, far off the floor -- and it pins the pass_qc_rate, since 1.0
    # would give 598.
    (34_886, 292, 0.9571, 625),
]


@pytest.mark.parametrize("n_pairs,n_genes,p_hat,expected", R_GROUP_COUNTS)
def test_group_count_matches_r(n_pairs, n_genes, p_hat, expected):
    assert n_synthetic_groups(n_pairs, n_genes, p_hat) == expected


def test_pass_qc_rate_changes_the_count_off_the_floor():
    # The rate is not cosmetic: on day0's parameters R's own 0.9571 yields its
    # actual 625 groups, while assuming every pair passes yields 598.
    assert n_synthetic_groups(34_886, 292, 0.9571) == 625
    assert n_synthetic_groups(34_886, 292, 1.0) == 598
    # ...but it cannot matter when the floor binds, which is why the first dataset probed was
    # insensitive to it and day0 is the case that pinned it down.
    assert n_synthetic_groups(33_135, 9_045, 0.967) == n_synthetic_groups(33_135, 9_045, 1.0)


def test_group_count_never_below_the_floor():
    # R's N_POSSIBLE_GROUPS_THRESHOLD. A request that needs far fewer groups
    # than the floor still gets the floor, which is what keeps a small
    # calibration check from resting on one or two synthetic targets.
    assert n_synthetic_groups(1, 1_000_000, 1.0) == 100


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_calibration_pairs": 0, "n_genes": 10, "pass_qc_rate": 1.0},
        {"n_calibration_pairs": 10, "n_genes": 0, "pass_qc_rate": 1.0},
        {"n_calibration_pairs": 10, "n_genes": 10, "pass_qc_rate": 0.0},
        {"n_calibration_pairs": 10, "n_genes": 10, "pass_qc_rate": 1.5},
    ],
)
def test_group_count_rejects_degenerate_inputs(kwargs):
    with pytest.raises(ValueError):
        n_synthetic_groups(**kwargs)


def test_sampled_groups_are_distinct_and_sorted():
    ids = [f"ntc{i}" for i in range(40)]
    groups = sample_ntc_groups(ids, 5, 30, np.random.default_rng(0))
    assert len(groups) == 30
    assert all(len(g) == 5 for g in groups)
    # Sorted members make the "&"-joined name canonical: the same set always
    # produces the same target name, so results merge against R's.
    assert all(list(g) == sorted(g) for g in groups)
    assert len({frozenset(g) for g in groups}) == 30


def test_group_size_cannot_exceed_available_ntcs():
    with pytest.raises(ValueError, match="exceeds the number of non-targeting"):
        sample_ntc_groups(["a", "b"], 3, 1, np.random.default_rng(0))


def test_group_sampling_is_reproducible_under_a_seed():
    ids = [f"ntc{i}" for i in range(50)]
    a = sample_ntc_groups(ids, 6, 20, np.random.default_rng(7))
    b = sample_ntc_groups(ids, 6, 20, np.random.default_rng(7))
    assert a == b


def test_group_cells_unions_and_deduplicates():
    # A cell carrying two of the group's gRNAs is one treated cell, not two.
    ntc = {"a": np.array([0, 1, 2]), "b": np.array([2, 3]), "c": np.array([9])}
    assert list(group_cells(["a", "b"], ntc, 10)) == [0, 1, 2, 3]
    assert list(group_cells(["a", "b", "c"], ntc, 10)) == [0, 1, 2, 3, 9]


def test_group_cells_rejects_unknown_grna():
    with pytest.raises(KeyError):
        group_cells(["nope"], {"a": np.array([0])}, 10)


def test_group_cells_rejects_out_of_range_index():
    with pytest.raises(ValueError, match="out of range"):
        group_cells(["a"], {"a": np.array([0, 99])}, 10)


def test_group_name_matches_r_join():
    assert group_name(["g1", "g2", "g3"]) == "g1&g2&g3"


def test_nonzero_counts_against_a_direct_computation():
    rng = np.random.default_rng(1)
    n_genes, n_cells = 12, 60
    X = sparse.random(n_genes, n_cells, density=0.4, format="csr", random_state=1)
    groups = [np.sort(rng.choice(n_cells, 15, replace=False)) for _ in range(5)]

    trt, cntrl = nonzero_counts(X, groups, n_cells)

    dense = X.toarray() != 0
    for j, cells in enumerate(groups):
        mask = np.zeros(n_cells, dtype=bool)
        mask[cells] = True
        assert np.array_equal(trt[:, j], dense[:, mask].sum(axis=1))
        # The control arm is the complement, which is why one pass suffices.
        assert np.array_equal(cntrl[:, j], dense[:, ~mask].sum(axis=1))


def test_nonzero_counts_ignores_magnitude():
    # Only the sparsity pattern matters for a nonzero count, so scaling the
    # data must not move the counts -- this is what lets the matrix be
    # binarized instead of densified.
    X = sparse.csr_matrix(np.array([[0.0, 3.0, 0.0], [5.0, 0.0, 1.0]]))
    big = sparse.csr_matrix(X.toarray() * 1000)
    groups = [np.array([0, 1])]
    assert np.array_equal(nonzero_counts(X, groups, 3)[0], nonzero_counts(big, groups, 3)[0])


def _toy(n_genes=40, n_cells=300, n_ntc=20, density=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = sparse.random(n_genes, n_cells, density=density, format="csr", random_state=seed)
    ntc = {f"ntc{i}": np.sort(rng.choice(n_cells, 30, replace=False)) for i in range(n_ntc)}
    return X, [f"g{i}" for i in range(n_genes)], ntc, n_cells


def test_built_pairs_all_clear_qc():
    X, gene_ids, ntc, n_cells = _toy()
    targets, pairs = build_negative_control_pairs(
        X,
        gene_ids,
        ntc,
        n_cells,
        n_calibration_pairs=200,
        calibration_group_size=4,
        n_nonzero_trt_thresh=7,
        n_nonzero_cntrl_thresh=7,
        rng=np.random.default_rng(0),
    )
    assert len(pairs) == 200
    # Unlike discovery, a constructed pair that fails QC is never returned --
    # there is no pass_qc column here because every row passes by construction.
    trt, cntrl = nonzero_counts(X, [targets[t] for t in targets], n_cells)
    idx = {t: j for j, t in enumerate(targets)}
    gidx = {g: i for i, g in enumerate(gene_ids)}
    for _, row in pairs.iterrows():
        j, i = idx[row.grna_target], gidx[row.response_id]
        assert trt[i, j] >= 7 and cntrl[i, j] >= 7


def test_built_pairs_are_unique_and_reference_returned_targets():
    X, gene_ids, ntc, n_cells = _toy()
    targets, pairs = build_negative_control_pairs(
        X,
        gene_ids,
        ntc,
        n_cells,
        n_calibration_pairs=150,
        calibration_group_size=3,
        rng=np.random.default_rng(3),
    )
    assert not pairs.duplicated().any()
    assert set(pairs.grna_target) <= set(targets)
    # Only targets that actually appear in a pair are returned, so the caller
    # never fits a synthetic target it will not test.
    assert set(targets) == set(pairs.grna_target)


def test_construction_is_reproducible_under_a_seed():
    X, gene_ids, ntc, n_cells = _toy()
    kw = dict(n_calibration_pairs=100, calibration_group_size=3)
    t1, p1 = build_negative_control_pairs(
        X, gene_ids, ntc, n_cells, rng=np.random.default_rng(11), **kw
    )
    t2, p2 = build_negative_control_pairs(
        X, gene_ids, ntc, n_cells, rng=np.random.default_rng(11), **kw
    )
    pd.testing.assert_frame_equal(p1, p2)
    assert list(t1) == list(t2)


def test_shortfall_returns_what_is_available_rather_than_raising():
    # Far more pairs requested than can pass QC. Returning a smaller check
    # beats discarding a usable one; the caller sees the shortfall in len().
    X, gene_ids, ntc, n_cells = _toy(n_genes=5, n_cells=60, n_ntc=6, density=0.9)
    _, pairs = build_negative_control_pairs(
        X,
        gene_ids,
        ntc,
        n_cells,
        n_calibration_pairs=10_000,
        calibration_group_size=2,
        rng=np.random.default_rng(0),
    )
    assert 0 < len(pairs) < 10_000


def test_raises_when_nothing_passes_qc():
    X, gene_ids, ntc, n_cells = _toy(n_genes=4, n_cells=40, n_ntc=4, density=0.05)
    with pytest.raises(ValueError, match="no negative control pair passes"):
        build_negative_control_pairs(
            X,
            gene_ids,
            ntc,
            n_cells,
            n_calibration_pairs=50,
            calibration_group_size=2,
            n_nonzero_trt_thresh=10_000,
            n_nonzero_cntrl_thresh=10_000,
            rng=np.random.default_rng(0),
        )


def test_raises_without_any_ntc_grnas():
    X, gene_ids, _, n_cells = _toy()
    with pytest.raises(ValueError, match="no non-targeting gRNAs"):
        build_negative_control_pairs(
            X,
            gene_ids,
            {},
            n_cells,
            n_calibration_pairs=10,
            calibration_group_size=2,
        )


def test_roundtrip_through_group_names():
    # The validation path: R's pair set is unseeded and differs run to run, so
    # comparing pair-by-pair means rebuilding R's groups from their names.
    X, gene_ids, ntc, n_cells = _toy()
    targets, pairs = build_negative_control_pairs(
        X,
        gene_ids,
        ntc,
        n_cells,
        n_calibration_pairs=120,
        calibration_group_size=4,
        rng=np.random.default_rng(5),
    )
    rebuilt = negative_control_pairs_from_names(pairs, ntc, n_cells)
    assert set(rebuilt) == set(targets)
    for name, cells in rebuilt.items():
        assert np.array_equal(cells, targets[name])
