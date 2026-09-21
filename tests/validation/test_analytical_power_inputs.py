"""The helpers that build `compute_power`'s inputs, against R.

`poscounts_size_factors` is checked against **DESeq2**, which defines the
estimator, rather than against a second transcription of the same arithmetic.
`bh_nominal_cutoff` and `cells_per_grna_from_assignments` are checked against
R's rule and against the many-to-many design the real data has.

The real-data checks against WattEG's own output live in
`test_analytical_power_day0.py`, behind the `realdata` marker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from pysceptre.analytical_power import (
    baseline_expression_stats,
    bh_nominal_cutoff,
    cells_per_grna_from_assignments,
    poscounts_size_factors,
)

RTOL = 1e-10


def _case_matrix(case: dict) -> sparse.csc_matrix:
    dense = np.array(case["counts_rowmajor"], dtype=float).reshape(case["n_genes"], case["n_cells"])
    return sparse.csc_matrix(dense)


def test_size_factors_match_deseq2_up_to_its_centring(poscounts_ground_truth):
    """Every ratio in the factor vector agrees with the package that defines poscounts.

    DESeq2 divides its factors by their geometric mean, so `sizeFactors()`
    comes back centred on 1; the implementation the estimator was validated
    against does not. Centring ours makes the two directly comparable, which
    checks all the information in the vector. The absolute scale is a separate
    claim, checked against real output in `test_analytical_power_day0.py` --
    see `docs/design.md`, "Analytical per-pair power".
    """
    for case in poscounts_ground_truth["cases"]:
        got = poscounts_size_factors(_case_matrix(case))
        centred = got / np.exp(np.mean(np.log(got)))
        np.testing.assert_allclose(
            centred, np.array(case["size_factors"]), rtol=RTOL, err_msg=case["label"]
        )


def test_the_offset_from_deseq2_is_exactly_one_constant(poscounts_ground_truth):
    """Pinned, because it is the difference somebody will try to "fix".

    If ours ever diverged from DESeq2 by anything other than a single scalar,
    the two would disagree about a *relative* normalisation, which would be a
    real bug rather than a convention. The constant itself is the geometric
    mean of our factors, and it is not small: 1.12 and 1.19 here.
    """
    seen = []
    for case in poscounts_ground_truth["cases"]:
        got = poscounts_size_factors(_case_matrix(case))
        ratio = got / np.array(case["size_factors"])
        assert np.ptp(ratio) < 1e-12, f"{case['label']}: offset is not constant"
        np.testing.assert_allclose(ratio[0], np.exp(np.mean(np.log(got))), rtol=1e-9)
        seen.append(float(ratio[0]))
    assert max(seen) > 1.1, "the fixture no longer exercises a non-trivial offset"


def test_normalised_mean_is_the_uncentred_normalisation(poscounts_ground_truth):
    """The mean the estimator consumes, built on uncentred factors.

    The fixture's `normalised_mean` divides by DESeq2's centred factors, so
    ours is that divided by the same constant. Asserting the relationship
    rather than equality is the honest form: it says exactly which convention
    this package follows and what it costs.
    """
    for case in poscounts_ground_truth["cases"]:
        m = _case_matrix(case)
        gene_ids = [f"g{i}" for i in range(case["n_genes"])]
        stats = baseline_expression_stats(m, gene_ids, np.ones(case["n_genes"]))
        got = poscounts_size_factors(m)
        constant = float(np.exp(np.mean(np.log(got))))
        np.testing.assert_allclose(
            stats["expression_mean"].to_numpy() * constant,
            np.array(case["normalised_mean"]),
            rtol=1e-9,
            err_msg=case["label"],
        )


def test_identical_cells_get_unit_size_factors(poscounts_ground_truth):
    """A guard the other two cases cannot give: no library-size differences, no correction."""
    uniform = [c for c in poscounts_ground_truth["cases"] if c["label"] == "uniform"]
    assert uniform, "fixture lost the uniform case"
    got = poscounts_size_factors(_case_matrix(uniform[0]))
    np.testing.assert_allclose(got, np.ones_like(got), rtol=1e-12)
    # And with no library-size differences the two conventions coincide.
    np.testing.assert_allclose(got, np.array(uniform[0]["size_factors"]), rtol=1e-12)


def test_a_gene_that_is_zero_everywhere_does_not_poison_the_factors(poscounts_ground_truth):
    """It has no geometric mean, so it is excluded rather than treated as mean 1."""
    mixed = [c for c in poscounts_ground_truth["cases"] if c["label"] == "mixed"][0]
    m = _case_matrix(mixed)
    zero_rows = np.flatnonzero(np.asarray(m.sum(axis=1)).ravel() == 0)
    assert zero_rows.size >= 1, "fixture lost the all-zero gene"
    got = poscounts_size_factors(m)
    assert np.all(np.isfinite(got)) and np.all(got > 0)
    gene_ids = [f"g{i}" for i in range(mixed["n_genes"])]
    stats = baseline_expression_stats(m, gene_ids, np.ones(mixed["n_genes"]))
    assert stats.loc[zero_rows[0], "expression_mean"] == 0.0


def test_a_cell_with_no_usable_nonzero_is_refused():
    """Better a loud error than a NaN size factor propagating into every gene's mean."""
    dense = np.zeros((3, 4))
    dense[:, :3] = [[5, 6, 7], [1, 2, 3], [4, 4, 4]]
    with pytest.raises(ValueError, match="no usable size factor"):
        poscounts_size_factors(sparse.csc_matrix(dense))


def test_size_factors_are_unchanged_by_sparse_format():
    rng = np.random.default_rng(3)
    dense = rng.poisson(5.0, size=(8, 15)).astype(float)
    a = poscounts_size_factors(sparse.csc_matrix(dense))
    b = poscounts_size_factors(sparse.csr_matrix(dense))
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_subsetting_the_output_is_not_subsetting_the_input():
    """The normalisation depends on the whole matrix, which is the helper's main trap.

    `gene_subset` returns fewer rows from statistics computed over everything.
    Passing a pre-subset matrix instead gives different size factors and so
    different means, and nothing in the output would say which had happened.
    """
    rng = np.random.default_rng(11)
    dense = rng.poisson(4.0, size=(10, 30)).astype(float)
    dense[5] *= 20  # one dominant gene, so dropping it moves every size factor
    gene_ids = [f"g{i}" for i in range(10)]
    thetas = np.full(10, 2.0)
    wanted = ["g0", "g1"]

    full = baseline_expression_stats(sparse.csc_matrix(dense), gene_ids, thetas, gene_subset=wanted)
    presubset = baseline_expression_stats(sparse.csc_matrix(dense[:2]), gene_ids[:2], thetas[:2])
    assert full["response_id"].tolist() == wanted
    assert not np.allclose(
        full["expression_mean"].to_numpy(), presubset["expression_mean"].to_numpy()
    )


def test_theta_comes_through_by_gene_id_and_a_gap_is_refused():
    rng = np.random.default_rng(5)
    dense = rng.poisson(4.0, size=(4, 12)).astype(float)
    gene_ids = ["a", "b", "c", "d"]
    stats = baseline_expression_stats(
        sparse.csc_matrix(dense), gene_ids, {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    )
    assert stats["expression_size"].tolist() == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(KeyError, match="missing"):
        baseline_expression_stats(sparse.csc_matrix(dense), gene_ids, {"a": 1.0})


# --------------------------------------------------------------- cells_per_grna ---


def _design() -> pd.DataFrame:
    # g2 sits in two overlapping elements, which is the real shape: a guide appears once per
    # target it belongs to. g4 is designed but has no cells.
    return pd.DataFrame(
        {
            "grna_id": ["g1", "g2", "g2", "g3", "g4"],
            "grna_target": ["A", "A", "B", "B", "B"],
        }
    )


def test_a_shared_guide_is_kept_under_every_target():
    cells = {"g1": np.arange(10), "g2": np.arange(20), "g3": np.arange(30)}
    out = cells_per_grna_from_assignments(_design(), cells)
    assert len(out) == 5
    g2 = out[out["grna_id"] == "g2"]
    assert set(g2["grna_target"]) == {"A", "B"}
    assert (g2["num_cells"] == 20).all()


def test_a_designed_guide_with_no_cells_is_a_zero_row_not_an_absent_one():
    """Dropped instead, `num_trt_cells_sq` would be over a smaller guide set than the screen."""
    cells = {"g1": np.arange(10), "g2": np.arange(20), "g3": np.arange(30)}
    out = cells_per_grna_from_assignments(_design(), cells)
    g4 = out[out["grna_id"] == "g4"]
    assert len(g4) == 1 and g4["num_cells"].iloc[0] == 0


def test_a_repeated_design_row_is_refused():
    design = pd.concat([_design(), _design().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="repeats 1"):
        cells_per_grna_from_assignments(design, {"g1": np.arange(10)})


def test_the_result_feeds_the_estimator_unchanged():
    from pysceptre.analytical_power import target_cell_counts

    cells = {"g1": np.arange(10), "g2": np.arange(20), "g3": np.arange(30)}
    out = cells_per_grna_from_assignments(_design(), cells)
    agg = target_cell_counts(out).set_index("grna_target")
    assert agg.loc["A", "num_trt_cells"] == 30.0  # 10 + 20
    assert agg.loc["B", "num_trt_cells"] == 50.0  # 20 + 30 + 0
    assert agg.loc["A", "num_trt_cells_sq"] == 10.0**2 + 20.0**2


# ----------------------------------------------------------------- BH cutoff ---


def test_bh_cutoff_matches_the_textbook_rule():
    """R's rule: the largest p with p_(k) <= k/m * alpha."""
    p = np.array([0.001, 0.008, 0.02, 0.2, 0.5])
    # m = 5, alpha = 0.1: bounds are 0.02, 0.04, 0.06, 0.08, 0.10; the last p below its bound
    # is 0.02 at k = 3.
    assert bh_nominal_cutoff(p, 0.1) == pytest.approx(0.02)


def test_bh_cutoff_agrees_with_r_p_adjust_on_random_inputs():
    """Cross-checked against the definition R's p.adjust implements, over many draws."""
    rng = np.random.default_rng(17)
    for _ in range(200):
        m = int(rng.integers(5, 300))
        p = np.sort(rng.beta(0.3, 3.0, size=m))
        alpha = float(rng.choice([0.01, 0.05, 0.1, 0.2]))
        # BH-adjusted p, the cumulative-minimum form p.adjust uses.
        adj = np.minimum.accumulate((p * m / np.arange(1, m + 1))[::-1])[::-1]
        sig = p[adj <= alpha]
        if sig.size == 0:
            with pytest.raises(ValueError, match="no pair is significant"):
                bh_nominal_cutoff(p, alpha)
        else:
            assert bh_nominal_cutoff(p, alpha) == pytest.approx(sig.max())


def test_nan_p_values_are_dropped_before_the_correction():
    """A pair that failed QC was never tested, so it must not inflate the denominator."""
    p = np.array([0.001, 0.008, 0.02, 0.2, 0.5])
    with_nans = np.concatenate([p, np.full(50, np.nan)])
    assert bh_nominal_cutoff(with_nans, 0.1) == bh_nominal_cutoff(p, 0.1)


def test_nothing_significant_raises_rather_than_returning_negative_infinity():
    """The behaviour this ports was written to replace: -Inf, then zero power everywhere."""
    with pytest.raises(ValueError, match="no pair is significant"):
        bh_nominal_cutoff(np.array([0.4, 0.6, 0.9]), 0.1)


def test_no_usable_p_value_raises():
    with pytest.raises(ValueError, match="no finite p-values"):
        bh_nominal_cutoff(np.array([np.nan, np.nan]), 0.1)


@pytest.mark.parametrize("alpha", [0.0, -0.1, 1.5])
def test_bh_cutoff_validates_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        bh_nominal_cutoff(np.array([0.01, 0.02]), alpha)
