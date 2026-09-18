"""Tests for the B2/B3 resampling budget derived in pipeline/api.py.

The reference values come from R's own sizing logic in
`s4_analysis_functs_1.R`: B2/B3 are set in `run_discovery_analysis` and, for
`no_approximation`, B3 is recomputed in `run_qc_pt_2` as
`ceiling(mult_fact * max(n_ok_pairs) / multiple_testing_alpha)` with
`mult_fact = 10` two-sided and `5` one-sided.
"""

import math

import numpy as np
import pandas as pd
import pytest

from pysceptre.pipeline.api import _resampling_budget, run_discovery_analysis
from pysceptre.test_statistic.resampling import run_low_level_test_full


def test_skew_normal_matches_rs_crt_values():
    # R: B2 <- 4999L; B3 <- if (permutations) 24999L else 0L. pysceptre is
    # CRT-only, so B3 is 0 -- this is parity with R, not a missing feature.
    assert _resampling_budget("skew_normal", 0, 33_066, 0.1) == (4999, 0)
    # n_pairs and alpha are irrelevant for skew_normal
    assert _resampling_budget("skew_normal", -1, 1, 0.5) == (4999, 0)


@pytest.mark.parametrize(
    ("side_code", "mult_fact"),
    [(0, 10), (-1, 5), (1, 5)],
)
def test_no_approximation_b3_matches_rs_formula(side_code, mult_fact):
    n_pairs, alpha = 33_066, 0.1
    B2, B3 = _resampling_budget("no_approximation", side_code, n_pairs, alpha)
    assert B2 == 0, "R sets B2 = 0 for no_approximation (no curve is fit)"
    assert B3 == math.ceil(mult_fact * n_pairs / alpha)


def test_no_approximation_b3_uses_ceiling_not_truncation():
    # 3 * 5 / 0.7 = 21.43... -> 22, matching R's ceiling()
    assert _resampling_budget("no_approximation", -1, 3, 0.7) == (0, 22)


def test_no_approximation_b3_scales_with_alpha():
    _, b3_lenient = _resampling_budget("no_approximation", -1, 100, 0.1)
    _, b3_strict = _resampling_budget("no_approximation", -1, 100, 0.01)
    assert b3_strict == 10 * b3_lenient


def _tiny_inputs(n_targets=2, n_genes=2, n_cells=400, seed=0):
    rng = np.random.default_rng(seed)
    covariate_matrix = np.column_stack([np.ones(n_cells), rng.normal(size=n_cells)])
    gene_ids = [f"gene_{i}" for i in range(n_genes)]
    response_matrix = rng.poisson(5.0, size=(n_genes, n_cells)).astype(float)
    grna_target_cells = {
        f"target_{t}": rng.choice(n_cells, size=40, replace=False) for t in range(n_targets)
    }
    pairs = pd.DataFrame(
        [{"response_id": g, "grna_target": t} for g in gene_ids for t in grna_target_cells]
    )
    return response_matrix, gene_ids, covariate_matrix, grna_target_cells, pairs


def test_no_approximation_runs_end_to_end_and_never_reports_stage_2():
    """With no curve fitting, every escalated pair must land in stage 3 (the
    empirical B3 batch) -- stage 2 is by definition a skew-normal fit."""
    resp, gene_ids, cov, cells, pairs = _tiny_inputs()
    result = run_discovery_analysis(
        response_matrix=resp,
        gene_ids=gene_ids,
        covariate_matrix=cov,
        grna_target_cells=cells,
        pairs=pairs,
        side="left",
        resampling_approximation="no_approximation",
        multiple_testing_alpha=0.1,
        seed=0,
    )
    assert len(result) == len(pairs)
    assert set(result["stage"]) <= {1, 3}
    assert result["p_value"].between(0, 1).all()


def test_no_approximation_resolves_p_at_the_b3_resolution():
    """The point of the B3 batch: stage 3 must read the B3 draws, not re-read
    the B1 draws (which is what the old fixed B3=0 did). Sized so B3 > B1:
    20 pairs one-sided at alpha=0.1 gives B3 = 1000, whose floor 1/1001 is
    finer than B1's 1/500 -- so a maximally extreme pair landing exactly on
    1/(B3+1) proves which batch was used."""
    resp, gene_ids, cov, cells, pairs = _tiny_inputs(n_genes=2, n_targets=10)
    _, B3 = _resampling_budget("no_approximation", -1, len(pairs), 0.1)
    assert B3 == 1000 > 499, "test needs B3 > B1 to distinguish the two batches"

    trt = cells["target_0"]
    # Near-total knockdown: strong enough to be maximally extreme, but
    # nonzero so fold_change > 0 and log2 stays finite.
    resp[0, trt] = 1.0

    result = run_discovery_analysis(
        response_matrix=resp,
        gene_ids=gene_ids,
        covariate_matrix=cov,
        grna_target_cells=cells,
        pairs=pairs,
        side="left",
        resampling_approximation="no_approximation",
        seed=0,
    )
    hit = result[(result.response_id == "gene_0") & (result.grna_target == "target_0")].iloc[0]
    assert hit["stage"] == 3
    assert hit["p_value"] < 1.0 / (499 + 1), "must beat the B1 floor"
    assert hit["p_value"] == pytest.approx(1.0 / (B3 + 1)), "p must sit at the B3 floor"


def test_no_approximation_b3_can_fall_below_b1_for_small_analyses():
    """A real property of R's formula, worth pinning: with few pairs, B3 is
    *smaller* than B1=499, so no_approximation is coarser than skew_normal
    there. 4 pairs one-sided at alpha=0.1 gives B3 = 200."""
    assert _resampling_budget("no_approximation", -1, 4, 0.1) == (0, 200)


def test_invalid_resampling_approximation_raises():
    resp, gene_ids, cov, cells, pairs = _tiny_inputs()
    with pytest.raises(ValueError, match="resampling_approximation"):
        run_discovery_analysis(
            response_matrix=resp,
            gene_ids=gene_ids,
            covariate_matrix=cov,
            grna_target_cells=cells,
            pairs=pairs,
            resampling_approximation="skew-normal",  # typo: hyphen, not underscore
        )


def test_invalid_side_raises():
    resp, gene_ids, cov, cells, pairs = _tiny_inputs()
    with pytest.raises(ValueError, match="side"):
        run_discovery_analysis(
            response_matrix=resp,
            gene_ids=gene_ids,
            covariate_matrix=cov,
            grna_target_cells=cells,
            pairs=pairs,
            side="two-sided",
        )


def test_no_approximation_matches_r_exactly(ground_truth):
    """The B3 path, value-for-value against R on R's own draws.

    This was the one code path with no reference comparison: every other
    fixture exercises `skew_normal`, which fits a curve to the B2 draws, so
    the B3 route to stage 3 had only been checked for self-consistency.

    The comparison is **exact**, not distributional, and that is the point.
    Everywhere else the CRT draws differ between implementations, because
    pysceptre seeds `numpy.random.Generator` and sceptre seeds
    `boost::mt19937`. Here R's own draws are fed back in, and an empirical
    p-value is then a deterministic function of those draws and `z_orig` --
    no RNG is left in the calculation, so anything other than equality is a
    porting error rather than Monte Carlo noise.
    """
    na = ground_truth["no_approximation_pair"]
    sp = ground_truth["signal_pair"]
    target = next(t for t in ground_truth["targets"] if t["target_id"] == na["target_id"])

    result = run_low_level_test_full(
        y=np.array(sp["y"], dtype=float),
        mu=np.array(sp["mu"]),
        a=np.array(sp["a"]),
        w=np.array(sp["w"]),
        D=np.array(sp["D"]),
        trt_idxs=np.array(target["trt_idxs_1based"]) - 1,
        synthetic_idxs=[np.array(i) for i in na["synthetic_idxs_0based"]],
        B1=na["B1"],
        B2=na["B2"],
        B3=na["B3"],
        fit_parametric_curve=False,
        side_code=0,
    )

    assert result.stage == 3, "no_approximation must never fit a curve"
    assert result.sn_params is None
    np.testing.assert_allclose(result.z_orig, na["z_orig"], rtol=1e-10)
    np.testing.assert_allclose(result.fold_change, na["fold_change"], rtol=1e-10)
    np.testing.assert_allclose(result.se_fold_change, na["se_fold_change"], rtol=1e-10)
    # Given identical draws this is deterministic, so it must match to the bit
    # of a float rather than approximately.
    np.testing.assert_allclose(result.p_value, na["p_value"], rtol=1e-12)


def test_no_approximation_p_sits_on_the_b3_grid(ground_truth):
    """An empirical p-value from B3 draws can only take values k/(B3+1).

    Pinning this separately from the value comparison above documents *why*
    `no_approximation` is slow: resolving a p-value of 1e-5 needs B3 on the
    order of 1e5 draws, which is what makes R size B3 as
    `n_pairs / multiple_testing_alpha`.
    """
    na = ground_truth["no_approximation_pair"]
    grid = round(na["p_value"] * (na["B3"] + 1))
    np.testing.assert_allclose(na["p_value"], grid / (na["B3"] + 1), rtol=1e-12)
    assert na["p_value"] >= 1.0 / (na["B3"] + 1)
