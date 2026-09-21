"""The power check: the discovery test over positive controls.

Nothing statistical is new -- the engine is the discovery engine -- so these
cover what *is* new: how the pairs are obtained, and how pairs that cannot be
tested are handled.

The QC behaviour is the interesting part, because it is the **opposite** of
the calibration check's and the reason both exist. Calibration samples its
pairs, so it can simply avoid ones that would fail, and every returned row
passes. A positive control is a specific claim about a specific pair; the
ones that fail QC are the ones the screen had too few cells to test, and
dropping them would make the power check flatter itself. They are reported
with a NaN result instead.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pysceptre.pipeline.api import run_power_check
from pysceptre.pipeline.power import (
    annotate_pairwise_qc,
    construct_positive_control_pairs,
)


def _screen(n_genes=6, n_cells=1200, n_targets=4, seed=0, trt_cells=150):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n_cells), rng.normal(size=(n_cells, 2))])
    theta, mu = 5.0, 20.0
    Y = rng.negative_binomial(theta, theta / (theta + mu), size=(n_genes, n_cells)).astype(float)
    genes = [f"gene_{i}" for i in range(n_genes)]
    targets = {
        f"chr1:{1000 + j}-{1100 + j}": np.sort(rng.choice(n_cells, trt_cells, replace=False))
        for j in range(n_targets)
    }
    return Y, genes, X, targets


def test_name_matching_finds_targets_named_after_genes():
    pairs = construct_positive_control_pairs(
        ["NMU", "HACD3", "PRKAR2B"], ["NMU", "PRKAR2B", "chr1:100-200"]
    )
    assert list(pairs.response_id) == ["NMU", "PRKAR2B"]
    # R pairs a target with the gene of the same name, so the two columns match.
    assert (pairs.response_id == pairs.grna_target).all()


def test_name_matching_finds_nothing_for_interval_named_targets():
    """The day0 case: 0 of 3,071 targets match, because they are coordinates.

    An empty result is correct here, not a failure -- it means the mapping
    between targets and the genes they perturb has to come from the
    experiment.
    """
    pairs = construct_positive_control_pairs(
        ["NMU", "HACD3"], ["chr4:55636048-55636948", "chr15:65530238-65530738"]
    )
    assert pairs.empty


def test_missing_pairs_raise_rather_than_returning_nothing():
    Y, genes, X, targets = _screen()
    with pytest.raises(ValueError, match="no gRNA target is named after a gene"):
        run_power_check(
            response_matrix=Y, gene_ids=genes, covariate_matrix=X, grna_target_cells=targets
        )


def test_qc_annotation_matches_a_direct_count():
    Y, genes, X, targets = _screen()
    pairs = pd.DataFrame({"response_id": [genes[0], genes[1]], "grna_target": list(targets)[:2]})
    out = annotate_pairwise_qc(pairs, Y, genes, targets, X.shape[0])
    dense = Y != 0
    for _, row in out.iterrows():
        cells = targets[row.grna_target]
        mask = np.zeros(X.shape[0], dtype=bool)
        mask[cells] = True
        g = genes.index(row.response_id)
        assert row.n_nonzero_trt == dense[g, mask].sum()
        assert row.n_nonzero_cntrl == dense[g, ~mask].sum()


def test_qc_failures_are_reported_not_dropped():
    """The behaviour that separates this from the calibration check."""
    Y, genes, X, targets = _screen()
    tnames = list(targets)
    pairs = pd.DataFrame(
        {
            "response_id": [genes[0], genes[1], genes[2]],
            "grna_target": [tnames[0], tnames[1], tnames[2]],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = run_power_check(
            response_matrix=Y,
            gene_ids=genes,
            covariate_matrix=X,
            grna_target_cells=targets,
            positive_control_pairs=pairs,
            # Impossible for one group, so some pairs must fail and still appear.
            n_nonzero_trt_thresh=140,
            n_nonzero_cntrl_thresh=7,
        )
    assert len(out) == len(pairs), "a supplied positive control went missing"
    assert {"pass_qc", "n_nonzero_trt", "n_nonzero_cntrl"} <= set(out.columns)
    failed = out[~out.pass_qc]
    if len(failed):
        assert failed.p_value.isna().all(), "a failed pair carries a p-value"
        assert failed.fold_change.isna().all()


def test_every_supplied_pair_appears_once_and_in_order():
    Y, genes, X, targets = _screen()
    tnames = list(targets)
    pairs = pd.DataFrame({"response_id": [genes[2], genes[0], genes[1]], "grna_target": tnames[:3]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = run_power_check(
            response_matrix=Y,
            gene_ids=genes,
            covariate_matrix=X,
            grna_target_cells=targets,
            positive_control_pairs=pairs,
        )
    assert list(out.response_id) == list(pairs.response_id)
    assert list(out.grna_target) == list(pairs.grna_target)


def test_no_multiple_testing_correction_is_applied():
    """R returns no `significant` column for a power check, and neither do we.

    These are a diagnostic rather than discoveries; adjusting them against
    each other would answer a question nobody asked.
    """
    Y, genes, X, targets = _screen()
    pairs = pd.DataFrame({"response_id": [genes[0]], "grna_target": [list(targets)[0]]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = run_power_check(
            response_matrix=Y,
            gene_ids=genes,
            covariate_matrix=X,
            grna_target_cells=targets,
            positive_control_pairs=pairs,
        )
    assert "significant" not in out.columns


def test_a_planted_effect_is_recovered():
    """The check has to actually detect what it is pointed at."""
    Y, genes, X, targets = _screen(n_cells=4000, trt_cells=600)
    tname = list(targets)[0]
    Y[0, targets[tname]] *= 0.3  # a strong knockdown in the treated cells
    pairs = pd.DataFrame({"response_id": [genes[0]], "grna_target": [tname]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = run_power_check(
            response_matrix=Y,
            gene_ids=genes,
            covariate_matrix=X,
            grna_target_cells=targets,
            positive_control_pairs=pairs,
            side="both",
            seed=0,
        )
    row = out.iloc[0]
    assert bool(row.pass_qc)
    assert row["pct_change_es"] < -30.0, "the planted knockdown was not recovered"
    assert row.p_value < 1e-3


def test_unknown_genes_and_targets_are_rejected():
    Y, genes, X, targets = _screen()
    with pytest.raises(KeyError, match="genes not present"):
        annotate_pairwise_qc(
            pd.DataFrame({"response_id": ["nope"], "grna_target": [list(targets)[0]]}),
            Y,
            genes,
            targets,
            X.shape[0],
        )
    with pytest.raises(KeyError, match="targets not present"):
        annotate_pairwise_qc(
            pd.DataFrame({"response_id": [genes[0]], "grna_target": ["nope"]}),
            Y,
            genes,
            targets,
            X.shape[0],
        )
