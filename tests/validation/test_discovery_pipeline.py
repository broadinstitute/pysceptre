import numpy as np
import pandas as pd

from pysceptre.pipeline.api import run_discovery_analysis


def _build_inputs(ground_truth, include_signal_gene: bool = False):
    X = np.array(ground_truth["X"])
    n_cells = X.shape[0]

    gene_ids = [g["gene_id"] for g in ground_truth["genes"]]
    ys = [np.array(g["y"], dtype=float) for g in ground_truth["genes"]]
    if include_signal_gene:
        sp = ground_truth["signal_pair"]
        gene_ids = gene_ids + [sp["gene_id"]]
        ys = ys + [np.array(sp["y"], dtype=float)]

    response_matrix = np.vstack(ys)

    grna_target_cells = {}
    for t in ground_truth["targets"]:
        grna_target_cells[t["target_id"]] = np.array(t["trt_idxs_1based"]) - 1

    pair_rows = [(g, t) for g in gene_ids for t in grna_target_cells]
    pairs = pd.DataFrame(pair_rows, columns=["response_id", "grna_target"])

    return response_matrix, gene_ids, X, grna_target_cells, pairs


def test_orchestration_matches_r_deterministic_fields_for_null_pairs(ground_truth):
    response_matrix, gene_ids, X, grna_target_cells, pairs = _build_inputs(ground_truth)

    result = run_discovery_analysis(
        response_matrix=response_matrix,
        gene_ids=gene_ids,
        covariate_matrix=X,
        grna_target_cells=grna_target_cells,
        pairs=pairs,
        seed=123,
    )
    result = result.set_index(["response_id", "grna_target"])

    for pair in ground_truth["pairs"]:
        row = result.loc[(pair["gene_id"], pair["target_id"])]
        # deterministic (no RNG): must match R closely
        np.testing.assert_allclose(row["z_orig"], pair["z_orig"], rtol=1e-5)
        np.testing.assert_allclose(row["fold_change"], pair["fold_change"], rtol=1e-6)
        # stochastic: every null pair here has R's p >> 0.02, so both R's and our
        # independent resample should stay at stage 1 (no escalation)
        assert row["stage"] == 1 == pair["stage"]
        assert row["p_value"] > 0.02


def test_orchestration_detects_strong_signal_end_to_end(ground_truth):
    response_matrix, gene_ids, X, grna_target_cells, pairs = _build_inputs(ground_truth, include_signal_gene=True)
    sp = ground_truth["signal_pair"]

    result = run_discovery_analysis(
        response_matrix=response_matrix,
        gene_ids=gene_ids,
        covariate_matrix=X,
        grna_target_cells=grna_target_cells,
        pairs=pairs,
        seed=7,
    )
    result = result.set_index(["response_id", "grna_target"])
    row = result.loc[(sp["gene_id"], sp["target_id"])]

    np.testing.assert_allclose(row["z_orig"], sp["z_orig"], rtol=1e-5)
    np.testing.assert_allclose(row["fold_change"], sp["fold_change"], rtol=1e-6)
    assert row["stage"] == 2
    assert row["p_value"] < 1e-10
