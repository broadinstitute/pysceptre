import numpy as np
import pandas as pd
import pytest

from pysceptre.pipeline.api import run_discovery_analysis, run_power_check


def _build_inputs(ground_truth, include_signal_gene: bool = False):
    X = np.array(ground_truth["X"])

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
    response_matrix, gene_ids, X, grna_target_cells, pairs = _build_inputs(
        ground_truth, include_signal_gene=True
    )
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


def _level_screen(n=2000, n_levels=3, raw_umis=False, seed=0):
    """One gene, an intercept, a UMI covariate and a batch, with targets made of batch levels.

    `raw_umis` puts the UMI counts in unlogged, thousands of times the scale of the batch
    columns: the design where the statistic's own denominator cannot tell a level apart.
    """
    rng = np.random.default_rng(seed)
    level = rng.integers(0, n_levels, n)
    umis = rng.lognormal(9, 0.6, n)
    X = np.column_stack(
        [np.ones(n), umis if raw_umis else np.log(umis)]
        + [(level == k).astype(float) for k in range(1, n_levels)]
    )
    mu = np.exp(-8 + np.log(umis) + 0.1 * (level % 3))
    Y = rng.negative_binomial(5, 5 / (5 + mu), size=(1, n)).astype(float)
    level2 = np.flatnonzero(level == 2)
    cells = {
        "level2": level2,
        "level0": np.flatnonzero(level == 0),
        "level2_but_one": level2[1:],
        "random": np.sort(rng.choice(n, 60, replace=False)),
    }
    pairs = pd.DataFrame({"response_id": "g0", "grna_target": list(cells)})
    return Y, ["g0"], X, cells, pairs


_SPANNED = ["level2", "level0"]
_RESULT = ["p_value", "fold_change", "se_fold_change", "z_orig", "stage"]


@pytest.mark.parametrize(
    ("mechanism", "raw_umis"), [("crt", False), ("permutations", False), ("crt", True)]
)
def test_a_target_the_covariates_explain_is_not_tested(mechanism, raw_umis):
    Y, genes, X, cells, pairs = _level_screen(
        n=6000 if raw_umis else 2000, n_levels=30 if raw_umis else 3, raw_umis=raw_umis
    )
    with pytest.warns(UserWarning, match="2 target"):
        r = run_discovery_analysis(
            response_matrix=Y,
            gene_ids=genes,
            covariate_matrix=X,
            grna_target_cells=cells,
            pairs=pairs,
            seed=0,
            resampling_mechanism=mechanism,
        ).set_index("grna_target")
    assert r.loc[_SPANNED, _RESULT].isna().all().all()
    assert np.isfinite(r.loc[["level2_but_one", "random"], "p_value"]).all()


def test_the_dose_test_flags_constant_weights_only():
    Y, genes, X, cells, pairs = _level_screen()
    rng = np.random.default_rng(1)
    constant = {t: np.full(c.size, 0.5) for t, c in cells.items()}
    varying = {t: rng.uniform(0.2, 1.0, c.size) for t, c in cells.items()}
    run = dict(
        response_matrix=Y,
        gene_ids=genes,
        covariate_matrix=X,
        grna_target_cells=cells,
        pairs=pairs,
        seed=0,
    )
    with pytest.warns(UserWarning, match="2 target"):
        r = run_discovery_analysis(**run, grna_target_weights=constant).set_index("grna_target")
    assert r.loc[_SPANNED, "p_value"].isna().all()
    r = run_discovery_analysis(**run, grna_target_weights=varying)
    assert np.isfinite(r["p_value"]).all()


def _with(case, Y, genes, cells, pairs):
    if case == "repeated pair":
        pairs = pd.concat([pairs, pairs.iloc[[0]]], ignore_index=True)
    elif case == "repeated gene id":
        Y, genes = np.vstack([Y, Y]), genes + genes
    elif case == "repeated cell":
        cells = {**cells, "random": np.r_[cells["random"], cells["random"][0]]}
    elif case == "negative cell":
        cells = {**cells, "random": np.r_[-1, cells["random"]]}
    elif case == "cell past the end":
        cells = {**cells, "random": np.r_[cells["random"], Y.shape[1]]}
    return Y, genes, cells, pairs


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("repeated pair", "more than once, e.g. \\('g0'"),
        ("repeated gene id", "more than one row"),
        ("repeated cell", "lists a cell more than once"),
        ("negative cell", "outside"),
        ("cell past the end", "outside"),
    ],
)
def test_inputs_that_would_be_counted_silently_are_refused(case, message):
    Y, genes, X, cells, pairs = _level_screen()
    Y, genes, cells, pairs = _with(case, Y, genes, cells, pairs)
    with pytest.raises(ValueError, match=message):
        run_discovery_analysis(
            response_matrix=Y,
            gene_ids=genes,
            covariate_matrix=X,
            grna_target_cells=cells,
            pairs=pairs,
            seed=0,
        )
    with pytest.raises(ValueError, match=message):
        run_power_check(
            response_matrix=Y,
            gene_ids=genes,
            covariate_matrix=X,
            grna_target_cells=cells,
            positive_control_pairs=pairs,
            seed=0,
        )
