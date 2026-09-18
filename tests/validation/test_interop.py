"""The scverse round trip: AnnData in, results back onto the same object.

`examples/scanpy_interop.py` is the paper's evidence that the ecosystem claim
is real rather than asserted, so it needs to keep working. This exercises the
plumbing it depends on -- covariates out of `obs`, assignments out of the gRNA
modality's `var`, results back onto `var` and `uns` -- at a size that runs in
a couple of seconds.

It does not re-check the statistics, which are validated against R elsewhere.
What it checks is that nothing in the object graph has to be copied,
reformatted or exported for the analysis to run.

Skipped without the `examples` extra, which CI does not install; the value
here is catching drift locally and on release, not on every push.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mudata", reason="the io extra is not installed")
pytest.importorskip("scanpy", reason="the examples extra is not installed")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from scanpy_interop import build_screen  # noqa: E402

import pysceptre  # noqa: E402


@pytest.fixture(scope="module")
def screen():
    return build_screen(n_cells=600, n_genes=8, n_targets=3, n_ntc=6, seed=0)


def test_screen_is_two_modalities_over_one_cell_set(screen):
    assert set(screen.mod) == {"rna", "grna"}
    assert screen["rna"].n_obs == screen["grna"].n_obs
    # The gRNA assay's var is what distinguishes a per-target union from an
    # individual non-targeting gRNA; without it the calibration check has
    # nothing to regroup.
    kinds = screen["grna"].var["unit_kind"].value_counts().to_dict()
    assert kinds == {"target": 3, "ntc_grna": 6}


def test_counts_transpose_to_the_engine_orientation_without_copying(screen):
    import scipy.sparse as sp

    rna = screen["rna"]
    counts = sp.csc_matrix(rna.X)
    transposed = counts.T
    # (cells, genes) CSC and (genes, cells) CSR are the same three buffers.
    assert transposed.shape == (rna.n_vars, rna.n_obs)
    assert np.shares_memory(counts.data, transposed.data)
    assert np.shares_memory(counts.indices, transposed.indices)


def test_analysis_runs_off_the_object_and_results_go_back_on(screen):
    import scanpy as sc

    rna, grna = screen["rna"], screen["grna"]
    rna = rna.copy()
    rna.layers["counts"] = rna.X.copy()
    sc.pp.calculate_qc_metrics(rna, inplace=True, percent_top=None, log1p=False)

    # Every covariate comes from columns scanpy itself populated.
    covariates = np.column_stack(
        [
            np.ones(rna.n_obs),
            np.log(rna.obs["total_counts"].to_numpy()),
            np.log(rna.obs["n_genes_by_counts"].to_numpy()),
        ]
    )

    assignments = np.asarray(grna.X)
    is_target = (grna.var["unit_kind"] == "target").to_numpy()
    targets = {
        u: np.flatnonzero(assignments[:, j]) for j, u in enumerate(grna.var_names) if is_target[j]
    }
    pairs = pd.DataFrame(
        [(g, t) for g in rna.var_names for t in targets],
        columns=["response_id", "grna_target"],
    )

    result = pysceptre.run_discovery_analysis(
        response_matrix=rna.layers["counts"].T,
        gene_ids=list(rna.var_names),
        covariate_matrix=covariates,
        grna_target_cells=targets,
        pairs=pairs,
        side="both",
        seed=0,
        chunk_memory_gb=4.0,
    )
    assert len(result) == len(pairs)
    assert result["p_value"].between(0, 1).all()

    # Back onto the object, indexed the way the rest of a workflow expects.
    best = result.loc[result.groupby("response_id")["p_value"].idxmin()]
    rna.var["sceptre_min_p"] = best.set_index("response_id")["p_value"].reindex(rna.var_names)
    assert rna.var["sceptre_min_p"].notna().all()
    assert list(rna.var_names) == [f"gene_{i}" for i in range(8)]

    # And the object is still a usable AnnData afterwards.
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.pca(rna, n_comps=5)
    assert rna.obsm["X_pca"].shape == (rna.n_obs, 5)
    assert rna.var["sceptre_min_p"].notna().all(), "results survived downstream steps"


def test_planted_effect_is_recovered(screen):
    """`build_screen` halves gene_0 in target_0's cells; the run must see it."""
    rna, grna = screen["rna"], screen["grna"]
    covariates = np.column_stack([np.ones(rna.n_obs), np.log(np.asarray(rna.X).sum(axis=1))])
    assignments = np.asarray(grna.X)
    targets = {
        u: np.flatnonzero(assignments[:, j])
        for j, u in enumerate(grna.var_names)
        if grna.var["unit_kind"].iloc[j] == "target"
    }
    pairs = pd.DataFrame(
        [(g, t) for g in rna.var_names for t in targets],
        columns=["response_id", "grna_target"],
    )
    result = pysceptre.run_discovery_analysis(
        response_matrix=np.asarray(rna.X).T.astype(float),
        gene_ids=list(rna.var_names),
        covariate_matrix=covariates,
        grna_target_cells=targets,
        pairs=pairs,
        side="both",
        seed=0,
        chunk_memory_gb=4.0,
    )
    planted = result[(result.response_id == "gene_0") & (result.grna_target == "target_0")].iloc[0]
    # A 50% knockdown, so roughly -50% with the interval covering it. Loose
    # bounds: this asserts the effect is found and signed correctly, not that
    # the estimator is unbiased on 600 cells.
    assert planted["pct_change_es"] < -25.0
    assert planted["pct_change_es_ci_high"] < 0.0
    assert planted["p_value"] < 1e-3
