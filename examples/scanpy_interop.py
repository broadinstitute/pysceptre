"""A CRISPR screen analysed end to end in Python, with no R in the loop.

The point of this example is not the statistics -- those are validated
against R elsewhere -- but the **plumbing**: that a `sceptre`-style analysis
can sit inside an ordinary `scanpy` workflow, operating on the same AnnData
object as every other step, instead of requiring the data to be exported into
a bespoke on-disk format and back.

That is the claim the paper makes about the ecosystem, and a claim of this
kind should be demonstrated rather than asserted, so this script runs.

What it does:

  1. builds a synthetic screen as MuData -- `rna` counts and `grna`
     assignments over one set of cells
  2. runs ordinary scanpy QC on the RNA modality, and derives the covariates
     pysceptre needs from `obs`, the same columns scanpy itself populates
  3. calls `run_discovery_analysis` on arrays taken straight out of the object
  4. writes the results back onto the object, so downstream steps see them
  5. carries on in scanpy -- normalize, PCA, neighbours, UMAP -- on the very
     same object

Step 3 is the only pysceptre-specific line, and it neither copies nor
reformats the matrix: `rna.X` is `(cells, genes)` CSC, whose transpose is the
`(genes, cells)` CSR the engine wants, sharing buffers.

Run with:  uv run --extra examples python examples/scanpy_interop.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import pysceptre


def build_screen(n_cells=4000, n_genes=60, n_targets=12, n_ntc=20, seed=0):
    """A small synthetic screen, as a MuData with two modalities."""
    import anndata as ad
    import mudata

    rng = np.random.default_rng(seed)

    # Cell-level nuisance structure a real screen has: sequencing depth and a
    # batch effect. Both end up as covariates, and both are things scanpy
    # would compute or carry anyway.
    depth = rng.lognormal(mean=0.0, sigma=0.35, size=n_cells)
    batch = rng.integers(0, 3, size=n_cells)
    base = rng.uniform(4.0, 40.0, size=n_genes)
    mu = np.outer(depth, base) * (1.0 + 0.15 * (batch[:, None] == 1))
    theta = 6.0
    counts = rng.negative_binomial(theta, theta / (theta + mu)).astype(np.int32)

    rna = ad.AnnData(
        X=counts,
        obs=pd.DataFrame(
            {"batch": pd.Categorical([f"batch_{b}" for b in batch])},
            index=[f"cell_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)]),
    )

    # One real effect, so the run has something to find: target_0 represses
    # gene_0 by half in the cells it treats.
    units, kinds, cells = [], [], []
    for t in range(n_targets):
        units.append(f"target_{t}")
        kinds.append("target")
        cells.append(np.sort(rng.choice(n_cells, 300, replace=False)))
    for g in range(n_ntc):
        units.append(f"ntc_{g}")
        kinds.append("ntc_grna")
        cells.append(np.sort(rng.choice(n_cells, 120, replace=False)))
    rna.X[np.ix_(cells[0], [0])] = (rna.X[np.ix_(cells[0], [0])] * 0.5).astype(np.int32)

    indicator = np.zeros((n_cells, len(units)), dtype=np.int8)
    for j, idx in enumerate(cells):
        indicator[idx, j] = 1
    grna = ad.AnnData(
        X=indicator,
        obs=rna.obs[[]].copy(),
        var=pd.DataFrame(
            {"unit_kind": kinds},
            index=pd.Index(units, name="unit_id"),
        ),
    )
    return mudata.MuData({"rna": rna, "grna": grna})


def main() -> None:
    import scanpy as sc

    mdata = build_screen()
    rna, grna = mdata["rna"], mdata["grna"]
    print(f"screen: {rna.n_obs:,} cells x {rna.n_vars} genes, {grna.n_vars} gRNA units")

    # --- 1. ordinary scanpy QC, on the object as it stands -------------------
    rna.layers["counts"] = rna.X.copy()  # the test needs raw counts, not logs
    sc.pp.calculate_qc_metrics(rna, inplace=True, percent_top=None, log1p=False)
    print(f"scanpy QC: median {rna.obs.total_counts.median():,.0f} UMIs/cell")

    # --- 2. covariates, from the columns scanpy just wrote -------------------
    # This is the design matrix R's model.matrix() would build from
    # ~ log(total_counts) + log(n_genes_by_counts) + batch. pysceptre does not
    # parse a formula, so it is assembled here -- but every input comes from
    # `obs`, not from a separate export.
    design = pd.get_dummies(rna.obs["batch"], drop_first=True, dtype=float)
    covariates = np.column_stack(
        [
            np.ones(rna.n_obs),
            np.log(rna.obs["total_counts"].to_numpy()),
            np.log(rna.obs["n_genes_by_counts"].to_numpy()),
            design.to_numpy(),
        ]
    )

    # --- 3. the gRNA modality, read off its own var --------------------------
    assignments = np.asarray(grna.X)
    is_target = (grna.var["unit_kind"] == "target").to_numpy()
    grna_target_cells = {
        unit: np.flatnonzero(assignments[:, j])
        for j, unit in enumerate(grna.var_names)
        if is_target[j]
    }
    ntc_grna_cells = {
        unit: np.flatnonzero(assignments[:, j])
        for j, unit in enumerate(grna.var_names)
        if not is_target[j]
    }

    pairs = pd.DataFrame(
        [(g, t) for g in rna.var_names for t in grna_target_cells],
        columns=["response_id", "grna_target"],
    )

    # --- 4. the analysis. (cells, genes) -> (genes, cells) shares buffers ----
    counts = rna.layers["counts"].T
    result = pysceptre.run_discovery_analysis(
        response_matrix=counts,
        gene_ids=list(rna.var_names),
        covariate_matrix=covariates,
        grna_target_cells=grna_target_cells,
        pairs=pairs,
        side="both",
        seed=0,
        n_jobs=-1,
    )
    print(f"discovery: {len(result):,} pairs tested")

    calib = pysceptre.run_calibration_check(
        response_matrix=counts,
        gene_ids=list(rna.var_names),
        covariate_matrix=covariates,
        ntc_grna_cells=ntc_grna_cells,
        n_calibration_pairs=len(pairs),
        calibration_group_size=5,
        side="both",
        seed=0,
        n_jobs=-1,
    )
    print(
        f"calibration: {len(calib):,} negative-control pairs, "
        f"median p {calib.p_value.median():.3f} (0.5 expected under the null)"
    )

    # --- 5. results back onto the object, where the rest of the workflow sees
    # them. `uns` keeps the per-pair table; `var` gets a per-gene summary, so
    # a gene's strongest hit is available anywhere `rna.var` is.
    mdata.uns["pysceptre_discovery"] = result
    mdata.uns["pysceptre_calibration"] = calib
    best = result.loc[result.groupby("response_id")["p_value"].idxmin()]
    rna.var["sceptre_min_p"] = best.set_index("response_id").p_value.reindex(rna.var_names)
    # Bracketed, not attribute access: `pct_change` is also a DataFrame
    # method, so `frame.pct_change` returns the method rather than the column.
    rna.var["sceptre_pct_change"] = best.set_index("response_id")["pct_change"].reindex(
        rna.var_names
    )

    hit = result.nsmallest(1, "p_value").iloc[0]
    print(
        f"top hit: {hit.response_id} x {hit.grna_target}  p = {hit.p_value:.2e}  "
        f"{hit['pct_change']:+.1f}% "
        f"[{hit['pct_change_ci_low']:+.1f}, {hit['pct_change_ci_high']:+.1f}]"
    )

    # --- 6. carry on in scanpy, on the same object ---------------------------
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    sc.pp.pca(rna, n_comps=10)
    sc.pp.neighbors(rna, n_neighbors=10)
    sc.tl.leiden(rna, flavor="igraph", n_iterations=2, directed=False)
    print(
        f"scanpy continues: PCA {rna.obsm['X_pca'].shape}, "
        f"{rna.obs.leiden.nunique()} leiden clusters"
    )
    print("\nno R involved at any point.")


if __name__ == "__main__":
    main()
