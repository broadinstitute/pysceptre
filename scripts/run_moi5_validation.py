"""Loads the real moi5 dataset (exported by export_moi5_for_pysceptre*.R) and
runs it through pysceptre's run_discovery_analysis, then compares against
R's actual sceptre discovery_result (results_crt.rds) on the same data.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from pysceptre.pipeline.api import run_discovery_analysis

DATA_DIR = "/mnt/disks/sw-dev-disk/pysceptre/tests/validation/moi5_real"


def load_data():
    gene_ids = [line.strip() for line in open(f"{DATA_DIR}/response_matrix.genes.txt")]
    n_genes = len(gene_ids)

    cov_cols = [line.strip() for line in open(f"{DATA_DIR}/covariate_matrix.cols.txt")]
    n_cov = len(cov_cols)

    cov_flat = np.fromfile(f"{DATA_DIR}/covariate_matrix.bin", dtype=np.float64)
    n_cells = cov_flat.size // n_cov
    covariate_matrix = cov_flat.reshape(n_cells, n_cov)

    resp_flat = np.fromfile(f"{DATA_DIR}/response_matrix.bin", dtype=np.float64)
    assert resp_flat.size == n_genes * n_cells, (resp_flat.size, n_genes, n_cells)
    response_matrix = resp_flat.reshape(n_genes, n_cells)

    target_cells_df = pd.read_csv(f"{DATA_DIR}/grna_target_cells.csv")
    grna_target_cells = {
        target: group["cell_index_0based"].to_numpy()
        for target, group in target_cells_df.groupby("grna_target")
    }

    pairs = pd.read_csv(f"{DATA_DIR}/pairs.csv")

    r_results = pd.read_csv(f"{DATA_DIR}/r_discovery_result.csv")

    print(f"Loaded: {n_genes} genes, {n_cells} cells, {n_cov} covariates, "
          f"{len(grna_target_cells)} targets, {len(pairs)} pairs")
    return response_matrix, gene_ids, covariate_matrix, grna_target_cells, pairs, r_results


def main():
    response_matrix, gene_ids, covariate_matrix, grna_target_cells, pairs, r_results = load_data()

    t0 = time.time()
    result = run_discovery_analysis(
        response_matrix=response_matrix,
        gene_ids=gene_ids,
        covariate_matrix=covariate_matrix,
        grna_target_cells=grna_target_cells,
        pairs=pairs,
        side="left",
        seed=0,
    )
    elapsed = time.time() - t0
    print(f"\npysceptre run_discovery_analysis: {len(pairs)} pairs in {elapsed:.1f}s "
          f"({elapsed / len(pairs) * 1000:.2f} ms/pair)")

    result.to_csv(f"{DATA_DIR}/pysceptre_discovery_result.csv", index=False)

    merged = result.merge(
        r_results, on=["response_id", "grna_target"], suffixes=("_py", "_r")
    )
    print(f"\nMerged {len(merged)} pairs (of {len(result)} pysceptre / {len(r_results)} R)")

    merged["neglog10_p_py"] = -np.log10(np.clip(merged["p_value_py"] if "p_value_py" in merged else merged["p_value"], 1e-300, None))

    # column naming: pysceptre's own output col is "p_value"/"fold_change"/"log_2_fold_change",
    # R's is the same names -- after merge with suffixes, ambiguous columns get _py/_r
    p_py_col = "p_value_py" if "p_value_py" in merged.columns else "p_value"
    p_r_col = "p_value_r" if "p_value_r" in merged.columns else "p_value"
    fc_py_col = "fold_change_py" if "fold_change_py" in merged.columns else "fold_change"
    fc_r_col = "fold_change_r" if "fold_change_r" in merged.columns else "fold_change"

    from scipy.stats import spearmanr, pearsonr

    spearman_p = spearmanr(merged[p_py_col], merged[p_r_col]).statistic
    pearson_fc = pearsonr(merged[fc_py_col], merged[fc_r_col]).statistic
    mean_abs_log2fc_diff = np.mean(np.abs(
        np.log2(merged[fc_py_col]) - np.log2(merged[fc_r_col])
    ))

    r_sig = r_results["significant"].fillna(False).astype(bool)
    r_sig_pairs = set(zip(r_results.loc[r_sig, "response_id"], r_results.loc[r_sig, "grna_target"]))
    # pysceptre doesn't do multiple-testing correction here, so define "significant" the same
    # nominal way R's own alpha=0.1 BH-threshold worked out to, approximated via matching p-value
    # rank; for a first look, just report p-value/fold-change agreement stats above, and the
    # overlap of "very significant" (p < 1e-4) calls as a sanity check on direction/strength.
    py_strong = set(zip(merged.loc[merged[p_py_col] < 1e-4, "response_id"], merged.loc[merged[p_py_col] < 1e-4, "grna_target"]))
    r_strong = set(zip(merged.loc[merged[p_r_col] < 1e-4, "response_id"], merged.loc[merged[p_r_col] < 1e-4, "grna_target"]))
    jaccard = len(py_strong & r_strong) / len(py_strong | r_strong) if (py_strong | r_strong) else float("nan")

    print(f"\nSpearman correlation (p-value):        {spearman_p:.4f}")
    print(f"Pearson correlation (fold_change):      {pearson_fc:.4f}")
    print(f"Mean |log2FC difference|:               {mean_abs_log2fc_diff:.4f}")
    print(f"p<1e-4 calls: pysceptre={len(py_strong)}  R={len(r_strong)}  "
          f"both={len(py_strong & r_strong)}  Jaccard={jaccard:.4f}")

    merged["abs_neglog10p_diff"] = np.abs(
        -np.log10(np.clip(merged[p_py_col], 1e-300, None)) - -np.log10(np.clip(merged[p_r_col], 1e-300, None))
    )
    print("\nTop 10 biggest -log10(p) disagreements:")
    print(merged.sort_values("abs_neglog10p_diff", ascending=False)
          [["response_id", "grna_target", p_py_col, p_r_col, fc_py_col, fc_r_col]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
