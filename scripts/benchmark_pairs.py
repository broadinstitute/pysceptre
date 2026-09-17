"""Benchmark at (roughly) the real dataset's scale: ~292 genes, ~3026 targets,
~34,886 QC-passing pairs, ~586k cells (element-gene-power-analysis, moi5/moi10
union sceptre objects). Uses synthetic NB-distributed data shaped like the
real thing, not the real data itself (which isn't available in this repo).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from pysceptre.pipeline.api import run_discovery_analysis

N_CELLS = 586_309
N_GENES = 292
N_TARGETS = 3_026
N_PAIRS = 34_886
MEDIAN_TRT_CELLS = 396


def build_synthetic_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)

    X = np.column_stack(
        [
            np.ones(N_CELLS),
            rng.normal(size=N_CELLS),
            rng.normal(size=N_CELLS),
        ]
    )

    beta = rng.normal(loc=[1.0, 0.2, -0.1], scale=0.1, size=(N_GENES, 3))
    theta_true = rng.uniform(1, 50, size=N_GENES)
    mu_true = np.exp(X @ beta.T)  # (n_cells, n_genes)
    response_matrix = rng.negative_binomial(
        n=theta_true[None, :], p=theta_true[None, :] / (theta_true[None, :] + mu_true)
    ).T.astype(float)  # (n_genes, n_cells)
    gene_ids = [f"gene_{i}" for i in range(N_GENES)]

    grna_target_cells = {}
    for t in range(N_TARGETS):
        n_trt = max(5, int(rng.lognormal(mean=np.log(MEDIAN_TRT_CELLS), sigma=0.5)))
        n_trt = min(n_trt, N_CELLS // 2)
        grna_target_cells[f"target_{t}"] = rng.choice(N_CELLS, size=n_trt, replace=False)

    target_ids = list(grna_target_cells.keys())
    pair_gene_idx = rng.integers(0, N_GENES, size=N_PAIRS)
    pair_target_idx = rng.integers(0, N_TARGETS, size=N_PAIRS)
    pairs = (
        pd.DataFrame(
            {
                "response_id": [gene_ids[i] for i in pair_gene_idx],
                "grna_target": [target_ids[i] for i in pair_target_idx],
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return response_matrix, gene_ids, X, grna_target_cells, pairs


def main():
    n_pairs_override = int(sys.argv[1]) if len(sys.argv) > 1 else None

    print(f"Building synthetic dataset: {N_GENES} genes x {N_CELLS} cells, {N_TARGETS} targets...")
    t0 = time.time()
    response_matrix, gene_ids, X, grna_target_cells, pairs = build_synthetic_dataset()
    if n_pairs_override:
        pairs = pairs.iloc[:n_pairs_override].reset_index(drop=True)
    print(f"  built in {time.time() - t0:.1f}s, {len(pairs)} pairs to test")

    t0 = time.time()
    result = run_discovery_analysis(
        response_matrix=response_matrix,
        gene_ids=gene_ids,
        covariate_matrix=X,
        grna_target_cells=grna_target_cells,
        pairs=pairs,
        seed=0,
    )
    elapsed = time.time() - t0
    print(
        f"\nrun_discovery_analysis: {len(pairs)} pairs in {elapsed:.1f}s ({elapsed / len(pairs) * 1000:.2f} ms/pair)"
    )
    print(result["stage"].value_counts())
    print(result.head())


if __name__ == "__main__":
    main()
