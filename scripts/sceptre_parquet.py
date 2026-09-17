"""Read the parquet bundle written by scripts/export_sceptre_to_parquet.R.

The response matrix is stored as sparse triplets and is loaded straight into a
scipy.sparse CSR matrix -- it is never densified, which matters at real scale
(the full moi5 gene set densifies to ~38 GiB and was OOM-killed once already).
`run_discovery_analysis` accepts the sparse matrix directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class SceptreExport:
    """Everything `run_discovery_analysis` needs, plus the analysis parameters
    the original sceptre object was configured with."""

    response_matrix: sparse.csr_matrix  # (n_genes, n_cells)
    gene_ids: list[str]
    covariate_matrix: np.ndarray  # (n_cells, p)
    grna_target_cells: dict[str, np.ndarray]
    pairs: pd.DataFrame
    metadata: dict
    discovery_result: pd.DataFrame | None = None

    @property
    def side(self) -> str:
        """Sidedness as `run_discovery_analysis` spells it."""
        return {-1: "left", 0: "both", 1: "right"}[self.metadata["side_code"]]

    def describe(self) -> str:
        m = self.metadata
        return (
            f"{m['n_genes']} genes x {m['n_cells']} cells, {m['n_targets']} targets, "
            f"{m['n_pairs']} pairs, side={self.side}, "
            f"resampling={'permutations' if m['run_permutations'] else 'crt'}, "
            f"B1/B2/B3={m['B1']}/{m['B2']}/{m['B3']}, sceptre {m['sceptre_version']}"
        )


def load_export(export_dir: str | Path) -> SceptreExport:
    export_dir = Path(export_dir)
    metadata = json.loads((export_dir / "metadata.json").read_text())

    gene_ids_df = pd.read_parquet(export_dir / "gene_ids.parquet").sort_values("gene_index")
    gene_ids = gene_ids_df["response_id"].tolist()

    triplets = pd.read_parquet(export_dir / "response_matrix.parquet")
    response_matrix = sparse.csr_matrix(
        (
            triplets["value"].to_numpy(),
            (triplets["gene_index"].to_numpy(), triplets["cell_index"].to_numpy()),
        ),
        shape=(metadata["n_genes"], metadata["n_cells"]),
    )

    covariate_matrix = pd.read_parquet(export_dir / "covariate_matrix.parquet").to_numpy(
        dtype=float
    )

    target_rows = pd.read_parquet(export_dir / "grna_target_cells.parquet")
    grna_target_cells = {
        str(target): group["cell_index"].to_numpy(dtype=np.int64)
        for target, group in target_rows.groupby("grna_target", sort=False)
    }

    pairs = pd.read_parquet(export_dir / "pairs.parquet")

    discovery_path = export_dir / "discovery_result.parquet"
    discovery_result = pd.read_parquet(discovery_path) if discovery_path.exists() else None

    _check_shapes(metadata, response_matrix, covariate_matrix, gene_ids, grna_target_cells)
    return SceptreExport(
        response_matrix=response_matrix,
        gene_ids=gene_ids,
        covariate_matrix=covariate_matrix,
        grna_target_cells=grna_target_cells,
        pairs=pairs,
        metadata=metadata,
        discovery_result=discovery_result,
    )


def _check_shapes(metadata, response_matrix, covariate_matrix, gene_ids, grna_target_cells):
    """Fail loudly here rather than deep inside a GLM fit."""
    n_cells = metadata["n_cells"]
    if response_matrix.shape != (len(gene_ids), n_cells):
        raise ValueError(
            f"response matrix is {response_matrix.shape}, expected ({len(gene_ids)}, {n_cells})"
        )
    if covariate_matrix.shape != (n_cells, metadata["n_covariates"]):
        raise ValueError(
            f"covariate matrix is {covariate_matrix.shape}, expected "
            f"({n_cells}, {metadata['n_covariates']})"
        )
    if response_matrix.nnz != metadata["n_nonzero"]:
        raise ValueError(
            f"response matrix has {response_matrix.nnz} nonzeros, metadata says "
            f"{metadata['n_nonzero']}"
        )
    for target, idxs in grna_target_cells.items():
        if idxs.size and (idxs.min() < 0 or idxs.max() >= n_cells):
            raise ValueError(f"target {target!r} has cell indices outside [0, {n_cells})")


if __name__ == "__main__":
    import sys

    export = load_export(sys.argv[1])
    print(export.describe())
