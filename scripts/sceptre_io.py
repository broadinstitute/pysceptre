"""Read pysceptre's input datasets.

**pysceptre's input format is h5ad.** It is what the single-cell Python
ecosystem already uses (`anndata`, `scanpy`, `muon`), and it stores sparse
matrices in CSR/CSC form natively -- `data`, `indices` and `indptr` as HDF5
datasets -- so a load reads straight into the final arrays. Measured on moi5:
0.41 GB of RSS and a 62 MB file.

`_load_intermediate` reads the columnar form `export_sceptre_dataset.R`
emits. That exists only so `scripts/make_h5ad.py` can convert a dataset out
of R once; it is not an input format, and it costs 1.17 GB of RSS because the
response matrix arrives as three 22.4M-element triplet columns that must be
materialized before the matrix can be built.

Orientation: the matrix is stored in AnnData's standard `(cells, genes)`
layout as CSC. A CSC matrix's transpose is the same buffers relabelled as CSR,
so `adata.X.T` yields the `(genes, cells)` CSR that
`run_discovery_analysis` wants with **no copy** -- verified with
`np.shares_memory`.

Neither format is ever densified. The full moi5 gene set densifies to ~38 GiB
and was OOM-killed once already.
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


def load_export(path: str | Path) -> SceptreExport:
    """Load a dataset: an .h5ad file, or a directory containing dataset.h5ad.

    Falls back to the R extraction's intermediate form if no h5ad is present,
    so an unconverted directory still works; run `scripts/make_h5ad.py` to
    convert it.
    """
    path = Path(path)
    if path.suffix == ".h5ad" or (path.is_file() and path.suffix in {".h5ad", ".h5"}):
        return load_h5ad(path)
    h5 = path / "dataset.h5ad"
    if h5.exists():
        return load_h5ad(h5)
    return _load_intermediate(path)


def load_h5ad(path: str | Path) -> SceptreExport:
    """Read an h5ad written by `write_h5ad`.

    The response matrix comes back as a zero-copy CSR view of the stored CSC,
    so nothing the size of the matrix is duplicated.
    """
    import anndata as ad

    path = Path(path)
    adata = ad.read_h5ad(path)

    metadata = dict(adata.uns["pysceptre"])
    gene_ids = list(adata.var_names)
    # Stored (cells, genes) CSC; .T relabels the same buffers as (genes, cells) CSR.
    X = adata.X
    if not sparse.isspmatrix_csc(X):
        X = sparse.csc_matrix(X)
    response_matrix = X.T.tocsr() if not sparse.isspmatrix_csr(X.T) else X.T

    covariate_matrix = adata.obs[list(metadata["covariate_names"])].to_numpy(dtype=float)

    assignments = adata.obsm["grna_assignments"]  # (cells, targets) indicator
    target_names = list(metadata["grna_targets"])
    csc = assignments.tocsc()
    grna_target_cells = {
        target_names[j]: csc.indices[csc.indptr[j] : csc.indptr[j + 1]].astype(np.int64)
        for j in range(len(target_names))
    }

    pairs = pd.DataFrame(
        {
            "response_id": np.asarray(adata.uns["pairs"]["response_id"], dtype=object),
            "grna_target": np.asarray(adata.uns["pairs"]["grna_target"], dtype=object),
        }
    )
    discovery = adata.uns.get("discovery_result")
    discovery_result = None
    if discovery is not None:
        discovery_result = _restore_bools(
            pd.DataFrame(dict(discovery)),
            list(adata.uns.get("discovery_result_bool_columns", [])),
        )

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


# Sentinel for a missing logical value, so NA survives the int8 encoding
# below rather than being silently flattened to FALSE.
_BOOL_NA = np.int8(-1)


def _is_bool_column(col: pd.Series) -> bool:
    if col.dtype == bool or isinstance(col.dtype, pd.BooleanDtype):
        return True
    if col.dtype == object:
        present = col.dropna()
        return bool(len(present)) and isinstance(present.iloc[0], (bool, np.bool_))
    return False


def _h5_safe(col: pd.Series) -> np.ndarray:
    """Coerce a column into something h5py will write.

    R's results carry logical columns that arrive as object dtype holding None
    for NA (`significant` and `pass_qc`), which h5py rejects with "Can't
    implicitly convert non-string objects to strings".

    Logicals are encoded as int8 with -1 for NA, and the column names are
    recorded in `uns` so `load_h5ad` can restore them as a nullable boolean.
    Encoding NA as FALSE would be a silent semantic change -- `pass_qc` is NA
    for pairs that were never evaluated, which is not the same as failing.
    """
    if _is_bool_column(col):
        out = np.full(len(col), _BOOL_NA, dtype=np.int8)
        present = col.notna().to_numpy()
        out[present] = col[present].to_numpy(dtype=bool).astype(np.int8)
        return out
    if col.dtype == object:
        return col.astype(str).to_numpy()
    return col.to_numpy()


def _restore_bools(frame: pd.DataFrame, bool_columns) -> pd.DataFrame:
    """Undo `_h5_safe`'s int8 encoding for the recorded logical columns."""
    for name in bool_columns:
        if name in frame:
            raw = frame[name].to_numpy()
            restored = pd.array(raw != 0, dtype="boolean")
            restored[raw == _BOOL_NA] = pd.NA
            frame[name] = restored
    return frame


def write_h5ad(export: SceptreExport, path: str | Path) -> Path:
    """Write a SceptreExport as h5ad, in AnnData's standard orientation.

    `X` is `(cells, genes)` CSC, cell covariates go in `obs`, gene ids in
    `var`, the gRNA assignments as a sparse `(cells, targets)` indicator in
    `obsm` -- which is where a `MuData` gRNA modality would naturally live --
    and the pair table and analysis parameters in `uns`.
    """
    import anndata as ad

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_cells = export.metadata["n_cells"]
    target_names = list(export.grna_target_cells)
    indicator = sparse.lil_matrix((n_cells, len(target_names)), dtype=np.int8)
    for j, target in enumerate(target_names):
        indicator[export.grna_target_cells[target], j] = 1

    metadata = dict(export.metadata)
    metadata["grna_targets"] = target_names

    adata = ad.AnnData(
        X=export.response_matrix.T.tocsc(),  # (cells, genes)
        obs=pd.DataFrame(
            export.covariate_matrix,
            columns=list(export.metadata["covariate_names"]),
            index=pd.RangeIndex(n_cells).astype(str),
        ),
        var=pd.DataFrame(index=pd.Index(export.gene_ids, name="response_id")),
        obsm={"grna_assignments": indicator.tocsr()},
        uns={
            "pysceptre": metadata,
            "pairs": {
                "response_id": export.pairs["response_id"].to_numpy().astype(object),
                "grna_target": export.pairs["grna_target"].to_numpy().astype(object),
            },
        },
    )
    if export.discovery_result is not None:
        result = export.discovery_result
        adata.uns["discovery_result"] = {c: _h5_safe(result[c]) for c in result.columns}
        # Recorded so load_h5ad can undo the int8 encoding; without it the
        # logicals come back as int8 and boolean indexing on them fails.
        adata.uns["discovery_result_bool_columns"] = np.array(
            [c for c in result.columns if _is_bool_column(result[c])], dtype=object
        )
    adata.write_h5ad(path, compression="gzip")
    return path


def _load_intermediate(export_dir: Path) -> SceptreExport:
    """Read the R extraction's columnar intermediate. Migration only -- see the
    module docstring; `scripts/make_h5ad.py` converts it to the real format."""
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
