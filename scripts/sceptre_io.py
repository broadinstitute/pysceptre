"""Read pysceptre's input datasets.

**pysceptre's input format is MuData (`.h5mu`).** A CRISPR screen measures two
modalities over one set of cells -- gene expression and gRNA assignment -- so
it is stored as two assays rather than as one matrix with the other smuggled
into `obsm`:

    rna    (cells, genes)  counts, CSC, `var` indexed by response_id
    grna   (cells, units)  0/1 assignments, CSC, `var` carrying `grna_target`
                           and `unit_kind`

That is what the single-cell Python ecosystem already uses (`mudata`, `muon`,
`anndata`, `scanpy`), and it stores sparse matrices natively -- `data`,
`indices` and `indptr` as HDF5 datasets -- so a load reads straight into the
final arrays.

**Why the gRNA assay needs an annotated `var`.** sceptre retains gRNA
assignments at two different resolutions, and only two: `grna_group_idxs` has
one entry per *target*, the union of that target's gRNAs, and
`indiv_nt_grna_idxs` has one entry per individual *non-targeting* gRNA.
Targeting gRNAs are never kept individually, because they are only ever used
as a union; NTCs are, because the calibration check regroups them into
synthetic targets. Crucially, `"non-targeting"` is **not a key** in the
target-keyed table (2,974 keys against 2,975 distinct targets on moi5), so a
target-keyed export drops every NTC -- not by oversight, but by construction.
Storing both kinds as rows of one `var` with a `unit_kind` column is what makes
the calibration check expressible at all.

`_load_intermediate` reads the columnar form `export_sceptre_dataset.R`
emits. That exists only so `scripts/make_h5mu.py` can convert a dataset out
of R once; it is not an input format, and it costs 1.17 GB of RSS because the
response matrix arrives as three 22.4M-element triplet columns that must be
materialized before the matrix can be built.

Orientation: `rna.X` is stored in AnnData's standard `(cells, genes)` layout as
CSC. A CSC matrix's transpose is the same buffers relabelled as CSR, so
`X.T` yields the `(genes, cells)` CSR that `run_discovery_analysis` wants with
**no copy** -- verified with `np.shares_memory`.

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
    # Individual non-targeting gRNAs -> their cells. Keyed by gRNA, not by
    # target: `grna_target_cells` collapses every NTC gRNA into one
    # "non-targeting" entry, and the calibration check needs them separable so
    # it can regroup them into synthetic negative-control targets. None when
    # the export predates this or the object had no NTC gRNAs.
    ntc_grna_cells: dict[str, np.ndarray] | None = None
    # R's calibration-check pairs and result, when the source object had been
    # through run_calibration_check. R's pair selection is unseeded, so these
    # are the only record of which pairs a given result used.
    negative_control_pairs: pd.DataFrame | None = None
    calibration_result: pd.DataFrame | None = None

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
            f"B1/B2/B3={m['B1']}/{m['B2']}/{m['B3']}, "
            f"ntc_grnas={len(self.ntc_grna_cells) if self.ntc_grna_cells else 0}, "
            f"sceptre {m['sceptre_version']}"
        )


class BackedResponseMatrix:
    """A `(n_genes, n_cells)` response matrix read from an `.h5mu` on demand.

    The dataset stores `rna/X` as `(cells, genes)` CSC, and a CSC matrix's
    buffers *are* its transpose's CSR buffers -- so `indptr` already indexes
    genes, and gene `j` is `data[indptr[j]:indptr[j+1]]`. A **contiguous range
    of genes is therefore one contiguous slice of each array**, which is
    exactly the access pattern `fit_all_genes` uses. That is why the orientation
    was chosen, and this class is what finally exploits it.

    The point is to stop paying for genes that are never touched. moi5's full
    matrix is 142.7M nonzeros -- 1.71 GB resident as float64, 0.86 GB once
    stored as counts -- while a calibration run reads 9,045 of its 38,606 genes
    and a discovery run reads 244.

    Reads are served through a bounded row cache because the per-pair loop is
    gene-outer *within* each target chunk, so it re-reads a gene once per chunk
    rather than once overall. Without the cache that is one decompression per
    (gene, chunk).

    Exposes just enough of the scipy.sparse surface for the engine: `shape`,
    `__getitem__` returning a row with `.toarray()`, and `rows()` for the
    contiguous fast path.
    """

    def __init__(self, path: str | Path, key: str = "mod/rna/X", cache_rows: int = 512):
        import h5py

        self._file = h5py.File(Path(path), "r")
        g = self._file[key]
        if g.attrs.get("encoding-type") not in ("csc_matrix", None):
            raise ValueError(
                f"expected {key} stored as CSC (cells, genes); got "
                f"{g.attrs.get('encoding-type')!r}. Backed reads rely on the CSC "
                "buffers being the transpose's CSR buffers."
            )
        self._data = g["data"]
        self._indices = g["indices"]
        # Small enough to hold: 38,607 int32 for moi5. Holding it avoids a
        # round trip per lookup, and every read needs two of its entries.
        self._indptr = g["indptr"][:]
        stored_shape = tuple(g.attrs["shape"])  # (n_cells, n_genes)
        self.shape = (int(stored_shape[1]), int(stored_shape[0]))
        self.dtype = self._data.dtype
        self.nnz = int(self._data.size)
        self._cache: dict[int, sparse.csr_matrix] = {}
        self._cache_rows = cache_rows

    @property
    def n_cells(self) -> int:
        return self.shape[1]

    def rows(self, start: int, stop: int) -> sparse.csr_matrix:
        """Genes `[start, stop)` as CSR, in one contiguous read per array."""
        start, stop = int(start), int(min(stop, self.shape[0]))
        if stop <= start:
            return sparse.csr_matrix((0, self.n_cells), dtype=self.dtype)
        lo, hi = int(self._indptr[start]), int(self._indptr[stop])
        indptr = self._indptr[start : stop + 1] - lo
        return sparse.csr_matrix(
            (self._data[lo:hi], self._indices[lo:hi], indptr),
            shape=(stop - start, self.n_cells),
        )

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.rows(i.start or 0, self.shape[0] if i.stop is None else i.stop)
        i = int(i)
        hit = self._cache.get(i)
        if hit is not None:
            return hit
        row = self.rows(i, i + 1)
        if len(self._cache) >= self._cache_rows:
            # Plain eviction, not LRU: the access pattern sweeps every gene in
            # order once per chunk, which is precisely the pattern LRU handles
            # worst, so recency carries no information here.
            self._cache.clear()
        self._cache[i] = row
        return row

    def toarray(self):
        raise NotImplementedError(
            "refusing to densify a backed response matrix; read genes with "
            "rows(start, stop) or index a single gene"
        )

    def close(self) -> None:
        self._cache.clear()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_export(path: str | Path, backed: bool = False) -> SceptreExport:
    """Load a dataset: an .h5mu file, or a directory containing dataset.h5mu.

    Falls back to the R extraction's intermediate form if no .h5mu is present,
    so an unconverted directory still works; run `scripts/make_h5mu.py` to
    convert it.
    """
    path = Path(path)
    if path.is_file():
        return load_h5mu(path, backed=backed)
    h5mu = path / "dataset.h5mu"
    if h5mu.exists():
        return load_h5mu(h5mu, backed=backed)
    if backed:
        raise ValueError(
            f"backed reads need a dataset.h5mu; {path} holds only the columnar "
            "intermediate. Convert it with scripts/make_h5mu.py first."
        )
    return _load_intermediate(path)


def load_h5mu(path: str | Path, backed: bool = False) -> SceptreExport:
    """Read a dataset written by `write_h5mu`.

    The response matrix comes back as a zero-copy CSR view of the stored CSC,
    so nothing the size of the matrix is duplicated.

    With `backed=True` it is not read at all: a `BackedResponseMatrix` serves
    genes from disk on demand, which is what keeps an all-genes dataset usable
    when a run touches a fraction of it. Everything else -- covariates, gRNA
    assignments, pairs -- is small and still read eagerly.
    """
    import mudata

    path = Path(path)
    if backed:
        # backed="r" leaves each modality's X on disk; the response matrix is
        # then served by BackedResponseMatrix, which understands the CSC
        # layout directly rather than going through AnnData's view machinery.
        mdata = mudata.read_h5mu(path, backed="r")
    else:
        mdata = mudata.read_h5mu(path)
    adata = mdata["rna"]
    grna = mdata["grna"]

    metadata = dict(mdata.uns["pysceptre"])
    gene_ids = list(adata.var_names)
    if backed:
        response_matrix = BackedResponseMatrix(path)
    else:
        # Stored (cells, genes) CSC; .T relabels the same buffers as (genes, cells) CSR.
        X = adata.X
        if not sparse.isspmatrix_csc(X):
            X = sparse.csc_matrix(X)
        response_matrix = X.T.tocsr() if not sparse.isspmatrix_csr(X.T) else X.T

    covariate_matrix = mdata.obs[list(metadata["covariate_names"])].to_numpy(dtype=float)

    # The gRNA assay: cells x assignment units, with `var` saying what each
    # unit is. `target` units are per-target unions; `ntc_grna` units are
    # individual non-targeting gRNAs -- the only two resolutions sceptre keeps.
    # The gRNA assay is materialized even under `backed`: it is ~1M nonzeros
    # against the response matrix's 142.7M, and every target's cell set is
    # needed up front to fit anything at all. Under backed="r" it arrives as an
    # on-disk sparse dataset rather than an array, so it is pulled into memory
    # explicitly -- handing that object to csc_matrix yields dtype object.
    grna_x = grna.X
    if hasattr(grna_x, "to_memory"):
        grna_x = grna_x.to_memory()
    csc = sparse.csc_matrix(grna_x)
    unit_ids = list(grna.var_names)
    kinds = grna.var["unit_kind"].astype(str).to_numpy()
    cells_of = {
        unit_ids[j]: csc.indices[csc.indptr[j] : csc.indptr[j + 1]].astype(np.int64)
        for j in range(len(unit_ids))
    }
    grna_target_cells = {
        u: cells_of[u] for u, k in zip(unit_ids, kinds, strict=True) if k == "target"
    }
    ntc_grna_cells = {
        u: cells_of[u] for u, k in zip(unit_ids, kinds, strict=True) if k == "ntc_grna"
    } or None

    pairs = pd.DataFrame(
        {
            "response_id": np.asarray(mdata.uns["pairs"]["response_id"], dtype=object),
            "grna_target": np.asarray(mdata.uns["pairs"]["grna_target"], dtype=object),
        }
    )

    def _uns_frame(key):
        raw = mdata.uns.get(key)
        if raw is None:
            return None
        return _restore_bools(
            pd.DataFrame(dict(raw)), list(mdata.uns.get(f"{key}_bool_columns", []))
        )

    negative_control_pairs = _uns_frame("negative_control_pairs")
    calibration_result = _uns_frame("calibration_result")

    discovery = mdata.uns.get("discovery_result")
    discovery_result = None
    if discovery is not None:
        discovery_result = _restore_bools(
            pd.DataFrame(dict(discovery)),
            list(mdata.uns.get("discovery_result_bool_columns", [])),
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
        ntc_grna_cells=ntc_grna_cells,
        negative_control_pairs=negative_control_pairs,
        calibration_result=calibration_result,
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
    recorded in `uns` so `load_h5mu` can restore them as a nullable boolean.
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


_COUNT_DTYPES = (np.uint8, np.uint16, np.uint32)


def count_dtype(data: np.ndarray, chunk: int = 8_000_000):
    """Smallest unsigned integer dtype that stores `data` exactly, or None.

    Returns None when the values are not non-negative integers, so anything
    that is not raw counts (normalized or transformed expression) is left in
    its own dtype rather than silently truncated.

    Scanned in chunks: testing integrality of a 142.7M-element float64 array
    with a whole-array comparison allocates another 1.14 GB, which is the cost
    this function exists to avoid.
    """
    if data.size == 0:
        return np.uint8
    lo = np.inf
    hi = -np.inf
    for i in range(0, data.size, chunk):
        c = data[i : i + chunk]
        if np.issubdtype(c.dtype, np.floating) and not np.array_equal(c, np.rint(c)):
            return None
        lo = min(lo, float(c.min()))
        hi = max(hi, float(c.max()))
    if lo < 0:
        return None
    for dt in _COUNT_DTYPES:
        if hi <= np.iinfo(dt).max:
            return dt
    return np.int64


def as_counts(matrix):
    """Re-type a count matrix to the smallest exact unsigned integer dtype.

    UMI counts are integers, and R hands them over as float64, which is 8 bytes
    per nonzero for values that on moi5 never exceed 1,610. Nothing downstream
    does arithmetic in the stored dtype -- the engine's `_get_row` casts every
    row to float on extraction, and the calibration check binarizes -- so this
    is purely a storage choice and cannot change a result.
    """
    dt = count_dtype(matrix.data)
    if dt is None or matrix.data.dtype == dt:
        return matrix
    out = matrix.copy()
    out.data = matrix.data.astype(dt, copy=False)
    return out


def write_h5mu(export: SceptreExport, path: str | Path, compression: str | None = None) -> Path:
    """Write a SceptreExport as MuData: two assays over one set of cells.

    The dataset genuinely has two measured modalities, so it is stored as two,
    rather than as one matrix with the other smuggled into `obsm`:

      `rna`   (cells, genes)  counts, CSC, `var` indexed by response_id.
      `grna`  (cells, units)  0/1 assignments, CSC, `var` carrying
                              `grna_target` and `unit_kind`.

    A unit is either a `target` -- the union of that target's gRNAs, which is
    all sceptre retains for targeting gRNAs -- or an `ntc_grna`, an individual
    non-targeting gRNA. Keeping both in one annotated assay is what makes the
    calibration check expressible: it selects the `ntc_grna` rows of `var` and
    regroups them, where a target-keyed table has no NTCs in it at all.

    Cell covariates are shared, so they live on the MuData's own `obs`. The
    pair table and analysis parameters go in `uns`.

    Storing `X` as (cells, genes) CSC means the transpose to the (genes, cells)
    CSR the engine wants is a zero-copy view.

    **Written uncompressed by default**, because this is a file to be read
    backed. A gzipped dataset must decompress a whole HDF5 chunk to serve any
    read inside it, which is the wrong trade when genes are fetched on demand.
    Compression also buys less here than it appears to: measured on moi5 at
    gzip level 4, `data` went 1142 -> 57 MB, but almost all of that was gzip
    eating the padding in float64 counts, which storing them as `uint16`
    removes at the source. `indices` -- the term that actually dominates an
    uncompressed file at 571 MB -- compressed only 2.7x, because cell indices
    are close to incompressible. Pass `compression="gzip"` for a file meant for
    archiving or transfer rather than analysis.
    """
    import anndata as ad
    import mudata

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_cells = export.metadata["n_cells"]
    units: list[tuple[str, str, np.ndarray]] = [
        (name, "target", cells) for name, cells in export.grna_target_cells.items()
    ]
    if export.ntc_grna_cells:
        units += [(name, "ntc_grna", cells) for name, cells in export.ntc_grna_cells.items()]

    # Built straight as CSC: column j is unit j's cells, already sorted, so the
    # indptr is a cumulative count and no lil_matrix intermediate is needed.
    counts = np.array([c.size for c in (u[2] for u in units)], dtype=np.int64)
    indptr = np.zeros(len(units) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    indices = np.concatenate([u[2] for u in units]) if units else np.empty(0, dtype=np.int64)
    assignments = sparse.csc_matrix(
        (np.ones(indices.size, dtype=np.int8), indices, indptr),
        shape=(n_cells, len(units)),
    )

    obs_index = pd.RangeIndex(n_cells).astype(str)
    rna = ad.AnnData(
        X=as_counts(export.response_matrix.T.tocsc()),  # (cells, genes) counts
        var=pd.DataFrame(index=pd.Index(export.gene_ids, name="response_id")),
        obs=pd.DataFrame(index=obs_index),
    )
    grna = ad.AnnData(
        X=assignments,
        var=pd.DataFrame(
            {
                "grna_target": [
                    name if kind == "target" else "non-targeting" for name, kind, _ in units
                ],
                "unit_kind": [kind for _, kind, _ in units],
            },
            index=pd.Index([name for name, _, _ in units], name="unit_id"),
        ),
        obs=pd.DataFrame(index=obs_index),
    )

    mdata = mudata.MuData({"rna": rna, "grna": grna})
    for j, name in enumerate(export.metadata["covariate_names"]):
        mdata.obs[name] = export.covariate_matrix[:, j]
    mdata.uns["pysceptre"] = dict(export.metadata)
    mdata.uns["pairs"] = {
        "response_id": export.pairs["response_id"].to_numpy().astype(object),
        "grna_target": export.pairs["grna_target"].to_numpy().astype(object),
    }
    for key, frame in (
        ("negative_control_pairs", export.negative_control_pairs),
        ("calibration_result", export.calibration_result),
    ):
        if frame is not None and len(frame):
            mdata.uns[key] = {c: _h5_safe(frame[c]) for c in frame.columns}
            mdata.uns[f"{key}_bool_columns"] = np.array(
                [c for c in frame.columns if _is_bool_column(frame[c])], dtype=object
            )
    if export.discovery_result is not None:
        result = export.discovery_result
        mdata.uns["discovery_result"] = {c: _h5_safe(result[c]) for c in result.columns}
        # Recorded so the loader can undo the int8 encoding; without it the
        # logicals come back as int8 and boolean indexing on them fails.
        mdata.uns["discovery_result_bool_columns"] = np.array(
            [c for c in result.columns if _is_bool_column(result[c])], dtype=object
        )
    if compression is None:
        mdata.write_h5mu(path)
    else:
        mdata.write_h5mu(path, compression=compression)
    return path


def _load_intermediate(export_dir: Path) -> SceptreExport:
    """Read the R extraction's columnar intermediate. Migration only -- see the
    module docstring; `scripts/make_h5mu.py` converts it to the real format."""
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

    # One assignment table plus its annotation, mirroring the `grna` assay.
    assignments = pd.read_parquet(export_dir / "grna_assignments.parquet")
    annotation = pd.read_parquet(export_dir / "grna_annotation.parquet")
    cells_of = {
        str(unit): group["cell_index"].to_numpy(dtype=np.int64)
        for unit, group in assignments.groupby("unit_id", sort=False)
    }
    kind_of = dict(
        zip(
            annotation["unit_id"].astype(str),
            annotation["unit_kind"].astype(str),
            strict=True,
        )
    )
    grna_target_cells = {u: c for u, c in cells_of.items() if kind_of.get(u) == "target"}
    ntc_grna_cells = {u: c for u, c in cells_of.items() if kind_of.get(u) == "ntc_grna"} or None

    pairs = pd.read_parquet(export_dir / "pairs.parquet")

    discovery_path = export_dir / "discovery_result.parquet"
    discovery_result = pd.read_parquet(discovery_path) if discovery_path.exists() else None

    def _optional(name):
        path = export_dir / name
        return pd.read_parquet(path) if path.exists() else None

    negative_control_pairs = _optional("negative_control_pairs.parquet")
    calibration_result = _optional("calibration_result.parquet")

    _check_shapes(metadata, response_matrix, covariate_matrix, gene_ids, grna_target_cells)
    return SceptreExport(
        response_matrix=response_matrix,
        gene_ids=gene_ids,
        covariate_matrix=covariate_matrix,
        grna_target_cells=grna_target_cells,
        pairs=pairs,
        metadata=metadata,
        discovery_result=discovery_result,
        ntc_grna_cells=ntc_grna_cells,
        negative_control_pairs=negative_control_pairs,
        calibration_result=calibration_result,
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
