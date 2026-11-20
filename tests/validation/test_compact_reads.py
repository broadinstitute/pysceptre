"""A compact read must give every row, and every result, of the eager read.

`load_export_compact` keeps only some genes and the cells in use, with 32-bit
cell indices, in blocks of consecutive genes. That is storage only: a row must
read back value for value, and an analysis must give the same numbers bit for
bit.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

pytest.importorskip("mudata", reason="the io extra is not installed")

import sceptre_io  # noqa: E402
from sceptre_io import SceptreExport, load_export, load_export_compact, write_h5mu  # noqa: E402

from pysceptre.pipeline.api import run_discovery_analysis  # noqa: E402
from pysceptre.pipeline.discovery import _get_row  # noqa: E402
from pysceptre.pipeline.pairwise_qc import nonzero_counts  # noqa: E402

N_GENES, N_CELLS = 40, 3_000


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """An all-cells export: integer counts, some cells out of use, units in all-cells space."""
    rng = np.random.default_rng(1)
    X = sparse.random(N_GENES, N_CELLS, density=0.3, format="csr", random_state=1)
    X.data = np.ceil(X.data * 30).astype(np.float64)
    in_use = rng.random(N_CELLS) > 0.1
    targets = {f"t{i}": np.sort(rng.choice(N_CELLS, 150, replace=False)) for i in range(6)}
    export = SceptreExport(
        response_matrix=X,
        gene_ids=[f"g{i}" for i in range(N_GENES)],
        covariate_matrix=np.column_stack([np.ones(N_CELLS), rng.normal(size=(N_CELLS, 2))]),
        grna_target_cells=targets,
        pairs=pd.DataFrame(
            [(f"g{g}", t) for g in range(0, N_GENES, 3) for t in targets],
            columns=["response_id", "grna_target"],
        ),
        metadata={
            "n_cells": N_CELLS,
            "n_genes": N_GENES,
            "n_targets": len(targets),
            "n_pairs": 0,
            "n_covariates": 3,
            "n_nonzero": X.nnz,
            "covariate_names": ["intercept", "a", "b"],
            "side_code": 0,
            "run_permutations": False,
            "B1": 499,
            "B2": 4999,
            "B3": 0,
            "sceptre_version": "0.10.3",
            "all_cells": True,
        },
        in_use=in_use,
    )
    path = write_h5mu(export, tmp_path_factory.mktemp("compact") / "dataset.h5mu")
    return path


def _same_rows(eager, compact, genes):
    row_e = {g: i for i, g in enumerate(eager.gene_ids)}
    for j, g in enumerate(compact.gene_ids):
        assert g in genes
        a = _get_row(eager.response_matrix, row_e[g])
        b = _get_row(compact.response_matrix, j)
        assert a.dtype == b.dtype and np.array_equal(a, b), g


@pytest.mark.parametrize("genes", [None, ["g3", "g4", "g5", "g17", "g39", "g0"]])
def test_rows_equal_the_eager_read(dataset, genes):
    eager = load_export(dataset)
    compact = load_export_compact(dataset, genes=genes)
    want = eager.gene_ids if genes is None else [g for g in eager.gene_ids if g in set(genes)]
    assert compact.gene_ids == want
    _same_rows(eager, compact, set(want))
    assert compact.response_matrix.shape == (len(want), eager.response_matrix.shape[1])
    assert np.array_equal(compact.covariate_matrix, eager.covariate_matrix)
    assert compact.grna_target_cells.keys() == eager.grna_target_cells.keys()
    for t, cells in eager.grna_target_cells.items():
        assert np.array_equal(compact.grna_target_cells[t], cells)


def test_many_blocks_and_ranges_across_them(dataset, monkeypatch):
    monkeypatch.setattr(sceptre_io, "_MAX_BLOCK_ENTRIES", 2_000)
    eager = load_export(dataset)
    compact = load_export_compact(dataset, read_entries=1_500)
    m = compact.response_matrix
    assert len(m._blocks) > 3
    assert all(b.indices.dtype == np.int32 for b in m._blocks)
    _same_rows(eager, compact, set(eager.gene_ids))
    a = eager.response_matrix[5:31].toarray()
    b = m.rows(5, 31).toarray()
    assert np.array_equal(a, b)


def test_qc_counts_and_discovery_are_identical(dataset):
    eager = load_export(dataset)
    genes = sorted(set(eager.pairs["response_id"]), key=eager.gene_ids.index)
    compact = load_export_compact(dataset, genes=genes)
    cells = list(eager.grna_target_cells.values())
    n = eager.covariate_matrix.shape[0]
    row_e = [eager.gene_ids.index(g) for g in genes]
    trt_e, ctl_e = nonzero_counts(eager.response_matrix, cells, n)
    trt_c, ctl_c = nonzero_counts(compact.response_matrix, cells, n)
    assert np.array_equal(trt_e[row_e], trt_c) and np.array_equal(ctl_e[row_e], ctl_c)
    runs = [
        run_discovery_analysis(
            ex.response_matrix,
            ex.gene_ids,
            ex.covariate_matrix,
            ex.grna_target_cells,
            eager.pairs,
            seed=5,
        )
        for ex in (eager, compact)
    ]
    pd.testing.assert_frame_equal(runs[0], runs[1], check_exact=True)


def test_unknown_gene_is_refused(dataset):
    with pytest.raises(KeyError, match="not in"):
        load_export_compact(dataset, genes=["g1", "nope"])


# --- a file holding only its cells in use ------------------------------------


@pytest.fixture(
    scope="module", params=[{}, {"compression": "gzip", "compression_opts": 1, "shuffle": True}]
)
def in_use_file(dataset, tmp_path_factory, request):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from write_in_use_h5mu import write_in_use

    return write_in_use(
        dataset, tmp_path_factory.mktemp("in_use") / "dataset.h5mu", **request.param
    )


def test_the_in_use_file_reads_as_the_compact_loader(dataset, in_use_file):
    compact = load_export_compact(dataset)
    backed = sceptre_io.load_h5mu(in_use_file, backed=True)
    assert backed.gene_ids == compact.gene_ids
    assert backed.response_matrix.shape == compact.response_matrix.shape
    for j in range(len(backed.gene_ids)):
        a = _get_row(compact.response_matrix, j)
        b = _get_row(backed.response_matrix, j)
        assert a.dtype == b.dtype and np.array_equal(a, b), j
    assert np.array_equal(backed.covariate_matrix, compact.covariate_matrix)
    # One layout for both files: BLAS gives row- and column-major different last bits.
    assert backed.covariate_matrix.flags.c_contiguous
    assert compact.covariate_matrix.flags.c_contiguous
    assert load_export(dataset).covariate_matrix.flags.c_contiguous
    assert list(backed.grna_target_cells) == list(compact.grna_target_cells)
    for t, cells in compact.grna_target_cells.items():
        assert np.array_equal(backed.grna_target_cells[t], cells)
    assert backed.pairs.equals(compact.pairs)
    assert backed.metadata["n_cells"] == compact.covariate_matrix.shape[0]
    assert backed.metadata["all_cells"] is False or backed.metadata["all_cells"] == 0


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_the_in_use_file_gives_the_same_discovery(dataset, in_use_file, n_jobs):
    eager = load_export(dataset)
    backed = sceptre_io.load_h5mu(in_use_file, backed=True)
    runs = [
        run_discovery_analysis(
            ex.response_matrix,
            ex.gene_ids,
            ex.covariate_matrix,
            ex.grna_target_cells,
            eager.pairs,
            seed=5,
            n_jobs=n_jobs,
        )
        for ex in (eager, backed)
    ]
    pd.testing.assert_frame_equal(runs[0], runs[1], check_exact=True)


def test_the_all_cells_file_is_refused_backed(dataset, in_use_file):
    with pytest.raises(ValueError, match="all-cells"):
        sceptre_io.load_h5mu(dataset, backed=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from write_in_use_h5mu import write_in_use

    with pytest.raises(FileExistsError):
        write_in_use(dataset, in_use_file)
    with pytest.raises(ValueError, match="only cells in use"):
        write_in_use(in_use_file, in_use_file.with_name("again.h5mu"))


# --- an nt_cells dataset: the NT-cell pool, gRNA counts and individual guides --


def _nt_pool_export(pool_cell_out_of_use: bool = False) -> SceptreExport:
    """An all-cells nt_cells export whose pool is in R's order: units in dict order, not sorted."""
    rng = np.random.default_rng(7)
    n_cells, n_genes = 800, 12
    X = sparse.random(n_genes, n_cells, density=0.3, format="csr", random_state=7)
    X.data = np.ceil(X.data * 30)
    in_use = rng.random(n_cells) > 0.15
    cells = rng.permutation(np.flatnonzero(in_use))
    if pool_cell_out_of_use:
        cells[0] = np.flatnonzero(~in_use)[0]
    ntc = {"nt_b": np.sort(cells[:120]), "nt_a": np.sort(cells[120:220])}
    targets = {"t0": np.sort(cells[220:300]), "t1": np.sort(cells[300:390])}
    guides = {"t0_g1": targets["t0"][::2], "t0_g2": targets["t0"][1::2], "t1_g1": targets["t1"]}
    counts = sparse.random(5, n_cells, density=0.2, format="csr", random_state=8)
    counts.data = np.ceil(counts.data * 9)
    return SceptreExport(
        response_matrix=X,
        gene_ids=[f"g{i}" for i in range(n_genes)],
        covariate_matrix=np.column_stack([np.ones(n_cells), rng.normal(size=n_cells)]),
        grna_target_cells=targets,
        pairs=pd.DataFrame({"response_id": ["g0", "g1", "g2"], "grna_target": ["t0", "t0", "t1"]}),
        metadata={
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_targets": len(targets),
            "n_pairs": 3,
            "n_covariates": 2,
            "n_nonzero": X.nnz,
            "covariate_names": ["intercept", "a"],
            "side_code": 0,
            "B1": 499,
            "B2": 4999,
            "B3": 0,
            "sceptre_version": "0.10.3",
            "all_cells": True,
            "low_moi": True,
            "control_group": "nt_cells",
            "resampling_mechanism": "crt",
        },
        ntc_grna_cells=ntc,
        targeting_grna_cells=guides,
        all_nt_idxs=np.concatenate(list(ntc.values())),
        in_use=in_use,
        grna_counts=counts,
        grna_count_ids=["nt_b", "nt_a", "t0_g1", "t0_g2", "t1_g1"],
    )


def _same_cells(a: dict, b: dict) -> None:
    assert list(a) == list(b)
    for unit, cells in b.items():
        assert a[unit].dtype == cells.dtype and np.array_equal(a[unit], cells), unit


@pytest.mark.parametrize("backed", [False, True])
def test_the_in_use_file_keeps_the_nt_pool(tmp_path, backed):
    from write_in_use_h5mu import write_in_use

    src = write_h5mu(_nt_pool_export(), tmp_path / "src.h5mu")
    dst = write_in_use(src, tmp_path / "dst.h5mu")
    want = load_export(src)
    got = sceptre_io.load_h5mu(dst, backed=backed)
    assert want.control_group == got.control_group == "nt_cells" and got.moi == "low"
    assert want.all_nt_idxs.dtype == got.all_nt_idxs.dtype == np.int64
    assert np.array_equal(got.all_nt_idxs, want.all_nt_idxs)
    # R's order, which the sort a remap could slip in would lose.
    assert not np.all(np.diff(got.all_nt_idxs) > 0)
    assert np.array_equal(got.all_nt_idxs, np.concatenate(list(got.ntc_grna_cells.values())))
    _same_cells(got.ntc_grna_cells, want.ntc_grna_cells)
    _same_cells(got.grna_target_cells, want.grna_target_cells)
    _same_cells(got.targeting_grna_cells, want.targeting_grna_cells)
    assert got.grna_count_ids == want.grna_count_ids
    assert got.grna_counts.dtype == want.grna_counts.dtype
    assert got.grna_counts.shape == want.grna_counts.shape
    assert (got.grna_counts != want.grna_counts).nnz == 0
    assert np.array_equal(got.covariate_matrix, want.covariate_matrix)
    assert got.metadata["n_cells"] == want.metadata["n_cells"] == got.covariate_matrix.shape[0]
    assert got.metadata["n_nonzero"] == want.metadata["n_nonzero"]


def test_the_in_use_file_keeps_the_global_var(tmp_path):
    import mudata
    from write_in_use_h5mu import write_in_use

    md = mudata.read_h5mu(write_h5mu(_nt_pool_export(), tmp_path / "plain.h5mu"))
    md.var["note"] = [f"v{i}" for i in range(md.n_vars)]
    md.write_h5mu(tmp_path / "src.h5mu")
    got = mudata.read_h5mu(write_in_use(tmp_path / "src.h5mu", tmp_path / "dst.h5mu"))
    assert got.var_names.equals(md.var_names)
    assert got.var["note"].astype(str).tolist() == md.var["note"].astype(str).tolist()


def _never(*args, **kwargs):
    raise AssertionError("a modality was read into memory before the refusal")


def test_a_pool_cell_out_of_use_is_refused(tmp_path, monkeypatch):
    import anndata
    from write_in_use_h5mu import write_in_use

    src = write_h5mu(_nt_pool_export(pool_cell_out_of_use=True), tmp_path / "pool.h5mu")
    monkeypatch.setattr(anndata.AnnData, "to_memory", _never)
    with pytest.raises(ValueError, match="NT-cell pool all_nt_idxs are not in use"):
        write_in_use(src, tmp_path / "pool_out.h5mu")
    assert sorted(f.name for f in tmp_path.iterdir()) == ["pool.h5mu"]


_EXTRA = ("uns['cell_ids']", "mod['rna'].uns['note']", "obsm['umap']", "obsp['knn']")
_EXTRA += ("varm['loadings']", "varp['corr']")


@pytest.mark.parametrize("entry", _EXTRA)
def test_what_cannot_be_renumbered_is_refused(tmp_path, monkeypatch, entry):
    import anndata
    import mudata
    from write_in_use_h5mu import write_in_use

    md = mudata.read_h5mu(write_h5mu(_nt_pool_export(), tmp_path / "plain.h5mu"))
    n, m = md.n_obs, md.n_vars
    where, key, value = {
        "uns['cell_ids']": (md.uns, "cell_ids", np.arange(10)),
        "mod['rna'].uns['note']": (md.mod["rna"].uns, "note", "x"),
        "obsm['umap']": (md.obsm, "umap", np.zeros((n, 2))),
        "obsp['knn']": (md.obsp, "knn", sparse.identity(n, format="csr")),
        "varm['loadings']": (md.varm, "loadings", np.zeros((m, 2))),
        "varp['corr']": (md.varp, "corr", sparse.identity(m, format="csr")),
    }[entry]
    where[key] = value
    md.write_h5mu(tmp_path / "extra.h5mu")
    monkeypatch.setattr(anndata.AnnData, "to_memory", _never)
    # The entry alone: the plain file holds nothing else that is refused.
    with pytest.raises(ValueError, match=re.escape(f"holds {entry}, which")):
        write_in_use(tmp_path / "extra.h5mu", tmp_path / "extra_out.h5mu")
    assert not (tmp_path / "extra_out.h5mu").exists()


# --- run_trans_background.py fit-genes: the reference run's genes, backed or not --


def test_fit_genes_takes_the_reference_genes_backed_or_not(dataset, in_use_file):
    pytest.importorskip("pyarrow", reason="the io extra is not installed")
    from run_trans_background import _reference_genes

    eager = load_export(dataset)
    wanted = set(eager.pairs["response_id"])
    want = [g for g in eager.gene_ids if g in wanted]
    compact = load_export_compact(dataset, genes="pairs")
    backed = sceptre_io.load_h5mu(in_use_file, backed=True)
    # The backed read lists every gene in the file, the compact one only the pairs'.
    assert backed.gene_ids == eager.gene_ids and compact.gene_ids == want != eager.gene_ids
    assert _reference_genes(backed) == _reference_genes(compact) == want


# --- run_trans_background.py combine: its checks gate the table, finished shards or not --


@pytest.mark.parametrize("repeated", [False, True])
def test_combine_writes_the_table_only_when_its_checks_pass(tmp_path, repeated):
    pytest.importorskip("pyarrow", reason="the io extra is not installed")
    import argparse
    import json

    from run_trans_background import cmd_combine

    pairs_dir, shard = tmp_path / "pairs", tmp_path / "shards" / "shard_0000"
    pairs_dir.mkdir()
    shard.mkdir(parents=True)
    (pairs_dir / "layout.json").write_text(
        json.dumps({"n_elements": 2, "fit_chunk_width": 1, "n_fit_chunks": 2, "layout_hash": "h"})
    )
    (pairs_dir / "shard_manifest.json").write_text(
        json.dumps({"layout_hash": "h", "shards": [[0], [1]]})
    )
    pd.DataFrame(
        {"target_code": [0, 1], "fit_chunk": [0, 1], "grna_target": ["e0", "e1"]}
    ).to_parquet(pairs_dir / "elements.parquet")
    pd.DataFrame(
        {
            "gene_code": np.repeat([0, 1, 2], 2).astype(np.int32),
            "target_code": np.tile([0, 1], 3).astype(np.int32),
            "pass_qc": True,
        }
    ).to_parquet(pairs_dir / "trans_pairs.parquet")
    genes = [0, 1, 2, 1] if repeated else [0, 1, 2]  # gene 1 twice: a rerun with another part size
    pd.DataFrame(
        {
            "gene_code": np.int32(genes),
            "target_code": np.int32(0),
            "p_value": 0.5,
            "fold_change": 0.9,
            "se_fold_change": 0.1,
            "z_orig": -0.3,
            "stage": 1.0,
            "p_value_floor": np.nan,
            "screen": np.int8(0),
            "n_null_evaluated": np.int32(499),
        }
    ).to_parquet(shard / "part_00000.parquet", index=False)
    (shard / "DONE.json").write_text(json.dumps({"host": "x", "reference_checked": 0}))
    out = tmp_path / "out"
    out.mkdir()
    (out / "trans_result.parquet").write_bytes(b"an earlier run's table")
    a = argparse.Namespace(
        pairs_dir=str(pairs_dir),
        shards=str(tmp_path / "shards"),
        out=str(out),
        manifest=None,
        allow_missing=True,  # shard 1 never ran
    )
    if repeated:
        with pytest.raises(SystemExit, match="checks failed"):
            cmd_combine(a)
    else:
        cmd_combine(a)
    assert (out / "summary.json").exists()
    assert (out / "trans_result.parquet").exists() is not repeated
