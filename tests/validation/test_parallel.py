"""Parallelism must not be observable in the output.

`n_jobs` distributes the per-pair tests, which the profiler puts at ~80% of
runtime. The guarantee is stronger than "results agree": they must be
**bit-identical** to a serial run, because only the genes *within* an
already-drawn target chunk are distributed. `fit_all_targets` still runs in
the parent, sequentially, so the resampling draws are made in exactly the same
order whatever the worker count. Nothing about the per-pair arithmetic changes
either -- unlike gene chunking, no batch width moves -- so exact equality is
the right assertion here, not a tolerance.

**Each backend is exercised on one platform only.** Linux forks (CI), macOS
threads (local development), so neither environment tests both. That is a
known gap, recorded here rather than papered over; the invariance assertions
below are what each platform contributes.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from pysceptre.pipeline.api import run_discovery_analysis
from pysceptre.pipeline.discovery import (
    parallel_backend,
    resolve_n_jobs,
)


def _dataset(n_genes=8, n_cells=1500, n_targets=6, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n_cells), rng.normal(size=(n_cells, 2))])
    theta, mu = 5.0, 20.0
    Y = rng.negative_binomial(theta, theta / (theta + mu), size=(n_genes, n_cells)).astype(float)
    genes = [f"g{i}" for i in range(n_genes)]
    targets = {f"t{j}": np.sort(rng.choice(n_cells, 120, replace=False)) for j in range(n_targets)}
    pairs = pd.DataFrame(
        [(g, t) for g in genes for t in targets], columns=["response_id", "grna_target"]
    )
    return Y, genes, X, targets, pairs


def _run(n_jobs, **kwargs):
    Y, genes, X, targets, pairs = _dataset(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_discovery_analysis(
            response_matrix=Y,
            gene_ids=genes,
            covariate_matrix=X,
            grna_target_cells=targets,
            pairs=pairs,
            seed=0,
            chunk_memory_gb=4.0,
            n_jobs=n_jobs,
        )


@pytest.mark.parametrize("n_jobs", [2, 3])
def test_parallel_results_are_bit_identical_to_serial(n_jobs):
    serial = _run(1)
    parallel = _run(n_jobs)
    assert list(parallel.columns) == list(serial.columns)
    for col in ("p_value", "fold_change", "se_fold_change", "z_orig", "stage"):
        np.testing.assert_array_equal(
            parallel[col].to_numpy(), serial[col].to_numpy(), err_msg=f"{col} moved"
        )


def test_parallel_preserves_row_order():
    # Row order is target-major and must not leak the fact that the inner
    # loops were inverted, nor the order workers happened to finish in.
    serial, parallel = _run(1), _run(3)
    np.testing.assert_array_equal(
        parallel["response_id"].to_numpy(), serial["response_id"].to_numpy()
    )
    np.testing.assert_array_equal(
        parallel["grna_target"].to_numpy(), serial["grna_target"].to_numpy()
    )


def test_parallel_with_more_workers_than_genes():
    # Workers are capped at the job count; asking for more must not hang or
    # produce empty results.
    serial = _run(1, n_genes=2)
    parallel = _run(16, n_genes=2)
    np.testing.assert_array_equal(parallel["p_value"].to_numpy(), serial["p_value"].to_numpy())


@pytest.mark.parametrize(
    "asked,expected",
    [(1, 1), (2, 2), (0, 1), (None, 1)],
)
def test_resolve_n_jobs(asked, expected):
    assert resolve_n_jobs(asked) == expected


def test_resolve_n_jobs_negative_means_all_cores():
    assert resolve_n_jobs(-1) == (os.cpu_count() or 1)


def test_parallel_backend_is_fork_on_linux_threads_elsewhere():
    backend = parallel_backend()
    assert backend in {"fork", "thread"}
    if sys.platform.startswith("linux"):
        # Threads scale poorly here -- NumPy holds the GIL through much of the
        # gather in the per-pair statistic -- so Linux must get processes.
        assert backend == "fork"
    else:
        # fork after Apple's Accelerate has run can deadlock, since the GCD
        # pools it uses are not fork-safe.
        assert backend == "thread"


def _read_row(args):
    path, row = args
    import sceptre_io

    m = sceptre_io.BackedResponseMatrix(path)
    try:
        return np.asarray(m[row].toarray()).ravel().tolist()
    finally:
        m.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork on this platform")
def test_backed_matrix_survives_a_fork(tmp_path):
    """An inherited HDF5 handle must not be reused in a forked child.

    HDF5 file handles are not fork-safe: children sharing one corrupt each
    other's reads, and do so *silently* -- the reads return wrong bytes rather
    than raising. An eagerly-loaded matrix survives fork fine under
    copy-on-write, so nothing else in the suite would catch this.
    """
    pytest.importorskip("mudata", reason="the io extra is not installed")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "scripts"))
    from sceptre_io import BackedResponseMatrix, SceptreExport, write_h5mu
    from scipy import sparse

    rng = np.random.default_rng(0)
    n_genes, n_cells = 12, 200
    X = sparse.random(n_genes, n_cells, density=0.4, format="csr", random_state=0)
    X.data = np.ceil(X.data * 30)
    export = SceptreExport(
        response_matrix=X,
        gene_ids=[f"g{i}" for i in range(n_genes)],
        covariate_matrix=rng.normal(size=(n_cells, 2)),
        grna_target_cells={"t0": np.arange(20)},
        pairs=pd.DataFrame({"response_id": ["g0"], "grna_target": ["t0"]}),
        metadata={
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_targets": 1,
            "n_pairs": 1,
            "n_covariates": 2,
            "n_nonzero": X.nnz,
            "covariate_names": ["a", "b"],
            "side_code": 0,
            "run_permutations": False,
            "B1": 499,
            "B2": 4999,
            "B3": 0,
            "sceptre_version": "0.10.3",
        },
    )
    path = write_h5mu(export, tmp_path / "dataset.h5mu")

    # Open in the parent first, so the children inherit a live handle -- which
    # is exactly the situation the analysis creates when it forks per chunk.
    parent = BackedResponseMatrix(path)
    expected = {i: np.asarray(parent[i].toarray()).ravel() for i in range(n_genes)}

    ctx = mp.get_context("fork")
    with ctx.Pool(3) as pool:
        got = pool.map(_read_row, [(str(path), i) for i in range(n_genes)])
    parent.close()

    for i, row in enumerate(got):
        np.testing.assert_array_equal(np.asarray(row), expected[i], err_msg=f"gene {i} corrupted")
