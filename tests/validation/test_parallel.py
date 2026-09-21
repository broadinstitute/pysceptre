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
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from pysceptre.pipeline import discovery
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


def test_n_jobs_one_really_means_one_thread():
    """A single-worker run must not quietly use a second core.

    Chunk preparation is prefetched onto a thread of its own, which is a
    1.31x win at `n_jobs=8` and, if left ungated, a 1.43x "win" at
    `n_jobs=1` that is only the extra thread: 831.0 s against 579.1 s on
    day0. `n_jobs` has to mean what it says -- a cgroup-limited container, a
    single-core benchmark and every matched-core comparison against R depend
    on it, and a contaminated single-core figure looks entirely plausible.

    `_map_jobs` imports `ThreadPoolExecutor` locally, so patching the module
    global catches the prefetch and nothing else.
    """
    Y, genes, X, targets, pairs = _dataset()
    kwargs = dict(
        response_matrix=Y,
        gene_ids=genes,
        covariate_matrix=X,
        grna_target_cells=targets,
        pairs=pairs,
        seed=0,
        target_chunk_size=2,  # several chunks, so prefetching is possible at all
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mock.patch.object(discovery, "ThreadPoolExecutor") as pool:
            run_discovery_analysis(n_jobs=1, **kwargs)
            assert not pool.called, "n_jobs=1 started a prefetch thread"
        with mock.patch.object(
            discovery, "ThreadPoolExecutor", wraps=discovery.ThreadPoolExecutor
        ) as pool:
            run_discovery_analysis(n_jobs=2, **kwargs)
            assert pool.called, "n_jobs=2 did not prefetch"


def test_concurrent_target_preparation_does_not_share_state():
    """`fit_all_targets` runs several at once, so its state cannot be a slot.

    Chunks of target state are prepared `_PREFETCH_DEPTH` ahead, so several
    `fit_all_targets` calls are in flight together. Its inputs live in a
    module global -- so that a forked child inherits them without copying --
    and with a single shared slot one call's `finally: clear()` deleted
    another's inputs mid-flight. That surfaced on day0 as
    `KeyError: 'chr2:201746583-201746884'`, and the existing suite missed it
    entirely: the race needs real overlap, and these fixtures are too small
    and too fast to collide by chance.

    So this drives the collision directly rather than hoping for it, and
    requires concurrent results to equal sequential ones.
    """
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(0)
    n_cells = 800
    X = np.column_stack([np.ones(n_cells), rng.normal(size=(n_cells, 2))])
    groups = [
        {f"c{c}t{j}": np.sort(rng.choice(n_cells, 60, replace=False)) for j in range(4)}
        for c in range(6)
    ]

    def prep(cells):
        return discovery.fit_all_targets(cells, X, B1=20, B2=20, B3=0, seed=0, n_jobs=1)

    expected = [prep(g) for g in groups]
    with ThreadPoolExecutor(max_workers=4) as ex:
        got = list(ex.map(prep, groups))

    for want, have in zip(expected, got, strict=True):
        assert set(want) == set(have)
        for tid in want:
            np.testing.assert_array_equal(want[tid].trt_idxs, have[tid].trt_idxs)
            np.testing.assert_allclose(
                want[tid].fitted_probabilities, have[tid].fitted_probabilities, rtol=0, atol=0
            )
    assert not discovery._TARGET_STATE, "per-call state leaked after the run"


def test_backend_override_is_honoured_but_cannot_make_fork_safe_on_macos():
    """`PYSCEPTRE_BACKEND` exists so the choice can be re-measured.

    The evidence for processes on Linux -- 1.85x threads against 3.54x
    processes -- was taken against a gather that no longer exists: the
    statistic is a sparse matmul now, and permutations reach it through a
    prefix scan, both of which release the GIL where the gather did not. A
    settled default nobody can re-test is a default that stays wrong.

    The override must not, however, be able to select `fork` off Linux: that
    restriction is about deadlocking after Accelerate, not performance.
    """
    import os

    with mock.patch.dict(os.environ, {"PYSCEPTRE_BACKEND": "thread"}):
        assert discovery.parallel_backend() == "thread"

    with mock.patch.dict(os.environ, {"PYSCEPTRE_BACKEND": "nonsense"}):
        assert discovery.parallel_backend() in ("fork", "thread")  # falls back

    with mock.patch.dict(os.environ, {"PYSCEPTRE_BACKEND": "fork"}):
        if sys.platform.startswith("linux"):
            assert discovery.parallel_backend() == "fork"
        else:
            with pytest.warns(UserWarning, match="only used on Linux"):
                assert discovery.parallel_backend() == "thread"


def test_gene_pool_uses_threads_when_there_are_many_chunks():
    """The backend follows the chunk count, because the cost does.

    A worker pool is built per chunk. Permutations run as one chunk and pay
    that once, so processes win there -- measured 70.3 s against threads'
    143.8 s, which are GIL-bound on that path. The CRT has 217 chunks at the
    default budget, so fork cost dominates and threads win: 601.9 s and
    6.93 GB against 679.7 s and 9.46 GB.

    Only those two points are measured, so the threshold sits in the gap
    between them rather than being fitted to either.
    """
    with mock.patch.object(discovery, "parallel_backend", return_value="fork"):
        assert discovery.gene_job_backend(1) is None  # permutations: keep fork
        assert discovery.gene_job_backend(217) == "thread"  # CRT: avoid 217 forks
        assert discovery.gene_job_backend(discovery._THREAD_ABOVE_N_CHUNKS) is None
        assert discovery.gene_job_backend(discovery._THREAD_ABOVE_N_CHUNKS + 1) == "thread"

    # Where the platform already uses threads there is nothing to choose, and
    # the function must not pretend otherwise.
    with mock.patch.object(discovery, "parallel_backend", return_value="thread"):
        assert discovery.gene_job_backend(1) is None
        assert discovery.gene_job_backend(217) is None
