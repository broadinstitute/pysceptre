"""Backed reads and integer count storage must not change a single number.

Both are storage-level changes: reading genes from disk on demand rather than
all at once, and storing UMI counts as integers rather than the float64 R hands
over. Neither may alter a result, so these tests are identity tests against the
eager float path rather than behavioural tests.

`mudata` lives in the `io` extra (dev-only, not a runtime dependency), so these
skip when it is absent -- which is the case in the CI test matrix, and is why
nothing here is allowed to be the only coverage of a code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

pytest.importorskip("mudata", reason="the io extra is not installed")

from sceptre_io import (  # noqa: E402
    BackedResponseMatrix,
    SceptreExport,
    as_counts,
    count_dtype,
    load_h5mu,
    write_h5mu,
)

from pysceptre.pipeline.calibration import nonzero_counts  # noqa: E402


@pytest.fixture
def dataset(tmp_path):
    """A small but structurally faithful dataset: integer counts, two assays."""
    rng = np.random.default_rng(0)
    n_genes, n_cells = 60, 400
    X = sparse.random(n_genes, n_cells, density=0.35, format="csr", random_state=0)
    # Integer counts, as a real response matrix holds.
    X.data = np.ceil(X.data * 40).astype(np.float64)
    targets = {f"t{i}": np.sort(rng.choice(n_cells, 40, replace=False)) for i in range(5)}
    ntc = {f"ntc{i}": np.sort(rng.choice(n_cells, 25, replace=False)) for i in range(8)}
    export = SceptreExport(
        response_matrix=X,
        gene_ids=[f"g{i}" for i in range(n_genes)],
        covariate_matrix=rng.normal(size=(n_cells, 3)),
        grna_target_cells=targets,
        pairs=pd.DataFrame({"response_id": ["g0"], "grna_target": ["t0"]}),
        metadata={
            "n_cells": n_cells,
            "n_genes": n_genes,
            "n_targets": len(targets),
            "n_pairs": 1,
            "n_covariates": 3,
            "n_nonzero": X.nnz,
            "covariate_names": ["a", "b", "c"],
            "side_code": 0,
            "run_permutations": False,
            "B1": 499,
            "B2": 4999,
            "B3": 0,
            "sceptre_version": "0.10.3",
        },
        ntc_grna_cells=ntc,
    )
    path = write_h5mu(export, tmp_path / "dataset.h5mu")
    return path, X, ntc, n_cells


def test_counts_are_stored_as_integers_losslessly(dataset):
    path, X, _, _ = dataset
    eager = load_h5mu(path)
    assert np.issubdtype(eager.response_matrix.dtype, np.integer)
    # Lossless: the values are unchanged, only their storage type is.
    assert (abs(eager.response_matrix - X) > 0).nnz == 0


def test_count_dtype_picks_the_smallest_exact_type():
    assert count_dtype(np.array([0.0, 1.0, 255.0])) == np.uint8
    assert count_dtype(np.array([0.0, 256.0])) == np.uint16
    assert count_dtype(np.array([0.0, 70_000.0])) == np.uint32


@pytest.mark.parametrize(
    "data",
    [
        np.array([0.5, 1.0]),  # not integral
        np.array([-1.0, 2.0]),  # negative
    ],
)
def test_count_dtype_refuses_non_counts(data):
    # Anything that is not a non-negative integer matrix keeps its own dtype,
    # rather than being silently truncated. Normalized expression must survive.
    assert count_dtype(data) is None
    m = sparse.csr_matrix(data.reshape(1, -1))
    assert as_counts(m).dtype == m.dtype


def test_count_dtype_is_chunk_size_invariant():
    # The scan is chunked to avoid a whole-array temporary; the answer must not
    # depend on where the chunk boundaries fall.
    data = np.arange(1000, dtype=np.float64)
    assert count_dtype(data, chunk=7) == count_dtype(data, chunk=100_000)


def test_backed_rows_match_eager_exactly(dataset):
    path, _, _, _ = dataset
    eager = load_h5mu(path).response_matrix
    with BackedResponseMatrix(path) as backed:
        assert backed.shape == eager.shape
        assert backed.nnz == eager.nnz
        for i in range(eager.shape[0]):
            assert np.array_equal(
                np.asarray(backed[i].toarray()).ravel(),
                np.asarray(eager[i].toarray()).ravel(),
            )


def test_backed_contiguous_range_matches_eager(dataset):
    path, _, _, _ = dataset
    eager = load_h5mu(path).response_matrix
    with BackedResponseMatrix(path) as backed:
        for start, stop in [(0, 10), (7, 23), (50, 60), (0, 60)]:
            assert np.array_equal(backed.rows(start, stop).toarray(), eager[start:stop].toarray())


def test_backed_range_clamps_and_empties(dataset):
    path, _, _, _ = dataset
    with BackedResponseMatrix(path) as backed:
        n = backed.shape[0]
        assert backed.rows(n - 2, n + 100).shape == (2, backed.shape[1])
        assert backed.rows(5, 5).shape == (0, backed.shape[1])
        assert backed.rows(9, 3).shape == (0, backed.shape[1])


def test_backed_cache_eviction_preserves_values(dataset):
    # The cache clears wholesale rather than evicting by recency; a gene read
    # after a clear must still be correct.
    path, _, _, _ = dataset
    eager = load_h5mu(path).response_matrix
    with BackedResponseMatrix(path, cache_rows=4) as backed:
        order = [0, 1, 2, 3, 4, 5, 0, 3, 59, 0]
        for i in order:
            assert np.array_equal(
                np.asarray(backed[i].toarray()).ravel(),
                np.asarray(eager[i].toarray()).ravel(),
            )


def test_backed_refuses_to_densify(dataset):
    path, _, _, _ = dataset
    with BackedResponseMatrix(path) as backed, pytest.raises(NotImplementedError):
        backed.toarray()


def test_nonzero_counts_identical_backed_and_eager(dataset):
    """The chunked sweep must agree with the single matmul, exactly."""
    path, _, ntc, n_cells = dataset
    eager = load_h5mu(path).response_matrix
    groups = [ntc["ntc0"], ntc["ntc1"], np.union1d(ntc["ntc2"], ntc["ntc3"])]

    trt_e, cntrl_e = nonzero_counts(eager, groups, n_cells)
    with BackedResponseMatrix(path) as backed:
        # A chunk size that does not divide the gene count, so the last slab is
        # partial -- the case an off-by-one in the sweep would hide behind.
        trt_b, cntrl_b = nonzero_counts(backed, groups, n_cells, gene_chunk=7)
    assert np.array_equal(trt_e, trt_b)
    assert np.array_equal(cntrl_e, cntrl_b)


def test_load_export_backed_flag(dataset, tmp_path):
    from sceptre_io import load_export

    path, _, _, _ = dataset
    eager = load_export(path.parent)
    backed = load_export(path.parent, backed=True)
    assert isinstance(backed.response_matrix, BackedResponseMatrix)
    assert backed.response_matrix.shape == eager.response_matrix.shape
    # Everything small is still read eagerly and must match.
    assert np.allclose(backed.covariate_matrix, eager.covariate_matrix)
    assert list(backed.grna_target_cells) == list(eager.grna_target_cells)
    assert list(backed.ntc_grna_cells) == list(eager.ntc_grna_cells)
    backed.response_matrix.close()


def test_backed_requires_a_converted_dataset(tmp_path):
    from sceptre_io import load_export

    (tmp_path / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="backed reads need a dataset.h5mu"):
        load_export(tmp_path, backed=True)
