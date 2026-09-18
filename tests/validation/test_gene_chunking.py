"""Tests for gene-chunked fitting in pipeline/discovery.py.

The Poisson IRLS needs dense (n_cells, k) arrays, so densifying every gene at
once is what made a genome-wide run unaffordable (38,606 genes x 131k cells is
a 40 GB response array, ~162 GB peak). Chunking bounds that; these tests pin
that it does not change results.
"""

import numpy as np
import pandas as pd
import pytest

from pysceptre.pipeline.api import run_discovery_analysis
from pysceptre.pipeline.discovery import fit_all_genes, gene_chunk_size_for_budget


def test_chunk_size_from_budget_matches_hand_calculation():
    # 10 arrays x 131,000 cells x 8 bytes = 10.48 MB per gene; 2 GB / that = 190.
    # The factor is measured, not counted -- see _IRLS_ARRAYS_PER_COLUMN.
    assert gene_chunk_size_for_budget(131_000, 10_000, 2.0) == 190


def test_chunk_size_never_exceeds_the_gene_count():
    assert gene_chunk_size_for_budget(1_000, 7, 100.0) == 7


def test_chunk_size_is_at_least_one_even_on_a_tiny_budget():
    assert gene_chunk_size_for_budget(131_000, 100, 1e-9) == 1


def test_moi5_scale_analysis_still_fits_in_one_chunk():
    """The default budget must not make normal runs chunk needlessly: the
    real moi5 analysis tested 244 genes over 131k cells. At the corrected
    per-column cost, 4 GB still holds all of them (381 would fit)."""
    assert gene_chunk_size_for_budget(131_000, 244, 4.0) == 244


def _inputs(n_genes=7, n_cells=400, n_targets=3, seed=0):
    """Responses are negative-binomial, i.e. overdispersed, deliberately.

    Exactly-Poisson counts make the NB dispersion unidentifiable (the true
    theta is unbounded), so `estimate_theta`'s Newton iteration wanders and
    the [0.01, 1000] clamp catches it -- a 1e-15 perturbation of the
    coefficients then moves theta by ~1000. That is a property of
    equidispersed data, not of chunking, and it would make these tests
    measure the wrong thing.
    """
    rng = np.random.default_rng(seed)
    cov = np.column_stack([np.ones(n_cells), rng.normal(size=n_cells)])
    gene_ids = [f"gene_{i}" for i in range(n_genes)]
    theta_true = 8.0
    mu_true = np.exp(cov @ np.array([1.6, 0.2]))
    resp = rng.negative_binomial(
        n=theta_true, p=theta_true / (theta_true + mu_true), size=(n_genes, n_cells)
    ).astype(float)
    cells = {f"target_{t}": rng.choice(n_cells, size=40, replace=False) for t in range(n_targets)}
    pairs = pd.DataFrame([{"response_id": g, "grna_target": t} for g in gene_ids for t in cells])
    return resp, gene_ids, cov, cells, pairs


@pytest.mark.parametrize("budget_gb", [1e-9, 1e-5, 100.0])
def test_fit_all_genes_agrees_across_chunk_sizes(budget_gb):
    """A gene's fit must not depend on which other genes share its batch.
    `fit_poisson_glm_batch` solves each column's normal equations
    independently, but via one batched `np.linalg.solve`, whose blocking
    depends on the batch width -- so agreement is to floating-point
    tolerance, not bit-for-bit. Measured spread is ~1e-15.
    """
    resp, gene_ids, cov, _, _ = _inputs()
    reference = fit_all_genes(resp, gene_ids, cov, chunk_memory_gb=100.0)
    other = fit_all_genes(resp, gene_ids, cov, chunk_memory_gb=budget_gb)

    assert list(reference) == list(other)
    for gid in gene_ids:
        r, o = reference[gid], other[gid]
        np.testing.assert_allclose(r.fitted_coefs, o.fitted_coefs, rtol=1e-10)
        np.testing.assert_allclose(r.theta, o.theta, rtol=1e-8)
        assert (r.theta_method, r.theta_clamped, r.glm_converged) == (
            o.theta_method,
            o.theta_clamped,
            o.glm_converged,
        )
        np.testing.assert_allclose(r.min_eigenvalue, o.min_eigenvalue, rtol=1e-8)
        np.testing.assert_allclose(r.max_eigenvalue, o.max_eigenvalue, rtol=1e-8)


def test_end_to_end_results_agree_across_gene_chunk_budgets():
    resp, gene_ids, cov, cells, pairs = _inputs()
    common = dict(
        response_matrix=resp,
        gene_ids=gene_ids,
        covariate_matrix=cov,
        grna_target_cells=cells,
        pairs=pairs,
        side="left",
        seed=11,
    )
    reference = run_discovery_analysis(**common, chunk_memory_gb=100.0)
    chunked = run_discovery_analysis(**common, chunk_memory_gb=1e-9)

    # p-values and stages must match exactly: they are rank-based, so the
    # ~1e-15 coefficient spread cannot move them unless a pair sits precisely
    # on the p <= 0.02 escalation boundary.
    pd.testing.assert_series_equal(reference.p_value, chunked.p_value)
    pd.testing.assert_series_equal(reference.stage, chunked.stage)
    for col in ("z_orig", "fold_change", "se_fold_change", "pct_change"):
        np.testing.assert_allclose(reference[col], chunked[col], rtol=1e-8)


def test_chunking_works_with_a_sparse_response_matrix():
    """_get_row handles scipy.sparse; chunking must not break that path."""
    sparse = pytest.importorskip("scipy.sparse")
    resp, gene_ids, cov, _, _ = _inputs()
    csr = sparse.csr_matrix(resp)
    dense_fit = fit_all_genes(resp, gene_ids, cov, chunk_memory_gb=100.0)
    sparse_fit = fit_all_genes(csr, gene_ids, cov, chunk_memory_gb=1e-9)
    for gid in gene_ids:
        np.testing.assert_allclose(
            dense_fit[gid].fitted_coefs, sparse_fit[gid].fitted_coefs, rtol=1e-12
        )


def test_fitting_a_subset_matches_fitting_all_genes():
    """Skipping untested genes must be a saving, not a change of result.

    `run_discovery_ntcells_complement` fits only the genes some pair mentions.
    That is safe because the batched IRLS solves each column independently --
    `A` is (k, p, p) and `b` is (k, p), solved per k -- so which other genes
    share the batch cannot affect a gene's own fit.

    **Compared to a tolerance, not bitwise.** Changing the subset changes the
    batch width, so `(k, n) @ (n, p*p)` is a differently-shaped matmul and BLAS
    may block and accumulate it differently. The results then differ in the
    last bits, and by how much depends on the BLAS: measured exactly 0.0 on
    macOS/Accelerate and nonzero on Linux/OpenBLAS, which is how an earlier
    version of this test passed locally and failed CI on 3.10 and 3.11. The
    invariant that matters is that the fits agree to floating-point precision.

    The data is negative-binomial with a real dispersion rather than Poisson.
    Under Poisson the dispersion MLE is unidentified, theta runs to
    `_THETA_BOUNDS` (measured 179.5 and 199.8 against a bound of 200), and the
    ill-conditioned fit amplifies exactly the rounding this test should be
    insensitive to.
    """
    rng = np.random.default_rng(0)
    n_genes, n_cells, p = 24, 300, 4
    X = np.column_stack([np.ones(n_cells), rng.normal(size=(n_cells, p - 1))])
    mu = 20.0
    theta = 5.0
    Y = rng.negative_binomial(theta, theta / (theta + mu), size=(n_genes, n_cells)).astype(float)
    gene_ids = [f"g{i}" for i in range(n_genes)]

    every = fit_all_genes(Y, gene_ids, X, chunk_memory_gb=1.0)

    subset_rows = [1, 4, 5, 6, 17, 23]  # scattered, and not starting at 0
    subset_ids = [gene_ids[i] for i in subset_rows]
    some = fit_all_genes(Y, subset_ids, X, chunk_memory_gb=1.0, gene_rows=subset_rows)

    assert set(some) == set(subset_ids)
    for gene_id in subset_ids:
        np.testing.assert_allclose(
            some[gene_id].fitted_coefs, every[gene_id].fitted_coefs, rtol=1e-9, atol=1e-12
        )
        assert some[gene_id].theta == pytest.approx(every[gene_id].theta, rel=1e-6)


def test_subset_fitting_reads_the_right_genes():
    """The saving must not come from fitting the wrong rows.

    A tolerance-based comparison would not catch an off-by-one in the row
    mapping if neighbouring genes happened to fit similarly, so this checks
    the mapping directly: each gene's fit must match fitting that gene alone.
    """
    rng = np.random.default_rng(1)
    n_genes, n_cells = 8, 200
    X = np.column_stack([np.ones(n_cells), rng.normal(size=(n_cells, 2))])
    theta = 5.0
    # Distinct means per gene, so mixing two rows up changes the answer.
    mu = np.linspace(5.0, 60.0, n_genes)[:, None]
    Y = rng.negative_binomial(theta, theta / (theta + mu), size=(n_genes, n_cells)).astype(float)
    ids = [f"g{i}" for i in range(n_genes)]

    for row in (0, 3, 7):
        alone = fit_all_genes(Y, [ids[row]], X, gene_rows=[row])
        both = fit_all_genes(
            Y, [ids[row], ids[(row + 1) % n_genes]], X, gene_rows=[row, (row + 1) % n_genes]
        )
        np.testing.assert_allclose(
            alone[ids[row]].fitted_coefs, both[ids[row]].fitted_coefs, rtol=1e-9, atol=1e-12
        )


def test_fit_all_genes_rejects_mismatched_row_mapping():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(50), rng.normal(size=(50, 2))])
    Y = rng.poisson(5.0, size=(3, 50)).astype(float)
    with pytest.raises(ValueError, match="gene_rows has 2 entries for 3 gene_ids"):
        fit_all_genes(Y, ["a", "b", "c"], X, gene_rows=[0, 1])
