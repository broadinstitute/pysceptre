"""Tests that degenerate gene fits are reported rather than swallowed.

`estimate_theta` returns a method code saying whether the dispersion MLE
succeeded, and the smallest eigenvalue of Zt_wZ says whether D is meaningful.
Both were previously computed and discarded, so a run could report a test
statistic built on a degenerate fit with no indication at all.
"""

import warnings

import numpy as np
import pytest

from pysceptre.pipeline.discovery import fit_all_genes, summarize_gene_fits
from pysceptre.precompute.pieces import compute_precomputation_pieces


def _nb_inputs(n_genes=4, n_cells=600, theta_true=8.0, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.column_stack([np.ones(n_cells), rng.normal(size=n_cells)])
    mu = np.exp(cov @ np.array([1.6, 0.2]))
    resp = rng.negative_binomial(
        n=theta_true, p=theta_true / (theta_true + mu), size=(n_genes, n_cells)
    ).astype(float)
    return resp, [f"gene_{i}" for i in range(n_genes)], cov


def test_healthy_overdispersed_fit_warns_about_nothing():
    """Records warnings and checks only for the degeneracy report, rather than
    promoting every warning to an error: older numpy versions emit unrelated
    RuntimeWarnings from the matmuls in precompute/pieces.py, which would make
    this fail on 3.10 for reasons that have nothing to do with the fit."""
    resp, gene_ids, cov = _nb_inputs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        precomps = fit_all_genes(resp, gene_ids, cov)
    degeneracy = [w for w in caught if "degenerate gene fits" in str(w.message)]
    assert not degeneracy, f"healthy fit should not be reported: {degeneracy}"
    assert summarize_gene_fits(precomps) == {
        "glm_not_converged": [],
        "theta_fallback": [],
        "theta_clamped": [],
        "singular_design": [],
    }


def test_equidispersed_counts_are_reported_as_degenerate():
    """Exactly-Poisson counts leave the NB dispersion unidentifiable. That
    surfaces either as the MLE failing and falling back, or as theta running
    away to the clamp -- which of the two depends on the data, so assert that
    it is reported at all rather than pinning one mechanism."""
    rng = np.random.default_rng(0)
    n_cells = 600
    cov = np.column_stack([np.ones(n_cells), rng.normal(size=n_cells)])
    resp = rng.poisson(np.exp(cov @ np.array([1.6, 0.2])), size=(3, n_cells)).astype(float)

    with pytest.warns(UserWarning, match="dispersion"):
        precomps = fit_all_genes(resp, [f"g{i}" for i in range(3)], cov)

    summary = summarize_gene_fits(precomps)
    flagged = set(summary["theta_clamped"]) | set(summary["theta_fallback"])
    assert flagged, "equidispersed counts should be reported as degenerate"
    for gene_id in summary["theta_clamped"]:
        assert precomps[gene_id].theta == 1000.0


def test_exactly_collinear_covariates_fail_fast_in_the_glm():
    """Worth pinning the boundary: an *exactly* singular design never reaches
    the reporting path, because the IRLS normal-equations solve raises first.
    That is the right behavior -- a hard error beats a silent result."""
    rng = np.random.default_rng(1)
    n_cells = 400
    x = rng.normal(size=n_cells)
    cov = np.column_stack([np.ones(n_cells), x, x])  # exactly collinear
    resp = rng.negative_binomial(8.0, 8.0 / (8.0 + 5.0), size=(2, n_cells)).astype(float)

    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        fit_all_genes(resp, ["g0", "g1"], cov)


def test_near_collinear_covariates_are_reported_as_rank_deficient():
    """The case the relative check exists for: near-collinearity survives the
    GLM solve but leaves Zt_wZ numerically rank deficient, so D's smallest
    direction is amplified roundoff. Previously silent."""
    rng = np.random.default_rng(1)
    n_cells = 400
    x = rng.normal(size=n_cells)
    cov = np.column_stack([np.ones(n_cells), x, x + 1e-11 * rng.normal(size=n_cells)])
    resp = rng.negative_binomial(8.0, 8.0 / (8.0 + 5.0), size=(2, n_cells)).astype(float)

    with pytest.warns(UserWarning, match="rank-deficient"):
        precomps = fit_all_genes(resp, ["g0", "g1"], cov)

    assert summarize_gene_fits(precomps)["singular_design"] == ["g0", "g1"]


def test_eigenvalues_are_recorded_on_the_pieces():
    rng = np.random.default_rng(2)
    n_cells = 300
    cov = np.column_stack([np.ones(n_cells), rng.normal(size=n_cells)])
    coefs = np.array([1.5, 0.1])
    y = rng.negative_binomial(8.0, 8.0 / (8.0 + np.exp(cov @ coefs))).astype(float)

    pieces = compute_precomputation_pieces(y, cov, coefs, 8.0)
    assert pieces.min_eigenvalue > 0
    assert pieces.max_eigenvalue >= pieces.min_eigenvalue
    healthy_ratio = pieces.min_eigenvalue / pieces.max_eigenvalue
    assert healthy_ratio > 1e-6

    # A collinear design is many orders of magnitude worse conditioned, but
    # note D stays finite -- which is exactly why the check is relative and
    # not an isfinite() test.
    collinear = np.column_stack([cov, cov[:, 1]])
    bad = compute_precomputation_pieces(y, collinear, np.r_[coefs, 0.0], 8.0)
    bad_ratio = bad.min_eigenvalue / bad.max_eigenvalue
    assert bad_ratio < 1e-15
    assert np.isfinite(bad.D).all()


def test_summary_keys_are_stable_for_an_empty_gene_set():
    assert set(summarize_gene_fits({})) == {
        "glm_not_converged",
        "theta_fallback",
        "theta_clamped",
        "singular_design",
    }
