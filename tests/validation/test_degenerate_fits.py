"""Tests that degenerate gene fits are reported rather than swallowed.

`estimate_theta` returns a method code saying whether the dispersion MLE
succeeded, and the smallest eigenvalue of Zt_wZ says whether D is meaningful.
Both were previously computed and discarded, so a run could report a test
statistic built on a degenerate fit with no indication at all.
"""

import warnings
from dataclasses import replace

import numpy as np
import pytest

from pysceptre.pipeline.discovery import (
    GenePrecomputation,
    _design_is_rank_deficient,
    fit_all_genes,
    summarize_gene_fits,
)
from pysceptre.precompute.pieces import (
    PrecomputationPieces,
    compute_precomputation_pieces,
)


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


def _pieces_with_eigenvalues(min_eig, max_eig, p=3, n=10):
    """A PrecomputationPieces carrying only what the rank check reads."""
    return PrecomputationPieces(
        mu=np.ones(n),
        w=np.ones(n),
        a=np.ones(n),
        D=np.zeros((p, n)),
        min_eigenvalue=min_eig,
        max_eigenvalue=max_eig,
    )


@pytest.mark.parametrize(
    ("min_eig", "max_eig", "expected"),
    [
        (1.0, 100.0, False),  # well conditioned
        (1e-6, 1.0, False),  # poorly conditioned but still full rank
        (4.3e-14, 540.0, True),  # measured values for exactly collinear covariates
        (0.0, 540.0, True),  # exactly singular
        (-1e-13, 540.0, True),  # slightly negative, as LAPACK can return
        (float("nan"), 540.0, True),  # eigendecomposition produced nothing usable
        (1.0, float("nan"), True),
        (0.0, 0.0, True),  # degenerate all round
    ],
)
def test_rank_deficiency_check_on_exact_eigenvalues(min_eig, max_eig, expected):
    """Tested on hand-built eigenvalues rather than a near-collinear design.

    Constructing a design whose conditioning lands in a specific window is not
    portable: the same matrix gives a slightly positive smallest eigenvalue on
    one LAPACK and a slightly negative one on another, which moved an earlier
    version of this test across the threshold between macOS and Linux CI.
    The threshold logic is what matters, so test it directly.
    """
    assert _design_is_rank_deficient(_pieces_with_eigenvalues(min_eig, max_eig)) is expected


def test_summarize_groups_each_degeneracy_independently():
    healthy = GenePrecomputation(
        y=np.ones(4),
        fitted_coefs=np.ones(2),
        theta=8.0,
        pieces=_pieces_with_eigenvalues(1.0, 10.0),
    )
    precomps = {
        "ok": healthy,
        "not_converged": replace(healthy, glm_converged=False),
        "theta_fell_back": replace(healthy, theta_method=2),
        "theta_at_bound": replace(healthy, theta_clamped=True),
        "singular": replace(healthy, pieces=_pieces_with_eigenvalues(0.0, 10.0)),
    }
    assert summarize_gene_fits(precomps) == {
        "glm_not_converged": ["not_converged"],
        "theta_fallback": ["theta_fell_back"],
        "theta_clamped": ["theta_at_bound"],
        "singular_design": ["singular"],
    }


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

    # A collinear design is many orders of magnitude worse conditioned. Whether
    # its D comes out finite is LAPACK-dependent -- macOS Accelerate returns a
    # tiny positive smallest eigenvalue and finite D, Linux OpenBLAS returns a
    # slightly negative one and non-finite D -- so only the conditioning is
    # asserted here. That variability is precisely why the rank check is
    # relative rather than an isfinite() test.
    collinear = np.column_stack([cov, cov[:, 1]])
    bad = compute_precomputation_pieces(y, collinear, np.r_[coefs, 0.0], 8.0)
    assert abs(bad.min_eigenvalue) / bad.max_eigenvalue < 1e-15


def test_summary_keys_are_stable_for_an_empty_gene_set():
    assert set(summarize_gene_fits({})) == {
        "glm_not_converged",
        "theta_fallback",
        "theta_clamped",
        "singular_design",
    }
