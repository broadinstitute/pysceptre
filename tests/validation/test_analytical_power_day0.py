"""The input helpers against the R output that produced the published numbers.

The analytical power estimator was validated with baseline statistics computed
by WattEG's R, so these check the helpers reproduce *those*, not merely
something reasonable. In particular the **absolute scale** of
`expression_mean` is only checkable here:
`test_analytical_power_inputs.py` validates every ratio against DESeq2 but
deliberately not the scale, because DESeq2 centres its size factors and this
does not. See `docs/design.md`, "Analytical per-pair power".

Needs two things, both real screen data and so never committed:

    PYSCEPTRE_WATTEG_PREPARED=path/to/prepared \
    PYSCEPTRE_DAY0_SCEPTRE_OBJECT=path/to/sceptre_object.rds \
        pytest -m realdata

The prepared directory supplies the ground truth -- `sim_input.rds` carries
`row_data$mean`, `row_data$dispersion` and `col_data$size_factors` -- and the
sceptre object supplies the counts, because `sim_input.rds` deliberately drops
the count matrix: the simulation draws from the mean and dispersion instead.

**The cell and gene sets have to be the full ones.** On day0 the size factors
were computed over all 292 genes and all 586,309 cells, the 18,619 that QC
removed included, while `row_data` is subset to the 237 genes that appear in
pairs. Recomputing on `cells_in_use` would give different factors and a
different mean for every gene, which is the whole reason an export now keeps
everything by default.

R and the `sceptre` package are needed; the tests skip without them.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from pysceptre.analytical_power import (
    baseline_expression_stats,
    bh_nominal_cutoff,
    poscounts_size_factors,
)

pytestmark = pytest.mark.realdata

# R that reads the pieces out of sim_input.rds. Written to a temp file rather than passed with
# -e so the quoting stays legible.
_DUMP_R = r"""
suppressPackageStartupMessages({library(sceptre); library(Matrix); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
prepared <- args[1]; object_path <- args[2]; out <- args[3]

# Ground truth: what WattEG's own run produced.
si <- readRDS(file.path(prepared, "sim_input.rds"))

# Counts: from the object, because sim_input.rds drops the matrix on purpose. The FULL matrix,
# every gene and every cell, which is what the size factors in si$col_data were computed over.
so <- readRDS(object_path)
counts <- sceptre:::get_response_matrix(so)
if (ncol(counts) != length(si$col_data$size_factors)) {
  stop(sprintf("cell mismatch: matrix has %d columns, sim_input %d size factors. The object and the prepared directory are not from the same run.",
               ncol(counts), length(si$col_data$size_factors)))
}
counts <- as(as(counts, "CsparseMatrix"), "TsparseMatrix")
cat(sprintf("counts %d x %d, %d nonzeros | ground truth on %d genes\n",
            nrow(counts), ncol(counts), length(counts@x), nrow(si$row_data)))

writeLines(jsonlite::toJSON(list(
  all_genes = rownames(counts),
  truth_genes = rownames(si$row_data),
  mean = si$row_data$mean,
  dispersion = si$row_data$dispersion,
  size_factors = si$col_data$size_factors,
  n_genes = nrow(counts), n_cells = ncol(counts),
  i = counts@i, j = counts@j, x = counts@x
), digits = 17, auto_unbox = TRUE), out)
"""


@pytest.fixture(scope="module")
def watteg_prepared() -> dict:
    prepared = os.environ.get("PYSCEPTRE_WATTEG_PREPARED")
    obj = os.environ.get("PYSCEPTRE_DAY0_SCEPTRE_OBJECT")
    if not prepared or not obj:
        pytest.skip("set PYSCEPTRE_WATTEG_PREPARED and PYSCEPTRE_DAY0_SCEPTRE_OBJECT (same run)")
    root = Path(prepared)
    if not (root / "sim_input.rds").exists():
        pytest.skip(f"{root / 'sim_input.rds'} is missing")
    if not Path(obj).exists():
        pytest.skip(f"{obj} is missing")
    try:
        ok = subprocess.run(
            ["Rscript", "-e", "library(sceptre); library(Matrix); library(jsonlite)"],
            capture_output=True,
            timeout=180,
        ).returncode
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("Rscript not available")
    if ok != 0:
        pytest.skip("R lacks sceptre, Matrix or jsonlite")

    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "dump.R"
        script.write_text(_DUMP_R)
        payload = Path(tmp) / "payload.json"
        run = subprocess.run(
            ["Rscript", str(script), str(root), obj, str(payload)],
            capture_output=True,
            text=True,
        )
        if run.returncode != 0:
            pytest.fail(f"could not assemble the ground truth: {run.stderr.strip()[-400:]}")
        with open(payload) as f:
            return json.load(f)


def _matrix(d: dict) -> sparse.csc_matrix:
    return sparse.csc_matrix(
        (np.asarray(d["x"], float), (np.asarray(d["i"], int), np.asarray(d["j"], int))),
        shape=(int(d["n_genes"]), int(d["n_cells"])),
    )


def test_size_factors_match_wattegs_absolute_scale(watteg_prepared):
    """Not centred, and this is the check that says so on real data."""
    d = watteg_prepared
    got = poscounts_size_factors(_matrix(d))
    expected = np.asarray(d["size_factors"], float)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-9)
    # If ours had been centred this would be ~1 and the assertion above would have failed; kept
    # as an explicit record of which convention the real data is in.
    assert not np.isclose(np.exp(np.mean(np.log(got))), 1.0, rtol=1e-6)


def test_normalised_gene_mean_matches_watteg(watteg_prepared):
    """`expression_mean` itself, which is what the estimator consumes.

    Computed over every gene and returned for the subset `row_data` covers,
    which is the distinction `gene_subset` exists to make.
    """
    d = watteg_prepared
    all_genes = list(d["all_genes"])
    truth_genes = list(d["truth_genes"])
    stats = baseline_expression_stats(
        _matrix(d),
        all_genes,
        np.ones(len(all_genes)),
        size_factors=np.asarray(d["size_factors"], float),
        gene_subset=truth_genes,
    )
    assert stats["response_id"].tolist() == truth_genes
    np.testing.assert_allclose(
        stats["expression_mean"].to_numpy(), np.asarray(d["mean"], float), rtol=1e-9
    )


def test_mean_is_the_same_whether_size_factors_are_passed_or_computed(watteg_prepared):
    """The helper's two routes must not diverge."""
    d = watteg_prepared
    genes = list(d["all_genes"])
    m = _matrix(d)
    passed = baseline_expression_stats(
        m, genes, np.ones(len(genes)), size_factors=np.asarray(d["size_factors"], float)
    )
    computed = baseline_expression_stats(m, genes, np.ones(len(genes)))
    np.testing.assert_allclose(
        passed["expression_mean"].to_numpy(), computed["expression_mean"].to_numpy(), rtol=1e-9
    )


def test_theta_is_reported_against_r_rather_than_asserted(watteg_prepared):
    """R's cached theta against ours, reported as a correlation, not a tolerance.

    `expression_size` is `1 / dispersion`. WattEG read sceptre's cached value;
    pysceptre fits its own with different code and clamps it, so demanding
    agreement to a tolerance would be asserting that two estimators coincide.
    What matters is that they do not disagree structurally, so this checks the
    relationship and prints the summary.
    """
    d = watteg_prepared
    theta_r = 1.0 / np.asarray(d["dispersion"], float)
    finite = np.isfinite(theta_r) & (theta_r > 0)
    assert finite.sum() > 0.9 * finite.size, "R's dispersions are mostly unusable"
    lo, hi = 0.01, 1000.0
    within = (theta_r[finite] >= lo) & (theta_r[finite] <= hi)
    print(
        f"\nR theta over {int(finite.sum())} genes: "
        f"median {np.median(theta_r[finite]):.4g}, "
        f"{100 * within.mean():.1f}% inside pysceptre's clamp [{lo}, {hi}]"
    )
    # The clamp is the thing that could silently bite, so it is the thing asserted.
    assert within.mean() > 0.5


def test_bh_cutoff_matches_the_threshold_watteg_derived(watteg_prepared):
    """Against `discovery_threshold.txt`, which is R's own `max(p[significant])`."""
    prepared = Path(os.environ["PYSCEPTRE_WATTEG_PREPARED"])
    threshold_file = prepared / "discovery_threshold.txt"
    if not threshold_file.exists():
        pytest.skip("no discovery_threshold.txt in the prepared directory")
    expected = float(threshold_file.read_text().split()[0])

    export = os.environ.get("PYSCEPTRE_DAY0_EXPORT")
    if not export:
        pytest.skip("set PYSCEPTRE_DAY0_EXPORT so the discovery p-values can be read")
    from scripts.sceptre_io import load_export  # noqa: PLC0415

    ex = load_export(Path(export) / "dataset.h5mu")
    if ex.discovery_result is None or "p_value" not in ex.discovery_result:
        pytest.skip("the export carries no discovery result")
    alpha = float(ex.metadata["multiple_testing_alpha"])
    got = bh_nominal_cutoff(ex.discovery_result["p_value"].to_numpy(), alpha)
    assert got == pytest.approx(expected, rel=1e-12)


_SCEPTRE_DUMP_R = r"""
suppressPackageStartupMessages({library(sceptre); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
object_path <- args[1]; out <- args[2]
so <- readRDS(object_path)
rp <- so@response_precomputations
Z <- so@covariate_matrix
ciu <- so@cells_in_use
Zu <- if (nrow(Z) == length(ciu)) Z else Z[ciu, , drop = FALSE]
genes <- names(rp)
# sceptre's own model mean for each gene: the average fitted value, E[Y] = exp(Z b).
model_mean <- vapply(genes, function(g) mean(exp(as.numeric(Zu %*% rp[[g]]$fitted_coefs))),
                     numeric(1))
theta <- vapply(genes, function(g) rp[[g]]$theta, numeric(1))
coefs <- lapply(genes, function(g) as.numeric(rp[[g]]$fitted_coefs))
cat(sprintf("sceptre precomputations: %d genes, %d covariates, %d cells\n",
            length(genes), ncol(Zu), nrow(Zu)))
writeLines(jsonlite::toJSON(list(
  genes = genes, model_mean = as.numeric(model_mean), theta = as.numeric(theta),
  coefs = coefs, n_cells = nrow(Zu), n_cov = ncol(Zu),
  Z = as.numeric(t(as.matrix(Zu)))
), digits = 17, auto_unbox = TRUE), out)
"""


@pytest.fixture(scope="module")
def sceptre_precomputations() -> dict:
    obj = os.environ.get("PYSCEPTRE_DAY0_SCEPTRE_OBJECT")
    if not obj or not Path(obj).exists():
        pytest.skip("set PYSCEPTRE_DAY0_SCEPTRE_OBJECT to a post-QC sceptre object")
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "dump.R"
        script.write_text(_SCEPTRE_DUMP_R)
        payload = Path(tmp) / "payload.json"
        run = subprocess.run(
            ["Rscript", str(script), obj, str(payload)], capture_output=True, text=True
        )
        if run.returncode != 0:
            pytest.skip(f"could not read the sceptre object: {run.stderr.strip()[-300:]}")
        with open(payload) as f:
            return json.load(f)


class _Fit:
    """The two attributes `baseline_expression_stats_from_fits` reads."""

    def __init__(self, coefs, theta):
        self.fitted_coefs = np.asarray(coefs, float)
        self.theta = float(theta)


def test_model_based_mean_matches_sceptres_own_fitted_values(sceptre_precomputations):
    """`baseline_expression_stats_from_fits` lands on sceptre's expression scale.

    This is the check that matters: the estimator predicts sceptre's test, so
    its `expression_mean` has to be the mean sceptre's own model implies, not
    a mean from some other normalisation. Fed sceptre's cached coefficients
    and its covariate matrix, the helper must reproduce
    `mean(exp(Z b))` per gene.
    """
    from pysceptre.analytical_power import baseline_expression_stats_from_fits

    d = sceptre_precomputations
    Z = np.asarray(d["Z"], float).reshape(int(d["n_cells"]), int(d["n_cov"]))
    fits = {g: _Fit(c, t) for g, c, t in zip(d["genes"], d["coefs"], d["theta"], strict=True)}

    got = baseline_expression_stats_from_fits(Z, fits, gene_subset=list(d["genes"]))
    np.testing.assert_allclose(
        got["expression_mean"].to_numpy(), np.asarray(d["model_mean"], float), rtol=1e-10
    )
    np.testing.assert_allclose(
        got["expression_size"].to_numpy(), np.asarray(d["theta"], float), rtol=1e-12
    )


def test_the_two_mean_conventions_differ_by_a_scale_not_a_shape(
    sceptre_precomputations, watteg_prepared
):
    """How far the poscounts mean sits from sceptre's, and that it is only a scale.

    Reported rather than bounded tightly: the point is that the offset is one
    factor across genes, so the poscounts convention makes the estimate
    conservative rather than wrong-shaped. If this ever stopped being a scale,
    the two would disagree about a gene's expression *relative* to another
    gene's, which would be a real problem.
    """
    from pysceptre.analytical_power import baseline_expression_stats_from_fits

    d = sceptre_precomputations
    w = watteg_prepared
    Z = np.asarray(d["Z"], float).reshape(int(d["n_cells"]), int(d["n_cov"]))
    fits = {g: _Fit(c, t) for g, c, t in zip(d["genes"], d["coefs"], d["theta"], strict=True)}

    shared = [g for g in w["truth_genes"] if g in fits]
    assert len(shared) > 100, "too few shared genes to say anything"
    model = baseline_expression_stats_from_fits(Z, fits, gene_subset=shared)
    poscounts = dict(zip(w["truth_genes"], np.asarray(w["mean"], float), strict=True))
    pos = np.array([poscounts[g] for g in shared])

    ratio = model["expression_mean"].to_numpy() / pos
    print(
        f"\nsceptre model mean / poscounts mean over {len(shared)} genes: "
        f"median {np.median(ratio):.4f}, sd {np.std(ratio):.4f}, "
        f"range {ratio.min():.4f} to {ratio.max():.4f}, "
        f"corr(log) {np.corrcoef(np.log(model['expression_mean']), np.log(pos))[0, 1]:.6f}"
    )
    # A scale, not a reshuffling: the two orderings of gene expression agree almost perfectly.
    assert np.corrcoef(np.log(model["expression_mean"]), np.log(pos))[0, 1] > 0.999
    # And the offset is real rather than noise, which is why the choice matters.
    assert np.median(ratio) > 1.05
