"""Regression tests against R sceptre on sceptredata's two real example screens.

sceptredata 0.99.0 ships two real screens, Papalexi 2021 (CRISPRko, low MOI)
and Gasperini 2019 (CRISPRi, high MOI). Real data is never committed, so these
are opt-in and deselected by default:

    PYSCEPTRE_SCEPTREDATA_DIR=<out_dir> pytest -m realdata \\
        tests/validation/test_sceptredata_realdata.py

`<out_dir>` is what `scripts/run_sceptredata_examples.R` wrote, one directory
per run (`<dataset>/<control_group>_<mechanism>/`, holding R's three result
CSVs and `run_info.json`), each with `export/dataset.h5mu` added by
`scripts/export_sceptre_dataset.R` and `scripts/make_h5mu.py`. A run without
an export is skipped. The machinery is `scripts/compare_sceptredata.py`'s,
which reports the same comparisons in full. R's own pairs are injected
throughout, so both sides always test the same thing.

Asserted, for each of the six runs:

- deterministic quantities: every pair's nonzero counts and `pass_qc`
  (discovery, calibration, power check) exactly; fold change to 1e-9 on the
  log2 scale; `z_orig` to 1e-6 relative, except on a pair whose dispersion fit
  is degenerate (see `test_fold_change_and_statistic_are_rs`)
- permutation runs, on R's own draws: stages identical, stage-1 and stage-3
  p-values bit-identical, stage-2 p-values within 1e-6 of R's or, where R's
  own skew-normal tail evaluation is the inaccurate one, within 1e-6 of an
  independent evaluation at R's fit; BH calls identical
- pysceptre's own draws (seed 0): rank agreement, BH calls, calibration
  uniformity and power, against floors well below what was measured

Every pair of every analysis is tested; nothing is subset. All of it runs at
`n_jobs=8`, which changes no result (the comparison script checks 1 against 8
bit for bit).
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

pytestmark = pytest.mark.realdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

RUNS = (
    "lowmoi/complement_crt",
    "lowmoi/complement_permutations",
    "lowmoi/nt_cells_crt",
    "lowmoi/nt_cells_permutations",
    "highmoi/complement_crt",
    "highmoi/complement_permutations",
)
PERMUTATION_RUNS = tuple(r for r in RUNS if r.endswith("_permutations"))
ANALYSES = ("calibration", "power", "discovery")
N_JOBS = 8

# Deterministic quantities.
LOG2_FC_ATOL = 1e-9
Z_RTOL = 1e-6
# sceptre's clamp on the dispersion estimate.
THETA_BOUNDS = (0.01, 1000.0)
# pysceptre's own draws.
SPEARMAN_FLOOR = 0.95
R_ONLY_CALLS = (3, 0.06)  # at most max(3, 6% of R's discoveries)
PYSCEPTRE_ONLY_CALLS = (3, 0.01)  # at most max(3, 1% of R's non-discoveries)
FD_SLACK = (5, 0.1)  # calibration false discoveries within max(5, 10% of R's)
POWER_P = 1e-3

_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


@pytest.fixture(scope="module")
def cmp():
    """`scripts/compare_sceptredata.py`, once the data and the io extra are present."""
    if not os.environ.get("PYSCEPTRE_SCEPTREDATA_DIR"):
        pytest.skip("set PYSCEPTRE_SCEPTREDATA_DIR to the run_sceptredata_examples.R out_dir")
    pytest.importorskip("mudata", reason="the io extra is not installed")
    import compare_sceptredata

    return compare_sceptredata


def _run(cmp, run: str) -> dict:
    """One run's inputs: the export, R's tables and settings, and the NT pool."""
    run_dir = Path(os.environ["PYSCEPTRE_SCEPTREDATA_DIR"]) / run
    if not (run_dir / "export" / "dataset.h5mu").exists():
        pytest.skip(f"{run_dir} has no export/dataset.h5mu")

    def build():
        info = json.loads((run_dir / "run_info.json").read_text())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            e = cmp.load_export(run_dir / "export")
        settings = cmp.check_settings(e, info)
        pool = None
        if settings["control_group"] == "nt_cells":
            pool = cmp.nt_cell_pool(e.ntc_grna_cells, settings["n_cells"])
        r = {a: cmp.read_r(run_dir, a) for a in ANALYSES}
        return {"e": e, "info": info, "settings": settings, "pool": pool, "r": r}

    return _cached(("run", run), build)


def _own(cmp, run: str, analysis: str):
    """pysceptre's own draws at seed 0: the result, and it merged with R's tested rows."""
    d = _run(cmp, run)

    def build():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = cmp.run_own(analysis, d["e"], d["r"][analysis], d["settings"], N_JOBS)
        return frame, cmp.merge(frame, cmp.r_tested(d["r"][analysis]))

    return _cached(("own", run, analysis), build)


def _on_rs_draws(cmp, run: str, analysis: str):
    """The analysis on R's own permutation draws, merged with R's tested rows."""
    d = _run(cmp, run)

    def build():
        specs = cmp.sampler_specs(d["e"], d["settings"], d["pool"])
        assert not cmp.check_specs_against_r(specs, d["info"], d["pool"])
        draws = cmp.make_draws(specs["calibration" if analysis == "calibration" else "discovery"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frame = cmp.run_r_exact(
                analysis, d["e"], d["r"][analysis], d["settings"], d["pool"], draws, N_JOBS
            )
        return cmp.merge(frame, cmp.r_tested(d["r"][analysis]))

    return _cached(("exact", run, analysis), build)


def _dispersion_is_degenerate(d: dict, analysis: str, gene: str, target: str) -> bool:
    """Whether pysceptre's dispersion MLE stopped past the upper clamp for this pair.

    That is the rounding accident docs/design.md describes ("A dispersion
    estimate can stop on a rounding accident"): an MLE at infinity that
    pysceptre's iteration stops on, where R's runs out and takes the method of
    moments. Nothing else is exempted.

    Refits the gene the way the engine does: on every cell for the complement,
    on the NT pool for the NT-cells calibration check, and on the target's cells
    with the NT cells for an NT-cells discovery or power pair.
    """
    from pysceptre.glm.irls import fit_poisson_glm_batch, x_outer_flat
    from pysceptre.glm.nb_theta import estimate_theta
    from pysceptre.pipeline.discovery import combined_cells

    e, pool = d["e"], d["pool"]
    n_cells = e.covariate_matrix.shape[0]
    row = np.asarray(e.response_matrix[e.gene_ids.index(gene)].toarray(), dtype=float).ravel()
    if pool is None:
        cells = np.arange(n_cells)
    elif analysis == "calibration":
        cells = pool
    else:
        cells = combined_cells(e.grna_target_cells[target], pool)
    X, y = e.covariate_matrix[cells], row[cells]
    fit = fit_poisson_glm_batch(X, y, X_outer_flat=x_outer_flat(X))
    mu = np.asarray(fit.fitted_values, dtype=float).ravel()
    theta, method = estimate_theta(y=y, mu=mu, dfr=X.shape[0] - X.shape[1])
    _, hi = THETA_BOUNDS
    return method == 1 and theta > hi


# --- deterministic -----------------------------------------------------------------------


@pytest.mark.parametrize("run", RUNS)
def test_the_inputs_are_rs(cmp, run):
    """The NT pool is R's `all_nt_idxs` in R's order, and R's pair lists are the export's."""
    d = _run(cmp, run)
    if d["pool"] is not None:
        np.testing.assert_array_equal(d["pool"], d["e"].all_nt_idxs)
    tested = d["r"]["discovery"].loc[d["r"]["discovery"]["pass_qc"], cmp.KEY]
    assert set(map(tuple, tested.to_numpy())) == set(map(tuple, d["e"].pairs.to_numpy()))
    neg = d["e"].negative_control_pairs[cmp.KEY]
    r_neg = d["r"]["calibration"][cmp.KEY]
    assert set(map(tuple, r_neg.to_numpy())) == set(map(tuple, neg.to_numpy()))


@pytest.mark.parametrize("run", RUNS)
def test_pairwise_qc_is_rs_exactly(cmp, run):
    """Nonzero counts and `pass_qc` for every discovery pair, tested or not, and for
    every calibration pair, counted within the calibration check's cells."""
    d = _run(cmp, run)
    for name, check in (
        ("discovery", cmp.discovery_pairwise_qc),
        ("calibration", cmp.calibration_pairwise_qc),
    ):
        q = check(d["e"], d["r"][name], d["pool"], d["settings"])
        n = q["pairs"]
        equal = (q["n_nonzero_trt_equal"], q["n_nonzero_cntrl_equal"], q["pass_qc_equal"])
        assert equal == (n, n, n), f"{name}: {q}"


@pytest.mark.parametrize("run", RUNS)
def test_power_check_reports_rs_counts(cmp, run):
    """Every positive control R reports, with its counts and `pass_qc`, none dropped."""
    d = _run(cmp, run)
    frame, _ = _own(cmp, run, "power")
    r = d["r"]["power"]
    full = cmp.merge(frame, r)
    assert len(full) == len(r) == len(frame)
    for col in ("n_nonzero_trt", "n_nonzero_cntrl", "pass_qc"):
        np.testing.assert_array_equal(full[f"{col}_py"], full[f"{col}_r"], err_msg=col)


@pytest.mark.parametrize("analysis", ANALYSES)
@pytest.mark.parametrize("run", RUNS)
def test_fold_change_and_statistic_are_rs(cmp, run, analysis):
    """Nothing resampled enters either, so they are compared as numbers.

    A pair may miss on `z_orig` only where its dispersion fit is degenerate.
    For a gene this close to Poisson the likelihood for theta is flat out to
    infinity, so its Newton iteration wanders, and whether it stops (pysceptre,
    theta then clamped to 1000) or runs out of iterations (R, which then uses the
    method of moments) is decided by rounding. The fold change does not involve
    theta and still agrees.
    """
    d = _run(cmp, run)
    _, m = _own(cmp, run, analysis)
    assert len(m) == len(cmp.r_tested(d["r"][analysis]))
    log2_py = np.log2(m["fold_change_py"].to_numpy(float))
    np.testing.assert_allclose(log2_py, m["log_2_fold_change"], rtol=0, atol=LOG2_FC_ATOL)
    z_py, z_r = m["z_orig_py"].to_numpy(float), m["z_orig_r"].to_numpy(float)
    off = np.flatnonzero(np.abs(z_py - z_r) > Z_RTOL * np.abs(z_r))
    assert off.size <= 0.01 * len(m), f"{off.size} of {len(m)} pairs off on z_orig"
    for i in off:
        gene, target = m["response_id"].iloc[i], m["grna_target"].iloc[i]
        assert _dispersion_is_degenerate(d, analysis, gene, target), (
            f"{gene} / {target}: z_orig {z_py[i]!r} against R's {z_r[i]!r}, with a sound fit"
        )


# --- R's own draws -----------------------------------------------------------------------


@pytest.mark.parametrize("analysis", ANALYSES)
@pytest.mark.parametrize("run", PERMUTATION_RUNS)
def test_p_values_on_rs_draws_are_rs(cmp, run, analysis):
    """With R's draws nothing random is left. Stages 1 and 3 count over the same
    statistics and must be identical; stage 2 evaluates the same skew-normal fit,
    so any difference is tail arithmetic, and the reference says whose."""
    d = _run(cmp, run)
    m = _on_rs_draws(cmp, run, analysis)
    assert len(m) == len(cmp.r_tested(d["r"][analysis]))
    np.testing.assert_array_equal(m["stage_py"], m["stage_r"])
    p_py, p_r = m["p_value_py"].to_numpy(float), m["p_value_r"].to_numpy(float)
    not_2 = m["stage_r"].to_numpy() != 2
    np.testing.assert_array_equal(p_py[not_2], p_r[not_2])

    c = cmp.stage_2_against_reference(m, cmp.SIDE_CODES[d["settings"]["side"]])
    assert c["unexplained"] == 0, c
    if c["stage_2"]:
        assert c["max_rel_err_vs_reference_py"] <= cmp.SN_RTOL, c
    if c.get("r_p_range_where_r_off_reference"):
        # R's tail evaluation gives out only far below anything a threshold would see.
        assert c["r_p_range_where_r_off_reference"][1] < 1e-12, c

    if "significant" in m:
        alpha = d["settings"]["multiple_testing_alpha"]
        np.testing.assert_array_equal(cmp.bh_calls(p_py, alpha), m["significant"].to_numpy(bool))


# --- pysceptre's own draws ---------------------------------------------------------------


@pytest.mark.parametrize("analysis", ("calibration", "discovery"))
@pytest.mark.parametrize("run", RUNS)
def test_own_draws_rank_like_rs(cmp, run, analysis):
    """Different draws on each side, so agreement is up to Monte Carlo error."""
    _, m = _own(cmp, run, analysis)
    rho = stats.spearmanr(m["p_value_py"], m["p_value_r"]).statistic
    assert rho > SPEARMAN_FLOOR, f"Spearman rho {rho:.4f}"


@pytest.mark.parametrize("run", RUNS)
def test_own_draws_make_rs_discovery_calls(cmp, run):
    """BH at R's alpha on pysceptre's p-values, against R's own `significant`,
    which BH on R's p-values reproduces exactly."""
    d = _run(cmp, run)
    _, m = _own(cmp, run, "discovery")
    alpha = d["settings"]["multiple_testing_alpha"]
    r_sig = m["significant"].to_numpy(bool)
    np.testing.assert_array_equal(cmp.bh_calls(m["p_value_r"], alpha), r_sig)
    t = cmp.two_by_two(cmp.bh_calls(m["p_value_py"], alpha), r_sig)
    n_r, n_not = int(r_sig.sum()), int((~r_sig).sum())
    assert t["r_only"] <= max(R_ONLY_CALLS[0], R_ONLY_CALLS[1] * n_r), t
    assert t["pysceptre_only"] <= max(PYSCEPTRE_ONLY_CALLS[0], PYSCEPTRE_ONLY_CALLS[1] * n_not), t


@pytest.mark.parametrize("run", RUNS)
def test_calibration_tracks_r(cmp, run):
    """Uniformity as R measures it on the same pairs, rather than ideal uniformity:
    the low-MOI complement is miscalibrated in R as well, and the port should say so.
    Two calibrations on different draws differ in KS by order 1/sqrt(n), hence the
    tolerance."""
    d = _run(cmp, run)
    _, m = _own(cmp, run, "calibration")
    alpha = d["settings"]["multiple_testing_alpha"]
    ks_py = stats.kstest(np.clip(m["p_value_py"], 0, 1), "uniform").statistic
    ks_r = stats.kstest(np.clip(m["p_value_r"], 0, 1), "uniform").statistic
    tolerance = max(0.01, 0.6 / np.sqrt(len(m)))
    assert abs(ks_py - ks_r) < tolerance, f"KS pysceptre {ks_py:.4f}, R {ks_r:.4f}"
    fd_py = int(cmp.bh_calls(m["p_value_py"], alpha).sum())
    fd_r = int(m["significant"].sum())
    assert abs(fd_py - fd_r) <= max(FD_SLACK[0], FD_SLACK[1] * fd_r), (fd_py, fd_r)


@pytest.mark.parametrize("run", RUNS)
def test_power_check_recovers_rs_effects(cmp, run):
    """The same effects, and every positive control R detects is detected."""
    _, m = _own(cmp, run, "power")
    assert len(m)
    np.testing.assert_allclose(
        np.median(np.log2(m["fold_change_py"])), np.median(m["log_2_fold_change"]), atol=1e-9
    )
    detected_by_r = m["p_value_r"].to_numpy(float) < POWER_P
    assert detected_by_r.any()
    assert np.all(m["p_value_py"].to_numpy(float)[detected_by_r] < POWER_P)
