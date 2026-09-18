"""Regression tests against R sceptre on the day0_grna20 screen.

Opt-in and slow: these need real screen data, which is never committed (see
`.gitignore`), and running 34,886 pairs over 567,690 cells takes minutes.
Deselected by default; run with

    PYSCEPTRE_DAY0_EXPORT=path/to/export pytest -m realdata

The export is a `.h5mu` written by `scripts/export_sceptre_dataset.R` plus
`scripts/make_h5mu.py`. It carries **R's own results alongside the inputs
they were computed from** -- `discovery_result`, `calibration_result` and
`negative_control_pairs` -- so the two implementations cannot disagree about
what they analysed. That mattered: an earlier comparison that re-derived the
gRNA assignments independently reproduced only 39 of 2,974 targets.

Thresholds are deliberately looser than the values measured when these were
written, so they catch regressions rather than tracking Monte Carlo noise.
Measured on the implementation as of this commit:

    Spearman rho, p-values            0.9865
    Pearson r, fold change            1.000000  (max |diff| 2.6e-12)
    sensitivity vs R's BH call        0.9882    (251 of 254 hits, 10 extra)
    specificity vs R's BH call        0.9997
    calibration KS vs U(0,1)          0.0260    (R: 0.0259)

The calibration p-values are **not** perfectly uniform, and the test asserts
that pysceptre tracks R rather than that either is ideal: R deviates by the
same amount, so the deviation belongs to the method on this data and not to
the port.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

pytestmark = pytest.mark.realdata

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))


@pytest.fixture(scope="module")
def export():
    path = os.environ.get("PYSCEPTRE_DAY0_EXPORT")
    if not path:
        pytest.skip("set PYSCEPTRE_DAY0_EXPORT to the day0 export directory")
    pytest.importorskip("mudata", reason="the io extra is not installed")
    from sceptre_io import load_export

    loaded = load_export(path, backed=True)
    yield loaded
    if hasattr(loaded.response_matrix, "close"):
        loaded.response_matrix.close()


@pytest.fixture(scope="module")
def discovery(export):
    """pysceptre's discovery result, merged against R's stored one."""
    import pysceptre

    if export.discovery_result is None or not len(export.discovery_result):
        pytest.skip("export carries no R discovery_result to compare against")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = pysceptre.run_discovery_analysis(
            response_matrix=export.response_matrix,
            gene_ids=export.gene_ids,
            covariate_matrix=export.covariate_matrix,
            grna_target_cells=export.grna_target_cells,
            pairs=export.pairs,
            side=export.side,
            seed=0,
            n_jobs=-1,
        )
    merged = got.merge(
        export.discovery_result, on=["response_id", "grna_target"], suffixes=("_py", "_r")
    )
    ok = np.isfinite(merged.p_value_py) & np.isfinite(merged.p_value_r)
    return merged[ok].reset_index(drop=True)


def test_every_pair_is_comparable(discovery, export):
    assert len(discovery) == len(export.pairs), "some pairs failed to match on (gene, target)"


def test_fold_change_agrees_to_numerical_precision(discovery):
    """The deterministic quantity. No resampling enters it, so agreement here
    is the check that both sides fit the same models to the same cells."""
    d = np.abs(discovery.fold_change_py - discovery.fold_change_r)
    assert d.max() < 1e-9, f"max fold-change difference {d.max():.3e}"
    r = stats.pearsonr(discovery.fold_change_py, discovery.fold_change_r).statistic
    assert r > 0.999999


def test_p_value_rank_correlation(discovery):
    """Stochastic on both sides, so rank correlation rather than equality.

    The ceiling is the Monte Carlo floor: pysceptre agrees with R about as
    well as it agrees with itself across seeds.
    """
    rho = stats.spearmanr(discovery.p_value_py, discovery.p_value_r).statistic
    assert rho > 0.98, f"Spearman rho {rho:.4f}"


def test_significance_calls_agree_on_rs_own_bh_threshold(discovery):
    """scipy's `false_discovery_control` reproduces R's `p.adjust(method="BH")`
    exactly, and R compares the adjusted value with `<`, so the two rejection
    rules are the same rule."""
    py = stats.false_discovery_control(np.clip(discovery.p_value_py, 0, 1), method="bh") < 0.1
    r = discovery.significant.fillna(False).to_numpy(dtype=bool)
    tp = int((py & r).sum())
    fn = int((~py & r).sum())
    tn = int((~py & ~r).sum())
    fp = int((py & ~r).sum())
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    assert sensitivity > 0.97, f"recovered {tp} of {tp + fn} of R's hits"
    assert specificity > 0.999, f"{fp} calls R did not make"


def test_discordant_pairs_are_threshold_cases_not_detection_failures(discovery):
    """Where the two disagree, both should still agree on the effect size.

    This is the claim the supplementary table makes, asserted rather than
    just reported: a disagreement is about which side of a cutoff a resampled
    p-value fell on, never about whether an effect exists.
    """
    py = stats.false_discovery_control(np.clip(discovery.p_value_py, 0, 1), method="bh") < 0.1
    r = discovery.significant.fillna(False).to_numpy(dtype=bool)
    disagree = discovery[py != r]
    if not len(disagree):
        pytest.skip("no discordant pairs on this run")
    # R stores log-2 fold change; compare on the same scale pysceptre reports.
    pct_r = (2.0**disagree.log_2_fold_change - 1.0) * 100
    assert np.abs(disagree["pct_change_es"] - pct_r).max() < 1e-6
    # And every one should be an effect some independent criterion supports.
    excludes_zero = (disagree["pct_change_es_ci_low"] > 0) | (disagree["pct_change_es_ci_high"] < 0)
    assert excludes_zero.all(), "a discordant pair whose interval covers no effect"


def test_calibration_tracks_r_rather_than_being_ideal(export):
    """Negative-control p-values, with R's own pairs injected.

    R's pair selection is unseeded and differs run to run, so only injected
    pairs make a pair-by-pair comparison meaningful. The assertion is that
    pysceptre's deviation from uniform matches R's -- not that either is
    uniform, because neither is.
    """
    import pysceptre
    from pysceptre.pipeline.calibration import GROUP_NAME_SEPARATOR  # noqa: F401

    r = export.calibration_result
    if r is None or "p_value" not in r:
        pytest.skip("export carries no R calibration_result")
    pairs = r[["response_id", "grna_target"]].reset_index(drop=True)
    meta = export.metadata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = pysceptre.run_calibration_check(
            response_matrix=export.response_matrix,
            gene_ids=export.gene_ids,
            covariate_matrix=export.covariate_matrix,
            ntc_grna_cells=export.ntc_grna_cells,
            negative_control_pairs=pairs,
            n_calibration_pairs=len(pairs),
            calibration_group_size=int(meta.get("calibration_group_size", 15)),
            n_nonzero_trt_thresh=int(meta.get("n_nonzero_trt_thresh", 7)),
            n_nonzero_cntrl_thresh=int(meta.get("n_nonzero_cntrl_thresh", 7)),
            side=export.side,
            seed=0,
            n_jobs=-1,
        )
    ks_py = stats.kstest(np.clip(got.p_value, 0, 1), "uniform").statistic
    ks_r = stats.kstest(np.clip(pd.Series(r.p_value).dropna(), 0, 1), "uniform").statistic
    assert abs(ks_py - ks_r) < 0.01, f"pysceptre KS {ks_py:.4f} against R's {ks_r:.4f}"
    merged = got.merge(r, on=["response_id", "grna_target"], suffixes=("_py", "_r"))
    rho = stats.spearmanr(merged.p_value_py, merged.p_value_r).statistic
    assert rho > 0.98, f"Spearman rho {rho:.4f}"
