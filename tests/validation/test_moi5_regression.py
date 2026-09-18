"""Regression tests against R sceptre on the real moi5 screen.

Opt-in and slow: these need the moi5 data, which is real screen data and is
never committed (see .gitignore), and running pysceptre over 33k pairs takes
minutes. Deselected by default; run with

    pytest -m moi5

pointing `PYSCEPTRE_MOI5_BENCH` at a directory produced by
`scripts/benchmark_vs_r.R` -- which contains both R's results and the exact
`export/` those results were computed from, so the two implementations cannot
disagree about their inputs.

The thresholds below are deliberately looser than the values measured when
these were written, so they catch regressions rather than tracking noise:

    Spearman rho, p-values           0.9965
    Pearson r, fold change           1.000000  (max |diff| ~1e-10)
    sensitivity vs R's BH call       0.9818  (216 of 220 hits, 7 extra)

scipy's `false_discovery_control` was verified to reproduce R's
`p.adjust(method="BH")` exactly on this dataset (max |difference| 0.0, same
rejection set, both giving 220), and R compares the adjusted value with `<`.
    sensitivity at p < 1e-6          0.9892  (0 extra)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

pytestmark = pytest.mark.moi5

DEFAULT_BENCH = Path("test_data/moi5/bench")


def _bench_dir() -> Path:
    path = Path(os.environ.get("PYSCEPTRE_MOI5_BENCH", DEFAULT_BENCH))
    if not (path / "export" / "metadata.json").exists():
        pytest.skip(
            f"no moi5 benchmark bundle at {path}; set PYSCEPTRE_MOI5_BENCH "
            "(produced by scripts/benchmark_vs_r.R prepare)"
        )
    return path


@pytest.fixture(scope="module")
def bench() -> Path:
    return _bench_dir()


@pytest.fixture(scope="module")
def bench_dir(bench: Path) -> Path:
    return bench


@pytest.fixture(scope="module")
def r_discovery(bench: Path) -> pd.DataFrame:
    path = bench / "r_discovery_result.parquet"
    if not path.exists():
        pytest.skip(f"no R discovery result at {path}")
    r = pd.read_parquet(path)
    return r[r["pass_qc"]] if "pass_qc" in r else r


@pytest.fixture(scope="module")
def py_discovery(bench: Path) -> pd.DataFrame:
    """Run pysceptre on the same inputs R was given."""
    import sys

    sys.path.insert(0, "scripts")
    from sceptre_io import load_export

    from pysceptre import run_discovery_analysis

    export = load_export(bench / "export")
    return run_discovery_analysis(
        response_matrix=export.response_matrix,
        gene_ids=export.gene_ids,
        covariate_matrix=export.covariate_matrix,
        grna_target_cells=export.grna_target_cells,
        pairs=export.pairs,
        side=export.side,
        seed=0,
    )


@pytest.fixture(scope="module")
def merged(py_discovery: pd.DataFrame, r_discovery: pd.DataFrame) -> pd.DataFrame:
    m = py_discovery.merge(r_discovery, on=["response_id", "grna_target"], suffixes=("_py", "_r"))
    assert len(m) == len(py_discovery), (
        f"only {len(m)} of {len(py_discovery)} pysceptre pairs matched R's"
    )
    return m


def test_fold_change_agrees_to_numerical_precision(merged: pd.DataFrame):
    """The benchmark setup invariant. Fold change involves no resampling, so with
    identical inputs it must agree to ~1e-11. If this fails, the two sides are
    not analysing the same data -- which is exactly the bug this suite exists
    to catch, and which a naive comparison missed by re-deriving the gRNA
    assignments (only 39 of 2,974 targets' cell sets matched).
    """
    diff = np.abs(merged["fold_change_py"] - merged["fold_change_r"])
    assert diff.max() < 1e-8, f"max |fold change difference| = {diff.max():.3e}"


def test_p_value_rank_correlation(merged: pd.DataFrame):
    """CRT p-values are stochastic and the RNG is deliberately not shared with
    R, so agreement is distributional. Rank correlation is the right measure."""
    rho = stats.spearmanr(merged["p_value_py"], merged["p_value_r"]).statistic
    assert rho > 0.99, f"Spearman rho = {rho:.4f}"


def _contingency(call_py: np.ndarray, call_r: np.ndarray) -> dict:
    """2x2 agreement on significance, R as reference.

    A single index such as Jaccard collapses this table and cannot tell
    "calls extra hits" from "misses hits" -- different failures with different
    consequences, so both are reported.

    Discordance, not error: R is a reference implementation rather than ground
    truth, and both sides estimate the same stochastic quantity from different
    draws.
    """
    tp = int(np.sum(call_py & call_r))
    fp = int(np.sum(call_py & ~call_r))
    fn = int(np.sum(~call_py & call_r))
    tn = int(np.sum(~call_py & ~call_r))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
    }


def test_significance_calls_agree_on_rs_own_bh_threshold(merged: pd.DataFrame, bench_dir: Path):
    """The scientifically meaningful comparison: R's own `significant` column
    (BH at its multiple_testing_alpha) against BH applied to pysceptre's
    p-values at the same level. pysceptre applies no correction itself.

    Measured when written:
                        R: sig   R: not sig
        py: sig            216            7
        py: not sig          4       32,908
        sensitivity 0.9818, specificity 0.9998
    """
    if "significant" not in merged:
        pytest.skip("R result has no `significant` column")
    # Read alpha from the object's own configuration rather than assuming: a
    # dataset set up with a different multiple_testing_alpha would otherwise
    # compare a corrected call against one corrected at the wrong level.
    alpha = float(
        json.loads((bench_dir / "export" / "metadata.json").read_text())["multiple_testing_alpha"]
    )
    call_r = merged["significant"].fillna(False).to_numpy(dtype=bool)
    call_py = (
        stats.false_discovery_control(
            np.clip(merged["p_value_py"].to_numpy(), 0.0, 1.0), method="bh"
        )
        < alpha
    )
    t = _contingency(call_py, call_r)
    assert t["sensitivity"] > 0.95, f"recovered only {t['sensitivity']:.4f} of R's hits: {t}"
    assert t["specificity"] > 0.999, f"specificity {t['specificity']:.4f}: {t}"


@pytest.mark.parametrize(
    ("threshold", "min_sensitivity", "max_extra"),
    [(1e-4, 0.93, 25), (1e-6, 0.95, 5)],
)
def test_significance_calls_agree_at_fixed_thresholds(
    merged: pd.DataFrame, threshold: float, min_sensitivity: float, max_extra: int
):
    """Agreement should tighten as pairs get more significant. Measured:
    sensitivity 0.9677 with 8 extra at 1e-4; 0.9892 with 0 extra at 1e-6."""
    t = _contingency(
        merged["p_value_py"].to_numpy() < threshold,
        merged["p_value_r"].to_numpy() < threshold,
    )
    assert t["sensitivity"] > min_sensitivity, f"at p<{threshold:g}: {t}"
    assert t["fp"] <= max_extra, f"at p<{threshold:g}, {t['fp']} pysceptre-only calls: {t}"


def test_no_pair_is_significant_on_one_side_only_by_orders_of_magnitude(
    merged: pd.DataFrame,
):
    """Guards the failure that would matter scientifically: a pair pysceptre
    calls a strong hit that R considers null, or the reverse. Disagreement on
    *how* small an already-tiny p-value is, is expected and allowed."""
    py, r = merged["p_value_py"].to_numpy(), merged["p_value_r"].to_numpy()
    contradictory = ((py < 1e-6) & (r > 1e-2)) | ((r < 1e-6) & (py > 1e-2))
    n = int(contradictory.sum())
    assert n == 0, (
        f"{n} pairs disagree qualitatively, e.g.\n"
        f"{merged.loc[contradictory, ['response_id', 'grna_target', 'p_value_py', 'p_value_r']].head()}"
    )


def test_r_calibration_result_is_calibrated(bench: Path):
    """Sanity check on the reference itself, and the target pysceptre's own
    calibration check will have to meet: negative control pairs should produce
    p-values with no excess of small values."""
    path = bench / "r_calibration_result.parquet"
    if not path.exists():
        pytest.skip(f"no R calibration result at {path}")
    cal = pd.read_parquet(path)
    p = cal["p_value"].dropna().to_numpy()
    assert len(p) > 1000
    # Under the null, about 1% of pairs fall below 0.01. Allow generous slack:
    # this is a smoke test for gross miscalibration, not a formal GOF test.
    frac = float(np.mean(p < 0.01))
    assert frac < 0.05, f"{frac:.3%} of negative control pairs below p=0.01"
    if "significant" in cal:
        n_sig = int(cal["significant"].fillna(False).sum())
        assert n_sig == 0, f"{n_sig} negative control pairs called significant"
