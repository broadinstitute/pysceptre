"""Is what `compute_power` computes worth computing?

`test_analytical_power.py` establishes that the port computes what PerturbPlan
computes, to a relative 1e-9. That is a different question from whether the
closed form agrees with the truth, and the only available truth is simulation:
WattEG's day0 sweep, 100 simulations per pair over 34,886 pairs at six effect
sizes.

Scored as the decision a reader takes -- would this screen have detected a
knockdown of this size? -- rather than as a distance, because a 10-point miss
at true power 0.55 changes nothing while a 2-point miss at 0.79 flips the
pair.

**Not a reproduction of the published comparison.** That scored PerturbPlan's
own R on a different screen analysed under sceptre's permutation test, which
an ondisc-backed response matrix forced. day0 ran under the CRT, which the
published work lists as untested.

    PYSCEPTRE_DAY0_SCEPTRE_OBJECT=... \\
    PYSCEPTRE_WATTEG_PREPARED=... \\
    PYSCEPTRE_DAY0_POWER_SUMMARY=.../power_summary.tsv \\
        pytest -m realdata
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysceptre.analytical_power import compute_power

pytestmark = pytest.mark.realdata

BAR = 0.8
DUMP_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "dump_day0_power_inputs.R"


@pytest.fixture(scope="module")
def scored() -> dict:
    obj = os.environ.get("PYSCEPTRE_DAY0_SCEPTRE_OBJECT")
    prepared = os.environ.get("PYSCEPTRE_WATTEG_PREPARED")
    summary_path = os.environ.get("PYSCEPTRE_DAY0_POWER_SUMMARY")
    if not (obj and prepared and summary_path):
        pytest.skip(
            "set PYSCEPTRE_DAY0_SCEPTRE_OBJECT, PYSCEPTRE_WATTEG_PREPARED and "
            "PYSCEPTRE_DAY0_POWER_SUMMARY (all one run)"
        )
    for path in (obj, prepared, summary_path):
        if not Path(path).exists():
            pytest.skip(f"{path} is missing")
    try:
        ok = subprocess.run(
            ["Rscript", "-e", "library(sceptre); library(jsonlite)"],
            capture_output=True,
            timeout=180,
        ).returncode
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("Rscript not available")
    if ok != 0:
        pytest.skip("R lacks sceptre or jsonlite")

    with tempfile.TemporaryDirectory() as tmp:
        payload = Path(tmp) / "inputs.json"
        run = subprocess.run(
            ["Rscript", str(DUMP_SCRIPT), obj, prepared, str(payload)],
            capture_output=True,
            text=True,
        )
        if run.returncode != 0:
            pytest.skip(f"could not assemble inputs: {run.stderr.strip()[-300:]}")
        with open(payload) as f:
            d = json.load(f)
    d["summary"] = pd.read_csv(summary_path, sep="\t")
    return d


def _confusion(predicted: np.ndarray, observed: np.ndarray) -> dict:
    p, o = predicted >= BAR, observed >= BAR
    tp, fp = int((p & o).sum()), int((p & ~o).sum())
    fn, tn = int((~p & o).sum()), int((~p & ~o).sum())
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "wrong_claims": fp,
        "missed": fn,
        "mcc": (tp * tn - fp * fn) / denom,
        "powered": int(o.sum()),
        "n": len(p),
    }


def _score(d: dict, es: int, convention: str) -> dict:
    design = (
        pd.DataFrame(d["design"])
        .drop_duplicates(subset=["grna_id", "grna_target"])
        .reset_index(drop=True)
    )
    theta = pd.DataFrame({"response_id": d["model_genes"], "expression_size": d["theta"]})
    if convention == "model":
        baseline = theta.assign(expression_mean=d["model_mean"])
    else:
        baseline = pd.DataFrame(
            {"response_id": d["poscounts_genes"], "expression_mean": d["poscounts_mean"]}
        ).merge(theta, on="response_id")

    summary = d["summary"]
    col = f"power_at_effect_size_{es}"
    pairs = summary[summary["response_id"].isin(set(baseline["response_id"]))]
    observed = pairs[col].to_numpy(dtype=float)
    usable = np.isfinite(observed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est = compute_power(
            pairs.loc[usable, ["response_id", "grna_target"]],
            design,
            baseline,
            fold_change_mean=1.0 - es / 100.0,
            fold_change_sd=0.13,
            cutoff=float(d["threshold"]) / 2.0,
            num_total_cells=int(d["n_cells"]),
        )
    return _confusion(est["power"].to_numpy(), observed[usable])


def test_the_estimate_agrees_with_simulation_at_the_decision(scored):
    """The headline, at the effect size real designs operate at.

    Floors rather than exact values: the ground truth is 100 simulations per
    pair, so it carries its own noise, and a tolerance would be asserting
    that noise. What matters is that the agreement does not quietly degrade.
    """
    c = _score(scored, 15, "model")
    print(
        f"\nday0, es=0.15, model mean: {c['n']} pairs, {c['powered']} powered by simulation | "
        f"sens {c['sensitivity']:.1%}, spec {c['specificity']:.2%}, "
        f"{c['wrong_claims']} wrong claims, {c['missed']} missed, MCC {c['mcc']:.3f}"
    )
    assert c["n"] > 30_000, "the sweep should cover the whole design"
    assert c["sensitivity"] > 0.90
    assert c["specificity"] > 0.95
    assert c["mcc"] > 0.90


def test_agreement_holds_across_the_effect_size_range(scored):
    """The published work found its own fitted model degraded as the signal grew.

    This checks the closed form does not: MCC must hold up from a 10% to a 50%
    knockdown. es=0.05 is excluded because only 274 of 34,886 pairs are
    powered there, so every rate on that row is a handful of pairs wide.
    """
    for es in (10, 15, 20, 25, 50):
        c = _score(scored, es, "model")
        print(f"  es={es / 100:.2f}  MCC {c['mcc']:.3f}  sens {c['sensitivity']:.1%}")
        assert c["mcc"] > 0.90, f"es={es}"


def test_the_model_mean_beats_the_normalised_mean_at_the_decision(scored):
    """The recommendation, tested rather than argued.

    `baseline_expression_stats_from_fits` is the documented default on the
    principle that the estimate should sit on the scale of the test it
    predicts. The two conventions differ by about 16% on day0, and this is
    where that shows up as a better or worse decision rather than as a number.
    """
    for es in (10, 15, 20, 25, 50):
        model = _score(scored, es, "model")
        poscounts = _score(scored, es, "poscounts")
        print(
            f"  es={es / 100:.2f}  model MCC {model['mcc']:.3f} vs poscounts {poscounts['mcc']:.3f}"
        )
        assert model["mcc"] > poscounts["mcc"], f"es={es}"


def test_the_normalised_mean_errs_conservative(scored):
    """And in the predicted direction, which is the other half of the claim.

    A 16% lower expression input means a lower power estimate, so it should
    miss powered pairs rather than invent them: more missed, fewer wrong
    claims, than the model mean on the same pairs.
    """
    model = _score(scored, 15, "model")
    poscounts = _score(scored, 15, "poscounts")
    assert poscounts["missed"] > model["missed"]
    assert poscounts["wrong_claims"] < model["wrong_claims"]
