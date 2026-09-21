#!/usr/bin/env python
"""Score `compute_power` against per-pair simulated power, at the bar a reader acts on.

The estimator is validated against PerturbPlan's R to a relative 1e-9, which establishes that the
port computes what it ports. This asks the separate question: is what it computes worth
computing? Ground truth is WattEG's simulated power, 100 simulations per pair, and the metric is
the decision -- would this screen have detected a knockdown of this size? -- rather than a
distance, because a 10-point miss at true power 0.55 changes nothing while a 2-point miss at 0.79
flips the pair.

**This is not a reproduction of the published comparison.** That one scored PerturbPlan's own R on
a different screen analysed under sceptre's permutation test, which an ondisc-backed response
matrix forced. day0 was analysed under the CRT, which the published work lists as untested, so
this is new evidence rather than a re-run. The two are also not directly comparable: the published
sweep fed its estimator a size-factor-normalised mean, and the simulation it was scored against
restored the size factors while the estimator did not.

Both expression conventions are scored here, because they differ by about 16% on day0 and the
question is whether that moves a decision taken at a 0.8 bar.

Usage:
  Rscript scripts/dump_day0_power_inputs.R <object.rds> <prepared/> inputs.json
  python scripts/score_power_against_simulation.py inputs.json <power_summary.tsv>
"""

from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd

from pysceptre.analytical_power import compute_power, target_cell_counts

BAR = 0.8


def confusion(predicted: np.ndarray, observed: np.ndarray) -> dict:
    """The two error directions kept apart, since only one licenses a false negative."""
    p, o = predicted >= BAR, observed >= BAR
    tp, fp = int((p & o).sum()), int((p & ~o).sum())
    fn, tn = int((~p & o).sum()), int((~p & ~o).sum())
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        "n": len(p),
        "powered": int(o.sum()),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "wrong_claims": fp,
        "missed": fn,
        "mcc": (tp * tn - fp * fn) / denom if denom else float("nan"),
    }


def main(inputs_path: str, summary_path: str) -> None:
    with open(inputs_path) as f:
        d = json.load(f)
    summary = pd.read_csv(summary_path, sep="\t")
    design = pd.DataFrame(d["design"])

    # day0's design lists 36 (grna_id, grna_target) pairs twice. `compute_power` refuses them,
    # rightly: a guide counted twice enters num_trt_cells and its squared sum twice, and unlike
    # the discovery path there is no R-parity argument for keeping it. Collapsed here explicitly
    # rather than worked around, which is the one-line fix the error message names.
    n_dup = int(design.duplicated(subset=["grna_id", "grna_target"]).sum())
    if n_dup:
        design = design.drop_duplicates(subset=["grna_id", "grna_target"]).reset_index(drop=True)
        print(f"collapsed {n_dup} duplicated design row(s) before counting cells")

    # The estimator wants the one-sided threshold: the sweep counted a simulation as a success
    # when the two-sided p cleared the nominal threshold and the fold change was negative.
    cutoff = float(d["threshold"]) / 2.0
    n_cells = int(d["n_cells"])
    print(f"cutoff {cutoff:.6g} (nominal {d['threshold']:.6g} halved), {n_cells} cells")

    baselines = {
        "model mean, exp(Z b)": pd.DataFrame(
            {
                "response_id": d["model_genes"],
                "expression_mean": d["model_mean"],
                "expression_size": d["theta"],
            }
        ),
        "poscounts mean": pd.DataFrame(
            {
                "response_id": d["poscounts_genes"],
                "expression_mean": d["poscounts_mean"],
            }
        ).merge(
            pd.DataFrame({"response_id": d["model_genes"], "expression_size": d["theta"]}),
            on="response_id",
        ),
    }

    effect_sizes = sorted(
        int(c.rsplit("_", 1)[1])
        for c in summary.columns
        if c.startswith("power_at_effect_size_") and c.rsplit("_", 1)[1].isdigit()
    )

    # Only pairs the estimator can speak to: a gene needs baseline statistics and a target needs
    # guides. Reported rather than silently dropped.
    have_target = set(target_cell_counts(design)["grna_target"])
    for label, baseline in baselines.items():
        have_gene = set(baseline["response_id"])
        pairs = summary[
            summary["grna_target"].isin(have_target) & summary["response_id"].isin(have_gene)
        ].reset_index(drop=True)
        print(
            f"\n=== {label} ===\n"
            f"{len(pairs)} of {len(summary)} sweep pairs scorable "
            f"({len(summary) - len(pairs)} lack a gene baseline or a target)"
        )
        print(
            f"{'es':>4}  {'pairs':>6} {'powered':>8} {'sens':>7} {'spec':>8} "
            f"{'wrong':>7} {'missed':>7} {'MCC':>7}"
        )
        for es in effect_sizes:
            col = f"power_at_effect_size_{es}"
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
                    cutoff=cutoff,
                    num_total_cells=n_cells,
                )
            c = confusion(est["power"].to_numpy(), observed[usable])
            print(
                f"{es / 100:>4.2f}  {c['n']:>6d} {c['powered']:>8d} "
                f"{c['sensitivity']:>6.1%} {c['specificity']:>7.2%} "
                f"{c['wrong_claims']:>7d} {c['missed']:>7d} {c['mcc']:>7.3f}"
            )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
