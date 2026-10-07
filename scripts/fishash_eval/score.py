"""Score every assignment run against the simulated truth, and check the replication criteria.

    python scripts/fishash_eval/score.py <sims_dir> <runs_dir> <tables_dir>

Writes to <tables_dir>:
    metrics.parquet     one row per (scenario, dataset, run): the confusion over every guide x
                        cell entry ("full") and over the entries with a count ("nonzero"), as
                        the fishash paper's bin/process_assignments.R computes it, plus the
                        run's timings
    summary.parquet     per scenario, parameter and run: median, first and third quartile of
                        F1, precision and recall over replicates (full subset), as the paper's
                        81_plot_simulation_results.R summarises them
    replication.json    the criteria R1 to R6 of the evaluation plan, each with its expected and
                        measured value
A run whose run.json records a guide block (crispat on large libraries) is scored on that block
of guides only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_lib import confusion, load_assigned, load_dataset, mean_infections  # noqa: E402

LABEL_PATTERNS = (
    ("moi", re.compile(r"^moi_(?P<moi>[0-9.]+)_iter_(?P<iter>\d+)$")),
    ("nguides", re.compile(r"^nguides_(?P<nguides>\d+)_iter_(?P<iter>\d+)$")),
    ("corr", re.compile(r"^corr_(?P<corr>[a-z]+)_unif_.*_iter_(?P<iter>\d+)$")),
)
VARY_MOI = "numCells20k_numGuides200_varyMOI"
HIGH = "numCells20k_medUmi100_snr4_endo75_varyNumGuides"
LOW = "numCells20k_medUmi20_snr1_endo25_varyNumGuides"
CORR = "numCells20k_varySignalNoiseCorr"
# The collaborator's panel A, read off the figure (median F1, full subset), MOI 0.1 to 10.
COLLABORATOR_MOIS = (0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
COLLABORATOR_FISHASH_PLUS = (0.965, 0.965, 0.964, 0.961, 0.948, 0.927, 0.883, 0.769)
COLLABORATOR_MIXTURE = (0.957, 0.926, 0.953, 0.948, 0.917, 0.888, 0.827, 0.690)


def parse_label(label: str) -> dict:
    for _, pat in LABEL_PATTERNS:
        m = pat.match(label)
        if m:
            out = m.groupdict()
            if "moi" in out:
                out["moi"] = float(out["moi"])
                out["mean_infections"] = float(mean_infections(out["moi"]))
            if "nguides" in out:
                out["nguides"] = int(out["nguides"])
            out["iter"] = int(out["iter"])
            return out
    return {}


def score_runs(sims_dir: Path, runs_dir: Path) -> pd.DataFrame:
    rows = []
    for scen_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        for label_dir in sorted(p for p in scen_dir.iterdir() if p.is_dir()):
            runs = [
                d
                for d in sorted(label_dir.iterdir())
                if (d / "run.json").exists()
                and json.loads((d / "run.json").read_text()).get("status") == "ok"
            ]
            if not runs:
                continue
            ds = load_dataset(sims_dir / scen_dir.name / label_dir.name)
            for run in runs:
                info = json.loads((run / "run.json").read_text())
                assigned = load_assigned(run, ds.counts.shape)
                truth, counts = ds.truth, ds.counts
                block = info.get("block")
                if block:
                    start, step = int(block[0]), int(block[1])
                    rows_kept = slice(start, min(start + step, counts.shape[0]))
                    assigned = assigned[rows_kept]
                    truth = truth.tocsr()[rows_kept]
                    counts = counts.tocsr()[rows_kept]
                conf = confusion(assigned, truth, counts)
                base = {
                    "scenario": scen_dir.name,
                    "sim_label": label_dir.name,
                    "run": run.name,
                    "seconds": info.get("seconds"),
                    "load_seconds": info.get("load_seconds"),
                    "block": json.dumps(block) if block else "",
                    **parse_label(label_dir.name),
                }
                for subset, c in conf.items():
                    rows.append({**base, "subset": subset, **c})
    return pd.DataFrame(rows)


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    full = metrics[metrics["subset"] == "full"]
    keys = ["scenario", "run"] + [
        k for k in ("moi", "mean_infections", "nguides", "corr") if k in full
    ]
    out = []
    for key, g in full.groupby(keys, dropna=False):
        row = dict(zip(keys, key, strict=True))
        row["n_replicates"] = len(g)
        for stat in ("f1", "precision", "recall"):
            v = g[stat].to_numpy(dtype=float)
            row[f"{stat}_median"] = float(np.nanmedian(v))
            row[f"{stat}_q1"] = float(np.nanquantile(v, 0.25))
            row[f"{stat}_q3"] = float(np.nanquantile(v, 0.75))
        row["seconds_median"] = float(np.nanmedian(g["seconds"].to_numpy(dtype=float)))
        out.append(row)
    return pd.DataFrame(out)


def _medians(summary, scenario, run, by):
    s = summary[(summary["scenario"] == scenario) & (summary["run"] == run)]
    return dict(zip(s[by], s["f1_median"], strict=True))


def replication(metrics: pd.DataFrame, summary: pd.DataFrame) -> dict:
    full = metrics[metrics["subset"] == "full"]
    out = {}
    for side in ("py", "r"):
        refit10, refit0 = f"{side}_fishash_refit10", f"{side}_fishash_refit0"
        res = {}
        ng = full[full["scenario"].isin([HIGH, LOW]) & (full["run"] == refit10)]
        res["R1"] = {
            "claim": "varyNumGuides <= 20k, both regimes: refit10 precision >= 0.95",
            "expected": "all datasets (paper: 100/100 incl. 80k)",
            "measured": f"{int((ng['precision'] >= 0.95).sum())}/{len(ng)}",
            "pass": bool(len(ng) and (ng["precision"] >= 0.95).all()),
        }
        corr = full[(full["scenario"] == CORR) & (full["run"] == refit0)]
        fails = {lvl: int((g["precision"] < 0.95).sum()) for lvl, g in corr.groupby("corr")}
        n = {lvl: len(g) for lvl, g in corr.groupby("corr")}
        res["R2"] = {
            "claim": "varySignalNoiseCorr: refit0 precision < 0.95",
            "expected": "low 20/20, mid 6/20",
            "measured": {
                lvl: f"{fails.get(lvl, 0)}/{n.get(lvl, 0)}" for lvl in ("low", "mid", "high")
            },
            "pass": fails.get("low") == 20 and fails.get("mid") == 6,
        }
        hi = full[(full["scenario"] == CORR) & (full["corr"] == "high")]
        m0 = float(np.median(hi[hi["run"] == refit0]["f1"])) if len(hi) else np.nan
        m10 = float(np.median(hi[hi["run"] == refit10]["f1"])) if len(hi) else np.nan
        res["R3"] = {
            "claim": "high-correlation median F1, refit0 vs refit10",
            "expected": "0.9426 vs 0.9421 (+-0.003; 4 decimals means our datasets are the paper's)",
            "measured": [round(m0, 6), round(m10, 6)],
            "pass": bool(abs(m0 - 0.9426) <= 0.003 and abs(m10 - 0.9421) <= 0.003),
            "four_decimal_match": bool(round(m0, 4) == 0.9426 and round(m10, 4) == 0.9421),
        }
        mix_run = "py_mixture_prob0.8" if side == "py" else "r_sceptre_mixture"
        candidates = {
            "fishash_refit10": _medians(summary, VARY_MOI, refit10, "moi"),
            "mixture": _medians(summary, VARY_MOI, mix_run, "moi"),
            "crispat_gauss": _medians(summary, VARY_MOI, "crispat_gauss", "moi"),
        }
        reps = {}
        for k, run in (
            ("fishash_refit10", refit10),
            ("mixture", mix_run),
            ("crispat_gauss", "crispat_gauss"),
        ):
            sel = summary[(summary["scenario"] == VARY_MOI) & (summary["run"] == run)]
            reps[k] = dict(zip(sel["moi"], sel["n_replicates"], strict=True))
        low_moi = [m for m in (0.1, 0.3, 0.5) if all(m in c for c in candidates.values())]
        complete = len(low_moi) == 3 and all(reps[k].get(m, 0) == 10 for k in reps for m in low_moi)
        res["R4"] = {
            "claim": "varyMOI, MOI <= 0.5: refit10 has the best median F1 of fishash, mixture, crispat",
            "measured": {m: {k: round(c[m], 4) for k, c in candidates.items()} for m in low_moi},
            "pass": bool(low_moi)
            and all(
                candidates["fishash_refit10"][m] >= max(c[m] for c in candidates.values())
                for m in low_moi
            ),
            "complete": complete,
        }
        med0 = _medians(summary, VARY_MOI, refit0, "moi")
        med10 = _medians(summary, VARY_MOI, refit10, "moi")
        medmix = candidates["mixture"]
        dev = {
            name: max(
                abs(med.get(m, np.nan) - v)
                for m, v in zip(COLLABORATOR_MOIS, COLLABORATOR_FISHASH_PLUS, strict=True)
            )
            for name, med in (("refit0", med0), ("refit10", med10))
        }
        mix_dev = max(
            abs(medmix.get(m, np.nan) - v)
            for m, v in zip(COLLABORATOR_MOIS, COLLABORATOR_MIXTURE, strict=True)
        )
        res["R5"] = {
            "claim": "collaborator's panel A: refit0 or refit10 within 0.01 at all 8 MOIs; mixture within 0.015",
            "max_abs_deviation": {k: round(v, 4) for k, v in dev.items()}
            | {"mixture": round(mix_dev, 4)},
            "fishash_plus_is": min(dev, key=dev.get),
            "pass": bool(min(dev.values()) <= 0.01 and mix_dev <= 0.015),
            "per_moi": {
                m: {
                    "collaborator_fishash_plus": v,
                    "refit0": round(med0.get(m, np.nan), 4),
                    "refit10": round(med10.get(m, np.nan), 4),
                    "collaborator_mixture": cm,
                    "mixture": round(medmix.get(m, np.nan), 4),
                }
                for m, v, cm in zip(
                    COLLABORATOR_MOIS, COLLABORATOR_FISHASH_PLUS, COLLABORATOR_MIXTURE, strict=True
                )
            },
        }
        fish = full[
            full["run"].isin([refit0, refit10]) & full["scenario"].isin([VARY_MOI, HIGH, LOW, CORR])
        ]
        res["R6"] = {
            "claim": "fishash step time up to 20k guides <= 10 s (memory is measured by time_methods.py)",
            "measured_max_seconds": float(fish["seconds"].max()) if len(fish) else None,
            "pass": bool(len(fish) and fish["seconds"].max() <= 10),
        }
        out[side] = res
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    sims_dir, runs_dir, tables_dir = (Path(a) for a in argv)
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics = score_runs(sims_dir, runs_dir)
    metrics.to_parquet(tables_dir / "metrics.parquet", index=False)
    summary = summarise(metrics)
    summary.to_parquet(tables_dir / "summary.parquet", index=False)
    rep = replication(metrics, summary)
    (tables_dir / "replication.json").write_text(json.dumps(rep, indent=2, default=float))
    for side, res in rep.items():
        for key, r in res.items():
            if r.get("complete") is False:
                status = "PENDING"
            else:
                status = "PASS" if r["pass"] else "FAIL"
            shown = r.get("measured", r.get("max_abs_deviation", r.get("measured_max_seconds")))
            print(side, key, status, shown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
