"""Compare pysceptre's assignment runs with R's, dataset by dataset (the scale-level check).

    python scripts/fishash_eval/compare_py_r.py <sims_dir> <runs_dir> <report.json>
        [--method fishash|mixture] [--scenario NAME ...]

For each dataset that has both runs (py_<method> from run_py_methods.py, r_<method> from
run_r_methods.R):

fishash (refit 0 and 10) passes when
    every call agrees, or each disagreement is a "boundary" flip (R's log p within the
        comparison tolerance of R's cut; see scripts/assignment_compare.py);
    the number of passes, and each pass's B and number of calls, are equal;
    the cuts agree to 1e-12 (or are both -Inf);
    every log p-value agrees to 1e-12 relative to max(1, |log p|), 1e-9 for counts of one.
A disagreement that is not a boundary flip is a "defect". In total, at most 10 boundary flips
are accepted across all fishash comparisons.

the mixture passes when
    every call agrees except at cells whose R posterior lies within 1e-4 of the 0.8 cut (listed
        in R's ti1s_band.parquet, written with --internals);
    when R's internals exist, every gRNA takes the same path (mixture or backup);
    the Jaccard index of the two call sets is at least 0.999 and F1 differs by at most 1e-3.

Writes the report and exits 1 if any dataset fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assignment_compare import REL_LOG_P, REL_LOG_P_COUNT1, flips  # noqa: E402
from eval_lib import confusion, load_assigned, load_dataset  # noqa: E402
from scipy import sparse  # noqa: E402

MAX_BOUNDARY_FLIPS = 10
PAIRS = {
    "fishash": [
        ("py_fishash_refit0", "r_fishash_refit0"),
        ("py_fishash_refit10", "r_fishash_refit10"),
    ],
    "mixture": [("py_mixture_prob0.8", "r_sceptre_mixture")],
}


def _log_p_matrix(run_dir: Path, shape) -> sparse.csr_matrix:
    t = pd.read_parquet(run_dir / "log_pval.parquet")
    return sparse.csr_matrix(
        (t["log_pval"].to_numpy(np.float64), (t["guide"].to_numpy(), t["cell"].to_numpy())),
        shape=shape,
    )


def _cut_equal(a: float, b: float) -> bool:
    if np.isneginf(a) or np.isneginf(b):
        return bool(np.isneginf(a) and np.isneginf(b))
    return abs(a - b) <= 1e-12


def compare_fishash(ds, py_dir: Path, r_dir: Path) -> dict:
    shape = ds.counts.shape
    py_info = json.loads((py_dir / "run.json").read_text())
    r_info = json.loads((r_dir / "run.json").read_text())
    a_py = load_assigned(py_dir, shape)
    a_r = load_assigned(r_dir, shape)
    lp_py = _log_p_matrix(py_dir, shape)
    lp_r = _log_p_matrix(r_dir, shape)
    table = flips(a_r, a_py, counts=ds.counts, log_p_ref=lp_r, cutoff_ref=r_info["cutoff"])

    # Log p at every nonzero count, compared in the counts' own pattern.
    c = sparse.csr_matrix(ds.counts)
    c.sort_indices()
    rows = np.repeat(np.arange(shape[0]), np.diff(c.indptr))
    cols = c.indices
    v_py = np.asarray(lp_py[rows, cols]).ravel()
    v_r = np.asarray(lp_r[rows, cols]).ravel()
    with np.errstate(invalid="ignore"):
        scale = np.maximum(1.0, np.abs(v_r))
        tol = np.where(c.data == 1, REL_LOG_P_COUNT1, REL_LOG_P) * scale
        diff = np.abs(v_py - v_r)
    same_special = (np.isnan(v_py) == np.isnan(v_r)) & (np.isinf(v_py) == np.isinf(v_r))
    finite = np.isfinite(v_r) & np.isfinite(v_py)
    log_p_ok = bool(same_special.all() and (diff[finite] <= tol[finite]).all())
    rel = diff[finite] / scale[finite]

    py_passes = py_info["passes"]
    r_passes = r_info["passes"]
    passes_ok = len(py_passes) == len(r_passes) and all(
        p["n_significant"] == (r["B"] if r.get("B") is not None else r["n_signif"])
        and p["n_assigned"] == r["n_assigned"]
        for p, r in zip(py_passes, r_passes, strict=False)
    )
    n_defect = int((table["flip_class"] == "defect").sum()) if len(table) else 0
    n_boundary = int((table["flip_class"] == "boundary").sum()) if len(table) else 0
    ok = (
        n_defect == 0
        and py_info["num_iter"] == r_info["num_iter"]
        and passes_ok
        and _cut_equal(py_info["cutoff"], r_info["cutoff"])
        and log_p_ok
    )
    return {
        "status": "ok" if ok else "fail",
        "n_flips": int(len(table)),
        "n_boundary_flips": n_boundary,
        "n_defect_flips": n_defect,
        "num_iter": [py_info["num_iter"], r_info["num_iter"]],
        "passes_equal": passes_ok,
        "cutoff_py": py_info["cutoff"],
        "cutoff_r": r_info["cutoff"],
        "log_p_within_tolerance": log_p_ok,
        "log_p_max_rel_diff": float(rel.max()) if rel.size else 0.0,
        "n_entries": int(c.nnz),
        "n_assigned": [int(a_py.nnz), int(a_r.nnz)],
        "seconds": [py_info.get("seconds"), r_info.get("seconds")],
        "flips": table.head(50).to_dict(orient="records"),
    }


def compare_mixture(ds, py_dir: Path, r_dir: Path) -> dict:
    shape = ds.counts.shape
    py_info = json.loads((py_dir / "run.json").read_text())
    r_info = json.loads((r_dir / "run.json").read_text())
    a_py = load_assigned(py_dir, shape)
    a_r = load_assigned(r_dir, shape)
    diff = (a_py != a_r).tocoo()
    band_path = r_dir / "ti1s_band.parquet"
    exempt = 0
    unexplained = int(diff.nnz)
    if band_path.exists() and diff.nnz:
        band = pd.read_parquet(band_path)
        near = band[np.abs(band["ti1"] - 0.8) <= 1e-4]
        keys = set(
            zip(near["guide"].to_numpy().tolist(), near["cell"].to_numpy().tolist(), strict=True)
        )
        exempt = sum(
            (g, c) in keys for g, c in zip(diff.row.tolist(), diff.col.tolist(), strict=True)
        )
        unexplained = int(diff.nnz) - exempt
    paths_equal = None
    per_grna = r_dir / "per_grna.parquet"
    if per_grna.exists():
        r_paths = pd.read_parquet(per_grna).sort_values("guide")["path"].to_numpy()
        py_fits = pd.read_parquet(py_dir / "fits.parquet").sort_values("guide")
        py_paths = py_fits["method"].to_numpy()
        r_simple = np.where(r_paths == "mixture", "mixture", "backup")
        paths_equal = bool(np.array_equal(r_simple, py_paths))
    inter = int(a_py.multiply(a_r).nnz)
    union = int(a_py.nnz + a_r.nnz - inter)
    jaccard = inter / union if union else 1.0
    f1_py = confusion(a_py, ds.truth, ds.counts)["full"]["f1"]
    f1_r = confusion(a_r, ds.truth, ds.counts)["full"]["f1"]
    ok = (
        unexplained == 0
        and paths_equal is not False
        and jaccard >= 0.999
        and abs(f1_py - f1_r) <= 1e-3
    )
    return {
        "status": "ok" if ok else "fail",
        "n_differing_calls": int(diff.nnz),
        "n_exempt_near_cut": int(exempt),
        "n_unexplained": unexplained,
        "paths_equal": paths_equal,
        "jaccard": jaccard,
        "f1": [f1_py, f1_r],
        "n_assigned": [int(a_py.nnz), int(a_r.nnz)],
        "fallback_used": [py_info.get("fallback_used"), r_info.get("fallback_used")],
        "seconds": [py_info.get("seconds"), r_info.get("seconds")],
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sims_dir", type=Path)
    ap.add_argument("runs_dir", type=Path)
    ap.add_argument("report", type=Path)
    ap.add_argument("--method", choices=sorted(PAIRS), action="append")
    ap.add_argument("--scenario", action="append")
    args = ap.parse_args(argv)
    methods = args.method or sorted(PAIRS)
    scenarios = args.scenario or sorted(p.name for p in args.runs_dir.iterdir() if p.is_dir())
    rows = []
    for scenario in scenarios:
        for label_dir in sorted((args.runs_dir / scenario).iterdir()):
            if not label_dir.is_dir():
                continue
            ds = None
            for method in methods:
                for py_name, r_name in PAIRS[method]:
                    py_dir, r_dir = label_dir / py_name, label_dir / r_name
                    ok_runs = all(
                        (d / "run.json").exists()
                        and json.loads((d / "run.json").read_text()).get("status") == "ok"
                        for d in (py_dir, r_dir)
                    )
                    if not ok_runs:
                        continue
                    if ds is None:
                        ds = load_dataset(args.sims_dir / scenario / label_dir.name)
                    fn = compare_fishash if method == "fishash" else compare_mixture
                    res = fn(ds, py_dir, r_dir)
                    res.update(scenario=scenario, sim_label=label_dir.name, py=py_name, r=r_name)
                    rows.append(res)
    n_boundary = sum(r.get("n_boundary_flips", 0) for r in rows)
    failures = [r for r in rows if r["status"] != "ok"]
    overall_ok = not failures and n_boundary <= MAX_BOUNDARY_FLIPS
    report = {
        "n_comparisons": len(rows),
        "n_failed": len(failures),
        "n_boundary_flips_total": n_boundary,
        "status": "ok" if overall_ok else "fail",
        "comparisons": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=float))
    print(
        f"{len(rows)} comparisons, {len(failures)} failed, {n_boundary} boundary flips: "
        f"{report['status']}"
    )
    for r in failures[:20]:
        print(
            "FAIL",
            r["scenario"],
            r["sim_label"],
            r["py"],
            {k: r[k] for k in r if k.startswith("n_")},
        )
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
