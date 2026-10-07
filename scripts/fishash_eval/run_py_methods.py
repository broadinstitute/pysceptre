"""Run pysceptre's gRNA assignment methods on the simulated datasets.

    python scripts/fishash_eval/run_py_methods.py <sims_dir> <scenario> <runs_dir> <method>
        [sim_label ...]

`method` is one of:
    fishash_refit0, fishash_refit10   pysceptre.assignment.assign_grnas_fishash, FDR 0.05 (as the
                                      fishash paper's bin/run_fishash.R)
    mixture_prob0.8                   pysceptre.assignment.assign_grnas_mixture on sceptre's
                                      default design, from the dataset's covariates.parquet
                                      (scripts/fishash_eval/gex_covariates.R); on a GLM failure,
                                      the reduced design of the paper's bin/run_sceptre_mixture.R
    mixture_reduced                   the mixture on the gRNA-only design (no gene expression)
    grid                              every normalization x caller of the normalization study,
                                      each written to py_grid__<norm>__<caller>/. Guides with no
                                      count are removed before normalizing and get no call.
                                      Callers: q99 and q95 (a per-guide quantile cut), q_oracle
                                      (the cut at the guide's true positive fraction, a ceiling),
                                      gmm_lab, and max (each cell's top guide, the low-MOI
                                      caller). The lab's CMO procedures are clr_cell x q95 and
                                      clr_cell x gmm_lab. gmm_lab does not run on
                                      20,000-guide datasets past replicate
                                      GMM_LAB_MAX_REPLICATE_AT_20K (no output is written);
                                      noise_size needs py_fishash_refit10/background.parquet

Outputs, in <runs_dir>/<scenario>/<sim_label>/py_<method>/, mirror run_r_methods.R:
    assigned.parquet    (guide, cell) of every call, 0-based int32, column-major
    log_pval.parquet    fishash: (guide, cell, log_pval) at every nonzero count
    background.parquet  fishash refit10: (cell, cell_size) of the last noise fit, and
                        (guide, guide_freq) in guide_freqs.parquet
    posterior.parquet   mixture: (guide, cell, ti1) at each gRNA's explicit cells, and
                        posterior_zero.parquet (guide, ti1) for its zero-count cells
    fits.parquet        mixture: one row per gRNA (path, EM and GLM diagnostics)
    thresholds.parquet  grid: (guide, threshold) for every guide, NaN for a removed one; the
                        quantile callers' cut, gmm_lab's display boundary, NaN for max
    run.json            status, seconds (the assignment call only), load_seconds, settings,
                        versions and the thread environment
A run whose run.json says "ok" is skipped; a grid run also needs the current guide filter in
its settings. Outputs must go to a git-ignored directory.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent))
from eval_lib import load_dataset  # noqa: E402

from pysceptre.assignment import assign_grnas_fishash  # noqa: E402
from pysceptre.assignment.design import design_from_covariates, mixture_design_matrix  # noqa: E402
from pysceptre.assignment.mixture import assign_grnas_mixture  # noqa: E402

METHODS = ("fishash_refit0", "fishash_refit10", "mixture_prob0.8", "mixture_reduced", "grid")
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
)
REPO = Path(__file__).resolve().parents[2]


def refuse_unless_ignored(out_dir: Path) -> None:
    probe = out_dir / "probe" / "run.json"
    r = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", str(probe)], capture_output=True
    )
    if r.returncode != 0:
        raise SystemExit(f"refusing to write to {out_dir}: it is not ignored by git")


def versions() -> dict:
    import numba
    import scipy

    sha = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", "--", "src"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {
        "pysceptre_sha": sha,
        "pysceptre_src_dirty": dirty,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def write_parquet(table: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(pa.Table.from_pandas(table, preserve_index=False), tmp)
    tmp.replace(path)


def write_json(obj: dict, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    tmp.replace(path)


def triplets(m, value_name: str | None = None) -> pd.DataFrame:
    """Column-major (guide, cell[, value]) triplets of a guides x cells sparse matrix."""
    c = sparse.csc_matrix(m)
    c.sort_indices()
    cols = np.repeat(np.arange(c.shape[1], dtype=np.int32), np.diff(c.indptr))
    out = pd.DataFrame({"guide": c.indices.astype(np.int32), "cell": cols})
    if value_name is not None:
        out[value_name] = c.data.astype(np.float64)
    return out


def run_fishash(ds, refit: int, out: Path) -> dict:
    t0 = time.perf_counter()
    res = assign_grnas_fishash(ds.counts, ds.grna_ids, refit=refit, padj_cutoff=0.05)
    seconds = time.perf_counter() - t0
    write_parquet(triplets(res.assigned), out / "assigned.parquet")
    write_parquet(triplets(res.log_pval, "log_pval"), out / "log_pval.parquet")
    if res.background_cell_sizes is not None:
        write_parquet(
            pd.DataFrame(
                {
                    "cell": np.arange(ds.counts.shape[1], dtype=np.int32),
                    "cell_size": res.background_cell_sizes,
                }
            ),
            out / "background.parquet",
        )
        write_parquet(
            pd.DataFrame(
                {
                    "guide": np.arange(ds.counts.shape[0], dtype=np.int32),
                    "guide_freq": res.background_guide_freqs,
                }
            ),
            out / "guide_freqs.parquet",
        )
    demux = pd.Series(res.demux_type).value_counts().to_dict()
    return {
        "seconds": seconds,
        "cutoff": res.log_pval_cutoff,
        "num_iter": res.num_iter,
        "n_assigned": int(res.assigned.nnz),
        "demux_type": {str(k): int(v) for k, v in demux.items()},
        "passes": [
            {
                "pass": i + 1,
                "logpval_cutoff": it.log_pval_cutoff,
                "n_significant": it.n_significant,
                "n_assigned": it.n_assigned,
                "impute_n_iter": it.impute_n_iter,
            }
            for i, it in enumerate(res.iterations)
        ],
        "settings": {
            "refit": refit,
            "padj_cutoff": 0.05,
            "padj_method": "GS",
            "min_count": 2,
            "exclude_empty": True,
        },
    }


def run_mixture(ds, dataset_dir: Path, reduced: bool, out: Path) -> dict:
    fallback_used = False
    if reduced:
        X, names = mixture_design_matrix(ds.counts)
    else:
        cov = pd.read_parquet(dataset_dir / "covariates.parquet")
        cov = cov.sort_values("cell").drop(columns="cell").reset_index(drop=True)
        X, names = design_from_covariates(cov)
    t0 = time.perf_counter()
    try:
        res = assign_grnas_mixture(ds.counts, ds.grna_ids, X)
    except ValueError:
        if reduced:
            raise
        # bin/run_sceptre_mixture.R retries with ~ log(grna_n_nonzero+1) + log(grna_n_umis+1).
        fallback_used = True
        X, names = mixture_design_matrix(ds.counts)
        res = assign_grnas_mixture(ds.counts, ds.grna_ids, X)
    seconds = time.perf_counter() - t0
    write_parquet(triplets(res.assigned), out / "assigned.parquet")
    write_parquet(triplets(res.posterior, "ti1"), out / "posterior.parquet")
    write_parquet(
        pd.DataFrame(
            {"guide": np.arange(len(res.grna_ids), dtype=np.int32), "ti1": res.posterior_zero}
        ),
        out / "posterior_zero.parquet",
    )
    fits = res.fits.copy()
    fits.insert(0, "guide", np.arange(len(fits), dtype=np.int32))
    write_parquet(fits, out / "fits.parquet")
    return {
        "seconds": seconds,
        "n_assigned": int(res.assigned.nnz),
        "fallback_used": fallback_used,
        "design_columns": names,
        "n_mixture": int((res.fits["method"] == "mixture").sum()),
        "n_backup": int((res.fits["method"] == "backup").sum()),
        "settings": {
            "probability_threshold": 0.8,
            "n_nonzero_cells_cutoff": 10,
            "backup_threshold": 5,
            "design": "reduced" if reduced else "default",
        },
    }


GRID_NORMS = ("raw", "clr_cell", "clr_guide", "depth", "noise_size")
GRID_CALLERS = ("q99", "q95", "q_oracle", "gmm_lab", "max")
GRID_QUANTILES = {"q99": 0.99, "q95": 0.95}
GRID_GUIDE_FILTER = "guides with no count removed before normalizing"
GMM_LAB_MAX_REPLICATE_AT_20K = 3


def _grid_done(info_path: Path) -> bool:
    if not info_path.exists():
        return False
    info = json.loads(info_path.read_text())
    return (
        info.get("status") == "ok"
        and info.get("settings", {}).get("guide_filter") == GRID_GUIDE_FILTER
    )


def run_grid(ds, label_dir: Path, label: str, meta_versions: dict) -> int:
    import eval_lib as el

    failures = 0
    n_guides = ds.counts.shape[0]
    replicate = int(label.rsplit("_", 1)[-1]) if label.rsplit("_", 1)[-1].isdigit() else 0
    keep = el.guides_with_counts(ds.counts)
    counts = ds.counts.tocsr()[keep]
    truth = ds.truth.tocsr()[keep]
    for norm in GRID_NORMS:
        todo = []
        for caller in GRID_CALLERS:
            out = label_dir / f"py_grid__{norm}__{caller}"
            if _grid_done(out / "run.json"):
                continue
            skip = (
                caller == "gmm_lab"
                and n_guides >= 20000
                and replicate > GMM_LAB_MAX_REPLICATE_AT_20K
            )
            if not skip:
                todo.append((caller, out))
        if not todo:
            continue
        t0 = time.perf_counter()
        if norm == "raw":
            values = el.norm_raw(counts)
        elif norm == "clr_cell":
            values = el.norm_clr_cell(counts)
        elif norm == "clr_guide":
            values = el.norm_clr_guide(counts)
        elif norm == "depth":
            values = el.norm_depth(counts)
        else:
            bg = label_dir / "py_fishash_refit10" / "background.parquet"
            if not bg.exists():
                print(f"{label} noise_size: no {bg}", flush=True)
                failures += 1
                continue
            sizes = pd.read_parquet(bg).sort_values("cell")["cell_size"].to_numpy()
            values = el.norm_noise_size(counts, sizes)
        norm_seconds = time.perf_counter() - t0
        for caller, out in todo:
            out.mkdir(parents=True, exist_ok=True)
            info = {
                "sim_label": label,
                "method": f"grid__{norm}__{caller}",
                "status": "error",
                "settings": {
                    "guide_filter": GRID_GUIDE_FILTER,
                    "n_guides": int(n_guides),
                    "n_guides_kept": int(keep.size),
                    "quantile": GRID_QUANTILES.get(caller),
                },
            }
            try:
                t1 = time.perf_counter()
                if caller in GRID_QUANTILES:
                    calls = el.call_q95(values, q=GRID_QUANTILES[caller])
                elif caller == "q_oracle":
                    calls = el.call_q_oracle(values, truth)
                elif caller == "max":
                    calls = el.call_max(values)
                else:
                    calls = el.call_gmm_lab(values)
                info["seconds"] = norm_seconds + (time.perf_counter() - t1)
                info["normalize_seconds"] = norm_seconds
                calls = el.expand_guides(calls, keep, n_guides)
                write_parquet(triplets(calls.assigned), out / "assigned.parquet")
                write_parquet(
                    pd.DataFrame(
                        {
                            "guide": np.arange(n_guides, dtype=np.int32),
                            "threshold": calls.thresholds,
                        }
                    ),
                    out / "thresholds.parquet",
                )
                info["n_assigned"] = int(calls.assigned.nnz)
                info["status"] = "ok"
            except Exception as e:  # noqa: BLE001 - recorded, and the exit code says so
                info["error"] = repr(e)
                failures += 1
            info["versions"] = meta_versions
            info["threads"] = {v: os.environ.get(v) for v in THREAD_VARS}
            write_json(info, out / "run.json")
            print(
                f"{label} grid {norm} x {caller}: {info['status']} {info.get('seconds', float('nan')):.2f}s",
                flush=True,
            )
    return failures


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[3] not in METHODS:
        print(__doc__, file=sys.stderr)
        return 2
    sims_dir, scenario, runs_dir, method = Path(argv[0]), argv[1], Path(argv[2]), argv[3]
    scen_dir = sims_dir / scenario
    manifest = json.loads((scen_dir / "manifest.json").read_text())
    labels = argv[4:] or [d["sim_label"] for d in manifest["datasets"]]
    refuse_unless_ignored(runs_dir)
    meta_versions = versions()
    failures = 0
    for label in labels:
        if method == "grid":
            failures += run_grid(
                load_dataset(scen_dir / label), runs_dir / scenario / label, label, meta_versions
            )
            continue
        out = runs_dir / scenario / label / f"py_{method}"
        info_path = out / "run.json"
        if info_path.exists() and json.loads(info_path.read_text()).get("status") == "ok":
            continue
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        ds = load_dataset(scen_dir / label)
        load_seconds = time.perf_counter() - t0
        info = {"scenario": scenario, "sim_label": label, "method": method, "status": "error"}
        try:
            if method.startswith("fishash"):
                info.update(run_fishash(ds, int(method.removeprefix("fishash_refit")), out))
            else:
                info.update(run_mixture(ds, scen_dir / label, method == "mixture_reduced", out))
            info["status"] = "ok"
        except Exception as e:  # noqa: BLE001 - recorded, and the exit code says so
            info["error"] = repr(e)
            failures += 1
        info["load_seconds"] = load_seconds
        info["versions"] = meta_versions
        info["threads"] = {v: os.environ.get(v) for v in THREAD_VARS}
        write_json(info, info_path)
        print(
            f"{scenario}/{label} {method}: {info['status']} {info.get('seconds', float('nan')):.3f}s",
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
