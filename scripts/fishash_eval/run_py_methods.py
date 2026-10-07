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

Outputs, in <runs_dir>/<scenario>/<sim_label>/py_<method>/, mirror run_r_methods.R:
    assigned.parquet    (guide, cell) of every call, 0-based int32, column-major
    log_pval.parquet    fishash: (guide, cell, log_pval) at every nonzero count
    background.parquet  fishash refit10: (cell, cell_size) of the last noise fit, and
                        (guide, guide_freq) in guide_freqs.parquet
    posterior.parquet   mixture: (guide, cell, ti1) at each gRNA's explicit cells, and
                        posterior_zero.parquet (guide, ti1) for its zero-count cells
    fits.parquet        mixture: one row per gRNA (path, EM and GLM diagnostics)
    run.json            status, seconds (the assignment call only), load_seconds, settings,
                        versions and the thread environment
A run whose run.json says "ok" is skipped. Outputs must go to a git-ignored directory.
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

METHODS = ("fishash_refit0", "fishash_refit10", "mixture_prob0.8", "mixture_reduced")
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
