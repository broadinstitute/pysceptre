"""Time discovery and the calibration check across worker counts.

Reports wall time, peak RSS and the speedup over `n_jobs=1`, and checks that
every worker count produced **identical** p-values -- parallelism is only
allowed to change how long the run takes.

**The backend depends on the platform, and so does the ceiling.** Linux forks
(processes); everything else uses threads, because `fork` after Apple's
Accelerate BLAS has run can deadlock. NumPy holds the GIL through much of the
gather in the per-pair statistic, so the thread ceiling is materially lower --
measured 1.85x against 3.54x for processes on one moi5-shaped call. A number
measured here on a Mac is a lower bound on what Linux does, not a prediction
of it.

Usage:
  benchmark_n_jobs.py <export_dir> [--steps discovery,calibration]
      [--jobs 1,2,4,8] [--budgets 1,2,4,8] [--chunk-memory-gb G]
      [--out results.json]
"""

from __future__ import annotations

import json
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from sceptre_io import load_export  # noqa: E402

import pysceptre  # noqa: E402
from pysceptre.pipeline.discovery import parallel_backend, resolve_n_jobs  # noqa: E402


def peak_rss_gb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (raw if sys.platform == "darwin" else raw * 1024) / 1e9


def r_calibration_pairs(export) -> pd.DataFrame | None:
    for frame in (export.negative_control_pairs, export.calibration_result):
        if frame is not None and len(frame):
            cols = [c for c in ("response_id", "grna_target") if c in frame]
            if len(cols) == 2:
                return frame[cols].reset_index(drop=True)
    return None


def run_step(step: str, export, n_jobs: int, chunk_memory_gb: float | None) -> tuple:
    meta = export.metadata
    kwargs = {} if chunk_memory_gb is None else {"chunk_memory_gb": chunk_memory_gb}
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if step == "discovery":
            result = pysceptre.run_discovery_analysis(
                response_matrix=export.response_matrix,
                gene_ids=export.gene_ids,
                covariate_matrix=export.covariate_matrix,
                grna_target_cells=export.grna_target_cells,
                pairs=export.pairs,
                side=export.side,
                seed=0,
                n_jobs=n_jobs,
                **kwargs,
            )
        else:
            pairs = r_calibration_pairs(export)
            if pairs is None:
                raise SystemExit("export carries no R calibration pairs to inject")
            result = pysceptre.run_calibration_check(
                response_matrix=export.response_matrix,
                gene_ids=export.gene_ids,
                covariate_matrix=export.covariate_matrix,
                ntc_grna_cells=export.ntc_grna_cells,
                negative_control_pairs=pairs,
                n_calibration_pairs=len(pairs),
                calibration_group_size=int(meta.get("calibration_group_size", 15)),
                n_nonzero_trt_thresh=int(meta.get("n_nonzero_trt_thresh", 7)),
                n_nonzero_cntrl_thresh=int(meta.get("n_nonzero_cntrl_thresh", 7)),
                pass_qc_rate=float(meta.get("discovery_pass_qc_rate") or 1.0),
                side=export.side,
                seed=0,
                n_jobs=n_jobs,
                **kwargs,
            )
    return result, time.perf_counter() - t0


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    export_dir = Path(sys.argv[1])
    argv = sys.argv[2:]

    def value(flag, cast, default=None):
        return cast(argv[argv.index(flag) + 1]) if flag in argv else default

    steps = value("--steps", lambda v: v.split(","), ["discovery", "calibration"])
    jobs = value("--jobs", lambda v: [int(x) for x in v.split(",")], [1, 2, 4, 8])
    budgets = value("--budgets", lambda v: [float(x) for x in v.split(",")], None)
    chunk_memory_gb = value("--chunk-memory-gb", float, None)
    out_path = value("--out", str, None)

    export = load_export(export_dir, backed=True)
    print(f"input: {export.describe()}")
    print(f"backend: {parallel_backend()}  |  cores: {resolve_n_jobs(-1)}\n")

    report: dict = {"export": str(export_dir), "backend": parallel_backend(), "steps": {}}
    for step in steps:
        print(f"== {step} ==")
        rows, baseline, reference = [], None, None
        grid = [(n, b) for b in (budgets or [chunk_memory_gb]) for n in jobs]
        for n, b in grid:
            result, wall = run_step(step, export, n, b)
            if baseline is None:
                baseline, reference = wall, result["p_value"].to_numpy()
                identical = True
            else:
                identical = bool(np.array_equal(reference, result["p_value"].to_numpy()))
            rows.append(
                {
                    "n_jobs": n,
                    "chunk_memory_gb": b,
                    "wall_seconds": wall,
                    "speedup": baseline / wall,
                    "peak_rss_gb": peak_rss_gb(),
                    "n_rows": len(result),
                    "p_values_identical_to_serial": identical,
                }
            )
            print(
                f"  budget={b if b else 'default':<7} n_jobs={n:<3} "
                f"{wall:8.1f}s ({wall / 60:5.2f} min)  "
                f"speedup {baseline / wall:5.2f}x  peak {peak_rss_gb():5.2f} GB  "
                f"identical={identical}"
            )
        report["steps"][step] = rows
        print()

    if hasattr(export.response_matrix, "close"):
        export.response_matrix.close()
    if out_path:
        Path(out_path).write_text(json.dumps(report, indent=2, default=float))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
