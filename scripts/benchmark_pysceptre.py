"""Run pysceptre on an export produced by benchmark_vs_r.R, with the same
instrumentation the R side records.

Each step runs as its own process (driven by benchmark_vs_r.sh or invoked
directly) so `ru_maxrss` reflects that step rather than a whole-pipeline
high-water mark.

Usage: benchmark_pysceptre.py <step> <export_dir> <out_dir> [chunk_memory_gb]
  step: discovery  (calibration and power are not implemented yet -- see
        the manuscript repo's status.md; R-side baselines already exist)
  chunk_memory_gb: optional working-set budget. Sweeping it and recording the
        resulting peak RSS is how the budget is calibrated against reality --
        it currently under-predicts (ROADMAP T1.4).
"""

from __future__ import annotations

import json
import resource
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sceptre_io import load_export  # noqa: E402

import pysceptre  # noqa: E402


def peak_rss_bytes() -> int:
    """ru_maxrss is bytes on macOS and kibibytes on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def record(
    step: str,
    out_dir: Path,
    wall: float,
    cpu_start,
    n_rows: int,
    chunk_memory_gb: float | None = None,
    load_wall: float | None = None,
) -> None:
    cpu_end = resource.getrusage(resource.RUSAGE_SELF)
    timing = {
        "step": step,
        "wall_seconds": wall,
        "user_seconds": cpu_end.ru_utime - cpu_start.ru_utime,
        "sys_seconds": cpu_end.ru_stime - cpu_start.ru_stime,
        "peak_rss_gb": peak_rss_bytes() / 1e9,
        "n_rows": n_rows,
        "chunk_memory_gb_budget": chunk_memory_gb,
        "load_seconds": load_wall,
        "pysceptre_version": pysceptre.__version__,
        "python_version": sys.version.split()[0],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / f"timing_pysceptre_{step}.json").write_text(json.dumps(timing, indent=2))
    print(
        f"  {step}: wall {wall:.1f}s ({wall / 60:.2f} min), "
        f"user {timing['user_seconds']:.1f}s, sys {timing['sys_seconds']:.1f}s, "
        f"peak RSS {timing['peak_rss_gb']:.2f} GB, {n_rows:,} rows"
    )


def main() -> None:
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(1)
    step, export_dir, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    chunk_memory_gb = float(sys.argv[4]) if len(sys.argv) > 4 else None
    out_dir.mkdir(parents=True, exist_ok=True)

    if step != "discovery":
        raise SystemExit(
            f"step {step!r} is not implemented in pysceptre yet; "
            "R-side baselines exist for comparison"
        )

    # Time the load separately from the analysis. R's benchmark does the same
    # -- its materialization and QC are a `prepare` step, and the discovery
    # timing excludes reading the post-QC object -- so counting pysceptre's
    # input load against its analysis would be an unfair asymmetry.
    load_start = time.perf_counter()
    export = load_export(export_dir)
    load_wall = time.perf_counter() - load_start
    print(f"input: {export.describe()}")
    print(f"  load: {load_wall:.2f}s, peak RSS after load {peak_rss_bytes() / 1e9:.2f} GB")
    kwargs = {} if chunk_memory_gb is None else {"chunk_memory_gb": chunk_memory_gb}
    label = step if chunk_memory_gb is None else f"{step}_mem{chunk_memory_gb:g}"
    print(f"budget: {'default' if chunk_memory_gb is None else f'{chunk_memory_gb} GB'}")

    cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pysceptre.run_discovery_analysis(
            response_matrix=export.response_matrix,
            gene_ids=export.gene_ids,
            covariate_matrix=export.covariate_matrix,
            grna_target_cells=export.grna_target_cells,
            pairs=export.pairs,
            side=export.side,
            seed=0,
            **kwargs,
        )
    wall = time.perf_counter() - t0

    record(label, out_dir, wall, cpu_start, len(result), chunk_memory_gb, load_wall)
    result.to_parquet(out_dir / f"pysceptre_{label}_result.parquet")
    print(f"  stages: {result.stage.value_counts().sort_index().to_dict()}")
    print(f"  p<1e-4: {int((result.p_value < 1e-4).sum())}")
    for w in caught:
        print(f"  WARN: {str(w.message)[:160]}")


if __name__ == "__main__":
    main()
