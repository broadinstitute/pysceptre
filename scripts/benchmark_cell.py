"""One cell of the implementation x mechanism x core-count matrix.

Usage: benchmark_cell.py <label> <export_dir> <out_dir> <mechanism> <n_jobs>
                        [chunk_memory_gb]

One configuration, one process, because `ru_maxrss` is a high-water mark for
the process lifetime: two configurations timed in one process make the
second inherit the first's peak. That mistake is recorded in
the manuscript repository; this script exists so it cannot recur.

Reports wall, peak RSS and **mean cores** -- `(user + sys) / wall`. Wall
alone cannot tell a real speedup from one bought by spending more cores, and
a run that claims to be single-core while reporting 1.9 cores is visibly
wrong rather than plausibly right.

**Both must account for forked workers, which `RUSAGE_SELF` does not.**
pysceptre uses processes on Linux and threads elsewhere, so on Linux every
worker's CPU time and memory belongs to `RUSAGE_CHILDREN` and is invisible
to the parent's own rusage. Measured on this matrix before the fix, an
8-worker run reported `cores 0.14` and 2.20 GB against a true ~6.5 cores and
~5.9 GB -- numbers that look like data and are nothing of the kind.

CPU time is therefore summed over self and children. Peak memory is read
from the container's cgroup, which is the only figure here that is a true
tree-wide peak: `RUSAGE_CHILDREN.ru_maxrss` is the maximum over reaped
children rather than their sum, so it understates a pool whose workers are
each large.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sceptre_io import load_export  # noqa: E402

import pysceptre  # noqa: E402
from pysceptre.pipeline.discovery import parallel_backend  # noqa: E402


def _cgroup_peak_bytes() -> int | None:
    """The container's true peak memory, workers included, or `None`.

    cgroup v2 exposes `memory.peak`; older kernels expose
    `memory.max_usage_in_bytes`. Absent outside a container, in which case
    the caller falls back to rusage.
    """
    for path in (
        "/sys/fs/cgroup/memory.peak",
        "/sys/fs/cgroup/memory/memory.max_usage_in_bytes",
    ):
        try:
            return int(Path(path).read_text().strip())
        except (OSError, ValueError):
            continue
    return None


def main() -> None:
    if len(sys.argv) < 6:
        print(__doc__)
        raise SystemExit(1)
    label, export_dir, out_dir, mechanism, n_jobs = (
        sys.argv[1],
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        sys.argv[4],
        int(sys.argv[5]),
    )
    # Optional, because the chunk budget is the one knob whose effect is
    # expected to differ by backend: it sets how many chunks there are, and
    # each chunk boundary costs a fork of the worker pool on Linux and
    # nothing at all on threads.
    chunk_gb = float(sys.argv[6]) if len(sys.argv) > 6 else None
    extra = {} if chunk_gb is None else {"chunk_memory_gb": chunk_gb}
    out_dir.mkdir(parents=True, exist_ok=True)

    export = load_export(str(export_dir))
    warnings.simplefilter("ignore")

    t0 = time.perf_counter()
    result = pysceptre.run_discovery_analysis(
        response_matrix=export.response_matrix,
        gene_ids=export.gene_ids,
        covariate_matrix=export.covariate_matrix,
        grna_target_cells=export.grna_target_cells,
        pairs=export.pairs,
        side=export.side,
        seed=0,
        n_jobs=n_jobs,
        resampling_mechanism=mechanism,
        **extra,
    )
    wall = time.perf_counter() - t0

    me = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    unit = 1 if sys.platform == "darwin" else 1024  # bytes on macOS, KiB on Linux
    rusage_peak = max(me.ru_maxrss, kids.ru_maxrss) * unit
    cgroup_peak = _cgroup_peak_bytes()
    peak = cgroup_peak if cgroup_peak else rusage_peak
    cpu = (me.ru_utime + me.ru_stime) + (kids.ru_utime + kids.ru_stime)
    record = {
        "label": label,
        "mechanism": mechanism,
        "n_jobs": n_jobs,
        "chunk_memory_gb": chunk_gb,
        # The platform default, which is not necessarily what the gene pool
        # used: `gene_job_backend` overrides it per chunk count. Named to say
        # so, after a cell reported `backend=fork` for a run whose gene pool
        # was threads.
        "platform_backend": parallel_backend(),
        "backend_env": os.environ.get("PYSCEPTRE_BACKEND", ""),
        "wall_seconds": wall,
        "peak_rss_gb": peak / 1e9,
        "peak_source": "cgroup" if cgroup_peak else "rusage",
        "peak_rss_rusage_gb": rusage_peak / 1e9,
        "mean_cores": cpu / wall,
        "n_rows": len(result),
        "stages": {int(k): int(v) for k, v in result.stage.value_counts().items()},
        "pysceptre_version": pysceptre.__version__,
        # Which checkout actually ran. The matrix mounts one ref per cell and
        # selects it with PYTHONPATH, and a silent fall-through to the image's
        # own installed copy would make two cells identical while looking
        # like a comparison.
        "pysceptre_path": str(Path(pysceptre.__file__).resolve().parent),
        "python_version": sys.version.split()[0],
    }
    (out_dir / f"cell_{label}.json").write_text(json.dumps(record, indent=2))
    result.to_parquet(out_dir / f"cell_{label}.parquet")
    print(
        f"  {label:28s} {wall:8.1f}s  peak {record['peak_rss_gb']:5.2f} GB "
        f"({record['peak_source']})  cores {record['mean_cores']:5.2f}  "
        f"rows {len(result):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
