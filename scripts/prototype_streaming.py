"""Prototype: stream CRT draws per target instead of per chunk (ROADMAP T1.1).

The current pipeline draws every target in a chunk up front, so the chunk holds
`chunk_size x (fit arrays + draws)`. Profiling moi5 showed the draws dominate
that -- 16.5 MB of the 20.7 MB per target -- and they are the reason a 4 GB
budget becomes a 9.8 GB peak.

Only the *batched binomial IRLS* benefits from a chunk. The draws do not: they
are independent per target. So keep the chunk for the fit, retain only the
fitted probabilities, and draw each target's samples when that target is
processed, discarding them immediately.

That forces the pair loop target-outer, where the current implementation is
gene-outer, so gene precomputations are rebuilt per pair rather than per chunk
-- 33,135 rebuilds instead of 3,660 on moi5, about +47 s.

Predicted on moi5: 3.97 GB -> 0.84 GB of working set, +12% wall time.

Results must be IDENTICAL: targets are drawn in the same order as before, so
the RNG stream is unchanged, and each pair's test depends only on its own
draws.

Usage: prototype_streaming.py <export_dir> <out_dir> [chunk_memory_gb]
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

from pysceptre.crt.sampler import crt_index_sampler_fast  # noqa: E402
from pysceptre.glm.irls import fit_binomial_glm_batch  # noqa: E402
from pysceptre.pipeline.discovery import (  # noqa: E402
    _DEFAULT_CHUNK_MEMORY_GB,
    _get_row,
    chunk_size_for_budget,
    fit_all_genes,
    irls_bytes_per_column,
)
from pysceptre.precompute.pieces import compute_precomputation_pieces  # noqa: E402
from pysceptre.test_statistic.resampling import run_low_level_test_full  # noqa: E402

B1, B2, B3 = 499, 4999, 0


def streamed_discovery(
    response_matrix,
    gene_ids,
    covariate_matrix,
    grna_target_cells,
    pairs,
    *,
    side_code=-1,
    seed=0,
    chunk_memory_gb=_DEFAULT_CHUNK_MEMORY_GB,
):
    rng = np.random.default_rng(seed)
    n_cells = covariate_matrix.shape[0]
    B_total = B1 + B2 + B3

    gene_precomps = fit_all_genes(
        response_matrix, gene_ids, covariate_matrix, chunk_memory_gb=chunk_memory_gb
    )
    gene_row = {g: i for i, g in enumerate(gene_ids)}

    pairs_by_target = {t: list(g["response_id"]) for t, g in pairs.groupby("grna_target")}
    target_ids = [t for t in grna_target_cells if t in pairs_by_target]

    # The chunk now sizes only the binomial fit -- the draws are no longer held
    # across it, so they do not enter the budget.
    chunk = chunk_size_for_budget(irls_bytes_per_column(n_cells), len(target_ids), chunk_memory_gb)

    rows = {}
    for start in range(0, len(target_ids), chunk):
        chunk_ids = target_ids[start : start + chunk]
        Y = np.zeros((len(chunk_ids), n_cells))
        for k, t in enumerate(chunk_ids):
            Y[k, grna_target_cells[t]] = 1.0
        fit = fit_binomial_glm_batch(covariate_matrix, Y)
        probs = fit.fitted_values  # (chunk, n_cells) -- the only thing retained
        del Y, fit

        for k, target_id in enumerate(chunk_ids):
            # Drawn in the same target order as the batched implementation, so
            # the RNG stream is identical.
            synthetic_idxs = crt_index_sampler_fast(probs[k], B_total, rng)
            trt_idxs = grna_target_cells[target_id]

            for gene_id in pairs_by_target[target_id]:
                gene = gene_precomps[gene_id]
                y = _get_row(response_matrix, gene_row[gene_id])
                pieces = compute_precomputation_pieces(
                    y, covariate_matrix, gene.fitted_coefs, gene.theta
                )
                r = run_low_level_test_full(
                    y=y,
                    mu=pieces.mu,
                    a=pieces.a,
                    w=pieces.w,
                    D=pieces.D,
                    trt_idxs=trt_idxs,
                    synthetic_idxs=synthetic_idxs,
                    B1=B1,
                    B2=B2,
                    B3=B3,
                    fit_parametric_curve=True,
                    side_code=side_code,
                )
                rows[(gene_id, target_id)] = {
                    "response_id": gene_id,
                    "grna_target": target_id,
                    "p_value": r.p_value,
                    "fold_change": r.fold_change,
                    "se_fold_change": r.se_fold_change,
                    "pct_change": (r.fold_change - 1.0) * 100.0,
                    "z_orig": r.z_orig,
                    "stage": r.stage,
                }
                del pieces, y
            del synthetic_idxs  # the whole point: one target's draws at a time
        del probs

    ordered = [
        rows[(row.response_id, target_id)]
        for target_id in target_ids
        for row in pairs[pairs.grna_target == target_id].itertuples(index=False)
    ]
    return pd.DataFrame(ordered)


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    export_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else _DEFAULT_CHUNK_MEMORY_GB
    out_dir.mkdir(parents=True, exist_ok=True)

    export = load_export(export_dir)
    print(f"input: {export.describe()}")
    print(f"budget: {budget} GB (streaming prototype)")

    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = streamed_discovery(
            export.response_matrix,
            export.gene_ids,
            export.covariate_matrix,
            export.grna_target_cells,
            export.pairs,
            side_code={-1: -1, 0: 0, 1: 1}[export.metadata["side_code"]],
            seed=0,
            chunk_memory_gb=budget,
        )
    wall = time.perf_counter() - t0
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    raw = cpu1.ru_maxrss
    peak_gb = (raw if sys.platform == "darwin" else raw * 1024) / 1e9

    timing = {
        "step": "discovery_streamed",
        "wall_seconds": wall,
        "user_seconds": cpu1.ru_utime - cpu0.ru_utime,
        "sys_seconds": cpu1.ru_stime - cpu0.ru_stime,
        "peak_rss_gb": peak_gb,
        "n_rows": len(result),
        "chunk_memory_gb_budget": budget,
    }
    (out_dir / "timing_pysceptre_discovery_streamed.json").write_text(json.dumps(timing, indent=2))
    print(
        f"  wall {wall:.1f}s ({wall / 60:.2f} min), peak RSS {peak_gb:.2f} GB, {len(result):,} rows"
    )
    print(f"  stages: {result.stage.value_counts().sort_index().to_dict()}")
    result.to_parquet(out_dir / "pysceptre_discovery_streamed_result.parquet")


if __name__ == "__main__":
    main()
