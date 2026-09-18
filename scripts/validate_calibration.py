"""Validate pysceptre's calibration check against R's, and against uniformity.

Two questions, which need different evidence:

1. **Does the engine agree with R on the same pairs?** R's calibration pair
   selection is unseeded -- nothing in its calibration path calls `set.seed`
   and `sceptre_object` has no seed slot -- so re-running R gives a different
   pair set (two runs on one screen shared only ~20 of each group's 331 genes). A
   pair-by-pair comparison is therefore only meaningful when R's own pairs are
   fed back in, which is what `--inject` does. Fold change involves no
   resampling and should agree to near machine precision; p-values are
   stochastic on both sides and are compared distributionally.

2. **Is the check itself calibrated?** Negative control pairs have no real
   effect, so a correct implementation returns p-values uniform on (0, 1).
   This is the question R cannot answer for us, and the one the calibration
   check exists to ask. Reported as a KS statistic against U(0, 1) and a BH
   rejection count, with a QQ plot.

Usage:
  validate_calibration.py <export_dir> [--inject] [--construct] [--plot out.png]
      [--seed N] [--chunk-memory-gb G] [--eager] [--n-jobs N]

`--inject` uses R's stored pairs (from the export's `negative_control_pairs` /
`calibration_result`, or from `--r-result <parquet>` when the object was saved
before the calibration check ran); `--construct` builds pysceptre's own. Both
run if neither is given and R's pairs are available.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from sceptre_io import load_export  # noqa: E402

import pysceptre  # noqa: E402

# Okabe-Ito, per CLAUDE.md.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
}


def peak_rss_gb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (raw if sys.platform == "darwin" else raw * 1024) / 1e9


def r_pairs(export, external: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """R's own negative-control pairs, from whichever source carried them.

    An object saved *before* `run_calibration_check` has no pairs in it -- some exports
    is exported from its post-QC object, so its R result exists only as a
    separate parquet. `--r-result` supplies that case.
    """
    for frame in (external, export.negative_control_pairs, export.calibration_result):
        if frame is not None and len(frame):
            cols = [c for c in ("response_id", "grna_target") if c in frame]
            if len(cols) == 2:
                return frame[cols].reset_index(drop=True)
    return None


def uniformity(p: np.ndarray) -> dict:
    """How close the negative-control p-values are to U(0, 1).

    A calibration check's whole claim is that these are uniform, so this is the
    primary result, not a diagnostic. The KS test is two-sided against the
    exact uniform CDF; `median` and `frac_below_05` are reported alongside
    because a KS statistic alone does not say which direction a deviation runs.
    """
    p = p[np.isfinite(p)]
    ks = stats.kstest(p, "uniform")
    return {
        "n": int(p.size),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "median": float(np.median(p)),
        "mean": float(np.mean(p)),
        "frac_below_05": float(np.mean(p < 0.05)),
        "frac_below_01": float(np.mean(p < 0.01)),
    }


def bh_rejections(p: np.ndarray, alpha: float) -> int:
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 0
    return int(np.sum(stats.false_discovery_control(np.clip(p, 0, 1), method="bh") < alpha))


def run(export, pairs, *, seed, chunk_memory_gb, label, n_jobs=1) -> tuple[pd.DataFrame, dict]:
    meta = export.metadata
    kwargs = {"n_jobs": n_jobs}
    if chunk_memory_gb is not None:
        kwargs["chunk_memory_gb"] = chunk_memory_gb
    t0 = time.perf_counter()
    result = pysceptre.run_calibration_check(
        response_matrix=export.response_matrix,
        gene_ids=export.gene_ids,
        covariate_matrix=export.covariate_matrix,
        ntc_grna_cells=export.ntc_grna_cells,
        negative_control_pairs=pairs,
        n_calibration_pairs=len(pairs) if pairs is not None else meta["n_pairs"],
        calibration_group_size=int(meta.get("calibration_group_size", 15)),
        n_nonzero_trt_thresh=int(meta.get("n_nonzero_trt_thresh", 7)),
        n_nonzero_cntrl_thresh=int(meta.get("n_nonzero_cntrl_thresh", 7)),
        # R's p_hat. Only matters for construction, and only off the floor --
        # but there it decides the group count exactly (day0: 625 vs 598).
        pass_qc_rate=float(meta.get("discovery_pass_qc_rate") or 1.0),
        side=export.side,
        multiple_testing_alpha=float(meta.get("multiple_testing_alpha", 0.1)),
        seed=seed,
        **kwargs,
    )
    wall = time.perf_counter() - t0
    timing = {"label": label, "wall_seconds": wall, "peak_rss_gb": peak_rss_gb(), "n": len(result)}
    print(f"  {label}: {wall:.1f}s ({wall / 60:.2f} min), peak RSS {timing['peak_rss_gb']:.2f} GB")
    return result, timing


def compare_to_r(
    result: pd.DataFrame, export, alpha: float, external: pd.DataFrame | None = None
) -> dict | None:
    """Pair-by-pair against R, on pairs both sides tested."""
    r = external if external is not None else export.calibration_result
    if r is None or not len(r):
        return None
    key = ["response_id", "grna_target"]
    m = result.merge(r, on=key, suffixes=("_py", "_r"), how="inner")
    if not len(m):
        return None
    out = {"n_merged": int(len(m))}
    ok = np.isfinite(m.p_value_py) & np.isfinite(m.p_value_r)
    out["spearman_p"] = float(stats.spearmanr(m.p_value_py[ok], m.p_value_r[ok]).statistic)
    nl = lambda v: -np.log10(np.clip(v, 1e-300, None))  # noqa: E731
    out["pearson_neglog10_p"] = float(
        stats.pearsonr(nl(m.p_value_py[ok]), nl(m.p_value_r[ok])).statistic
    )
    if "fold_change_py" in m and "fold_change_r" in m:
        f = np.isfinite(m.fold_change_py) & np.isfinite(m.fold_change_r)
        out["max_abs_fc_diff"] = float(np.max(np.abs(m.fold_change_py[f] - m.fold_change_r[f])))
        out["pearson_fc"] = float(stats.pearsonr(m.fold_change_py[f], m.fold_change_r[f]).statistic)
    # Both sides' own calibration quality, on the same pairs.
    out["py_bh_rejections"] = bh_rejections(m.p_value_py.to_numpy(), alpha)
    out["r_bh_rejections"] = bh_rejections(m.p_value_r.to_numpy(), alpha)
    return out


def qq_plot(results: dict[str, np.ndarray], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = [OKABE_ITO["sky_blue"], OKABE_ITO["orange"], OKABE_ITO["bluish_green"]]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lim = 0.0
    for (label, p), colour in zip(results.items(), colours, strict=False):
        p = np.sort(p[np.isfinite(p)])
        n = p.size
        # Expected order statistics of U(0,1): i/(n+1), on the -log10 scale
        # where the tail that matters is legible.
        expected = -np.log10(np.arange(1, n + 1) / (n + 1))
        observed = -np.log10(np.clip(p, 1e-300, None))
        lim = max(lim, expected.max(), observed.max())
        ax.scatter(
            expected,
            observed,
            s=5,
            alpha=0.5,
            color=colour,
            label=f"{label} (n={n:,})",
            edgecolors="none",
        )
    ax.plot([0, lim], [0, lim], color=OKABE_ITO["black"], lw=1, ls="--", zorder=0)
    ax.set_xlabel("expected  $-\\log_{10}(p)$ under $U(0,1)$")
    ax.set_ylabel("observed  $-\\log_{10}(p)$")
    ax.set_title("Calibration check: negative-control p-values")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    export_dir = Path(sys.argv[1])
    argv = sys.argv[2:]

    def flag(name):
        return name in argv

    def value(name, cast, default=None):
        return cast(argv[argv.index(name) + 1]) if name in argv else default

    seed = value("--seed", int, 0)
    n_jobs = value("--n-jobs", int, 1)
    r_result_path = value("--r-result", str, None)
    external = pd.read_parquet(r_result_path) if r_result_path else None
    if external is not None:
        print(f"R result from {r_result_path}: {len(external):,} rows")
    chunk_memory_gb = value("--chunk-memory-gb", float, None)

    # Backed by default: an all-genes dataset is mostly genes a given run never
    # reads, and holding them costs 0.86 GB for nothing on a transcriptome-wide screen.
    backed = "--eager" not in argv
    export = load_export(export_dir, backed=backed)
    print(f"reading {'backed' if backed else 'eagerly'}")
    print(f"input: {export.describe()}")
    if not export.ntc_grna_cells:
        raise SystemExit(
            "export carries no individual NTC gRNAs; re-export with a build that writes "
            "the grna assay's ntc_grna units (see scripts/sceptre_export_lib.R)"
        )
    alpha = float(export.metadata.get("multiple_testing_alpha", 0.1))

    injected = r_pairs(export, external)
    do_inject = flag("--inject") or (not flag("--construct") and injected is not None)
    do_construct = flag("--construct") or not flag("--inject")

    curves: dict[str, np.ndarray] = {}
    report: dict = {"export": str(export_dir), "alpha": alpha, "seed": seed}

    if do_inject:
        if injected is None:
            raise SystemExit("--inject given but the export carries no R pairs")
        print(f"\n== injected: R's own {len(injected):,} pairs ==")
        result, timing = run(
            export,
            injected,
            seed=seed,
            chunk_memory_gb=chunk_memory_gb,
            label="injected",
            n_jobs=n_jobs,
        )
        result.to_parquet(export_dir / "pysceptre_calibration_injected.parquet")
        report["injected"] = {
            "timing": timing,
            "uniformity": uniformity(result.p_value.to_numpy()),
            "bh_rejections": bh_rejections(result.p_value.to_numpy(), alpha),
            "vs_r": compare_to_r(result, export, alpha, external),
        }
        curves["pysceptre (R's pairs)"] = result.p_value.to_numpy()
        r_frame = external if external is not None else export.calibration_result
        if r_frame is not None and "p_value" in r_frame:
            curves["R sceptre"] = r_frame.p_value.to_numpy()

    if do_construct:
        print("\n== constructed: pysceptre's own pairs ==")
        result, timing = run(
            export,
            None,
            seed=seed,
            chunk_memory_gb=chunk_memory_gb,
            label="constructed",
            n_jobs=n_jobs,
        )
        result.to_parquet(export_dir / "pysceptre_calibration_constructed.parquet")
        report["constructed"] = {
            "timing": timing,
            "uniformity": uniformity(result.p_value.to_numpy()),
            "bh_rejections": bh_rejections(result.p_value.to_numpy(), alpha),
        }
        curves["pysceptre (own pairs)"] = result.p_value.to_numpy()

    print("\n" + "=" * 64)
    print(json.dumps(report, indent=2, default=float))
    (export_dir / "calibration_validation.json").write_text(
        json.dumps(report, indent=2, default=float)
    )
    if "--plot" in argv and curves:
        qq_plot(curves, argv[argv.index("--plot") + 1])


if __name__ == "__main__":
    main()
