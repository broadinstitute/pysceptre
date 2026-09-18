"""Recreate sceptre's four-panel calibration-check figure.

Ports `sceptre:::plot_run_calibration_check` (0.10.3), which lays out:

  A  QQ plot (bulk)    observed vs expected null p-value, both axes reversed
                       and linear, with a pointwise confidence band
  B  QQ plot (tail)    the same points on a reversed log10 axis, where the
                       small p-values that decide anything are legible
  C  Log fold changes  histogram of log-2 fold change, restricted to
                       |lfc| < 0.6, with a line at zero
  D  summary           false discoveries at alpha, and mean log-2 fold change

Faithful to the R construction, which was read out of the installed package
rather than guessed:

  * expected quantiles are `ppoints(n)` -- `(i - a) / (n + 1 - 2a)` with
    `a = 0.5` for `n > 10`, `3/8` otherwise -- not `i / (n + 1)`.
  * the band is **pointwise**, from the exact distribution of the r-th order
    statistic of n uniforms: `Beta(r, n + 1 - r)`, at `ci_level = 0.95`. It is
    not a simultaneous band, so excursions outside it are expected somewhere
    along a 34,886-point curve and only a sustained departure means anything.
  * observed p-values are clamped below at `1e-8` (R's `ymin`), so a p-value of
    zero plots at the axis edge instead of vanishing.
  * panel C filters to `|log_2_fold_change| < 0.6` and uses binwidth 0.02 when
    there are more than 10,000 pairs, 0.05 otherwise.

Three deliberate departures, all noted because the figure is meant to be
comparable to R's, not pixel-identical:

  * **Colours are Okabe-Ito, not firebrick2**, per CLAUDE.md. The repo requires
    a colourblind-safe palette and sceptre's red is not one.
  * **Every point is drawn.** R thins to one point per cell of a 500x500 grid,
    which is a renderer concession; matplotlib handles 34,886 points, and
    thinning a QQ plot's tail is exactly where information would be lost.
  * **Panel C's y axis is labelled "Count"**, which is what it holds. R labels
    it "Density" while plotting counts.

Usage:
  plot_calibration_check.py <result.parquet> [--compare <r_result.parquet>]
      [--export DIR] [--alpha A] [--out fig.png] [--title T]

`--export` reads R's stored calibration result from a dataset directory, so
the comparison needs no second file. With no comparison the figure is
sceptre's, for one result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

# Okabe-Ito, per CLAUDE.md.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
}
# R clamps observed p-values here so an exact zero still plots.
P_FLOOR = 1e-8


def ppoints(n: int, a: float | None = None) -> np.ndarray:
    """R's `stats::ppoints`: `(i - a) / (n + 1 - 2a)`, `a = 0.5` for n > 10."""
    if a is None:
        a = 0.5 if n > 10 else 3.0 / 8.0
    return (np.arange(1, n + 1) - a) / (n + 1 - 2 * a)


def qq_frame(p: np.ndarray, ci_level: float = 0.95) -> pd.DataFrame:
    """Sorted observed p-values with expected quantiles and a pointwise band.

    The band is the central `ci_level` interval of the r-th order statistic of
    n uniforms, which is exactly `Beta(r, n + 1 - r)` -- no approximation, and
    the same call R makes.
    """
    p = np.sort(np.clip(p[np.isfinite(p)], P_FLOOR, 1.0))
    n = p.size
    r = np.arange(1, n + 1)
    lo = (1 - ci_level) / 2
    return pd.DataFrame(
        {
            "expected": ppoints(n),
            "observed": p,
            "lower": stats.beta.ppf(lo, r, n + 1 - r),
            "upper": stats.beta.ppf(1 - lo, r, n + 1 - r),
        }
    )


def _log2_fc(frame: pd.DataFrame) -> np.ndarray | None:
    """log-2 fold change, computed from `fold_change` when not stored.

    pysceptre drops `log_2_fold_change` from its output: it is derivable, and
    `pct_change_es` is the effect size actually read off. sceptre's panel C is
    defined on the log-2 scale, so it is reconstructed here rather than the
    panel being silently redefined.
    """
    if "log_2_fold_change" in frame:
        return frame["log_2_fold_change"].to_numpy(dtype=float)
    if "fold_change" in frame:
        fc = frame["fold_change"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log2(np.where(fc > 0, fc, np.nan))
    return None


def _bh_rejections(p: np.ndarray, alpha: float) -> int:
    p = p[np.isfinite(p)]
    if not p.size:
        return 0
    return int(np.sum(stats.false_discovery_control(np.clip(p, 0, 1), method="bh") < alpha))


def _panel_qq(ax, series: dict[str, np.ndarray], *, tail: bool) -> None:
    colours = [OKABE_ITO["sky_blue"], OKABE_ITO["orange"], OKABE_ITO["bluish_green"]]
    band_drawn = False
    for (label, p), colour in zip(series.items(), colours, strict=False):
        q = qq_frame(p)
        if not band_drawn:
            # One band: it depends only on n, which the series share.
            ax.fill_between(
                _t(q.expected, tail),
                _t(q.lower, tail),
                _t(q.upper, tail),
                color=OKABE_ITO["black"],
                alpha=0.15,
                lw=0,
                label="95% pointwise band",
                zorder=0,
            )
            band_drawn = True
        ax.scatter(
            _t(q.expected, tail),
            _t(q.observed, tail),
            s=1.5,
            alpha=0.55,
            color=colour,
            edgecolors="none",
            label=f"{label} (n={q.shape[0]:,})",
        )

    if tail:
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, hi], [0, hi], color=OKABE_ITO["black"], lw=0.9, zorder=1)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        # R labels these axes with p-values via `revlog_trans` and leaves the
        # transform implicit. Labelled as the transform here instead: the axis
        # holds -log10(p), and saying so is clearer than printing 1e-4 at a
        # position of 4 and leaving the reader to infer the mapping.
        ax.set_xlabel("expected  $-\\log_{10}(p)$ under $U(0,1)$")
        ax.set_ylabel("observed  $-\\log_{10}(p)$")
        ax.set_title("QQ plot (tail)")
    else:
        ax.plot([0, 1], [0, 1], color=OKABE_ITO["black"], lw=0.9, zorder=1)
        # Reversed, as R does: p = 1 at the origin, small p away from it.
        ax.set_xlim(1, 0)
        ax.set_ylim(1, 0)
        ax.set_xlabel("expected $p$ under $U(0,1)$")
        ax.set_ylabel("observed $p$")
        ax.set_title("QQ plot (bulk)")
    ax.spines[["top", "right"]].set_visible(False)


def _t(p, tail: bool):
    """Axis transform: identity for the bulk panel, -log10 for the tail."""
    return -np.log10(np.clip(p, P_FLOOR, 1.0)) if tail else p


def _panel_lfc(ax, series: dict[str, np.ndarray], n_pairs: int) -> None:
    binwidth = 0.02 if n_pairs > 10_000 else 0.05
    edges = np.arange(-0.6, 0.6 + binwidth, binwidth)
    colours = [OKABE_ITO["sky_blue"], OKABE_ITO["orange"]]
    # First series filled, the rest as outlines on top. With two
    # implementations the distributions coincide almost exactly, so drawing
    # both as outlines hides one entirely and reads as a missing series.
    for i, ((label, lfc), colour) in enumerate(zip(series.items(), colours, strict=False)):
        lfc = lfc[np.isfinite(lfc)]
        inside = lfc[np.abs(lfc) < 0.6]
        if i == 0:
            ax.hist(
                inside,
                bins=edges,
                facecolor=colour if len(series) > 1 else "#E6E6E6",
                edgecolor=OKABE_ITO["black"],
                lw=0.4,
                alpha=0.85 if len(series) > 1 else 1.0,
                label=label,
            )
        else:
            ax.hist(
                inside,
                bins=edges,
                histtype="step",
                edgecolor=colour,
                lw=1.1,
                label=label,
            )
    ax.axvline(0.0, color=OKABE_ITO["vermillion"], lw=1.5)
    ax.set_title("Log fold changes")
    ax.set_xlabel("Estimated log-2 fold change")
    # R labels this "Density" but plots counts; labelled for what it holds.
    ax.set_ylabel("Count")
    ax.margins(y=0)
    ax.spines[["top", "right"]].set_visible(False)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=7)


def _panel_summary(ax, results: dict[str, pd.DataFrame], alpha: float) -> None:
    ax.axis("off")
    lines = []
    for label, frame in results.items():
        p = frame["p_value"].to_numpy(dtype=float)
        # R reads its own `significant` column; recomputed here so both sides
        # are judged by the same rule at the same alpha.
        n_rej = _bh_rejections(p, alpha)
        lfc = _log2_fc(frame)
        mean_lfc = float(np.nanmean(lfc)) if lfc is not None else float("nan")
        ks = stats.kstest(np.clip(p[np.isfinite(p)], 0, 1), "uniform")
        lines.append(
            f"{label.upper()}\n"
            f"False discoveries (BH, alpha={alpha:g}): {n_rej}\n"
            f"Mean log-2 fold change: {mean_lfc:.3g}\n"
            f"Median p-value: {np.nanmedian(p):.3f}\n"
            f"KS vs U(0,1): {ks.statistic:.4f}"
        )
    ax.text(
        0.02,
        0.95,
        "\n\n".join(lines),
        ha="left",
        va="top",
        fontsize=8.5,
        transform=ax.transAxes,
        linespacing=1.5,
    )


def figure(results: dict[str, pd.DataFrame], alpha: float, title: str | None):
    p_series = {k: v["p_value"].to_numpy(dtype=float) for k, v in results.items()}
    lfc_series = {k: lfc for k, v in results.items() if (lfc := _log2_fc(v)) is not None}
    n_pairs = max(len(v) for v in results.values())

    fig, axes = plt.subplots(2, 2, figsize=(9, 8.2), height_ratios=[0.55, 0.45])
    _panel_qq(axes[0][0], p_series, tail=False)
    _panel_qq(axes[0][1], p_series, tail=True)
    _panel_lfc(axes[1][0], lfc_series, n_pairs)
    _panel_summary(axes[1][1], results, alpha)
    axes[0][1].legend(frameon=False, fontsize=7, loc="upper left")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    argv = sys.argv[1:]

    def value(flag, cast, default=None):
        return cast(argv[argv.index(flag) + 1]) if flag in argv else default

    alpha = value("--alpha", float, 0.1)
    out = value("--out", str, "calibration_check.png")
    title = value("--title", str, None)

    results = {"pysceptre": pd.read_parquet(argv[0])}
    compare = value("--compare", str)
    export_dir = value("--export", str)
    if compare:
        results["R sceptre"] = pd.read_parquet(compare)
    elif export_dir:
        from sceptre_io import load_export

        export = load_export(export_dir, backed=True)
        if export.calibration_result is not None and "p_value" in export.calibration_result:
            results["R sceptre"] = export.calibration_result
        if hasattr(export.response_matrix, "close"):
            export.response_matrix.close()

    # Compare only where both tested the same pair, otherwise the panels
    # describe different pair sets and the overlay is meaningless.
    if len(results) > 1:
        key = ["response_id", "grna_target"]
        a, b = results["pysceptre"], results["R sceptre"]
        if all(c in a for c in key) and all(c in b for c in key):
            shared = a[key].merge(b[key], on=key)
            before = len(a), len(b)
            results["pysceptre"] = a.merge(shared, on=key)
            results["R sceptre"] = b.merge(shared, on=key)
            print(f"matched {len(shared):,} pairs (from {before[0]:,} and {before[1]:,})")
        else:
            print("NOTE: no shared pair key; panels describe different pair sets")

    fig = figure(results, alpha, title)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
