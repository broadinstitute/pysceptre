"""Compare a pysceptre discovery result against R sceptre's, pair by pair.

Both sides must have been run with the SAME resampling mechanism for this to
mean anything. CRT p-values are stochastic and pysceptre's RNG is deliberately
not bit-compatible with R's (different algorithm and seeding), so the target is
distributional agreement, not equality: rank correlation on p-values, and exact
agreement on fold changes, which involve no resampling at all.

Usage:
  compare_pysceptre_vs_r.py <pysceptre_result> <r_result>
      [--alpha A] [--export DIR] [--plot out.png]

`--alpha` must match the sceptre object's `multiple_testing_alpha`; pass
`--export` to read it from an export's metadata.json instead of assuming.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Okabe-Ito, per CLAUDE.md: colorblind-safe categorical palette.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
STAGE_COLORS = {1: OKABE_ITO["sky_blue"], 2: OKABE_ITO["orange"], 3: OKABE_ITO["vermillion"]}


def load(py_path: str, r_path: str) -> pd.DataFrame:
    py = pd.read_parquet(py_path)
    r = pd.read_csv(r_path) if r_path.endswith(".csv") else pd.read_parquet(r_path)
    key = ["response_id", "grna_target"]
    for frame, name in ((py, "pysceptre"), (r, "R")):
        missing = [c for c in key if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} result is missing {missing}; has {list(frame.columns)}")
    merged = py.merge(r, on=key, suffixes=("_py", "_r"), how="inner")
    print(f"pysceptre {len(py):,} pairs | R {len(r):,} pairs | merged {len(merged):,}")
    if len(merged) < min(len(py), len(r)):
        print(f"  NOTE: {min(len(py), len(r)) - len(merged):,} pairs did not match on {key}")
    return merged


def compare(m: pd.DataFrame, alpha: float = 0.1) -> dict:
    p_py, p_r = m["p_value_py"].to_numpy(), m["p_value_r"].to_numpy()
    ok = np.isfinite(p_py) & np.isfinite(p_r)
    out = {"n": int(ok.sum())}

    out["spearman_p"] = float(stats.spearmanr(p_py[ok], p_r[ok]).statistic)
    # Compare on the -log10 scale too: that is where hits live, and a rank
    # correlation over 33k mostly-null pairs is dominated by the null bulk.
    nl_py = -np.log10(np.clip(p_py[ok], 1e-300, None))
    nl_r = -np.log10(np.clip(p_r[ok], 1e-300, None))
    out["pearson_neglog10_p"] = float(stats.pearsonr(nl_py, nl_r).statistic)

    # Compare fold change itself rather than a log of it: it is the quantity
    # both sides compute, it involves no resampling, and agreement here is the
    # check that the two are analysing identical inputs.
    if "fold_change_py" in m and "fold_change_r" in m:
        f_py = m["fold_change_py"].to_numpy()[ok]
        f_r = m["fold_change_r"].to_numpy()[ok]
        good = np.isfinite(f_py) & np.isfinite(f_r)
        out["pearson_fc"] = float(stats.pearsonr(f_py[good], f_r[good]).statistic)
        out["max_abs_fc_diff"] = float(np.max(np.abs(f_py[good] - f_r[good])))

    out["tables"] = {}
    # R's own `significant` column (BH at its multiple_testing_alpha) is the
    # scientifically meaningful call, so prefer it when present. pysceptre
    # deliberately applies no multiple-testing correction, so BH is applied
    # here at the same level to make the two comparable.
    if "significant" in m:
        # Verified against this dataset: scipy's false_discovery_control
        # reproduces R's p.adjust(method="BH") exactly (max |diff| 0.0), R
        # compares with `<`, and both yield the same rejection set. The alpha
        # must match the sceptre object's multiple_testing_alpha -- it is a
        # parameter here rather than a constant so a dataset configured
        # differently cannot silently mismatch.
        call_r = m["significant"].fillna(False).to_numpy(dtype=bool)[ok]
        out["alpha"] = alpha
        out["tables"][f"BH {alpha:g} (R's own call)"] = _contingency(_bh(p_py[ok], alpha), call_r)
    for thresh in (1e-2, 1e-4, 1e-6):
        out["tables"][f"p < {thresh:g}"] = _contingency(p_py[ok] < thresh, p_r[ok] < thresh)
    return out


def _bh(p: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg rejections at `alpha`."""
    return stats.false_discovery_control(np.clip(p, 0.0, 1.0), method="bh") < alpha


def _contingency(call_py: np.ndarray, call_r: np.ndarray) -> dict:
    """2x2 agreement on significance calls, treating R as the reference.

    Reported as rates rather than a single index (a Jaccard coefficient
    collapses the table and cannot distinguish calling extra hits from
    missing hits, which are different failures).

    These are *discordance* rates, not true Type I/II error rates: R is a
    reference implementation, not ground truth, and both sides estimate the
    same stochastic quantity from different random draws. A pair they disagree
    on is not necessarily one either got wrong.
    """
    tp = int(np.sum(call_py & call_r))
    fp = int(np.sum(call_py & ~call_r))
    fn = int(np.sum(~call_py & call_r))
    tn = int(np.sum(~call_py & ~call_r))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
        "false_negative_rate": fn / (fn + tp) if fn + tp else float("nan"),
        "n": tp + fp + fn + tn,
    }


def report(m: pd.DataFrame, stats_out: dict) -> None:
    print(f"\n{'=' * 62}\nPAIR-BY-PAIR AGREEMENT  (n = {stats_out['n']:,})\n{'=' * 62}")
    print(f"  Spearman rho, p-values          : {stats_out['spearman_p']:.4f}")
    print(f"  Pearson r, -log10(p)            : {stats_out['pearson_neglog10_p']:.4f}")
    if "pearson_fc" in stats_out:
        print(f"  Pearson r, fold change          : {stats_out['pearson_fc']:.6f}")
        print(f"  max |fold change difference|    : {stats_out['max_abs_fc_diff']:.3e}")
    for label, t in stats_out["tables"].items():
        print(f"\n  Significance calls -- {label}   (R as reference)")
        print(f"    {'':22s} {'R: sig':>9} {'R: not sig':>11}")
        print(f"    {'pysceptre: sig':22s} {t['tp']:>9,} {t['fp']:>11,}")
        print(f"    {'pysceptre: not sig':22s} {t['fn']:>9,} {t['tn']:>11,}")
        print(f"    sensitivity (R's hits recovered) : {t['sensitivity']:.4f}")
        print(f"    specificity                      : {t['specificity']:.4f}")
        print(
            f"    discordance, pysceptre-only (~I) : {t['false_positive_rate']:.2e}"
            f"   ({t['fp']:,} pairs)"
        )
        print(
            f"    discordance, R-only        (~II) : {t['false_negative_rate']:.4f}"
            f"   ({t['fn']:,} pairs)"
        )

    if "stage" in m:
        print("\n  pysceptre stage distribution:")
        for stage, n in m["stage"].value_counts().sort_index().items():
            print(f"    stage {stage}: {n:,}")

    disagree = m.assign(
        d=np.abs(
            -np.log10(np.clip(m["p_value_py"], 1e-300, None))
            + np.log10(np.clip(m["p_value_r"], 1e-300, None))
        )
    ).nlargest(5, "d")
    print("\n  Largest -log10(p) disagreements:")
    cols = ["response_id", "grna_target", "p_value_py", "p_value_r"]
    if "stage" in disagree:
        cols.append("stage")
    print(disagree[cols].to_string(index=False, float_format=lambda v: f"{v:.3e}"))


def plot(m: pd.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nl_py = -np.log10(np.clip(m["p_value_py"], 1e-300, None))
    nl_r = -np.log10(np.clip(m["p_value_r"], 1e-300, None))
    fig, ax = plt.subplots(figsize=(6, 6))
    if "stage" in m:
        for stage, colour in STAGE_COLORS.items():
            sel = m["stage"] == stage
            if sel.any():
                ax.scatter(
                    nl_r[sel],
                    nl_py[sel],
                    s=6,
                    alpha=0.5,
                    color=colour,
                    label=f"stage {stage} (n={int(sel.sum()):,})",
                    edgecolors="none",
                )
        ax.legend(frameon=False, loc="upper left")
    else:
        ax.scatter(nl_r, nl_py, s=6, alpha=0.5, color=OKABE_ITO["sky_blue"], edgecolors="none")
    lim = max(nl_py.max(), nl_r.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=OKABE_ITO["black"], lw=1, ls="--", zorder=0)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("R sceptre  $-\\log_{10}(p)$")
    ax.set_ylabel("pysceptre  $-\\log_{10}(p)$")
    ax.set_title("Discovery p-values, moi5 (CRT, both sides)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\n  wrote {path}")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)
    merged = load(sys.argv[1], sys.argv[2])
    alpha = None
    if "--alpha" in sys.argv:
        alpha = float(sys.argv[sys.argv.index("--alpha") + 1])
    elif "--export" in sys.argv:
        import json

        meta = json.loads(
            (Path(sys.argv[sys.argv.index("--export") + 1]) / "metadata.json").read_text()
        )
        alpha = float(meta["multiple_testing_alpha"])
        print(f"  alpha {alpha:g} read from the export metadata")
    if alpha is None:
        alpha = 0.1
        print(f"  NOTE: assuming alpha={alpha:g}; pass --alpha or --export to be sure")
    stats_out = compare(merged, alpha=alpha)
    report(merged, stats_out)
    if "--plot" in sys.argv:
        plot(merged, sys.argv[sys.argv.index("--plot") + 1])


if __name__ == "__main__":
    main()
