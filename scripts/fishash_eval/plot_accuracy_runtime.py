"""The collaborator's two-panel figure, re-made from our runs.

    python scripts/fishash_eval/plot_accuracy_runtime.py <tables_dir> <figures_dir>

Panel A: median F1 (all guide x cell entries) against the mean number of infections per cell,
on the fishash paper's varyMOI simulations, with the interquartile range over replicates as a
band. Panel B: median seconds per dataset against the number of guides, on the high-gRNA
varyNumGuides simulations, log-log. Methods that have not run yet are left out. The values read
off the collaborator's figure are overlaid as black markers. Okabe-Ito colours; Python solid,
R dashed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from score import (  # noqa: E402
    COLLABORATOR_FISHASH_PLUS,
    COLLABORATOR_MIXTURE,
    COLLABORATOR_MOIS,
    HIGH,
    VARY_MOI,
)

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
# run name -> (label, colour, line style)
RUNS = {
    "py_fishash_refit10": ("fishash, refit 10 (Python)", OKABE_ITO["blue"], "-"),
    "r_fishash_refit10": ("fishash, refit 10 (R)", OKABE_ITO["blue"], "--"),
    "py_fishash_refit0": ("fishash, refit 0 (Python)", OKABE_ITO["sky_blue"], "-"),
    "r_fishash_refit0": ("fishash, refit 0 (R)", OKABE_ITO["sky_blue"], "--"),
    "py_mixture_prob0.8": ("sceptre mixture (Python)", OKABE_ITO["vermillion"], "-"),
    "r_sceptre_mixture": ("sceptre mixture (R)", OKABE_ITO["vermillion"], "--"),
    "crispat_gauss": ("crispat Gaussian mixture", OKABE_ITO["bluish_green"], "-"),
    "py_gmm_raw": ("per-guide GMM", OKABE_ITO["reddish_purple"], "-"),
    "py_cmo_clr_q95": ("CMO CLR + 95th percentile", OKABE_ITO["orange"], "-"),
}


def mean_infections(moi: np.ndarray) -> np.ndarray:
    return moi / (1 - np.exp(-moi)) * 0.9


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    tables, figures = Path(argv[0]), Path(argv[1])
    figures.mkdir(parents=True, exist_ok=True)
    summary = pd.read_parquet(tables / "summary.parquet")
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.6))

    moi = summary[summary["scenario"] == VARY_MOI]
    for run, (label, colour, style) in RUNS.items():
        d = moi[moi["run"] == run].sort_values("moi")
        if len(d) < 2:
            continue
        x = d["mean_infections"].to_numpy()
        ax_a.plot(x, d["f1_median"], style, color=colour, marker="o", ms=3, label=label)
        if style == "-":
            ax_a.fill_between(x, d["f1_q1"], d["f1_q3"], color=colour, alpha=0.15, lw=0)
    xc = mean_infections(np.asarray(COLLABORATOR_MOIS))
    ax_a.plot(
        xc,
        COLLABORATOR_FISHASH_PLUS,
        "^",
        color=OKABE_ITO["black"],
        label='collaborator "fishash+"',
    )
    ax_a.plot(
        xc,
        COLLABORATOR_MIXTURE,
        "s",
        mfc="none",
        color=OKABE_ITO["black"],
        label="collaborator mixture",
    )
    ax_a.set_xscale("log")
    ax_a.set_xticks(xc, [f"{v:.2f}" for v in xc], rotation=45)
    ax_a.minorticks_off()
    ax_a.set_xlabel("mean infections per cell (MOI 0.1 to 10)")
    ax_a.set_ylabel("median F1, all guide x cell entries")
    ax_a.set_title("A. Accuracy across MOI (200 guides, 20k cells)")
    ax_a.legend(fontsize=7, loc="lower left")

    ng = summary[summary["scenario"] == HIGH]
    for run, (label, colour, style) in RUNS.items():
        d = ng[ng["run"] == run].sort_values("nguides")
        if len(d) < 2:
            continue
        ax_b.plot(
            d["nguides"], d["seconds_median"], style, color=colour, marker="o", ms=3, label=label
        )
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel("number of guides (20k cells)")
    ax_b.set_ylabel("seconds per dataset (median)")
    ax_b.set_title("B. Runtime (accuracy runs; controlled timing pending)")
    ax_b.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures / f"accuracy_runtime.{ext}", dpi=160)
    print("wrote", figures / "accuracy_runtime.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
