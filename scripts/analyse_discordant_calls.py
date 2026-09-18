"""Tabulate the pairs where pysceptre and R disagree on significance.

Both implementations are compared at R's own `multiple_testing_alpha` using
BH, and for every discordant pair the table reports each side's percent change
effect size with a 95% Wald interval.

The point of the effect columns: fold change and its standard error involve no
resampling, so they are deterministic given the inputs. If the two
implementations are analysing the same data they must agree on them to
numerical precision, whatever their p-values do. That turns the effect size
into an independent read on pairs the p-values disagree about.

Caveat worth carrying into any writeup: the Wald interval is a normal
approximation and is *not* sceptre's test -- sceptre exists partly because
parametric approximations are miscalibrated on this data. It is a useful
independent criterion, not ground truth.

Usage: analyse_discordant_calls.py <name> <export_dir> <pysceptre.parquet> <r.parquet> [...]
       (arguments repeat in groups of four, one per dataset)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from sceptre_io import load_export  # noqa: E402

from pysceptre.pipeline.discovery import _get_row, fit_all_genes  # noqa: E402
from pysceptre.precompute.pieces import compute_precomputation_pieces  # noqa: E402
from pysceptre.test_statistic.fold_change import estimate_log_fold_change  # noqa: E402

CI_Z = 1.959963984540054


def as_pct(fold_change: float) -> float:
    return (fold_change - 1.0) * 100.0


def discordant(name: str, export_dir: str, py_path: str, r_path: str) -> pd.DataFrame:
    export = load_export(export_dir)
    alpha = float(export.metadata["multiple_testing_alpha"])
    py = pd.read_parquet(py_path)
    r = pd.read_parquet(r_path)
    if "pass_qc" in r:
        r = r[r["pass_qc"].astype(bool)]
    merged = py.merge(r, on=["response_id", "grna_target"], suffixes=("_py", "_r"))

    sig_r = merged["significant"].astype(bool).to_numpy()
    sig_py = (
        stats.false_discovery_control(
            np.clip(merged["p_value_py"].to_numpy(), 0.0, 1.0), method="bh"
        )
        < alpha
    )
    disc = merged[sig_r != sig_py].copy()
    disc["called_by"] = np.where(sig_py[sig_r != sig_py], "pysceptre", "R")

    # Recompute pysceptre's effect size from the deterministic path rather than
    # relying on a stored run, so the two sides are demonstrably independent.
    genes = sorted(disc["response_id"].unique())
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = fit_all_genes(
            export.response_matrix[[export.gene_ids.index(g) for g in genes]],
            genes,
            export.covariate_matrix,
        )
    for _, pair in disc.iterrows():
        fit = fits[pair["response_id"]]
        y = _get_row(export.response_matrix, export.gene_ids.index(pair["response_id"]))
        pieces = compute_precomputation_pieces(
            y, export.covariate_matrix, fit.fitted_coefs, fit.theta
        )
        fc, se = estimate_log_fold_change(
            y, pieces.mu, export.grna_target_cells[pair["grna_target"]]
        )
        r_fc, r_se = pair["fold_change_r"], pair["se_fold_change"]
        rows.append(
            {
                "dataset": name,
                "response_id": pair["response_id"],
                "grna_target": pair["grna_target"],
                "called_by": pair["called_by"],
                "py_pct": as_pct(fc),
                "py_lo": as_pct(fc - CI_Z * se),
                "py_hi": as_pct(fc + CI_Z * se),
                "r_pct": as_pct(r_fc),
                "r_lo": as_pct(r_fc - CI_Z * r_se),
                "r_hi": as_pct(r_fc + CI_Z * r_se),
                "p_py": pair["p_value_py"],
                "p_r": pair["p_value_r"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = sys.argv[1:]
    if not args or len(args) % 4:
        print(__doc__)
        raise SystemExit(1)
    frames = [discordant(*args[i : i + 4]) for i in range(0, len(args), 4)]
    out = pd.concat(frames, ignore_index=True)
    out["py_excludes_zero"] = (out.py_lo > 0) | (out.py_hi < 0)
    out["r_excludes_zero"] = (out.r_lo > 0) | (out.r_hi < 0)

    def fmt(p, lo, hi):
        return f"{p:+8.2f}%  [{lo:+7.2f}, {hi:+7.2f}]"

    head = (
        f"{'#':>3}  {'data':5s} {'called by':9s}  {'pysceptre  % change [95% CI]':30s}  "
        f"{'R          % change [95% CI]':30s}  {'p (py)':>9}  {'p (R)':>9}"
    )
    print(head)
    print("-" * len(head))
    for i, row in out.iterrows():
        print(
            f"{i + 1:>3}  {row.dataset:5s} {row.called_by:9s}  "
            f"{fmt(row.py_pct, row.py_lo, row.py_hi):30s}  "
            f"{fmt(row.r_pct, row.r_lo, row.r_hi):30s}  {row.p_py:9.2e}  {row.p_r:9.2e}"
        )
    print("-" * len(head))
    worst = max(
        np.abs(out.py_pct - out.r_pct).max(),
        np.abs(out.py_lo - out.r_lo).max(),
        np.abs(out.py_hi - out.r_hi).max(),
    )
    print(
        f"  {int(out.py_excludes_zero.sum())}/{len(out)} pysceptre CIs and "
        f"{int(out.r_excludes_zero.sum())}/{len(out)} R CIs exclude zero.  "
        f"Max |pysceptre - R| over pct and both bounds: {worst:.1e}"
    )
    out.to_csv("discordant_calls.csv", index=False)
    print("  wrote discordant_calls.csv")


if __name__ == "__main__":
    main()
