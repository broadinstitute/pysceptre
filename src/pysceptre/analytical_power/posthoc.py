"""Per-pair analytical power for a completed screen.

Port of PerturbPlan's `compute_power_posthoc()` (`Katsevich-Lab/perturbplan`,
MIT -- see `THIRD_PARTY_LICENSES`), restricted to the post-hoc,
complement-control-group, explicit-cutoff path.

Given a screen that has already been analysed, it answers per pair: if this
element really did reduce this gene by X%, would this screen have detected it?
No simulation and no fitted parameters. None of its inputs is a property of
the *pair* -- expression and dispersion belong to the gene, cell counts to the
element -- so it answers that question for pairs the screen never tested too.

Several plausible generalisations are deliberately not offered, and several
arguments deliberately have no default. `docs/design.md`, "Analytical
per-pair power", says which and why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .closed_form import SIDES, qc_failure_prob, rejection_prob, test_stat_distribution

_NON_TARGETING = "non-targeting"


def target_cell_counts(cells_per_grna: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-gRNA cell counts to the per-target terms the formula needs.

    `cells_per_grna` needs columns `grna_target` and `num_cells`, one row per
    individual gRNA. Rows whose target is `"non-targeting"` are dropped, as
    R does -- NTCs are not a target and the complement control group does not
    use them.

    Returns `grna_target`, `num_trt_cells` (the **sum** over the target's
    gRNAs) and `num_trt_cells_sq` (the sum of their squares). The sum is not
    the size of the union of perturbed cells, and in high MOI the two differ.
    See `docs/design.md`, "Analytical per-pair power".
    """
    missing = [c for c in ("grna_target", "num_cells") if c not in cells_per_grna.columns]
    if missing:
        raise KeyError(f"cells_per_grna is missing column(s): {missing}")
    counts = cells_per_grna.loc[cells_per_grna["grna_target"] != _NON_TARGETING]
    n = counts["num_cells"].to_numpy(dtype=float)
    if np.any(n < 0) or not np.all(np.isfinite(n)):
        raise ValueError("cells_per_grna['num_cells'] must be finite and non-negative")
    agg = (
        counts.assign(num_cells=n, _sq=n**2)
        .groupby("grna_target", sort=False, as_index=False)
        .agg(num_trt_cells=("num_cells", "sum"), num_trt_cells_sq=("_sq", "sum"))
    )
    return agg


def compute_power_posthoc(
    discovery_pairs: pd.DataFrame,
    cells_per_grna: pd.DataFrame,
    baseline_expression_stats: pd.DataFrame,
    *,
    fold_change_mean: float,
    fold_change_sd: float,
    cutoff: float,
    num_total_cells: int,
    side: str = "left",
    n_nonzero_trt_thresh: int = 0,
    n_nonzero_cntrl_thresh: int = 0,
) -> pd.DataFrame:
    """Analytical power for each pair in `discovery_pairs`.

    Args:
        discovery_pairs: columns `grna_target` and `response_id`, the pairs to
            score. They need not be pairs the screen tested.
        cells_per_grna: one row per individual gRNA, columns `grna_target` and
            `num_cells` (and conventionally `grna_id`). Per-gRNA granularity
            is required, not optional: the across-gRNA variance term needs the
            sum of squared counts, which no target-level total can supply.
        baseline_expression_stats: one row per gene, columns `response_id`,
            `expression_mean` and `expression_size`. `expression_size` is the
            NB size -- theta, i.e. `1 / dispersion`. The validated
            `expression_mean` is the size-factor-normalised mean.
        fold_change_mean: a *multiplier*, not an effect size: a 15% knockdown
            is `0.85`.
        fold_change_sd: across-gRNA sd of the fold change. Required, with no
            default; see `docs/design.md`, "Analytical per-pair power".
        cutoff: the nominal p-value threshold this screen's test is read at.
            Required, and it must be *this* screen's threshold rather than a
            borrowed one, because it carries the multiple-testing correction;
            see `docs/design.md`, "Analytical per-pair power".
        num_total_cells: cells in the analysis. The complement control group
            is `num_total_cells - num_trt_cells`.
        side: `"left"` (the default, paired with `cutoff = alpha / 2`),
            `"right"`, or `"both"` (paired with `cutoff = alpha`). The default
            is the configuration the estimator was validated in.
        n_nonzero_trt_thresh: sceptre's pairwise nonzero-count threshold for
            the treated cells. Defaults to 0, which makes the QC factor 1 --
            correct for pairs that already passed QC. Raise it to score pairs
            that were never tested.
        n_nonzero_cntrl_thresh: the same, for the complement group.

    Returns:
        `discovery_pairs` with `power` appended, in the order supplied, plus
        the inputs each row's estimate was built from (`num_trt_cells`,
        `num_trt_cells_sq`, `num_cntrl_cells`, `expression_mean`,
        `expression_size`, `mean_test_stat`, `sd_test_stat`, `qc_failure_prob`)
        so a surprising number can be traced without recomputing it.

    Raises:
        KeyError: a pair names a target with no gRNA counts, or a gene with no
            baseline statistics. Never filled in silently: a missing input
            would otherwise become a NaN power indistinguishable from a real
            one.
        ValueError: an argument is out of range, a key is duplicated, or the
            summed treated count leaves the complement group empty.
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {list(SIDES)}, got {side!r}")
    if not 0.0 < cutoff < 1.0:
        raise ValueError(f"cutoff must lie in (0, 1), got {cutoff!r}")
    if not np.isfinite(fold_change_mean) or fold_change_mean <= 0:
        raise ValueError(
            f"fold_change_mean is a multiplier and must be positive, got {fold_change_mean!r} "
            "(a 15% knockdown is 0.85, not 0.15)"
        )
    if not np.isfinite(fold_change_sd) or fold_change_sd < 0:
        raise ValueError(f"fold_change_sd must be finite and non-negative, got {fold_change_sd!r}")
    if n_nonzero_trt_thresh < 0 or n_nonzero_cntrl_thresh < 0:
        raise ValueError("nonzero-count thresholds must be non-negative")

    for frame, name, cols in (
        (discovery_pairs, "discovery_pairs", ("grna_target", "response_id")),
        (
            baseline_expression_stats,
            "baseline_expression_stats",
            ("response_id", "expression_mean", "expression_size"),
        ),
    ):
        absent = [c for c in cols if c not in frame.columns]
        if absent:
            raise KeyError(f"{name} is missing column(s): {absent}")

    # R joins these with `relationship = "many-to-one"`, which errors on a repeated key rather
    # than quietly taking one of them. Match that: a duplicated gene would silently pick a row,
    # and a duplicated gRNA would be counted twice into the target's sum.
    dup_genes = baseline_expression_stats["response_id"].duplicated()
    if dup_genes.any():
        offenders = sorted(baseline_expression_stats.loc[dup_genes, "response_id"].unique())
        raise ValueError(
            f"baseline_expression_stats has {len(offenders)} duplicated response_id(s), "
            f"including: {offenders[:5]}"
        )
    if "grna_id" in cells_per_grna.columns:
        dup_grnas = cells_per_grna["grna_id"].duplicated()
        if dup_grnas.any():
            offenders = sorted(cells_per_grna.loc[dup_grnas, "grna_id"].unique())
            raise ValueError(
                f"cells_per_grna has {len(offenders)} duplicated grna_id(s), which would be "
                f"counted twice into their target's cell count, including: {offenders[:5]}"
            )

    targets = target_cell_counts(cells_per_grna).set_index("grna_target")
    genes = baseline_expression_stats.set_index("response_id")

    out = discovery_pairs.loc[:, ["grna_target", "response_id"]].reset_index(drop=True)
    unknown_targets = sorted(set(out["grna_target"]) - set(targets.index))
    if unknown_targets:
        raise KeyError(
            f"{len(unknown_targets)} target(s) in discovery_pairs have no rows in "
            f"cells_per_grna, including: {unknown_targets[:5]}"
        )
    unknown_genes = sorted(set(out["response_id"]) - set(genes.index))
    if unknown_genes:
        raise KeyError(
            f"{len(unknown_genes)} gene(s) in discovery_pairs have no rows in "
            f"baseline_expression_stats, including: {unknown_genes[:5]}"
        )

    num_trt_cells = targets["num_trt_cells"].reindex(out["grna_target"]).to_numpy()
    num_trt_cells_sq = targets["num_trt_cells_sq"].reindex(out["grna_target"]).to_numpy()
    expression_mean = genes["expression_mean"].reindex(out["response_id"]).to_numpy(dtype=float)
    expression_size = genes["expression_size"].reindex(out["response_id"]).to_numpy(dtype=float)

    if np.any(expression_mean <= 0) or not np.all(np.isfinite(expression_mean)):
        raise ValueError("baseline_expression_stats['expression_mean'] must be finite and positive")
    if np.any(expression_size <= 0) or not np.all(np.isfinite(expression_size)):
        raise ValueError(
            "baseline_expression_stats['expression_size'] must be finite and positive "
            "(it is the NB size, theta = 1/dispersion, not the dispersion)"
        )

    num_cntrl_cells = float(num_total_cells) - num_trt_cells
    # The sum over gRNAs can exceed the cell count where cells carry several of a target's gRNAs,
    # which would silently make the control group empty or negative rather than merely wrong.
    if np.any(num_cntrl_cells <= 0) or np.any(num_trt_cells <= 0):
        raise ValueError(
            "every pair needs a non-empty treated and complement group, but "
            f"{int(np.sum(num_cntrl_cells <= 0))} pair(s) have num_total_cells "
            f"({num_total_cells}) at or below the summed per-gRNA treated count, and "
            f"{int(np.sum(num_trt_cells <= 0))} have no treated cells. Note that "
            "num_trt_cells is the sum over the target's gRNAs, not the union, so it can "
            "exceed the number of distinct perturbed cells."
        )

    mean_stat, sd_stat = test_stat_distribution(
        num_trt_cells=num_trt_cells,
        num_cntrl_cells=num_cntrl_cells,
        num_trt_cells_sq=num_trt_cells_sq,
        expression_mean=expression_mean,
        expression_size=expression_size,
        fold_change_mean=fold_change_mean,
        fold_change_sd=fold_change_sd,
    )
    p_fail_qc = qc_failure_prob(
        fold_change_mean=fold_change_mean,
        expression_mean=expression_mean,
        expression_size=expression_size,
        num_trt_cells=num_trt_cells,
        num_cntrl_cells=num_cntrl_cells,
        n_nonzero_trt_thresh=n_nonzero_trt_thresh,
        n_nonzero_cntrl_thresh=n_nonzero_cntrl_thresh,
    )
    power = rejection_prob(mean_stat, sd_stat, side=side, cutoff=cutoff) * (1.0 - p_fail_qc)

    out["num_trt_cells"] = num_trt_cells
    out["num_trt_cells_sq"] = num_trt_cells_sq
    out["num_cntrl_cells"] = num_cntrl_cells
    out["expression_mean"] = expression_mean
    out["expression_size"] = expression_size
    out["mean_test_stat"] = mean_stat
    out["sd_test_stat"] = sd_stat
    out["qc_failure_prob"] = p_fail_qc
    out["power"] = power
    return out
