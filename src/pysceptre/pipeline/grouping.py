"""gRNA integration strategies: what counts as one treated unit.

sceptre tests whatever it calls a `grna_group`, and
`grna_integration_strategy` only decides what a group is
(`update_dfs_based_on_grouping_strategy`):

* `union` -- a group is a target, and its cells are the union of its guides';
* `singleton` -- a group is an individual guide, tested on its own cells;
* `bonferroni` -- tested per guide exactly as `singleton`, then aggregated
  back to the target (`apply_grouping_to_result`).

**Nothing statistical changes between them.** The same CRT, the same score
statistic, the same skew-normal escalation; only the treated cell set differs,
and in `bonferroni`'s case what happens to the results afterwards. That is why
this module holds only pair bookkeeping and one aggregation, and the engine
never learns which strategy is in play.

See `docs/design.md`, "gRNA integration strategies".
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

__all__ = ["singleton_pairs", "aggregate_bonferroni"]

_NON_TARGETING = "non-targeting"


def singleton_pairs(
    pairs: pd.DataFrame,
    grna_target_data_frame: pd.DataFrame,
    *,
    drop_duplicate_design_rows: bool = False,
) -> pd.DataFrame:
    """Fan each (response, target) pair out to one row per guide of that target.

    Args:
        pairs: columns `response_id` and `grna_target`, the union-shaped pair
            list, which is the same input R takes.
        grna_target_data_frame: the screen's design, one row per
            (`grna_id`, `grna_target`).
        drop_duplicate_design_rows: collapse a guide listed more than once
            against the same target. The default keeps it, which is what R
            does; either way a warning names the count. See the note below on
            why this is a choice and not a fix.

    Returns:
        `response_id`, `grna_id`, `grna_target`, one row per pair per guide of
        that pair's target, in the order R produces.

    The join is **many-to-many**, matching R's, because the design map is: a
    guide inside two overlapping candidate elements belongs to both targets
    and has to appear under each. On one real screen 1,673 of 43,736 guides
    do, and 36,450 pairs expand to 515,972.

    Non-targeting guides are never expanded. R keeps their `grna_group` as
    `"non-targeting"` rather than their id, and since a discovery pair names
    a target rather than a guide they cannot enter the output anyway.

    A **duplicated** `(grna_id, grna_target)` row is kept by default,
    matching R, and warned about either way. It is not cosmetic: such a guide
    is expanded twice, so it appears twice in the result and is counted twice
    in `aggregate_bonferroni`'s correction factor, which inflates the
    corrected p-value. `drop_duplicate_design_rows=True` collapses them, at
    the cost of no longer reproducing R's row count. Note that the `union`
    strategy is immune either way: R takes `unique()` of the treated cells, so
    a guide listed twice contributes the same set once.

    Raises:
        KeyError: a column is missing, or a pair names a target with no guides
            in the design. **R differs here**: its left join leaves
            `grna_group` as `NA` and keeps the row, which then looks for the
            cells of a group called `NA`. An untestable row that survives into
            a result is the failure this package refuses everywhere else, so
            it is refused here too.
    """
    for frame, name, cols in (
        (pairs, "pairs", ("response_id", "grna_target")),
        (grna_target_data_frame, "grna_target_data_frame", ("grna_id", "grna_target")),
    ):
        absent = [c for c in cols if c not in frame.columns]
        if absent:
            raise KeyError(f"{name} is missing column(s): {absent}")

    design = grna_target_data_frame.loc[:, ["grna_id", "grna_target"]].astype(str)
    design = design[design["grna_target"] != _NON_TARGETING]

    # **Duplicated design rows are kept, because R keeps them**, and that is a result and not
    # just a row count: a guide listed twice against one target is counted twice in
    # `aggregate_bonferroni`'s `sum(pass_qc)` factor, so the corrected p-value is inflated. Real
    # designs have them -- day0 has 36 duplicate rows of 45,463, which fan out to 288 duplicate
    # pairs. Deduplicating here was the first implementation and it disagreed with R by exactly
    # those 288 rows. Warned about rather than silently propagated, since it is a defect in the
    # design table that the caller can fix and nothing else will tell them.
    n_duplicated = int(design.duplicated().sum())
    if n_duplicated and drop_duplicate_design_rows:
        design = design.drop_duplicates()
        warnings.warn(
            f"dropped {n_duplicated} duplicated (grna_id, grna_target) row(s) from "
            "grna_target_data_frame. The expansion no longer matches R's row count, which is "
            "what drop_duplicate_design_rows=True asks for.",
            stacklevel=2,
        )
    elif n_duplicated:
        warnings.warn(
            f"grna_target_data_frame has {n_duplicated} duplicated (grna_id, grna_target) "
            f"row(s) of {len(design)}. They are kept, matching R, so those pairs are expanded "
            "more than once and a bonferroni correction counts those guides more than once, "
            "inflating the corrected p-value. Pass drop_duplicate_design_rows=True to collapse "
            "them, or drop them from the design yourself. The union strategy is unaffected "
            "either way.",
            stacklevel=2,
        )

    wanted = pairs.loc[:, ["response_id", "grna_target"]].astype(str)
    orphans = sorted(set(wanted["grna_target"]) - set(design["grna_target"]))
    if orphans:
        raise KeyError(
            f"{len(orphans)} target(s) in pairs have no guides in grna_target_data_frame, "
            f"including: {orphans[:5]}. Under a singleton strategy every pair is expanded to "
            "its target's guides, so such a pair has nothing to test."
        )

    out = wanted.merge(design, on="grna_target", how="left")
    return out.loc[:, ["response_id", "grna_id", "grna_target"]].reset_index(drop=True)


def aggregate_bonferroni(result: pd.DataFrame) -> pd.DataFrame:
    """Collapse a per-guide result back to one row per (response, target).

    Port of the `bonferroni` branch of `apply_grouping_to_result`. Within each
    (`response_id`, `grna_target`) group:

    * if **no** guide passed QC, the row is a non-result: `pass_qc = False`,
      a NaN p-value, and the *largest* nonzero counts of the group, which is
      the most favourable evidence there was and so the honest thing to report
      about why nothing could be tested;
    * otherwise the smallest p-value is taken, multiplied by the number of
      guides that passed QC and capped at 1, and every other reported quantity
      comes from that same guide's row.

    `grna_id` is dropped, because the aggregate is not about one guide. Every
    other column pysceptre reports is carried from the smallest-p guide, which
    is what R does for the fold change and the counts.

    Args:
        result: a singleton result, with `response_id`, `grna_target`,
            `p_value` and `pass_qc`.

    Returns:
        One row per (`response_id`, `grna_target`).
    """
    absent = [c for c in ("response_id", "grna_target", "p_value") if c not in result.columns]
    if absent:
        raise KeyError(f"result is missing column(s): {absent}")

    # `pass_qc` is not a column `run_discovery_analysis` produces: it is handed pairs that have
    # already passed QC, so every row it returns was testable. When the column is absent every
    # row therefore counts as passing, and the Bonferroni factor is the number of guides in the
    # group. Supply it -- `pipeline/pairwise_qc.py` computes it at guide resolution -- to get R's
    # behaviour for a target whose guides individually fail, which is the case this cannot see.
    result = result.copy()
    synthesised_pass_qc = "pass_qc" not in result.columns
    if synthesised_pass_qc:
        result["pass_qc"] = True

    carried = [
        c
        for c in result.columns
        if c not in ("response_id", "grna_target", "grna_id")
        and not (synthesised_pass_qc and c == "pass_qc")
    ]
    count_cols = [c for c in ("n_nonzero_trt", "n_nonzero_cntrl") if c in result.columns]

    rows = []
    for (response_id, grna_target), group in result.groupby(
        ["response_id", "grna_target"], sort=False
    ):
        passing = group[group["pass_qc"].astype(bool)]
        row = {"response_id": response_id, "grna_target": grna_target}
        if passing.empty:
            for c in carried:
                row[c] = np.nan
            if not synthesised_pass_qc:
                row["pass_qc"] = False
            # The best evidence in the group, so the numbers say how close it came.
            for c in count_cols:
                row[c] = group[c].max()
        else:
            best = passing.loc[passing["p_value"].idxmin()]
            for c in carried:
                row[c] = best[c]
            if not synthesised_pass_qc:
                row["pass_qc"] = True
            row["p_value"] = min(len(passing) * float(best["p_value"]), 1.0)
        rows.append(row)

    columns = ["response_id", "grna_target"] + carried
    return pd.DataFrame(rows, columns=columns)


def sort_like_r(result: pd.DataFrame) -> pd.DataFrame:
    """Order a singleton or bonferroni result the way R leaves it.

    `setorderv(c("p_value", "response_id"), na.last = TRUE)`. R applies this
    only on the singleton and bonferroni branches; a union result comes back
    in whatever order the engine produced, so this is not applied there.
    """
    return result.sort_values(
        ["p_value", "response_id"], na_position="last", kind="stable"
    ).reset_index(drop=True)
