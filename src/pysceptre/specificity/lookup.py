"""The TSS lookup and the detour check (part 3 of the check).

Any target within a window of a measured gene's TSS knocks that gene down,
whether or not the design labelled it a TSS control. Its effects on every other
measured gene give a gene-to-gene table, and a far link E -> H is a detour when
E also lowers some G whose TSS knockdown lowers H. See `docs/design.md`,
"Specificity check".
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd

from .background import called_down, element_table, tested_pairs

__all__ = ["DETOUR_STATUSES", "detour_check", "gene_lookup", "tss_targets"]

DETOUR_STATUSES = ("detour", "detour (nominal)", "no detour", "can't check")


def tss_targets(
    targets: Collection[str],
    element_positions: pd.DataFrame,
    gene_positions: pd.DataFrame,
    measured_genes: Collection[str],
    *,
    window: float = 1_000.0,
) -> pd.DataFrame:
    """Targets within `window` bp of a measured gene's TSS, and that gene.

    A target near several measured TSSs is assigned the nearest.

    Returns:
        `grna_target`, `tss_gene`, `tss_distance`.

    Raises:
        KeyError: a target has no position.
    """
    elements = element_table(element_positions)
    targets = pd.Index(pd.unique(np.asarray(list(targets), dtype=object)))
    missing = sorted(set(targets) - set(elements.index))
    if missing:
        raise KeyError(
            f"element_positions has no entry for {len(missing)} target(s): {missing[:5]}"
        )
    measured = gene_positions.loc[
        gene_positions["response_id"].isin(set(measured_genes)), ["response_id", "chrom", "tss"]
    ]
    t = elements.loc[targets].rename_axis("grna_target").reset_index()
    m = t.merge(measured, on="chrom")
    m["tss_distance"] = (m["centre"] - m["tss"]).abs()
    m = (
        m.loc[m["tss_distance"] <= window]
        .sort_values(["tss_distance", "grna_target", "response_id"], kind="mergesort")
        .drop_duplicates("grna_target")
    )
    return (
        m[["grna_target", "response_id", "tss_distance"]]
        .rename(columns={"response_id": "tss_gene"})
        .reset_index(drop=True)
    )


def gene_lookup(
    tss: pd.DataFrame,
    cis_result: pd.DataFrame,
    trans_result: pd.DataFrame,
    cutoff: float,
    *,
    nominal: float = 0.05,
) -> pd.DataFrame:
    """What each TSS knockdown lowers: gene G knocked down -> gene H.

    Every tested pair of a TSS target, from either result, against any gene
    other than its own. Several TSS targets of one gene are pooled.

    Returns:
        `knocked_down`, `gene`, `tested`, `lowers` (called at `cutoff` with a
        negative effect by any of G's TSS targets), `lowers_nominal` (the same
        at `p < nominal`) and `best_lfc` (the most negative log2 fold change).
    """
    both = pd.concat(
        [tested_pairs(cis_result, "cis_result"), tested_pairs(trans_result, "trans_result")],
        ignore_index=True,
    )
    both = both.merge(tss[["grna_target", "tss_gene"]], on="grna_target")
    both = both.loc[both["response_id"] != both["tss_gene"]]
    both = both.assign(
        lowers=called_down(both, cutoff),
        lowers_nominal=(both["p_value"].to_numpy() < nominal)
        & (both["log_2_fold_change"].to_numpy() < 0),
    )
    return (
        both.groupby(["tss_gene", "response_id"])
        .agg(
            tested=("lowers", "size"),
            lowers=("lowers", "any"),
            lowers_nominal=("lowers_nominal", "any"),
            best_lfc=("log_2_fold_change", "min"),
        )
        .reset_index()
        .rename(columns={"tss_gene": "knocked_down", "response_id": "gene"})
    )


def detour_check(
    links: pd.DataFrame,
    tss: pd.DataFrame,
    lookup: pd.DataFrame,
    *,
    far_distance: float = 1e5,
) -> pd.DataFrame:
    """For each far link E -> H, whether it can run through another gene E lowers.

    The genes E lowers are its links at any distance, plus the gene whose TSS
    E sits on (knocked down even when that pair was not tested).

    Returns:
        The far links (`distance > far_distance`) with `element_lowers` and a
        `status`: `"detour"` when some G that E lowers has a TSS knockdown
        that lowers H at the cutoff, `"detour (nominal)"` when only at the
        nominal level, `"no detour"` when such G were tested against H and
        none lowers it, and `"can't check"` when no G was tested against H.
        `through` names the G that explain a detour.
    """
    link_rows = links.loc[links["link"]]
    lowered = link_rows.groupby("grna_target")["response_id"].agg(set).to_dict()
    for target, gene in zip(tss["grna_target"], tss["tss_gene"], strict=True):
        lowered.setdefault(target, set()).add(gene)
    by_pair = lookup.set_index(["gene", "knocked_down"]).sort_index()

    far = link_rows.loc[
        link_rows["distance"] > far_distance, ["grna_target", "response_id", "distance"]
    ]
    rows = []
    for target, gene, distance in far.itertuples(index=False, name=None):
        via = sorted(lowered.get(target, set()) - {gene})
        hit = by_pair.loc[[(gene, v) for v in via if (gene, v) in by_pair.index]]
        if hit["lowers"].any():
            status, through = "detour", hit.index.get_level_values(1)[hit["lowers"].to_numpy()]
        elif hit["lowers_nominal"].any():
            status, through = (
                "detour (nominal)",
                hit.index.get_level_values(1)[hit["lowers_nominal"].to_numpy()],
            )
        elif len(hit):
            status, through = "no detour", []
        else:
            status, through = "can't check", []
        rows.append(
            {
                "grna_target": target,
                "response_id": gene,
                "distance": distance,
                "element_lowers": ", ".join(via),
                "status": status,
                "through": ", ".join(through),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["grna_target", "response_id", "distance", "element_lowers", "status", "through"],
    )
