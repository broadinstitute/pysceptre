"""The matched background rate and the broad-effect flag (parts 1 and 2 of the check).

A test between an element and a gene on another chromosome cannot be direct
regulation, so how often those tests are called at the cis cutoff, with a
negative effect, is how often a test is called without it. Every function here
takes result tables and returns tables; none needs the count matrix. See
`docs/design.md`, "Specificity check".
"""

from __future__ import annotations

import warnings
from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd
from scipy import sparse, stats

__all__ = [
    "DEFAULT_DISTANCE_BINS",
    "above_background_by_distance",
    "background_pairs",
    "broad_effect_elements",
    "cis_links",
    "nearest_tss",
    "per_gene_background",
]

DEFAULT_DISTANCE_BINS: tuple[float, ...] = (0.0, 1e3, 1e4, 5e4, 1e5, np.inf)
_RESULT_COLUMNS = ("grna_target", "response_id", "p_value", "log_2_fold_change")
# Draws times pairs held at once by the bootstrap; bounds memory, not the answer.
_BOOTSTRAP_BLOCK = 1 << 22


def _require(frame: pd.DataFrame, name: str, columns: Sequence[str]) -> None:
    absent = [c for c in columns if c not in frame.columns]
    if absent:
        raise KeyError(f"{name} is missing column(s): {absent}")


def tested_pairs(result: pd.DataFrame, name: str) -> pd.DataFrame:
    """The rows of a discovery result that were tested: `pass_qc` true, finite p-value."""
    _require(result, name, _RESULT_COLUMNS)
    keep = np.isfinite(result["p_value"].to_numpy(dtype=float))
    if "pass_qc" in result.columns:
        keep &= result["pass_qc"].astype(bool).to_numpy()
    return result.loc[keep, list(_RESULT_COLUMNS)].reset_index(drop=True)


def called_down(result: pd.DataFrame, cutoff: float) -> np.ndarray:
    """Called at `cutoff` with a negative effect: a link on the cis side, a background call on the other."""
    return (result["p_value"].to_numpy() <= cutoff) & (result["log_2_fold_change"].to_numpy() < 0)


def element_table(element_positions: pd.DataFrame) -> pd.DataFrame:
    """`element_positions` indexed by `grna_target`, validated."""
    _require(element_positions, "element_positions", ("grna_target", "chrom", "centre"))
    table = element_positions.loc[:, ["grna_target", "chrom", "centre"]]
    dup = table["grna_target"].duplicated()
    if dup.any():
        raise ValueError(
            f"element_positions lists {int(dup.sum())} target(s) twice, including: "
            f"{table.loc[dup, 'grna_target'].tolist()[:5]}"
        )
    return table.set_index("grna_target")


def gene_table(gene_positions: pd.DataFrame, genes: Collection[str]) -> pd.DataFrame:
    """One position per gene in `genes`, indexed by `response_id`.

    `gene_positions` may be a whole annotation. Only the genes asked for have
    to be present and unique there: a symbol on two chromosomes makes its
    distances ambiguous, so it is refused rather than resolved.
    """
    _require(gene_positions, "gene_positions", ("response_id", "chrom", "tss"))
    wanted = set(genes)
    table = gene_positions.loc[
        gene_positions["response_id"].isin(wanted), ["response_id", "chrom", "tss"]
    ]
    dup = table["response_id"].duplicated(keep=False)
    if dup.any():
        raise ValueError(
            f"gene_positions has more than one position for {table.loc[dup, 'response_id'].nunique()} "
            f"gene(s), including: {sorted(set(table.loc[dup, 'response_id']))[:5]}"
        )
    missing = sorted(wanted - set(table["response_id"]))
    if missing:
        raise KeyError(f"gene_positions has no entry for {len(missing)} gene(s): {missing[:5]}")
    return table.set_index("response_id")


def cis_links(
    cis_result: pd.DataFrame,
    element_positions: pd.DataFrame,
    gene_positions: pd.DataFrame,
    cutoff: float,
    *,
    control_targets: Collection[str] = (),
) -> pd.DataFrame:
    """The tested element-gene pairs, with their distance and whether each is a link.

    Args:
        cis_result: a discovery result (`grna_target`, `response_id`,
            `p_value`, `log_2_fold_change`, optionally `pass_qc`) on pairs with
            the element and the gene on the same chromosome.
        element_positions: `grna_target`, `chrom`, `centre`.
        gene_positions: `response_id`, `chrom`, `tss`.
        cutoff: the nominal p-value threshold the screen was called at.
        control_targets: targets that are not elements, such as TSS
            positive controls. Their pairs are left out.

    Returns:
        `grna_target`, `response_id`, `p_value`, `log_2_fold_change`,
        `distance` (element centre to TSS) and `link` (called with a negative
        effect), one row per tested pair.

    Raises:
        KeyError: a target or gene has no position.
        ValueError: a pair has its element and gene on different chromosomes.
    """
    pairs = tested_pairs(cis_result, "cis_result")
    pairs = pairs.loc[~pairs["grna_target"].isin(set(control_targets))].reset_index(drop=True)
    elements = element_table(element_positions)
    missing = sorted(set(pairs["grna_target"]) - set(elements.index))
    if missing:
        raise KeyError(
            f"element_positions has no entry for {len(missing)} target(s): {missing[:5]}"
        )
    genes = gene_table(gene_positions, pairs["response_id"].unique())

    e_chrom = elements["chrom"].reindex(pairs["grna_target"]).to_numpy()
    g_chrom = genes["chrom"].reindex(pairs["response_id"]).to_numpy()
    apart = e_chrom != g_chrom
    if apart.any():
        raise ValueError(
            f"{int(apart.sum())} cis_result pair(s) put the element and the gene on different "
            "chromosomes, so they have no distance. Pass those as trans_result."
        )
    centre = elements["centre"].reindex(pairs["grna_target"]).to_numpy(dtype=float)
    tss = genes["tss"].reindex(pairs["response_id"]).to_numpy(dtype=float)
    return pairs.assign(distance=np.abs(centre - tss), link=called_down(pairs, cutoff))


def background_pairs(
    trans_result: pd.DataFrame,
    links: pd.DataFrame,
    element_positions: pd.DataFrame,
    gene_positions: pd.DataFrame,
    cutoff: float,
    *,
    exclude_gene_chroms: Collection[str] = ("chrY",),
) -> pd.DataFrame:
    """The tests the background is measured on, and which of them are called.

    The cis elements against the cis genes, gene on another chromosome, gene
    chromosome not in `exclude_gene_chroms`, tested. Restricting to the cis
    sets is what lets the two sides be matched gene for gene.

    Returns:
        `grna_target`, `response_id`, `p_value`, `log_2_fold_change` and
        `background` (called at the cis `cutoff` with a negative effect).
    """
    pairs = tested_pairs(trans_result, "trans_result")
    pairs = pairs.loc[
        pairs["grna_target"].isin(set(links["grna_target"]))
        & pairs["response_id"].isin(set(links["response_id"]))
    ].reset_index(drop=True)
    elements = element_table(element_positions)
    genes = gene_table(gene_positions, pairs["response_id"].unique())
    g_chrom = genes["chrom"].reindex(pairs["response_id"]).to_numpy()
    e_chrom = elements["chrom"].reindex(pairs["grna_target"]).to_numpy()
    keep = (g_chrom != e_chrom) & ~np.isin(g_chrom, list(exclude_gene_chroms))
    pairs = pairs.loc[keep].reset_index(drop=True)
    return pairs.assign(background=called_down(pairs, cutoff))


def per_gene_background(
    background: pd.DataFrame, *, genes: Collection[str] | None = None
) -> pd.DataFrame:
    """`response_id`, `trans_pairs`, `background_calls`, `background_rate`.

    With `genes`, one row per gene in that order; a gene with no background
    test has zero pairs and calls and a NaN rate.
    """
    out = background.groupby("response_id")["background"].agg(
        trans_pairs="size", background_calls="sum", background_rate="mean"
    )
    if genes is not None:
        out = out.reindex(list(genes)).rename_axis("response_id")
        out["trans_pairs"] = out["trans_pairs"].fillna(0).astype(int)
        out["background_calls"] = out["background_calls"].fillna(0).astype(int)
    return out.reset_index()


def _bin_labels(edges: Sequence[float]) -> list[str]:
    def kb(x: float) -> str:
        return f"{x / 1000:g}"

    return [
        f"> {kb(a)} kb" if np.isinf(b) else f"{kb(a)} to {kb(b)} kb"
        for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]


def above_background_by_distance(
    links: pd.DataFrame,
    background: pd.DataFrame,
    *,
    distance_bins: Sequence[float] = DEFAULT_DISTANCE_BINS,
    n_bootstrap: int = 2000,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Per distance bin: cis rate, matched background and links above background.

    The background of a bin is the mean, over its tested pairs, of each
    pair's gene's background rate, so a gene counts as often as it is tested
    there. Links above background is `(cis rate - background) / cis rate`.
    Intervals are a bootstrap over **elements**: an element's cis and
    background tests are drawn together, and every gene's rate is recomputed
    in each draw.

    Args:
        links: `cis_links` output.
        background: `background_pairs` output for the same elements.
        distance_bins: bin edges in bp, increasing. Bins are closed on the
            right, and the first also includes its left edge.
        n_bootstrap: bootstrap draws.
        seed: an int, or a Generator to continue drawing from.

    Returns:
        One row per bin: `distance_bin`, `pairs`, `links`, then `cis_rate`,
        `background` and `above`, each with `_lo` and `_hi` (the 2.5 and 97.5
        percentiles of the draws).

    Raises:
        ValueError: a distance falls outside the bins, or the bins do not
            increase.
    """
    edges = np.asarray(distance_bins, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError(
            f"distance_bins must be at least two increasing edges, got {distance_bins!r}"
        )
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be positive, got {n_bootstrap!r}")
    rng = np.random.default_rng(seed)
    n_bins = edges.size - 1

    elems = np.sort(links["grna_target"].unique())
    genes = np.sort(links["response_id"].unique())
    e_index, g_index = pd.Index(elems), pd.Index(genes)
    n_e, n_g = elems.size, genes.size

    # Removing an element from `links` must remove it from the background too; see
    # docs/design.md, "Broad-effect elements".
    bg = background.loc[
        background["grna_target"].isin(elems) & background["response_id"].isin(genes)
    ]
    te = e_index.get_indexer(bg["grna_target"])
    tg = g_index.get_indexer(bg["response_id"])
    tested = sparse.csc_matrix((np.ones(len(bg)), (te, tg)), shape=(n_e, n_g))
    calls = sparse.csc_matrix((bg["background"].to_numpy(dtype=float), (te, tg)), shape=(n_e, n_g))

    ce = e_index.get_indexer(links["grna_target"])
    cg = g_index.get_indexer(links["response_id"])
    cb = pd.cut(links["distance"], edges, include_lowest=True, labels=False).to_numpy()
    if np.isnan(cb).any():
        raise ValueError(
            f"{int(np.isnan(cb).sum())} pair(s) have a distance outside distance_bins "
            f"[{edges[0]}, {edges[-1]}]"
        )
    cb = cb.astype(int)
    n_pairs = len(links)
    bin_of_pair = sparse.csr_matrix(
        (np.ones(n_pairs), (np.arange(n_pairs), cb)), shape=(n_pairs, n_bins)
    )
    pairs_eb = np.zeros((n_e, n_bins))
    links_eb = np.zeros((n_e, n_bins))
    np.add.at(pairs_eb, (ce, cb), 1.0)
    np.add.at(links_eb, (ce, cb), links["link"].to_numpy(dtype=float))

    def estimate(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rate = np.nan_to_num((calls.T @ w.T).T / (tested.T @ w.T).T)
        n = w @ pairs_eb
        cis = (w @ links_eb) / n
        bg_sum = np.empty((w.shape[0], n_bins))
        step = max(1, _BOOTSTRAP_BLOCK // max(n_pairs, 1))
        for s in range(0, w.shape[0], step):
            block = w[s : s + step, ce] * rate[s : s + step, cg]
            bg_sum[s : s + step] = (bin_of_pair.T @ block.T).T
        back = bg_sum / n
        return cis, back, (cis - back) / cis

    with np.errstate(divide="ignore", invalid="ignore"):
        point = estimate(np.ones((1, n_e)))
        w = rng.multinomial(n_e, np.full(n_e, 1.0 / n_e), n_bootstrap).astype(float)
        draws = estimate(w)

    out = pd.DataFrame(
        {
            "distance_bin": _bin_labels(edges),
            "pairs": pairs_eb.sum(0).astype(int),
            "links": links_eb.sum(0).astype(int),
        }
    )
    with warnings.catch_warnings():
        # A bin no draw can estimate (no pairs) is NaN, not an error.
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, p, d in zip(("cis_rate", "background", "above"), point, draws, strict=True):
            out[name] = p[0]
            out[f"{name}_lo"], out[f"{name}_hi"] = np.nanpercentile(d, [2.5, 97.5], axis=0)
    return out


def nearest_tss(
    chrom: pd.Series, position: pd.Series, gene_positions: pd.DataFrame
) -> pd.DataFrame:
    """The nearest TSS in `gene_positions` on the same chromosome, for each position.

    Returns `nearest_gene` and `tss_distance`, indexed like `chrom`; NaN on a
    chromosome with no gene. A tie goes to the upstream TSS.
    """
    _require(gene_positions, "gene_positions", ("response_id", "chrom", "tss"))
    out = pd.DataFrame(
        {"nearest_gene": pd.Series(np.nan, index=chrom.index, dtype=object), "tss_distance": np.nan}
    )
    for c, idx in chrom.groupby(chrom).groups.items():
        g = gene_positions.loc[gene_positions["chrom"] == c].sort_values("tss", kind="mergesort")
        if g.empty:
            continue
        t = g["tss"].to_numpy(dtype=float)
        p = position.loc[idx].to_numpy(dtype=float)
        if t.size == 1:
            j = np.zeros(p.size, dtype=int)
        else:
            i = np.clip(np.searchsorted(t, p), 1, t.size - 1)
            j = np.where(np.abs(t[i - 1] - p) <= np.abs(t[i] - p), i - 1, i)
        out.loc[idx, "nearest_gene"] = g["response_id"].to_numpy()[j]
        out.loc[idx, "tss_distance"] = np.abs(t[j] - p)
    return out


def broad_effect_elements(
    background: pd.DataFrame,
    element_positions: pd.DataFrame,
    gene_positions: pd.DataFrame,
    *,
    p_threshold: float = 1e-3,
) -> pd.DataFrame:
    """Elements with more background calls than the screen's background rate allows.

    Each element's background calls are tested against the pooled rate over
    all background tests (one-sided binomial). Such an element makes its own
    background, which a per-gene rate cannot see.

    Returns:
        `grna_target`, `trans_pairs`, `trans_calls`, `p_excess`,
        `broad_effect` (`p_excess < p_threshold`), `nearest_gene` and
        `tss_distance` (the nearest TSS in `gene_positions`), one row per
        element with background tests.
    """
    if not 0.0 < p_threshold < 1.0:
        raise ValueError(f"p_threshold must lie in (0, 1), got {p_threshold!r}")
    per = (
        background.groupby("grna_target", sort=True)["background"]
        .agg(trans_calls="sum", trans_pairs="size")
        .reset_index()
    )
    pooled = float(background["background"].mean())
    per["p_excess"] = stats.binom.sf(per["trans_calls"] - 1, per["trans_pairs"], pooled)
    per["broad_effect"] = per["p_excess"] < p_threshold
    elements = element_table(element_positions)
    near = nearest_tss(
        elements["chrom"].reindex(per["grna_target"]).reset_index(drop=True),
        elements["centre"].reindex(per["grna_target"]).reset_index(drop=True),
        gene_positions,
    )
    per = per.join(near)
    return per[
        [
            "grna_target",
            "trans_pairs",
            "trans_calls",
            "p_excess",
            "broad_effect",
            "nearest_gene",
            "tss_distance",
        ]
    ]
