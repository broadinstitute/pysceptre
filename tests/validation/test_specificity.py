"""The specificity check, on synthetic screens with a known answer.

There is no R for this check. These tests pin the identities the method has to
satisfy, the bookkeeping a shorter version would get wrong, and agreement of
the bootstrap with the dense formulation it was ported from. The comparison
against that notebook on real screens is `test_specificity_days.py`, behind
the `realdata` marker.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pysceptre import run_specificity_check
from pysceptre.specificity import (
    above_background_by_distance,
    background_pairs,
    cis_links,
    detour_check,
    gene_lookup,
    tss_targets,
)

CUTOFF = 0.01
EDGES = [0, 1_000, 10_000, 50_000, 100_000, np.inf]


def _screen(seed: int = 0, n_elements: int = 40, n_genes: int = 12, link_rate: float = 0.2):
    """Two chromosomes, elements and genes on both; cis within, trans across."""
    rng = np.random.default_rng(seed)
    chroms = ["chr1", "chr2"]
    genes = pd.DataFrame(
        {
            "response_id": [f"G{k}" for k in range(n_genes)],
            "chrom": [chroms[k % 2] for k in range(n_genes)],
            "tss": rng.integers(1_000_000, 2_000_000, n_genes).astype(float),
        }
    )
    elements = pd.DataFrame(
        {
            "grna_target": [f"E{k}" for k in range(n_elements)],
            "chrom": [chroms[k % 2] for k in range(n_elements)],
            "centre": rng.integers(800_000, 2_200_000, n_elements).astype(float),
        }
    )
    pairs = elements.merge(genes, on="chrom", suffixes=("", "_g"))
    cis = pairs[["grna_target", "response_id"]].copy()
    hit = rng.random(len(cis)) < link_rate
    cis["p_value"] = np.where(
        hit, rng.uniform(0, CUTOFF / 2, len(cis)), rng.uniform(CUTOFF, 1, len(cis))
    )
    cis["log_2_fold_change"] = np.where(hit, -0.5, rng.normal(0, 0.2, len(cis)))
    cis["pass_qc"] = True
    trans = elements.merge(genes, how="cross", suffixes=("", "_g"))
    trans = trans.loc[trans["chrom"] != trans["chrom_g"], ["grna_target", "response_id"]].copy()
    trans["p_value"] = rng.uniform(0, 1, len(trans)) ** 0.8
    trans["log_2_fold_change"] = rng.normal(0, 0.2, len(trans))
    trans["pass_qc"] = True
    return cis.reset_index(drop=True), trans.reset_index(drop=True), elements, genes


def _dense_reference(links, background, rng, n_boot):
    """The notebook's formulation: a dense elements x bins x genes tensor and one einsum."""
    elems = np.sort(links["grna_target"].unique())
    genes = np.sort(links["response_id"].unique())
    ei = {e: i for i, e in enumerate(elems)}
    gi = {g: i for i, g in enumerate(genes)}
    bg = background[background["response_id"].isin(genes)]
    tn = np.zeros((len(elems), len(genes)))
    tc = np.zeros((len(elems), len(genes)))
    te, tg = bg["grna_target"].map(ei).to_numpy(), bg["response_id"].map(gi).to_numpy()
    np.add.at(tn, (te, tg), 1)
    np.add.at(tc, (te, tg), bg["background"].to_numpy())
    ce, cg = links["grna_target"].map(ei).to_numpy(), links["response_id"].map(gi).to_numpy()
    cb = pd.cut(links["distance"], EDGES, include_lowest=True, labels=False).to_numpy()
    nb = len(EDGES) - 1
    cn = np.zeros((len(elems), nb))
    cl = np.zeros((len(elems), nb))
    cbg = np.zeros((len(elems), nb, len(genes)))
    np.add.at(cn, (ce, cb), 1)
    np.add.at(cl, (ce, cb), links["link"].to_numpy())
    np.add.at(cbg, (ce, cb, cg), 1)

    def estimate(w):
        rate = np.nan_to_num((w @ tc) / (w @ tn))
        n = w @ cn
        cis = (w @ cl) / n
        back = np.einsum("de,ebg,dg->db", w, cbg, rate, optimize=True) / n
        return cis, back, (cis - back) / cis

    with np.errstate(divide="ignore", invalid="ignore"):
        point = estimate(np.ones((1, len(elems))))
        w = rng.multinomial(len(elems), np.full(len(elems), 1 / len(elems)), n_boot).astype(float)
        draws = estimate(w)
    out = {}
    with warnings.catch_warnings():
        # A bin with no pairs in some draw is NaN there, as in the code under test.
        warnings.simplefilter("ignore", RuntimeWarning)
        for name, p, d in zip(("cis_rate", "background", "above"), point, draws, strict=True):
            out[name] = p[0]
            out[f"{name}_lo"], out[f"{name}_hi"] = np.nanpercentile(d, [2.5, 97.5], axis=0)
    return pd.DataFrame(out)


def _links_and_background(seed=0, **kw):
    cis, trans, elements, genes = _screen(seed, **kw)
    links = cis_links(cis, elements, genes, CUTOFF)
    background = background_pairs(trans, links, elements, genes, CUTOFF)
    return links, background


def test_the_bootstrap_matches_the_dense_formulation_it_was_ported_from():
    links, background = _links_and_background(seed=1, n_elements=60, n_genes=14)
    got = above_background_by_distance(links, background, n_bootstrap=300, seed=7)
    want = _dense_reference(links, background, np.random.default_rng(7), 300)
    for col in want.columns:
        np.testing.assert_allclose(got[col], want[col], rtol=1e-12, atol=1e-15, err_msg=col)


def test_the_point_estimate_is_the_per_gene_matched_mean_by_hand():
    links, background = _links_and_background(seed=2)
    got = above_background_by_distance(links, background, n_bootstrap=10, seed=0)
    rate = background.groupby("response_id")["background"].mean()
    bins = pd.cut(links["distance"], EDGES, include_lowest=True, labels=False)
    for b, rows in links.groupby(bins):
        cis_rate = rows["link"].mean()
        matched = rows["response_id"].map(rate).fillna(0).mean()
        row = got.iloc[int(b)]
        assert row["pairs"] == len(rows)
        np.testing.assert_allclose(row["cis_rate"], cis_rate, rtol=1e-12)
        np.testing.assert_allclose(row["background"], matched, rtol=1e-12)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.testing.assert_allclose(row["above"], (cis_rate - matched) / cis_rate, rtol=1e-12)


def test_no_links_beyond_background_gives_zero_above():
    """When every cis call is exactly what the gene's background predicts, nothing is above it."""
    links = pd.DataFrame(
        {
            "grna_target": ["E0", "E0", "E1", "E1"],
            "response_id": ["A", "B", "A", "B"],
            "distance": [500.0, 500.0, 500.0, 500.0],
            "link": [True, False, False, False],
        }
    )
    # A is called in 1 of 2 background tests and B in 0: the matched background is 0.25, the cis
    # rate 1 of 4.
    background = pd.DataFrame(
        {
            "grna_target": ["E0", "E1", "E0", "E1"],
            "response_id": ["A", "A", "B", "B"],
            "background": [True, False, False, False],
        }
    )
    got = above_background_by_distance(links, background, n_bootstrap=5, seed=0)
    np.testing.assert_allclose(
        got.loc[0, ["cis_rate", "background", "above"]].to_numpy(float), [0.25, 0.25, 0.0]
    )


def test_bins_are_labelled_and_closed_on_the_right():
    links = pd.DataFrame(
        {
            "grna_target": ["E0", "E0", "E1"],
            "response_id": ["A", "A", "A"],
            "distance": [0.0, 1_000.0, 1_000.5],
            "link": [True, True, False],
        }
    )
    background = pd.DataFrame({"grna_target": ["E0"], "response_id": ["A"], "background": [False]})
    got = above_background_by_distance(links, background, n_bootstrap=2, seed=0)
    assert got["distance_bin"].tolist() == [
        "0 to 1 kb",
        "1 to 10 kb",
        "10 to 50 kb",
        "50 to 100 kb",
        "> 100 kb",
    ]
    assert got["pairs"].tolist() == [2, 1, 0, 0, 0]


def test_a_seed_reproduces_and_a_generator_continues():
    cis, trans, elements, genes = _screen(3)
    a = run_specificity_check(cis, trans, elements, genes, CUTOFF, n_bootstrap=50, seed=11)
    b = run_specificity_check(cis, trans, elements, genes, CUTOFF, n_bootstrap=50, seed=11)
    pd.testing.assert_frame_equal(a.by_distance, b.by_distance)

    rng = np.random.default_rng(11)
    c = run_specificity_check(cis, trans, elements, genes, CUTOFF, n_bootstrap=50, seed=rng)
    d = run_specificity_check(cis, trans, elements, genes, CUTOFF, n_bootstrap=50, seed=rng)
    pd.testing.assert_frame_equal(a.by_distance, c.by_distance)
    assert not d.by_distance.equals(c.by_distance)


def test_a_broad_effect_element_is_flagged_and_removed_from_both_sides():
    cis, trans, elements, genes = _screen(4, n_elements=40)
    loud = trans["grna_target"] == "E0"
    trans.loc[loud, "p_value"] = 1e-6
    trans.loc[loud, "log_2_fold_change"] = -1.0
    res = run_specificity_check(cis, trans, elements, genes, CUTOFF, n_bootstrap=20, seed=0)
    flagged = set(res.broad_effect.loc[res.broad_effect["broad_effect"], "grna_target"])
    assert flagged == {"E0"}
    kept = res.links.loc[res.links["grna_target"] != "E0"]
    assert res.by_distance_without_broad["pairs"].sum() == len(kept)
    assert res.by_distance["pairs"].sum() == len(res.links)
    assert res.by_distance_without_broad["background"].max() < res.by_distance["background"].max()


def test_the_trans_side_is_other_chromosomes_only_and_drops_chry():
    cis, trans, elements, genes = _screen(5)
    genes.loc[genes["response_id"] == "G1", "chrom"] = "chrY"
    cis = cis.loc[cis["response_id"] != "G1"]
    extra = cis.head(3).assign(p_value=1e-9, log_2_fold_change=-2.0)
    res = run_specificity_check(
        cis,
        pd.concat([trans, extra], ignore_index=True),
        elements,
        genes,
        CUTOFF,
        n_bootstrap=5,
        seed=0,
    )
    g_chrom = genes.set_index("response_id")["chrom"]
    e_chrom = elements.set_index("grna_target")["chrom"]
    bg = res.background
    assert (bg["response_id"].map(g_chrom) != bg["grna_target"].map(e_chrom)).all()
    assert "G1" not in set(bg["response_id"])


def test_control_targets_are_not_elements_but_are_tss_knockdowns():
    cis, trans, elements, genes = _screen(6)
    g0 = genes.iloc[0]
    control = pd.DataFrame(
        {"grna_target": ["TSS_G0"], "chrom": [g0["chrom"]], "centre": [g0["tss"] + 100]}
    )
    elements = pd.concat([elements, control], ignore_index=True)
    ctrl_pairs = pd.DataFrame(
        {
            "grna_target": "TSS_G0",
            "response_id": genes.loc[genes["chrom"] == g0["chrom"], "response_id"],
            "p_value": 1e-8,
            "log_2_fold_change": -1.0,
            "pass_qc": True,
        }
    )
    cis = pd.concat([cis, ctrl_pairs], ignore_index=True)
    res = run_specificity_check(
        cis, trans, elements, genes, CUTOFF, control_targets={"TSS_G0"}, n_bootstrap=5, seed=0
    )
    assert "TSS_G0" not in set(res.links["grna_target"])
    assert "TSS_G0" not in set(res.background["grna_target"])
    assert res.tss_targets.set_index("grna_target").loc["TSS_G0", "tss_gene"] == "G0"
    assert (res.gene_lookup.loc[res.gene_lookup["knocked_down"] == "G0", "gene"] != "G0").all()


def test_each_detour_status():
    """E -> H far. E lowers G (a link) or sits on G's TSS; G's TSS knockdown may lower H."""
    links = pd.DataFrame(
        {
            "grna_target": ["E1", "E1", "E2", "E2", "E3", "E4"],
            "response_id": ["H", "G", "H", "G", "H", "H"],
            "distance": [500_000.0, 1_000.0, 500_000.0, 1_000.0, 500_000.0, 500_000.0],
            "link": [True, True, True, True, True, True],
        }
    )
    tss = pd.DataFrame(
        {"grna_target": ["T_G", "E3"], "tss_gene": ["G", "K"], "tss_distance": [0.0, 10.0]}
    )
    cis = pd.DataFrame(
        {
            "grna_target": ["T_G", "E3"],
            "response_id": ["H", "H"],
            "p_value": [0.001, 0.5],
            "log_2_fold_change": [-1.0, -0.1],
        }
    )
    lookup = gene_lookup(tss, cis, cis.iloc[0:0], cutoff=CUTOFF)
    got = detour_check(links, tss, lookup).set_index("grna_target")["status"]
    assert got["E1"] == "detour"  # lowers G, and G's TSS knockdown lowers H
    assert got["E3"] == "no detour"  # sits on K's TSS, K's knockdown tested against H, no effect
    assert got["E4"] == "can't check"  # lowers nothing else

    nominal = gene_lookup(tss, cis.assign(p_value=[0.03, 0.5]), cis.iloc[0:0], cutoff=CUTOFF)
    assert (
        detour_check(links, tss, nominal).set_index("grna_target").loc["E2", "status"]
        == "detour (nominal)"
    )


def test_tss_targets_take_the_nearest_measured_gene_within_the_window():
    elements = pd.DataFrame(
        {
            "grna_target": ["A", "B", "C"],
            "chrom": ["chr1", "chr1", "chr2"],
            "centre": [1_000.0, 5_000.0, 1_000.0],
        }
    )
    genes = pd.DataFrame(
        {
            "response_id": ["X", "Y", "Z", "U"],
            "chrom": ["chr1", "chr1", "chr2", "chr1"],
            "tss": [1_400.0, 1_200.0, 9_000.0, 1_000.0],
        }
    )
    got = tss_targets(["A", "B", "C"], elements, genes, measured_genes=["X", "Y", "Z"])
    assert got.set_index("grna_target")["tss_gene"].to_dict() == {"A": "Y"}  # U is not measured


def test_inputs_that_would_silently_change_the_answer_raise():
    cis, trans, elements, genes = _screen(7)
    with pytest.raises(ValueError, match="cutoff"):
        run_specificity_check(cis, trans, elements, genes, 1.5)
    with pytest.raises(KeyError, match="element_positions"):
        run_specificity_check(cis, trans, elements.iloc[1:], genes, CUTOFF)
    dup = pd.concat([genes, genes.iloc[[0]].assign(chrom="chr9")], ignore_index=True)
    with pytest.raises(ValueError, match="more than one position"):
        run_specificity_check(cis, trans, elements, dup, CUTOFF)
    crossed = pd.concat([cis, trans.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="different chromosomes"):
        run_specificity_check(crossed, trans, elements, genes, CUTOFF)
    links, background = _links_and_background(7)
    with pytest.raises(ValueError, match="outside distance_bins"):
        above_background_by_distance(links, background, distance_bins=[0, 10], n_bootstrap=2)


def test_removing_an_element_from_the_links_removes_its_background_too():
    """Keeping a removed element's background calls would inflate the background."""
    links, background = _links_and_background(8)
    drop = links["grna_target"] != "E0"
    one = above_background_by_distance(links[drop], background, n_bootstrap=20, seed=0)
    both = above_background_by_distance(
        links[drop], background[background["grna_target"] != "E0"], n_bootstrap=20, seed=0
    )
    pd.testing.assert_frame_equal(one, both)
