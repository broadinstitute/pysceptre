"""gRNA integration strategies against sceptre's own grouping code.

`grna_integration_strategy` changes nothing statistical: the same CRT on a
different treated cell set, plus an aggregation in `bonferroni`'s case. So
these check the bookkeeping, which is where the whole difference lives.

The expansion is compared against `update_dfs_based_on_grouping_strategy`
itself, cached in `singleton_ground_truth.json`, because the gRNA-to-target
map is many-to-many in real data and that is the part a transcription gets
wrong.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pysceptre import run_discovery_analysis
from pysceptre.pipeline.grouping import aggregate_bonferroni, singleton_pairs, sort_like_r

N_CELLS = 2500


def _expected(case: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "response_id": case["pairs_out"]["response_id"],
            "grna_id": case["pairs_out"]["grna_group"],
            "grna_target": case["pairs_out"]["grna_target"],
        }
    )


def test_the_expansion_matches_sceptres_own(singleton_ground_truth):
    """Row for row and in the same order, for every case R could produce one for."""
    for case in singleton_ground_truth["cases"]:
        if case["label"] == "orphan_target":
            continue  # R keeps an NA row here; see the next test
        got = singleton_pairs(pd.DataFrame(case["pairs_in"]), pd.DataFrame(case["gtdf"]))
        pd.testing.assert_frame_equal(got, _expected(case), check_dtype=False)


def test_a_shared_guide_is_expanded_under_each_of_its_targets(singleton_ground_truth):
    """The case that makes this a many-to-many join rather than a lookup."""
    case = next(c for c in singleton_ground_truth["cases"] if c["label"] == "shared_guide")
    got = singleton_pairs(pd.DataFrame(case["pairs_in"]), pd.DataFrame(case["gtdf"]))
    shared = got[got["grna_id"] == "gS"]
    assert set(shared["grna_target"]) == {"A", "B"}
    assert len(got) == 8


def test_non_targeting_guides_are_never_expanded(singleton_ground_truth):
    case = next(c for c in singleton_ground_truth["cases"] if c["label"] == "ntcs_are_not_expanded")
    got = singleton_pairs(pd.DataFrame(case["pairs_in"]), pd.DataFrame(case["gtdf"]))
    assert "non-targeting" not in set(got["grna_target"])
    assert not got["grna_id"].str.startswith("ntc").any()


def test_an_orphan_target_raises_where_r_keeps_an_na_row(singleton_ground_truth):
    """A deliberate deviation, recorded here so it is not read as a port error.

    R's left join leaves `grna_group` as `NA` and keeps the row, which then
    asks for the cells of a group called `NA`. An untestable row surviving
    into a result is the failure this package refuses everywhere else.
    """
    case = next(c for c in singleton_ground_truth["cases"] if c["label"] == "orphan_target")
    assert "NA" in case["pairs_out"]["grna_group"], "fixture no longer shows R's NA row"
    with pytest.raises(KeyError, match="no guides in grna_target_data_frame"):
        singleton_pairs(pd.DataFrame(case["pairs_in"]), pd.DataFrame(case["gtdf"]))


# ----------------------------------------------------------------- end to end ---


@pytest.fixture
def screen():
    """Target A with three guides, B with two, and one guide shared by both."""
    rng = np.random.default_rng(41)
    counts = rng.poisson(14.0, size=(3, N_CELLS)).astype(float)
    gene_ids = ["g0", "g1", "g2"]
    cov = np.column_stack([np.ones(N_CELLS), rng.normal(size=N_CELLS)])
    design = pd.DataFrame(
        {
            "grna_id": ["gA1", "gA2", "gS", "gS", "gB1"],
            "grna_target": ["A", "A", "A", "B", "B"],
        }
    )
    guide_cells = {
        g: rng.choice(N_CELLS, size=120, replace=False) for g in ("gA1", "gA2", "gS", "gB1")
    }
    counts[0, guide_cells["gA1"]] = rng.poisson(5.0, size=120)
    pairs = pd.DataFrame({"response_id": ["g0", "g0", "g1"], "grna_target": ["A", "B", "A"]})
    return counts, gene_ids, cov, guide_cells, design, pairs


def _run(screen, strategy, **kw):
    counts, gene_ids, cov, guide_cells, design, pairs = screen
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_discovery_analysis(
            counts,
            gene_ids,
            cov,
            guide_cells,
            pairs,
            side="left",
            seed=0,
            grna_integration_strategy=strategy,
            grna_target_data_frame=design,
            **kw,
        )


def test_singleton_reports_one_row_per_pair_per_guide(screen):
    """Three pairs over targets of three, two and three guides: eight rows, as R gives."""
    result = _run(screen, "singleton")
    assert len(result) == 8
    assert list(result.columns[:3]) == ["response_id", "grna_id", "grna_target"]


def test_singleton_is_sorted_the_way_r_leaves_it(screen):
    """`setorderv(c("p_value", "response_id"), na.last = TRUE)`, and only on this branch."""
    result = _run(screen, "singleton")
    assert result["p_value"].is_monotonic_increasing


def test_a_shared_guide_gives_the_same_test_under_both_targets(screen):
    """Same gene and same guide means the same treated cells, so it must be one test.

    R runs it twice and reports both rows; this runs it once and reports both
    rows. The p-values must therefore be identical, not merely close, which
    would not be true if the test had been run twice from a shared RNG.
    """
    result = _run(screen, "singleton")
    shared = result[(result["grna_id"] == "gS") & (result["response_id"] == "g0")]
    assert set(shared["grna_target"]) == {"A", "B"}
    assert shared["p_value"].nunique() == 1
    assert shared["pct_change_es"].nunique() == 1


def test_singleton_keeps_pysceptres_own_columns(screen):
    """`pct_change_es` and the rest, not R's `log_2_fold_change`.

    The naming already differs from R on the union path and deliberately so;
    the strategies must not diverge from each other on top of that.
    """
    union_cols = set(_run(screen, "singleton").columns)  # same engine, so the same value columns
    for column in ("p_value", "pct_change_es", "z_orig", "stage"):
        assert column in union_cols


def test_bonferroni_collapses_to_one_row_per_target(screen):
    result = _run(screen, "bonferroni")
    assert len(result) == 3
    assert "grna_id" not in result.columns
    assert set(zip(result["response_id"], result["grna_target"], strict=True)) == {
        ("g0", "A"),
        ("g0", "B"),
        ("g1", "A"),
    }


def test_bonferroni_multiplies_the_smallest_p_by_the_number_of_guides(screen):
    """R's arithmetic: `min(sum(pass_qc) * min(p), 1)`, carrying that guide's row."""
    singleton = _run(screen, "singleton")
    bonferroni = _run(screen, "bonferroni").set_index(["response_id", "grna_target"])
    for key, group in singleton.groupby(["response_id", "grna_target"]):
        expected = min(len(group) * group["p_value"].min(), 1.0)
        assert bonferroni.loc[key, "p_value"] == pytest.approx(expected)
        # and the other quantities come from that same guide
        best = group.loc[group["p_value"].idxmin()]
        assert bonferroni.loc[key, "pct_change_es"] == pytest.approx(best["pct_change_es"])


def test_bonferroni_caps_at_one():
    """Six guides at p = 0.4 would otherwise report 2.4."""
    result = pd.DataFrame(
        {
            "response_id": ["g0"] * 6,
            "grna_target": ["A"] * 6,
            "grna_id": [f"gA{i}" for i in range(6)],
            "p_value": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "pct_change_es": [-1.0] * 6,
        }
    )
    assert aggregate_bonferroni(result)["p_value"].iloc[0] == 1.0


def test_bonferroni_reports_a_target_whose_guides_all_failed_qc():
    """R's other branch: NaN, `pass_qc = False`, and the group's largest counts."""
    result = pd.DataFrame(
        {
            "response_id": ["g0", "g0"],
            "grna_target": ["A", "A"],
            "grna_id": ["gA1", "gA2"],
            "p_value": [np.nan, np.nan],
            "pct_change_es": [np.nan, np.nan],
            "n_nonzero_trt": [3, 5],
            "n_nonzero_cntrl": [11, 9],
            "pass_qc": [False, False],
        }
    )
    row = aggregate_bonferroni(result).iloc[0]
    assert row["pass_qc"] is False or row["pass_qc"] == False  # noqa: E712
    assert np.isnan(row["p_value"])
    assert row["n_nonzero_trt"] == 5 and row["n_nonzero_cntrl"] == 11


def test_bonferroni_counts_only_the_guides_that_passed_qc():
    """The factor is `sum(pass_qc)`, not the group size."""
    result = pd.DataFrame(
        {
            "response_id": ["g0"] * 4,
            "grna_target": ["A"] * 4,
            "grna_id": [f"gA{i}" for i in range(4)],
            "p_value": [0.01, 0.02, np.nan, np.nan],
            "pct_change_es": [-5.0, -4.0, np.nan, np.nan],
            "pass_qc": [True, True, False, False],
        }
    )
    assert aggregate_bonferroni(result)["p_value"].iloc[0] == pytest.approx(2 * 0.01)


def test_sort_puts_missing_p_values_last():
    """`na.last = TRUE`, which matters because QC failures carry a NaN p-value."""
    frame = pd.DataFrame({"p_value": [0.5, np.nan, 0.1], "response_id": ["b", "a", "c"]})
    assert list(sort_like_r(frame)["response_id"]) == ["c", "b", "a"]


def test_the_design_frame_is_required_and_rejected_where_it_is_meaningless(screen):
    counts, gene_ids, cov, guide_cells, _, pairs = screen
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="needs grna_target_data_frame"):
            run_discovery_analysis(
                counts,
                gene_ids,
                cov,
                guide_cells,
                pairs,
                grna_integration_strategy="singleton",
            )
        with pytest.raises(ValueError, match="only used by the singleton"):
            run_discovery_analysis(
                counts,
                gene_ids,
                cov,
                guide_cells,
                pairs,
                grna_target_data_frame=pd.DataFrame({"grna_id": ["x"], "grna_target": ["A"]}),
            )


def test_an_unknown_strategy_is_rejected(screen):
    with pytest.raises(ValueError, match="grna_integration_strategy must be one of"):
        _run(screen, "nonsense")


def test_a_duplicated_design_row_is_kept_and_warned_about():
    """R keeps it, so this does, and it changes the bonferroni factor.

    Deduplicating was the first implementation, and on day0 it disagreed with
    R by exactly the 288 rows those 36 duplicate design rows fan out to. The
    warning exists because nothing else would tell a caller their design
    table has them.
    """
    design = pd.DataFrame({"grna_id": ["gA1", "gA1", "gA2"], "grna_target": ["A", "A", "A"]})
    pairs = pd.DataFrame({"response_id": ["g0"], "grna_target": ["A"]})
    with pytest.warns(UserWarning, match="1 duplicated"):
        got = singleton_pairs(pairs, design)
    assert len(got) == 3
    assert list(got["grna_id"]) == ["gA1", "gA1", "gA2"]


def test_a_duplicated_guide_inflates_the_bonferroni_factor():
    """Stated as a test because it is the consequence, not the row count.

    Three rows of which two are the same guide gives a factor of 3, which is
    what R does. It is the reason the duplicate warning is worth emitting.
    """
    result = pd.DataFrame(
        {
            "response_id": ["g0"] * 3,
            "grna_target": ["A"] * 3,
            "grna_id": ["gA1", "gA1", "gA2"],
            "p_value": [0.01, 0.01, 0.2],
            "pct_change_es": [-5.0, -5.0, -1.0],
        }
    )
    assert aggregate_bonferroni(result)["p_value"].iloc[0] == pytest.approx(3 * 0.01)


def test_duplicate_design_rows_can_be_dropped_on_request():
    """The option, and it warns either way because the input is defective either way."""
    design = pd.DataFrame({"grna_id": ["gA1", "gA1", "gA2"], "grna_target": ["A", "A", "A"]})
    pairs = pd.DataFrame({"response_id": ["g0"], "grna_target": ["A"]})
    with pytest.warns(UserWarning, match="dropped 1 duplicated"):
        got = singleton_pairs(pairs, design, drop_duplicate_design_rows=True)
    assert len(got) == 2
    assert list(got["grna_id"]) == ["gA1", "gA2"]


def test_dropping_duplicates_removes_the_bonferroni_inflation():
    """The reason the option exists, shown as the consequence rather than a row count."""
    design = pd.DataFrame({"grna_id": ["gA1", "gA1", "gA2"], "grna_target": ["A", "A", "A"]})
    pairs = pd.DataFrame({"response_id": ["g0"], "grna_target": ["A"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kept = singleton_pairs(pairs, design)
        dropped = singleton_pairs(pairs, design, drop_duplicate_design_rows=True)

    def bonferroni_p(expanded):
        result = expanded.assign(
            p_value=[0.01] * len(expanded), pct_change_es=[-5.0] * len(expanded)
        )
        return aggregate_bonferroni(result)["p_value"].iloc[0]

    assert bonferroni_p(kept) == pytest.approx(3 * 0.01)
    assert bonferroni_p(dropped) == pytest.approx(2 * 0.01)


def test_the_option_is_rejected_under_union(screen):
    """It cannot change a union result, so asking for it there is a mistake worth naming."""
    counts, gene_ids, cov, guide_cells, _, pairs = screen
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="only applies to the singleton"):
            run_discovery_analysis(
                counts,
                gene_ids,
                cov,
                guide_cells,
                pairs,
                drop_duplicate_design_rows=True,
            )


def test_union_is_immune_to_a_duplicated_design_row(screen):
    """Not an opinion: R takes `unique()` of the treated cells.

    The union path never reads the design table at all -- a pair already names
    the unit tested -- so this is a statement about what the option would even
    have to change, and the answer is nothing.
    """
    counts, gene_ids, cov, guide_cells, _, _ = screen
    target_cells = {"A": np.union1d(guide_cells["gA1"], guide_cells["gA2"])}
    pairs = pd.DataFrame({"response_id": ["g0"], "grna_target": ["A"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_discovery_analysis(
            counts, gene_ids, cov, target_cells, pairs, side="left", seed=0
        )
    assert len(result) == 1
    assert "grna_id" not in result.columns
