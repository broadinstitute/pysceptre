"""sceptre's thresholding and maximum assignment, against sceptre 0.10.3 itself.

The fixture (`scripts/dump_assignment_rules_ground_truth.R`) runs sceptre's public path,
`import_data` -> `set_analysis_parameters` -> `assign_grnas`, on synthetic counts with cells on
every cut: ties for the top gRNA, empty cells, top shares of exactly 0.8 and 0.9, gRNA UMI
totals of 4 and 5, and counts equal to each threshold. Every value compared is exact, so every
comparison is equality.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from pysceptre.assignment import (
    MaximumResult,
    assign_grnas,
    assign_grnas_fishash,
    assign_grnas_maximum,
    cells_w_zero_or_twoplus_grnas,
)

DUMPER = Path(__file__).resolve().parents[2] / "scripts" / "dump_assignment_rules_ground_truth.R"
RUNS = (
    "maximum_default",
    "maximum_frac0.5_umis0",
    "maximum_frac0.9_umis10",
    "default_low",
    "thresholding_5_low",
    "thresholding_1_low",
    "thresholding_7_low",
    "thresholding_5_high",
    "thresholding_none_low",
)


def _counts(gt: dict) -> sparse.csr_matrix:
    c = gt["counts"]
    return sparse.csr_matrix(
        (np.asarray(c["count"], dtype=np.float64), (np.asarray(c["grna"]), np.asarray(c["cell"]))),
        shape=(gt["n_grnas"], gt["n_cells"]),
    )


def _num(values) -> np.ndarray:
    return np.array([np.nan if v == "NA" else float(v) for v in values])


def _run(gt: dict, label: str) -> dict:
    return next(r for r in gt["runs"] if r["label"] == label)


def _hyper(run: dict) -> dict:
    return run["hyperparameters"] or {}


def test_fixture_is_from_this_dumper(assignment_rules_ground_truth):
    recorded = assignment_rules_ground_truth["provenance"]["dumper_md5"]
    current = hashlib.md5(DUMPER.read_bytes()).hexdigest()
    assert recorded == current, (
        "assignment_rules_ground_truth.json.gz was made by a different version of "
        "scripts/dump_assignment_rules_ground_truth.R; delete the .gz and rerun the tests"
    )


def test_fixture_provenance_and_runs(assignment_rules_ground_truth):
    gt = assignment_rules_ground_truth
    assert gt["provenance"]["sceptre"]["version"] == "0.10.3"
    assert tuple(r["label"] for r in gt["runs"]) == RUNS


def test_top_grna_share_and_umis_are_import_datas(assignment_rules_ground_truth):
    gt = assignment_rules_ground_truth
    res = assign_grnas_maximum(_counts(gt), gt["grna_ids"])
    assert np.array_equal(res.max_grna, np.asarray(gt["max_grna"]))
    np.testing.assert_array_equal(res.max_grna_frac_umis, _num(gt["max_grna_frac_umis"]))
    np.testing.assert_array_equal(res.grna_n_umis, _num(gt["grna_n_umis"]))


@pytest.mark.parametrize("label", RUNS)
def test_assignments_and_flagged_cells_match_r(assignment_rules_ground_truth, label):
    gt = assignment_rules_ground_truth
    run = _run(gt, label)
    res = assign_grnas(
        _counts(gt), gt["grna_ids"], method=run["method"], moi=run["moi"], **_hyper(run)
    )
    a = res.assigned.tocsr()
    a.sort_indices()
    for r, grna_id in enumerate(gt["grna_ids"]):
        got = a.indices[a.indptr[r] : a.indptr[r + 1]]
        assert np.array_equal(got, np.asarray(run["assignments"][grna_id])), (label, grna_id)

    expected = np.asarray(run["cells_w_zero_or_twoplus_grnas"], dtype=np.int64)
    if isinstance(res, MaximumResult):
        assert np.array_equal(res.cells_w_zero_or_twoplus_grnas, expected)
    elif run["moi"] == "low" and a.nnz:
        assert np.array_equal(cells_w_zero_or_twoplus_grnas(a), expected)
    elif run["moi"] == "low":
        # Nothing assigned: sceptre's indexing flags no cell; the port flags them all
        # (docs/design.md, "Thresholding and maximum").
        assert expected.size == 0
        assert np.array_equal(cells_w_zero_or_twoplus_grnas(a), np.arange(gt["n_cells"]))
    else:
        assert expected.size == 0  # sceptre flags cells in low MOI only


def test_inputs_sceptre_refuses_are_refused(assignment_rules_ground_truth):
    gt = assignment_rules_ground_truth
    counts = _counts(gt)
    for case in gt["refusals"]:
        assert case["refused"], case["label"]
        with pytest.raises(ValueError):
            assign_grnas(
                counts,
                gt["grna_ids"],
                method=case["method"],
                moi=case["moi"],
                **(case["hyperparameters"] or {}),
            )


def test_assign_grnas_routes_and_refuses():
    counts = np.array([[0, 3, 5, 0], [2, 0, 0, 6], [0, 0, 1, 0]])
    ids = ["a", "b", "c"]
    via = assign_grnas(counts, ids, method="fishash", refit=0)
    direct = assign_grnas_fishash(counts, ids, refit=0)
    assert (via.assigned != direct.assigned).nnz == 0
    for kwargs in (
        {"method": "default"},
        {"method": "mixture"},
        {"method": "thresholding", "covariate_matrix": np.ones((4, 1))},
        {"method": "nearest"},
        {"method": "thresholding", "moi": "medium"},
        {"method": "thresholding", "threshold": float("nan")},
    ):
        with pytest.raises(ValueError):
            assign_grnas(counts, ids, **kwargs)
    with pytest.raises(TypeError):
        assign_grnas(counts, ids, method="maximum", threshold=5)
