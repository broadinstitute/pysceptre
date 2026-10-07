"""The fishash port (`pysceptre.assignment.fishash`) against fishash's own R, value for value.

The fixture (`scripts/dump_fishash_ground_truth.R`) runs fishash 0.99.5 on small synthetic
count matrices -- simulated with its own simulator, or built by hand to reach edge cases -- and
records each pass's internals through `trace()`. Three layers are checked:

- full runs: final log p-values, odds ratios, cut, calls, per-cell type and assignment strings,
  number of passes and each pass's B (or n_signif) and number of calls;
- stepwise: each pass of the deep run is fed R's own background for that pass, so an error in
  one pass cannot hide behind or cascade into another;
- the noise imputation, called directly and inside the deep run.

Tolerances: log p 1e-12 relative to max(1, |R|) (the same algorithm on the same rounded
arguments, so only libm's last bits differ); odds ratios 1e-14 relative at refit 0 and 1e-10
once they inherit an imputed background; the imputation's factors and values 1e-10 relative
(at most ten fixed-point sweeps, sums in R's order); calls, cuts' B and n_signif, pass counts,
cell types and strings exactly. The dumper keeps every log p-value at least 1e-6 from its cut,
so any flip is a defect. Both the numba kernel and the numpy fallback run.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from pysceptre.assignment import fishash as fh

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from assignment_compare import describe, flips  # noqa: E402

RTOL_LOG_P = 1e-12
RTOL_ODDS_REFIT0 = 1e-14
RTOL_ODDS_REFIT = 1e-10
RTOL_IMPUTE = 1e-10
DUMPER = Path(__file__).resolve().parents[2] / "scripts" / "dump_fishash_ground_truth.R"

_SPECIAL = {"Inf": np.inf, "-Inf": -np.inf, "NaN": np.nan, "NA": np.nan}


def _num(values) -> np.ndarray:
    if not isinstance(values, list):
        values = [values]
    return np.array([_SPECIAL[v] if isinstance(v, str) else v for v in values], dtype=np.float64)


def _matrix(trip, dims) -> sparse.csc_matrix:
    i = np.asarray(trip["i"], dtype=np.int64)
    j = np.asarray(trip["j"], dtype=np.int64)
    x = _num(trip["x"])
    return sparse.csc_matrix((x, (i, j)), shape=tuple(dims))


def _assert_close(py, r, rtol, label):
    py = np.asarray(py, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    assert py.shape == r.shape, f"{label}: shapes {py.shape} and {r.shape}"
    np.testing.assert_array_equal(np.isnan(py), np.isnan(r), err_msg=f"{label}: NaN pattern")
    np.testing.assert_array_equal(np.isposinf(py), np.isposinf(r), err_msg=f"{label}: +inf")
    np.testing.assert_array_equal(np.isneginf(py), np.isneginf(r), err_msg=f"{label}: -inf")
    fin = np.isfinite(r)
    if fin.any():
        err = np.abs(py[fin] - r[fin]) / np.maximum(1.0, np.abs(r[fin]))
        assert err.max() <= rtol, f"{label}: max relative error {err.max():.3g} (tolerance {rtol})"


def _run_kwargs(case, run):
    a = run["args"]
    kwargs = {
        "padj_cutoff": a.get("padj_cutoff", 0.05),
        "padj_method": a.get("padj_method", "GS"),
        "refit": int(a.get("refit", 10)),
        "exclude_empty": bool(a.get("exclude_empty", True)),
        "min_count": a.get("min_count", 2),
        "min_frac": a.get("min_frac", 0.0),
    }
    if run.get("background"):
        kwargs["background"] = _matrix(case["background"], case["dims"])
    return kwargs


def _entries(case):
    """R's entry order and the stored counts (stored zeros included)."""
    i = np.asarray(case["counts"]["i"], dtype=np.int64)
    j = np.asarray(case["counts"]["j"], dtype=np.int64)
    x = _num(case["counts"]["x"])
    return i, j, x


def _at(mat, i, j) -> np.ndarray:
    if i.size == 0:
        return np.zeros(0)
    return np.asarray(sparse.csr_matrix(mat)[i, j]).ravel()


def _positions_to_matrix(positions, i, j, dims) -> sparse.csr_matrix:
    pos = np.asarray(positions, dtype=np.int64)
    return sparse.csr_matrix((np.ones(pos.size, dtype=bool), (i[pos], j[pos])), shape=tuple(dims))


def test_fixture_is_from_this_dumper(fishash_ground_truth):
    recorded = fishash_ground_truth["provenance"]["dumper_md5"]
    current = hashlib.md5(DUMPER.read_bytes()).hexdigest()
    assert recorded == current, (
        "fishash_ground_truth.json.gz was made by a different version of "
        "scripts/dump_fishash_ground_truth.R; delete the .gz and rerun the tests to regenerate it"
    )


def test_fixture_provenance(fishash_ground_truth):
    prov = fishash_ground_truth["provenance"]
    assert prov["fishash"]["version"] == "0.99.5"
    assert prov["fishash"]["remote_sha"] == "5eabd3c5d93b08a8f54ad9803a17f235975a0c61"


def test_full_runs_match_r(fishash_ground_truth, assignment_kernel):
    failures = []
    n_runs = 0
    for case in fishash_ground_truth["cases"]:
        counts = _matrix(case["counts"], case["dims"])
        ids = case["grna_ids"]
        i, j, x = _entries(case)
        nz = x != 0
        for r_idx, run in enumerate(case["runs"]):
            label = f"{case['id']} run {r_idx} {run['args']}"
            kwargs = _run_kwargs(case, run)
            if run.get("error"):
                with pytest.raises((ValueError, RuntimeError)):
                    fh.assign_grnas_fishash(counts, ids, **kwargs)
                n_runs += 1
                continue
            try:
                res = fh.assign_grnas_fishash(counts, ids, **kwargs)
            except Exception as e:  # noqa: BLE001 - collected and reported together
                failures.append(f"{label}: raised {e!r}")
                continue
            n_runs += 1
            try:
                assert res.num_iter == run["num_iter"], (
                    f"num_iter {res.num_iter} vs {run['num_iter']}"
                )
                r_cut = _num(run["cutoff"])[0]
                assert (np.isneginf(r_cut) and np.isneginf(res.log_pval_cutoff)) or abs(
                    res.log_pval_cutoff - r_cut
                ) <= 1e-12, f"cutoff {res.log_pval_cutoff} vs {r_cut}"
                for p, (it, rp) in enumerate(zip(res.iterations, run["passes"], strict=True)):
                    r_sig = rp["B"] if rp["B"] != "NA" else rp["n_signif"]
                    assert it.n_significant == int(r_sig), f"pass {p + 1}: B/n_signif"
                    assert it.n_assigned == len(rp["assigned"]), f"pass {p + 1}: n_assigned"
                for it, rim in zip(res.iterations[1:], run["imputes"], strict=True):
                    assert it.impute_n_iter == rim["n_iter"], "imputation inner iterations"
                _assert_close(_at(res.log_pval, i[nz], j[nz]), _num(run["log_pval"])[nz],
                              RTOL_LOG_P, "log_pval")  # fmt: skip
                if "odds_ratio" in run:
                    rtol = (
                        RTOL_ODDS_REFIT0
                        if res.num_iter == 1 and not run.get("background")
                        else RTOL_ODDS_REFIT
                    )
                    for key in ("odds_ratio", "odds_ratio_regularized"):
                        _assert_close(
                            _at(getattr(res, key), i[nz], j[nz]), _num(run[key])[nz], rtol, key
                        )
                r_assigned = _positions_to_matrix(run["assigned"], i, j, case["dims"])
                table = flips(r_assigned, res.assigned, counts=counts)
                assert table.empty, describe(table)
                assert list(res.demux_type) == run["demux_type"], "demux_type"
                assert list(res.assignment) == run["assignment"], "assignment strings"
            except AssertionError as e:
                failures.append(f"{label} [{assignment_kernel}]: {e}")
    assert n_runs > 30
    assert not failures, "\n".join(failures)


def _deep_run(fishash_ground_truth):
    case = next(c for c in fishash_ground_truth["cases"] if c["id"] == "sim_a")
    run = case["runs"][0]
    assert run["args"]["refit"] == 10 and "background" in run["passes"][0]
    return case, run


def test_each_pass_from_rs_background(fishash_ground_truth, assignment_kernel):
    case, run = _deep_run(fishash_ground_truth)
    counts = fh.as_count_matrix(_matrix(case["counts"], case["dims"]))
    e = fh._Entries.from_csc(counts)
    i, j, _ = _entries(case)
    x = np.floor((e.data - 1.0) + 1e-7)
    k = np.rint(e.col_sums[e.cols])
    assert len(run["passes"]) >= 5
    for p, rp in enumerate(run["passes"], start=1):
        bg = _num(rp["background"])
        out = fh._one_pass(
            e,
            bg,
            np.bincount(e.rows, weights=bg, minlength=e.n_grnas),
            np.bincount(e.cols, weights=bg, minlength=e.n_cells),
            fh._seqsum(bg),
            x,
            k,
            None,
            0.05,
            "GS",
            2,
            0.0,
            True,
        )
        _assert_close(out.log_pval, _num(rp["log_pval"]), RTOL_LOG_P, f"pass {p} log_pval")
        assert abs(out.cutoff - _num(rp["cutoff"])[0]) <= 1e-12, f"pass {p} cutoff"
        assert out.n_significant == int(rp["B"]), f"pass {p} B"
        np.testing.assert_array_equal(
            np.flatnonzero(out.assigned), np.asarray(rp["assigned"]), err_msg=f"pass {p} calls"
        )


def test_deep_run_imputations(fishash_ground_truth):
    case, run = _deep_run(fishash_ground_truth)
    counts = fh.as_count_matrix(_matrix(case["counts"], case["dims"]))
    e = fh._Entries.from_csc(counts)
    assert len(run["imputes"]) >= 4
    for q, rim in enumerate(run["imputes"], start=2):
        mask = np.zeros(e.data.size, dtype=bool)
        mask[np.asarray(rim["mask"], dtype=np.int64)] = True
        values, gf, cs, n_iter = fh._impute_aligned(e, e.data, mask, eps=1e-4, max_iter=10)
        assert n_iter == rim["n_iter"], f"before pass {q}: inner iterations"
        _assert_close(gf, _num(rim["guide_freqs"]), RTOL_IMPUTE, f"before pass {q}: guide_freqs")
        _assert_close(cs, _num(rim["cell_sizes"]), RTOL_IMPUTE, f"before pass {q}: cell_sizes")
        _assert_close(values[mask], _num(rim["imputed"]), RTOL_IMPUTE, f"before pass {q}: imputed")


def test_impute_masked_counts_matches_r(fishash_ground_truth):
    for unit in fishash_ground_truth["impute_unit"]:
        dims = unit["dims"]
        counts = _matrix(unit["counts"], dims)
        mask = _matrix(unit["mask"], dims)
        out = fh.impute_masked_counts(counts, mask, eps=unit["eps"], max_iter=unit["max_iter"])
        label = unit["label"]
        assert out.n_iter == unit["n_iter"], f"{label}: n_iter"
        _assert_close(
            out.guide_freqs, _num(unit["guide_freqs"]), RTOL_IMPUTE, f"{label}: guide_freqs"
        )
        _assert_close(out.cell_sizes, _num(unit["cell_sizes"]), RTOL_IMPUTE, f"{label}: cell_sizes")
        expected = _matrix(unit["out"], dims).toarray()
        _assert_close(out.counts.toarray(), expected, RTOL_IMPUTE, f"{label}: imputed matrix")


def test_integer_and_float_counts_agree(fishash_ground_truth):
    case = next(c for c in fishash_ground_truth["cases"] if c["id"] == "sim_a")
    counts = _matrix(case["counts"], case["dims"])
    ids = case["grna_ids"]
    base = fh.assign_grnas_fishash(counts, ids)
    as_int = fh.assign_grnas_fishash(counts.astype(np.int64), ids)
    as_dense = fh.assign_grnas_fishash(counts.toarray(), ids)
    for other in (as_int, as_dense):
        assert (base.assigned != other.assigned).nnz == 0
        assert base.num_iter == other.num_iter


def test_non_integer_counts_are_refused():
    with pytest.raises(ValueError, match="integer"):
        fh.assign_grnas_fishash(sparse.csc_matrix([[1.5, 2.0], [3.0, 0.0]]), ["a", "b"])


def test_background_with_refit_is_refused():
    m = sparse.csc_matrix([[3.0, 0.0], [0.0, 4.0]])
    with pytest.raises(ValueError, match="background"):
        fh.assign_grnas_fishash(m, ["a", "b"], refit=1, background=m)
