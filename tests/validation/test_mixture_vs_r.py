"""The port of sceptre's mixture assignment (`pysceptre.assignment.mixture`) against sceptre.

The fixture (`scripts/dump_mixture_ground_truth.R`) runs sceptre 0.10.3 on synthetic gRNA rows
built to reach each path -- the EM, its pi > 0.5 flip, its 1e-100 floor, the backup rule for
too few cells, and an EM that fails after glm.fit itself fails to converge -- and records
sceptre's internals. The layers checked, and their tolerances:

- the ten starting constants: exactly;
- the default design, on the main data and three variants: column names exactly, values to
  1e-15 relative (logs of whole numbers);
- the EM fed R's own fitted means: a test-only dense replica of the C++ (`r_mixture.py`)
  against R, posteriors to 1e-9 and log-likelihoods to 1e-12 relative, with the chosen start
  and convergence flags exact unless two starts tie within 1e-9; the package's EM, which
  groups the zero-count cells, against the replica to 1e-10;
- the Poisson GLM against glm.fit: the existing pysceptre-against-glm.fit tolerance (rtol
  1e-6, atol 1e-8), for gRNAs where glm.fit converged;
- end to end with R's design: each gRNA's path and assigned cells exactly (the fixture keeps
  every EM-path posterior at least 1e-3 from the cut), posteriors to 1e-4.

Both the numba EM and its numpy fallback run, through the `assignment_kernel` fixture.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.special import gammaln

from pysceptre.assignment import design as dz
from pysceptre.assignment import mixture as mx
from pysceptre.glm.irls import fit_poisson_glm_batch

sys.path.insert(0, str(Path(__file__).parent))
from r_mixture import run_reduced_em  # noqa: E402

DUMPER = Path(__file__).resolve().parents[2] / "scripts" / "dump_mixture_ground_truth.R"
_SPECIAL = {"Inf": np.inf, "-Inf": -np.inf, "NaN": np.nan, "NA": np.nan}


def _num(values) -> np.ndarray:
    if not isinstance(values, list):
        values = [values]
    return np.array([_SPECIAL[v] if isinstance(v, str) else v for v in values], dtype=np.float64)


def _dense_row(rec, n_cells) -> np.ndarray:
    g = np.zeros(n_cells)
    g[np.asarray(rec["j"], dtype=np.int64)] = _num(rec["x"])
    return g


def _grna_matrix(gt) -> sparse.csr_matrix:
    n = gt["n_cells"]
    return sparse.csr_matrix(np.vstack([_dense_row(r, n) for r in gt["grnas"]]))


def _design(gt) -> np.ndarray:
    return np.column_stack([_num(col) for col in gt["design"]])


def _near_tie(rec) -> bool:
    gap = rec.get("start_gap")
    if gap is None or isinstance(gap, str):
        return False
    return gap <= 1e-9 * abs(_num(rec["em"]["outer_log_lik"])[0])


def test_fixture_is_from_this_dumper(mixture_ground_truth):
    recorded = mixture_ground_truth["provenance"]["dumper_md5"]
    current = hashlib.md5(DUMPER.read_bytes()).hexdigest()
    assert recorded == current, (
        "mixture_ground_truth.json.gz was made by a different version of "
        "scripts/dump_mixture_ground_truth.R; delete the .gz and rerun the tests to regenerate it"
    )


def test_fixture_provenance(mixture_ground_truth):
    prov = mixture_ground_truth["provenance"]
    assert prov["sceptre"]["version"] == "0.10.3"
    assert prov["sceptre"]["remote_sha"] == "552c342b0c0215353931ca7ef4fd6f2b37270bf9"
    assert mixture_ground_truth["public_equals_internal"] is True


def test_starting_guesses_are_sceptres(mixture_ground_truth):
    sg = mixture_ground_truth["starting_guesses"]
    assert list(mx.PI_GUESSES) == list(_num(sg["pi"]))
    assert list(mx.G_PERT_GUESSES) == list(_num(sg["g_pert"]))


def test_cell_count_covariates(mixture_ground_truth):
    cov = mixture_ground_truth["covariates"]
    n_nonzero, n_umis = dz.cell_count_covariates(_grna_matrix(mixture_ground_truth))
    np.testing.assert_array_equal(n_nonzero, _num(cov["grna_n_nonzero"]))
    np.testing.assert_array_equal(n_umis, _num(cov["grna_n_umis"]))


def test_default_design_matches_model_matrix(mixture_ground_truth):
    gt = mixture_ground_truth
    cov = gt["covariates"]
    X, names = dz.mixture_design_matrix(
        _grna_matrix(gt),
        response_n_nonzero=_num(cov["response_n_nonzero"]),
        response_n_umis=_num(cov["response_n_umis"]),
    )
    assert names == gt["design_columns"]
    np.testing.assert_allclose(X, _design(gt), rtol=1e-15, atol=0)


@pytest.mark.parametrize("label", ["logx", "extra", "reduced"])
def test_design_variants(mixture_ground_truth, label):
    variant = next(v for v in mixture_ground_truth["design_variants"] if v["label"] == label)
    expected = np.column_stack([_num(c) for c in variant["X"]])
    if label == "reduced":
        X, names = dz.mixture_design_matrix(_grna_matrix(mixture_ground_truth))
    else:
        frame = pd.DataFrame(
            {
                k: (_num(v) if not isinstance(v[0], str) or v[0] in _SPECIAL else v)
                for k, v in variant["covariates"].items()
            }
        )
        X, names = dz.design_from_covariates(frame)
    assert names == variant["columns"], label
    np.testing.assert_allclose(X, expected, rtol=1e-15, atol=0, err_msg=label)


def _em_records(gt):
    return [r for r in gt["grnas"] if "em" in r]


def test_replica_matches_r_em(mixture_ground_truth):
    gt = mixture_ground_truth
    n = gt["n_cells"]
    for rec in _em_records(gt):
        g = _dense_row(rec, n)
        mu0 = _num(rec["glm"]["fitted"])
        out = run_reduced_em(mx.PI_GUESSES, mx.G_PERT_GUESSES, g, mu0, gammaln(g + 1.0))
        em = rec["em"]
        label = rec["grna_id"]
        assert out["outer_converged"] == em["outer_converged"], label
        r_ll = _num(em["outer_log_lik"])[0]
        if np.isfinite(r_ll):
            np.testing.assert_allclose(out["outer_log_lik"], r_ll, rtol=1e-12, err_msg=label)
        else:
            assert out["outer_log_lik"] == r_ll, label
        if not _near_tie(rec):
            assert out["outer_i"] == em["outer_i"], label
        np.testing.assert_allclose(
            out["outer_Ti1s"], _num(em["outer_ti1s"]), rtol=1e-9, atol=1e-12, err_msg=label
        )
        for b, (s_py, s_r) in enumerate(zip(out["starts"], em["starts"], strict=True)):
            assert s_py["converged"] == s_r["converged"], f"{label} start {b}"
            ll = _num(s_r["log_lik"])[0]
            if np.isfinite(ll):
                np.testing.assert_allclose(s_py["log_lik"], ll, rtol=1e-12, err_msg=f"{label} {b}")


def test_grouped_em_matches_replica(mixture_ground_truth, assignment_kernel):
    gt = mixture_ground_truth
    n = gt["n_cells"]
    for rec in _em_records(gt):
        g = _dense_row(rec, n)
        mu0 = _num(rec["glm"]["fitted"])
        cells = np.flatnonzero(g)
        ref = run_reduced_em(mx.PI_GUESSES, mx.G_PERT_GUESSES, g, mu0, gammaln(g + 1.0))
        out = mx._run_em(cells, g[cells], mu0, mx.PI_GUESSES, mx.G_PERT_GUESSES)
        label = f"{rec['grna_id']} [{assignment_kernel}]"
        assert out["converged"] == ref["outer_converged"], label
        if ref["outer_converged"]:
            np.testing.assert_allclose(
                out["log_lik"], ref["outer_log_lik"], rtol=1e-10, err_msg=label
            )
            np.testing.assert_allclose(
                out["ti_all"], ref["outer_Ti1s"], rtol=1e-10, atol=1e-14, err_msg=label
            )
            if not _near_tie(rec):
                assert out["best_start"] == ref["outer_i"], label
        np.testing.assert_array_equal(
            out["start_converged"], [s["converged"] for s in ref["starts"]], err_msg=label
        )


def test_glm_matches_glm_fit(mixture_ground_truth):
    gt = mixture_ground_truth
    n = gt["n_cells"]
    X = _design(gt)
    xo = None
    checked = 0
    for rec in _em_records(gt):
        if not rec["glm"]["converged"]:
            continue
        g = _dense_row(rec, n)
        fit = fit_poisson_glm_batch(X, g, X_outer_flat=xo)
        np.testing.assert_allclose(
            fit.fitted_values,
            _num(rec["glm"]["fitted"]),
            rtol=1e-6,
            atol=1e-8,
            err_msg=rec["grna_id"],
        )
        np.testing.assert_allclose(
            fit.coefs,
            _num(rec["glm"]["coefficients"]),
            rtol=1e-6,
            atol=1e-8,
            err_msg=rec["grna_id"],
        )
        checked += 1
    assert checked >= 10


def test_end_to_end_matches_r(mixture_ground_truth, assignment_kernel):
    gt = mixture_ground_truth
    n = gt["n_cells"]
    res = mx.assign_grnas_mixture(_grna_matrix(gt), gt["grna_ids"], _design(gt))
    failures = []
    for r, rec in enumerate(gt["grnas"]):
        label = f"{rec['grna_id']} [{assignment_kernel}]"
        fit = res.fits.iloc[r]
        r_path = {"mixture": "mixture", "backup_em": "backup", "backup_n_nonzero": "backup"}[
            rec["path"]
        ]
        if fit["method"] != r_path:
            failures.append(f"{label}: path {fit['method']} vs R {rec['path']}")
        got = np.sort(res.assigned.getrow(r).indices)
        if not np.array_equal(got, np.asarray(rec["assigned"], dtype=np.int64)):
            failures.append(f"{label}: assigned {got.tolist()} vs R {rec['assigned']}")
        if rec["path"] == "mixture":
            ti = np.full(n, res.posterior_zero[r])
            row = res.posterior.getrow(r)
            ti[row.indices] = row.data
            try:
                np.testing.assert_allclose(ti, _num(rec["em"]["outer_ti1s"]), rtol=1e-4, atol=1e-6)
            except AssertionError as e:
                failures.append(f"{label}: posterior {e}")
    assert not failures, "\n".join(failures)


def test_backup_and_cutoff_rules():
    g = sparse.csr_matrix(np.array([[0, 6, 0, 9, 1, 0, 2, 0, 0, 7, 0, 0]], dtype=float))
    X = np.column_stack([np.ones(12), np.linspace(0.0, 1.0, 12)])
    res = mx.assign_grnas_mixture(g, ["g"], X)
    assert res.fits.loc[0, "method"] == "backup"
    assert res.fits.loc[0, "backup_reason"] == "n_nonzero"
    np.testing.assert_array_equal(res.assigned.getrow(0).indices, [1, 3, 9])
