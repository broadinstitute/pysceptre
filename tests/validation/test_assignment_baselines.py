"""The fishash evaluation's baselines and scoring (`scripts/fishash_eval/eval_lib.py`).

Not a sceptre comparison. What is checked:

- `cmo_clr_q95` against the lab's CMO code, pasted below verbatim except that its hard-coded
  25 tags read the matrix width: every CLR value, geometric mean, threshold, call, count,
  argmax and status, bit for bit, on shapes, dtypes and chunk sizes picked to reach each edge
  of the chunking (all-zero cells and tags, a tag whose 95th percentile is 0, duplicated
  cells, a tie at the threshold, chunk sizes that do not divide the shapes);
- the quantile callers and the normalizations against that reference and each other;
- `confusion` against a dense count of the same quantities;
- `call_gmm` against crispat's `fit_em`: a replica of its code wherever scikit-learn is
  installed, and the package itself where crispat is (the evaluation's own venv,
  `test_data/fishash_eval/venvs/eval`; not the repo's).

The numpy-only tests run in CI. Dense matrices appear only here, on small shapes, to build
the references.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "fishash_eval"))
import eval_lib  # noqa: E402

# ---------------------------------------------------------------------------------------------
# The lab's CMO code. The two functions are pasted verbatim (marimo cells take `np` as an
# argument); the only edit is `/ 25` -> `/ _cmo.shape[1]` in the generalised one. The
# assignment step is the lab's one-liner, written out.


def cmo_clr_normalization(adata_cmo, np):
    _cmo = adata_cmo.X
    _cmo = _cmo.toarray() if hasattr(_cmo, "toarray") else np.asarray(_cmo)
    # 1. Compute Seurat-style geometric mean per cell across the 25 features
    # Seurat sums log1p of non-zero values, divides by total features (25), then takes exp
    _log_counts = np.log1p(np.where(_cmo > 0, _cmo, 0))
    _gm = np.exp(np.sum(_log_counts, axis=1) / _cmo.shape[1])
    # 2. Divide raw counts by the geometric mean, then apply log1p
    clr = np.log1p(_cmo / _gm[:, None])
    return (clr,)


def cmo_clr_normalization_25(adata_cmo, np):
    _cmo = adata_cmo.X
    _cmo = _cmo.toarray() if hasattr(_cmo, "toarray") else np.asarray(_cmo)
    # 1. Compute Seurat-style geometric mean per cell across the 25 features
    # Seurat sums log1p of non-zero values, divides by total features (25), then takes exp
    _log_counts = np.log1p(np.where(_cmo > 0, _cmo, 0))
    _gm = np.exp(np.sum(_log_counts, axis=1) / 25)
    # 2. Divide raw counts by the geometric mean, then apply log1p
    clr = np.log1p(_cmo / _gm[:, None])
    return (clr,)


def find_cmo_thresholds(clr, np):
    positive_quantile_threshold = 0.95
    thresholds = np.quantile(clr, positive_quantile_threshold, axis=0)
    positive = clr > thresholds
    n_pos = positive.sum(axis=1)
    return n_pos, positive, positive_quantile_threshold, thresholds


def lab_reference(x: np.ndarray) -> dict:
    """The lab's procedure on a dense C-ordered cells x tags matrix, as `cmo_clr_q95` reports it."""
    (clr,) = cmo_clr_normalization(SimpleNamespace(X=x), np)
    n_pos, positive, _, thresholds = find_cmo_thresholds(clr, np)
    _top_cmo_idx = np.argmax(clr, axis=1)
    status = np.array(["Negative" if n == 0 else "Singlet" if n == 1 else "Doublet" for n in n_pos])
    # The geometric mean, by the lines of cmo_clr_normalization that compute it.
    _cmo = np.asarray(x)
    _log_counts = np.log1p(np.where(_cmo > 0, _cmo, 0))
    _gm = np.exp(np.sum(_log_counts, axis=1) / _cmo.shape[1])
    return {
        "clr": clr,
        "clr_gm": _gm,
        "thresholds": thresholds,
        "positive": positive,
        "n_pos": n_pos,
        "top_idx": _top_cmo_idx,
        "status": status,
    }


def same_bits(a, b) -> bool:
    """Equal shape, equal dtype and identical bytes (so 0.0 and -0.0 differ)."""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def cmo_counts(n_cells: int, n_tags: int, seed: int, *, integer: bool) -> np.ndarray:
    """A dense C-ordered cells x tags matrix shaped like CMO counts, with the edge cases.

    One dominant tag in most cells, a second in some (doublets), low noise elsewhere; then, as
    the shape allows, all-zero cells, a block of identical cells (ties in every column), an
    all-zero tag and a tag with at most 2% nonzero cells (its 95th percentile is 0). With
    `integer=False` every nonzero value is scaled by a random non-integer factor.
    """
    rng = np.random.default_rng(seed)
    x = rng.poisson(0.6, size=(n_cells, n_tags)) * (rng.random((n_cells, n_tags)) < 0.35)
    rows = np.arange(n_cells)
    x[rows, rng.integers(0, n_tags, n_cells)] += rng.poisson(25, n_cells) * (
        rng.random(n_cells) < 0.85
    )
    x[rows, rng.integers(0, n_tags, n_cells)] += rng.poisson(20, n_cells) * (
        rng.random(n_cells) < 0.1
    )
    if n_cells >= 20:
        x[: max(1, n_cells // 50)] = 0
        n_dup = n_cells // 10
        x[n_cells - n_dup :] = x[n_cells // 3]
    if n_tags >= 3:
        x[:, 1] = 0
        x[:, 2] = 0
        few = rng.choice(n_cells, size=n_cells // 50, replace=False)
        x[few, 2] = rng.poisson(30, few.size) + 1
    if integer:
        return np.ascontiguousarray(x, dtype=np.int64)
    return np.ascontiguousarray(x * rng.uniform(0.5, 2.0, size=x.shape), dtype=np.float64)


def guides_by_cells(x: np.ndarray, fmt: str):
    """The cells x tags `x` as a guides x cells sparse matrix in format `fmt`."""
    return {"csr": sparse.csr_matrix, "csc": sparse.csc_matrix, "coo": sparse.coo_matrix}[fmt](x.T)


SHAPES = [(300, 25), (500, 7), (400, 200), (50, 1), (1, 10)]
CHUNKS = [(None, None), (7, 3), (3, 7)]


@pytest.mark.parametrize("chunks", CHUNKS, ids=["default", "cells7_guides3", "cells3_guides7"])
@pytest.mark.parametrize("integer", [True, False], ids=["int", "float"])
@pytest.mark.parametrize("shape", SHAPES, ids=[f"{c}x{t}" for c, t in SHAPES])
def test_cmo_matches_the_lab_code(shape, integer, chunks):
    n_cells, n_tags = shape
    x = cmo_counts(n_cells, n_tags, seed=n_cells * 1000 + n_tags, integer=integer)
    ref = lab_reference(x)
    fmt = ["csr", "csc", "coo"][(n_cells + n_tags) % 3]
    got = eval_lib.cmo_clr_q95(
        guides_by_cells(x, fmt), chunk_cells=chunks[0], chunk_guides=chunks[1]
    )

    assert same_bits(got["clr"].toarray().T, ref["clr"])
    assert same_bits(got["clr_gm"], ref["clr_gm"])
    assert same_bits(got["thresholds"], ref["thresholds"])
    assert np.array_equal(got["positive"].toarray().T, ref["positive"])
    assert np.array_equal(got["n_pos"], ref["n_pos"])
    assert np.array_equal(got["top_idx"], ref["top_idx"])
    assert np.array_equal(got["status"], ref["status"])
    # The CLR keeps the counts' pattern: an absent entry is a zero count.
    assert got["clr"].nnz == np.count_nonzero(x)


def test_cmo_cases_reach_their_edges():
    """The generated matrices do contain the cases the equivalence test claims to cover."""
    x = cmo_counts(300, 25, seed=300 * 1000 + 25, integer=True)
    ref = lab_reference(x)
    clr, thr, pos = ref["clr"], ref["thresholds"], ref["positive"]
    assert np.any(~x.any(axis=1)), "an all-zero cell"
    assert np.any(~x.any(axis=0)), "an all-zero tag"
    assert np.any((thr == 0) & x.any(axis=0)), "a tag with nonzero counts and threshold 0"
    assert np.any((clr == thr) & (thr > 0)), "a positive threshold tied with a cell's value"
    assert {"Negative", "Singlet", "Doublet"} <= set(ref["status"])
    assert np.any(pos.any(axis=0))


def test_strict_cut_excludes_a_tie_at_the_threshold():
    """Ten identical cells hold a tag's ten highest values: the cut lands on them, strictly."""
    rng = np.random.default_rng(7)
    x = np.zeros((100, 4), dtype=np.int64)
    x[:90, 0] = 2
    x[:90, 1:] = rng.poisson(3, size=(90, 3))
    x[90:] = [50, 1, 0, 3]
    ref = lab_reference(x)
    got = eval_lib.cmo_clr_q95(sparse.csr_matrix(x.T), chunk_cells=7, chunk_guides=3)
    assert ref["thresholds"][0] == ref["clr"][95, 0]
    assert not ref["positive"][:, 0].any()
    assert np.count_nonzero(ref["clr"][:, 0] >= ref["thresholds"][0]) == 10
    assert same_bits(got["thresholds"], ref["thresholds"])
    assert np.array_equal(got["positive"].toarray().T, ref["positive"])


def test_generalised_reference_is_the_lab_code_at_25_tags():
    x = cmo_counts(300, 25, seed=11, integer=True)
    for given in (x, sparse.csr_matrix(x)):
        (general,) = cmo_clr_normalization(SimpleNamespace(X=given), np)
        (literal,) = cmo_clr_normalization_25(SimpleNamespace(X=given), np)
        assert same_bits(general, literal)


@pytest.mark.parametrize("shape", [(300, 25), (400, 200), (50, 1)], ids=str)
def test_call_q95_on_the_clr_is_the_cmo_cut(shape):
    x = cmo_counts(*shape, seed=5, integer=True)
    counts = sparse.csc_matrix(x.T)
    cmo = eval_lib.cmo_clr_q95(counts, chunk_cells=7, chunk_guides=3)
    clr_cell = eval_lib.norm_clr_cell(counts, chunk_cells=5)
    assert same_bits(clr_cell.toarray(), cmo["clr"].toarray())
    assert np.array_equal(clr_cell.indices, cmo["clr"].indices)
    assert np.array_equal(clr_cell.indptr, cmo["clr"].indptr)
    for values in (cmo["clr"], clr_cell):
        calls = eval_lib.call_q95(values, chunk_guides=4)
        assert same_bits(calls.thresholds, cmo["thresholds"])
        assert (calls.assigned != cmo["positive"]).nnz == 0
        assert calls.fitted.all() and calls.converged is None


def random_counts(n_guides: int, n_cells: int, seed: int) -> sparse.csr_matrix:
    """Sparse integer guides x cells counts: noise, a signal in a few cells per guide, ties."""
    rng = np.random.default_rng(seed)
    noise = rng.poisson(0.4, size=(n_guides, n_cells)) * (rng.random((n_guides, n_cells)) < 0.3)
    carriers = rng.random((n_guides, n_cells)) < rng.uniform(0.0, 0.15, size=(n_guides, 1))
    signal = carriers * rng.poisson(rng.uniform(2, 40, size=(n_guides, 1)), (n_guides, n_cells))
    x = (noise + signal).astype(np.int64)
    x[0] = 0
    x[1] = 0
    x[1, :3] = 4
    return sparse.csr_matrix(x)


def test_clr_per_guide_does_not_change_the_quantile_calls():
    """A transform strictly increasing within a guide keeps the cut between the same values."""
    c = random_counts(60, 3000, seed=3)
    raw = eval_lib.call_q95(eval_lib.norm_raw(c))
    clr = eval_lib.call_q95(eval_lib.norm_clr_guide(c, chunk_guides=7), chunk_guides=9)
    assert raw.assigned.nnz > 0
    assert (raw.assigned != clr.assigned).nnz == 0


def test_clr_per_guide_is_the_lab_formula_on_the_other_margin():
    c = random_counts(40, 500, seed=4)
    (ref,) = cmo_clr_normalization(SimpleNamespace(X=c.toarray()), np)
    got = eval_lib.norm_clr_guide(c, chunk_guides=7)
    assert same_bits(got.toarray(), ref)


def test_normalizations_keep_the_pattern_and_follow_their_formulas():
    c = random_counts(30, 400, seed=8)
    d = c.toarray().astype(np.float64)
    nz = d > 0
    canonical = sparse.csr_matrix(c, dtype=np.float64)
    canonical.eliminate_zeros()
    canonical.sort_indices()

    totals = d.sum(axis=0)
    med = np.median(totals)
    kappa = np.random.default_rng(1).uniform(0.2, 3.0, size=d.shape[1])
    kappa[:5] = [0.0, np.nan, np.inf, -1.0, 0.0]
    scale = np.where(np.isfinite(kappa) & (kappa > 0), kappa, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = {
            "raw": np.log1p(d),
            "depth": np.where(nz, np.log1p(d * med / totals), 0.0),
            "noise_size": np.log1p(d / scale),
        }
    got = {
        "raw": eval_lib.norm_raw(c),
        "depth": eval_lib.norm_depth(c),
        "noise_size": eval_lib.norm_noise_size(c, kappa),
        "clr_cell": eval_lib.norm_clr_cell(c, chunk_cells=7),
        "clr_guide": eval_lib.norm_clr_guide(c, chunk_guides=7),
    }
    for name, m in got.items():
        assert isinstance(m, sparse.csr_matrix), name
        assert np.array_equal(m.indptr, canonical.indptr), name
        assert np.array_equal(m.indices, canonical.indices), name
        if name in expected:
            assert same_bits(m.toarray(), expected[name]), name
    with pytest.raises(ValueError, match="cell_sizes"):
        eval_lib.norm_noise_size(c, None)
    with pytest.raises(ValueError, match="negative"):
        eval_lib.norm_raw(-c)


def test_oracle_cut_calls_each_guides_true_count():
    """With no ties, the oracle calls exactly as many cells per guide as are true."""
    rng = np.random.default_rng(12)
    n_guides, n_cells = 8, 300
    values = rng.random((n_guides, n_cells))
    values[values < 0.5] = 0.0
    values[values > 0] += np.arange(np.count_nonzero(values)) * 1e-6
    truth = rng.random((n_guides, n_cells)) < rng.uniform(0, 0.3, size=(n_guides, 1))
    truth[0] = False
    calls = eval_lib.call_q_oracle(sparse.csr_matrix(values), sparse.csr_matrix(truth))
    called = np.asarray(calls.assigned.sum(axis=1)).ravel()
    expected = np.minimum(truth.sum(axis=1), (values > 0).sum(axis=1))
    assert np.array_equal(called, expected)
    assert called[0] == 0 and calls.thresholds[0] == values[0].max()


def dense_confusion(est: np.ndarray, true: np.ndarray, counts: np.ndarray) -> dict:
    """R's get_confusion on dense matrices, both subsets."""

    def one(e, t):
        tp = int(np.sum(e * t))
        fn = int(np.sum(t)) - tp
        fp = int(np.sum(e)) - tp
        tn = e.size - tp - fn - fp
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = float(np.float64(tp) / np.float64(tp + fp))
            recall = float(np.float64(tp) / np.float64(tp + fn))
            f1 = float(np.float64(2 * precision * recall) / np.float64(precision + recall))
        return {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    nz = counts > 0
    return {"full": one(est, true), "nonzero": one(est[nz], true[nz])}


@pytest.mark.parametrize("case", ["random", "empty", "perfect", "disjoint"])
def test_confusion_matches_a_dense_count(case):
    rng = np.random.default_rng(21)
    shape = (30, 40)
    counts = rng.poisson(0.8, size=shape) * (rng.random(shape) < 0.5)
    true = rng.random(shape) < 0.15
    est = {
        "random": rng.random(shape) < 0.2,
        "empty": np.zeros(shape, dtype=bool),
        "perfect": true.copy(),
        "disjoint": ~true & (rng.random(shape) < 0.1),
    }[case]
    # Both a true entry and an assignment at a zero count, outside the "nonzero" subset.
    assert np.any(true & (counts == 0)) and (case != "random" or np.any(est & (counts == 0)))
    got = eval_lib.confusion(
        sparse.csr_matrix(est), sparse.csc_matrix(true), sparse.coo_matrix(counts)
    )
    expected = dense_confusion(est, true, counts)
    for subset in ("full", "nonzero"):
        assert set(got[subset]) == set(expected[subset])
        for key, value in expected[subset].items():
            assert type(got[subset][key]) is type(value), (subset, key)
            np.testing.assert_equal(got[subset][key], value, err_msg=f"{subset} {key}")
    if case == "empty":
        assert math.isnan(got["full"]["precision"]) and math.isnan(got["full"]["f1"])
    if case == "perfect":
        assert got["full"]["f1"] == 1.0 and got["full"]["FP"] == 0


def test_confusion_refuses_mismatched_shapes():
    a = sparse.csr_matrix((3, 4), dtype=bool)
    with pytest.raises(ValueError, match="shapes differ"):
        eval_lib.confusion(a, a, sparse.csr_matrix((4, 3)))


def test_mean_infections_is_fishash_simulators_formula():
    for moi in (0.1, 0.3, 1.0, 10.0):
        expected = moi / (1 - math.exp(-moi)) * (1 - 0.1)
        assert eval_lib.mean_infections(moi) == pytest.approx(expected, rel=1e-15)
    many = eval_lib.mean_infections(np.array([0.1, 1.0, 10.0]), hurdle=0.0)
    assert many.shape == (3,) and many[2] == pytest.approx(10.0, rel=1e-4)


# ---------------------------------------------------------------------------------------------
# Loaders (pyarrow)


def write_triplets(path: Path, m: sparse.spmatrix, *, logical: bool, order=None) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    coo = sparse.coo_matrix(m)
    keep = coo.data != 0
    guide, cell, value = coo.row[keep], coo.col[keep], coo.data[keep]
    o = np.lexsort((guide, cell)) if order is None else order(guide.size)
    columns = {"guide": guide[o].astype(np.int32), "cell": cell[o].astype(np.int32)}
    if not logical:
        columns["value"] = value[o].astype(np.int32)
    pq.write_table(pa.table(columns), path)


def make_dataset_dir(d: Path, *, order=None):
    c = random_counts(12, 50, seed=2)
    truth = sparse.csr_matrix(np.random.default_rng(3).random(c.shape) < 0.1)
    signal = c.multiply(truth).tocsr()
    d.mkdir(parents=True, exist_ok=True)
    write_triplets(d / "counts.parquet", c, logical=False, order=order)
    write_triplets(d / "ground_truth.parquet", truth, logical=True)
    write_triplets(d / "counts_signal.parquet", signal, logical=False)
    meta = {
        "n_guides": c.shape[0],
        "n_cells": c.shape[1],
        "nnz_counts": int(c.count_nonzero()),
        "nnz_truth": int(truth.count_nonzero()),
    }
    (d / "meta.json").write_text(json.dumps(meta))
    return c, truth, signal, meta


def test_load_dataset_reads_what_simulate_writes(tmp_path):
    c, truth, signal, meta = make_dataset_dir(tmp_path / "ds")
    ds = eval_lib.load_dataset(tmp_path / "ds")
    assert isinstance(ds.counts, sparse.csc_matrix) and ds.counts.dtype == np.float64
    assert ds.truth.dtype == bool and ds.signal.dtype == np.float64
    assert np.array_equal(ds.counts.toarray(), c.toarray())
    assert np.array_equal(ds.truth.toarray(), truth.toarray())
    assert np.array_equal(ds.signal.toarray(), signal.toarray())
    assert ds.meta == meta
    assert ds.grna_ids == [f"feature_{i}" for i in range(1, 13)]


def test_load_dataset_refuses_triplets_out_of_order(tmp_path):
    make_dataset_dir(tmp_path / "ds", order=lambda n: np.arange(n)[::-1])
    with pytest.raises(ValueError, match="column-major"):
        eval_lib.load_dataset(tmp_path / "ds")


def test_load_assigned_round_trip_and_duplicates(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    guide = np.array([0, 2, 1, 2], dtype=np.int32)
    cell = np.array([0, 0, 3, 4], dtype=np.int32)
    pq.write_table(pa.table({"guide": guide, "cell": cell}), tmp_path / "assigned.parquet")
    m = eval_lib.load_assigned(tmp_path, (3, 5))
    assert isinstance(m, sparse.csr_matrix) and m.dtype == bool
    assert sorted(zip(*m.nonzero(), strict=True)) == sorted(
        zip(guide.tolist(), cell.tolist(), strict=True)
    )
    pq.write_table(
        pa.table({"guide": np.r_[guide, 0].astype(np.int32), "cell": np.r_[cell, 0]}),
        tmp_path / "assigned.parquet",
    )
    with pytest.raises(ValueError, match="more than once"):
        eval_lib.load_assigned(tmp_path, (3, 5))
    with pytest.raises(ValueError, match="outside"):
        eval_lib.load_assigned(tmp_path, (2, 5))


# ---------------------------------------------------------------------------------------------
# The Gaussian mixture caller against crispat's fit_em


def gmm_counts() -> sparse.csr_matrix:
    """Guides x cells counts for the mixture tests, with each skip rule reached once."""
    rng = np.random.default_rng(31)
    n_guides, n_cells = 8, 1500
    x = rng.poisson(0.3, size=(n_guides, n_cells)) * (rng.random((n_guides, n_cells)) < 0.4)
    carriers = rng.random((n_guides, n_cells)) < 0.06
    x = x + carriers * rng.poisson(rng.uniform(8, 30, size=(n_guides, 1)), (n_guides, n_cells))
    x[5] = 0  # no nonzero cell
    x[6] = 0
    x[6, 17] = 9  # one nonzero cell
    x[7] = 0
    x[7, :40] = 3  # nonzero cells that all hold the same value
    return sparse.csr_matrix(x.astype(np.int64))


def log10p(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    v = sparse.csr_matrix(counts, dtype=np.float64)
    v.data = np.log10(v.data + 1)
    return v


def fit_em_replica(column: np.ndarray, nonzero: bool) -> np.ndarray:
    """crispat 0.9.10 `gauss.fit_em` (MIT) on one guide's `(n_cells, 1)` integer counts.

    The body of `fit_em` from its `toarray()` on, with the AnnData indexing replaced by cell
    positions. Returns the called cell positions.
    """
    from sklearn import mixture

    data = column
    data = np.log10(data + 1)
    if nonzero:
        data = data[data != 0]
    gmm = mixture.GaussianMixture(n_components=2, n_init=10, covariance_type="tied", random_state=0)
    gmm.fit(data)
    posterior = gmm.predict_proba(data)
    high_component = np.argmax(gmm.means_)
    perturbed = posterior[:, high_component] > 0.5
    return np.flatnonzero(perturbed)


def test_call_gmm_is_fit_em_on_all_cells():
    pytest.importorskip("sklearn")
    from threadpoolctl import threadpool_limits

    counts = gmm_counts()
    calls = eval_lib.call_gmm(log10p(counts), nonzero=False)
    assert calls.fitted.tolist() == [True] * 5 + [False, False, True]
    assert np.isnan(calls.thresholds).all()
    dense = counts.toarray()
    for g in np.flatnonzero(calls.fitted):
        # One OpenMP thread, as call_gmm uses, so only the model is compared.
        with threadpool_limits(limits=1, user_api="openmp"):
            expected = fit_em_replica(dense[g].reshape(-1, 1), nonzero=False)
        assert np.array_equal(calls.assigned[g].indices, expected), g
    assert calls.assigned[:5].nnz > 0


def test_call_gmm_nonzero_calls_only_nonzero_cells():
    pytest.importorskip("sklearn")
    counts = gmm_counts()
    calls = eval_lib.call_gmm(log10p(counts), nonzero=True)
    assert calls.fitted.tolist() == [True] * 5 + [False] * 3
    outside = calls.assigned.astype(np.int8) - calls.assigned.multiply(counts > 0)
    assert sparse.csr_matrix(outside).count_nonzero() == 0
    assert calls.assigned[5:].nnz == 0 and calls.assigned[:5].nnz > 0
    assert calls.converged[calls.fitted].all() and not calls.converged[~calls.fitted].any()


def test_fit_em_replica_with_nonzero_cannot_run():
    """crispat's nonzero path hands scikit-learn the 1-D `data[data != 0]`."""
    pytest.importorskip("sklearn")
    column = gmm_counts()[0].toarray().reshape(-1, 1)
    with pytest.raises(ValueError, match="Expected 2D array, got 1D array"):
        fit_em_replica(column, nonzero=True)


@pytest.fixture(scope="module")
def crispat_gauss():
    pytest.importorskip("sklearn")
    pytest.importorskip("anndata")
    crispat = pytest.importorskip("crispat", reason="crispat is in the evaluation's venv only")
    return crispat.gauss


def crispat_adata(counts: sparse.csr_matrix):
    """The AnnData the fishash analysis repository's `bin/convert_to_anndata.R` writes.

    `X` is the transposed counts (cells x guides) as reticulate converts a dgCMatrix, a CSC
    matrix, cast to int as that script does; `obs["batch"]` is 0. Returned as the batch subset
    `crispat.ga_gauss` hands to `fit_em`.
    """
    import anndata as ad

    adata = ad.AnnData(X=sparse.csc_matrix(counts.T).astype(np.int64))
    adata.obs_names = [f"cell_{i}" for i in range(1, counts.shape[1] + 1)]
    adata.var_names = [f"feature_{i}" for i in range(1, counts.shape[0] + 1)]
    adata.obs["batch"] = 0
    return adata[adata.obs["batch"] == 0]


def test_call_gmm_matches_crispat_fit_em(crispat_gauss):
    from threadpoolctl import threadpool_limits

    counts = gmm_counts()
    adata = crispat_adata(counts)
    calls = eval_lib.call_gmm(log10p(counts), nonzero=False)
    for g in np.flatnonzero(calls.fitted):
        with threadpool_limits(limits=1, user_api="openmp"):
            got = crispat_gauss.fit_em(f"feature_{g + 1}", adata, nonzero=False)
        expected = {f"cell_{c + 1}" for c in calls.assigned[g].indices}
        assert set(got["cell"]) == expected, g
        assert (got["gRNA"] == f"feature_{g + 1}").all()


def test_crispat_fit_em_with_nonzero_raises(crispat_gauss):
    adata = crispat_adata(gmm_counts())
    with pytest.raises(ValueError, match="Expected 2D array, got 1D array"):
        crispat_gauss.fit_em("feature_1", adata, nonzero=True)
