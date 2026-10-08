"""The dose test: discovery with per-cell weights (`grna_target_weights`), not from sceptre.

No R ground truth exists for it. What is checked: weights of 1 give sceptre's result exactly; a
constant weight gives sceptre's statistic and p-values (the scale cancels) and a fold change whose
effect is scaled by the inverse weight; the weighted statistic matches its dense formula, a cell
placed twice included; the settings it does not support are refused; and on null data whose weights
follow a covariate its p-values are uniform. Targets on both sides of the CRT sampler's 0.2%
threshold are covered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse, stats

from pysceptre import run_discovery_analysis
from pysceptre.precompute.pieces import compute_precomputation_pieces
from pysceptre.test_statistic.score_stat import (
    WeightedListDraws,
    compute_null_statistics_from_draws,
    compute_observed_full_statistic,
    stack_pieces,
)

N_CELLS = 20_000


def _data(seed=0, n_genes=3):
    rng = np.random.default_rng(seed)
    lib = rng.lognormal(8.0, 0.5, N_CELLS)
    X = np.column_stack([np.ones(N_CELLS), np.log(lib)])
    scale = rng.uniform(0.5, 2.0, size=(n_genes, 1))
    mu = 1e-3 * lib[None, :] * scale
    Y = sparse.csr_matrix(rng.negative_binomial(5, 5 / (5 + mu)).astype(np.float64))
    genes = [f"g{i}" for i in range(n_genes)]
    big = np.sort(rng.choice(N_CELLS, 2000, replace=False))  # above 0.2% of cells: exact sampler
    rest = np.setdiff1d(np.arange(N_CELLS), big)
    small = np.sort(rng.choice(rest, 30, replace=False))  # below: fast sampler
    cells = {"big": big, "small": small}
    pairs = pd.DataFrame(
        {"response_id": genes * 2, "grna_target": ["big"] * n_genes + ["small"] * n_genes}
    )
    return Y, genes, X, cells, pairs, lib


def test_weights_of_one_give_sceptres_result_exactly():
    Y, genes, X, cells, pairs, _ = _data()
    plain = run_discovery_analysis(Y, genes, X, cells, pairs, seed=1)
    ones = run_discovery_analysis(
        Y,
        genes,
        X,
        cells,
        pairs,
        seed=1,
        grna_target_weights={t: np.ones(c.size) for t, c in cells.items()},
    )
    pd.testing.assert_frame_equal(plain, ones)


def test_a_constant_weight_gives_sceptres_statistic_and_p_values():
    Y, genes, X, cells, pairs, _ = _data()
    plain = run_discovery_analysis(Y, genes, X, cells, pairs, seed=1)
    half = run_discovery_analysis(
        Y,
        genes,
        X,
        cells,
        pairs,
        seed=1,
        grna_target_weights={t: np.full(c.size, 0.5) for t, c in cells.items()},
    )
    np.testing.assert_allclose(half["z_orig"], plain["z_orig"], rtol=1e-10)
    np.testing.assert_allclose(half["p_value"], plain["p_value"], rtol=1e-10)
    assert half["stage"].tolist() == plain["stage"].tolist()
    # The fold change is the effect at full weight: twice the effect at weight 0.5.
    np.testing.assert_allclose(half["fold_change"] - 1, 2 * (plain["fold_change"] - 1), rtol=1e-10)


def test_weighted_statistic_matches_its_dense_formula():
    Y, genes, X, cells, pairs, _ = _data()
    rng = np.random.default_rng(3)
    y = Y[0].toarray().ravel()
    fit = np.linalg.lstsq(X, np.log1p(y), rcond=None)[0]  # any coefficients define valid pieces
    pieces = compute_precomputation_pieces(y, X, fit, 5.0)
    a, w, D = pieces.a, pieces.w, pieces.D
    idx = cells["big"]
    t = rng.uniform(0.1, 1.0, idx.size)
    dense_t = np.zeros(N_CELLS)
    dense_t[idx] = t
    expected = (dense_t @ a) / np.sqrt((dense_t**2) @ w - np.sum((D @ dense_t) ** 2))
    assert compute_observed_full_statistic(a, w, D, idx, t) == pytest.approx(expected, rel=1e-12)
    # Null statistics: one resample places a cell twice, which then counts twice.
    draws = [np.array([5, 9, 9, 40]), np.array([1, 2, 3])]
    weights = [np.array([0.2, 0.7, 0.7, 1.0]), np.array([0.5, 0.25, 1.0])]
    got = compute_null_statistics_from_draws(
        stack_pieces(a, w, D), WeightedListDraws(draws, weights, N_CELLS).slice(0, 2)
    )
    for b, (i, tw) in enumerate(zip(draws, weights, strict=True)):
        top = np.sum(tw * a[i])
        left = np.sum(tw * tw * w[i])
        right = np.sum((D[:, i] @ tw) ** 2)
        assert got[b] == pytest.approx(top / np.sqrt(left - right), rel=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"resampling_mechanism": "permutations"},
        {"grna_integration_strategy": "bonferroni"},
        {"weights": "short"},
        {"weights": "zero"},
    ],
)
def test_unsupported_settings_and_bad_weights_are_refused(kwargs):
    Y, genes, X, cells, pairs, _ = _data()
    weights = {t: np.ones(c.size) for t, c in cells.items()}
    kind = kwargs.pop("weights", None)
    if kind == "short":
        weights["big"] = weights["big"][:-1]
    if kind == "zero":
        weights["big"][0] = 0.0
    extra = {}
    if kwargs.get("grna_integration_strategy") == "bonferroni":
        extra["grna_target_data_frame"] = pd.DataFrame(
            {"grna_id": ["big", "small"], "grna_target": ["big", "small"]}
        )
    with pytest.raises(ValueError):
        run_discovery_analysis(
            Y, genes, X, cells, pairs, seed=1, grna_target_weights=weights, **kwargs, **extra
        )


def test_dose_test_is_calibrated_on_null_data_with_covariate_dependent_weights():
    """Weights rise with the cells' library size, which also drives expression; no effect exists."""
    Y, genes, X, cells, pairs, lib = _data(seed=7, n_genes=120)
    idx = cells["big"]
    rank = stats.rankdata(lib[idx]) / idx.size
    weights = {"big": 0.1 + 0.9 * rank, "small": np.full(cells["small"].size, 1.0)}
    big_pairs = pairs[pairs.grna_target == "big"].reset_index(drop=True)
    res = run_discovery_analysis(
        Y, genes, X, {"big": idx}, big_pairs, seed=2, grna_target_weights={"big": weights["big"]}
    )
    p = res["p_value"].dropna().to_numpy()
    assert p.size == 120
    assert stats.kstest(p, "uniform").pvalue > 0.01
