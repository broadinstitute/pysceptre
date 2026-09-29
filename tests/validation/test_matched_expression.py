"""`matched_expression_stats`, on synthetic fits with a known answer.

There is no PerturbPlan R for this input, so nothing here compares against R.
These check the identities the construction has to satisfy and the
bookkeeping a shorter version would get wrong. The comparison against the
script it was ported from, on a real screen, is
`test_matched_expression_dctap.py`, behind the `realdata` marker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from pysceptre.analytical_power import (
    baseline_expression_stats_from_fits,
    compute_power,
    matched_expression_stats,
)

N_CELLS = 3000


@dataclass
class _Fit:
    fitted_coefs: np.ndarray
    theta: float


def _design(rng: np.random.Generator) -> np.ndarray:
    batch = rng.random(N_CELLS) < 0.4
    log_lib = rng.normal(0.0, 0.3, N_CELLS)
    return np.column_stack([np.ones(N_CELLS), batch.astype(float), log_lib])


def _case(seed: int = 0):
    rng = np.random.default_rng(seed)
    Z = _design(rng)
    fits = {
        "flat": _Fit(np.array([0.5, 0.0, 1.0]), 5.0),
        # Nearly absent from the batch, the CYBA shape.
        "batchy": _Fit(np.array([1.5, -4.0, 1.0]), 3.0),
        "bright": _Fit(np.array([3.0, 0.2, 1.0]), 20.0),
    }
    targets = {
        f"t{k}": np.sort(rng.choice(N_CELLS, size=60 + 20 * k, replace=False)) for k in range(3)
    }
    # Two gRNAs per target whose counts sum above the union, as in high MOI.
    cells_per_grna = pd.DataFrame(
        [
            {"grna_id": f"{t}_g{g}", "grna_target": t, "num_cells": idx.size // 2 + 5 * (g + 1)}
            for t, idx in targets.items()
            for g in range(2)
        ]
    )
    pairs = pd.DataFrame(
        [(t, g) for t in targets for g in fits], columns=["grna_target", "response_id"]
    ).sample(frac=1.0, random_state=seed)
    return Z, fits, targets, cells_per_grna, pairs


def test_an_intercept_only_design_gives_back_the_plain_fitted_mean():
    """With no covariate every cell carries the same information, so nothing is matched."""
    _, fits, targets, cells_per_grna, pairs = _case()
    Z = np.ones((N_CELLS, 1))
    flat = {g: _Fit(f.fitted_coefs[:1], f.theta) for g, f in fits.items()}
    got = matched_expression_stats(pairs, Z, flat, targets, cells_per_grna, N_CELLS)
    plain = baseline_expression_stats_from_fits(Z, flat).set_index("response_id")
    np.testing.assert_allclose(
        got["expression_mean"], plain.loc[got["response_id"], "expression_mean"], rtol=1e-12
    )
    np.testing.assert_allclose(got["info_ratio_pert"], 1.0, rtol=1e-12)


def test_matches_a_dense_reference_including_the_union_and_sum_split():
    """Averages over the union of cells, counts from the per-gRNA sum, rows in input order."""
    Z, fits, targets, cells_per_grna, pairs = _case()
    num_total_cells = N_CELLS + 7
    got = matched_expression_stats(pairs, Z, fits, targets, cells_per_grna, num_total_cells)

    sums = cells_per_grna.groupby("grna_target")["num_cells"].sum()
    for row, (t, g) in zip(
        got.itertuples(index=False), pairs.itertuples(index=False, name=None), strict=True
    ):
        assert (row.grna_target, row.response_id) == (t, g)
        fit = fits[g]
        mu = np.exp(Z @ fit.fitted_coefs)
        w = mu / (1 + mu / fit.theta)
        in_p = np.zeros(N_CELLS, dtype=bool)
        in_p[targets[t]] = True
        f_p, f_c = w[in_p].mean(), w[~in_p].mean()
        n_p = sums[t]
        n_c = num_total_cells - n_p
        f = (1 / n_p + 1 / n_c) / (1 / (n_p * f_p) + 1 / (n_c * f_c))
        np.testing.assert_allclose(row.expression_mean, f / (1 - f / fit.theta), rtol=1e-12)
        np.testing.assert_allclose(row.info_ratio_pert, f_p / w.mean(), rtol=1e-12)
        assert row.expression_size == fit.theta


def test_a_batch_effect_lowers_the_mean_for_a_random_treated_set():
    """Concavity of w: matching the information reads a batchy gene below its average mean."""
    Z, fits, targets, cells_per_grna, pairs = _case()
    got = matched_expression_stats(pairs, Z, fits, targets, cells_per_grna, N_CELLS)
    plain = baseline_expression_stats_from_fits(Z, fits).set_index("response_id")["expression_mean"]
    batchy = got.loc[got["response_id"] == "batchy"]
    assert np.all(batchy["expression_mean"].to_numpy() < 0.9 * plain["batchy"])
    np.testing.assert_allclose(got["info_ratio_pert"], 1.0, atol=0.25)


def test_perturbed_cells_with_larger_libraries_carry_more_information():
    Z, fits, _, _, _ = _case()
    big = np.flatnonzero(Z[:, 2] > 0.3)[:80]
    targets = {"big": big}
    cells_per_grna = pd.DataFrame(
        {"grna_id": ["g1"], "grna_target": ["big"], "num_cells": [big.size]}
    )
    pairs = pd.DataFrame({"grna_target": ["big"] * 3, "response_id": list(fits)})
    got = matched_expression_stats(pairs, Z, fits, targets, cells_per_grna, N_CELLS)
    assert np.all(got["info_ratio_pert"] > 1.2)


def test_the_output_feeds_compute_power_as_per_pair_statistics():
    Z, fits, targets, cells_per_grna, pairs = _case()
    stats = matched_expression_stats(pairs, Z, fits, targets, cells_per_grna, N_CELLS)
    res = compute_power(
        pairs,
        cells_per_grna,
        stats,
        fold_change_mean=0.85,
        fold_change_sd=0.0,
        cutoff=1e-3,
        num_total_cells=N_CELLS,
        side="both",
    )
    np.testing.assert_array_equal(res["expression_mean"], stats["expression_mean"])
    assert np.all((res["power"] >= 0) & (res["power"] <= 1))


def test_inputs_that_would_silently_answer_a_different_question_raise():
    Z, fits, targets, cells_per_grna, pairs = _case()

    with pytest.raises(KeyError, match="gene_fits"):
        matched_expression_stats(
            pd.concat([pairs, pd.DataFrame({"grna_target": ["t0"], "response_id": ["nope"]})]),
            Z,
            fits,
            targets,
            cells_per_grna,
            N_CELLS,
        )
    with pytest.raises(KeyError, match="grna_target_cells"):
        matched_expression_stats(
            pairs, Z, fits, {k: v for k, v in targets.items() if k != "t0"}, cells_per_grna, N_CELLS
        )
    with pytest.raises(KeyError, match="cells_per_grna"):
        matched_expression_stats(
            pairs, Z, fits, targets, cells_per_grna[cells_per_grna["grna_target"] != "t1"], N_CELLS
        )
    with pytest.raises(ValueError, match="not the same design"):
        matched_expression_stats(pairs, Z[:, :2], fits, targets, cells_per_grna, N_CELLS)
    with pytest.raises(ValueError, match="non-empty"):
        matched_expression_stats(
            pairs, Z, fits, {**targets, "t0": np.array([], dtype=int)}, cells_per_grna, N_CELLS
        )
    with pytest.raises(ValueError, match="summed count"):
        matched_expression_stats(pairs, Z, fits, targets, cells_per_grna, 50)
