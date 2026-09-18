"""Tests for the single memory budget in pipeline/discovery.py.

Every per-chunk cost here is linear in the chunk size -- the dense IRLS
arrays, and the CRT draws a target holds -- so one budget and one division
bounds all of it. The guarantee under test is that `target_chunk_size` is an
upper bound rather than a mandate: no value a caller passes can exhaust
memory.

The cases that made this necessary, both measured at real scale:
  - the target binomial fit was unbounded: 13.1 GB at target_chunk_size=200
    over 586,309 cells;
  - no_approximation sizes B3 as mult * n_pairs / alpha, which at real dataset scale
    (33,066 pairs, one-sided, alpha=0.1) is 1,653,300 draws per target --
    5.2 GB each, or ~1 TB at the default chunk size.
"""

import numpy as np
import pandas as pd
import pytest

from pysceptre.pipeline.api import run_discovery_analysis
from pysceptre.pipeline.discovery import (
    _resolve_target_chunk_size,
    chunk_size_for_budget,
    estimate_draw_memory_bytes,
    gene_chunk_size_for_budget,
    irls_bytes_per_column,
    target_bytes_per_item,
    target_chunk_size_for_budget,
)

REAL_CELLS = 131_000
BENCH_CELLS = 586_309
N_TRT = [396] * 100
SKEW_NORMAL_B = 499 + 4999
NO_APPROX_B = 499 + 5 * 33_066 // 0.1  # one-sided, alpha=0.1


# --- the shared primitive -------------------------------------------------


def test_chunk_size_is_budget_over_cost_per_item():
    assert chunk_size_for_budget(bytes_per_item=1e9, n_items=100, chunk_memory_gb=4.0) == 4


def test_chunk_size_never_exceeds_the_item_count():
    assert chunk_size_for_budget(1.0, 7, 1000.0) == 7


def test_chunk_size_is_at_least_one_however_small_the_budget():
    assert chunk_size_for_budget(1e12, 100, 1e-9) == 1


def test_chunk_size_handles_a_zero_cost_without_dividing_by_zero():
    assert chunk_size_for_budget(0.0, 12, 4.0) == 12


# --- what each stage costs ------------------------------------------------


def test_irls_cost_uses_the_measured_per_column_factor():
    """10, not the 4 obvious arrays. Counting mu, weights, working response and
    the responses under-predicted peak RSS by about 2.5x; the factor is
    calibrated against measured growth (see _IRLS_ARRAYS_PER_COLUMN)."""
    assert irls_bytes_per_column(BENCH_CELLS) == 10 * BENCH_CELLS * 8


def test_target_cost_is_the_fit_plus_its_draws():
    got = target_bytes_per_item(BENCH_CELLS, SKEW_NORMAL_B, N_TRT)
    assert got == irls_bytes_per_column(BENCH_CELLS) + SKEW_NORMAL_B * 396 * 8


def test_draw_estimate_uses_the_median_treated_cell_count():
    # median of [10, 100, 1000] is 100, not the mean (370)
    assert estimate_draw_memory_bytes(1, [10, 100, 1000], 1) == 100 * 8


def test_draw_estimate_is_zero_without_draws_or_targets():
    assert estimate_draw_memory_bytes(0, [400], 10) == 0.0
    assert estimate_draw_memory_bytes(SKEW_NORMAL_B, [], 10) == 0.0


# --- the real-scale cases that motivated this -----------------------------


def test_unbounded_target_fit_was_the_problem_being_fixed():
    """13.1 GB measured at chunk=200 over 586k cells; the budget must cut it."""
    unbounded = target_bytes_per_item(BENCH_CELLS, SKEW_NORMAL_B, N_TRT) * 200
    assert unbounded / 1e9 > 5.0
    bounded = target_chunk_size_for_budget(BENCH_CELLS, SKEW_NORMAL_B, N_TRT, 3026, 4.0)
    assert bounded < 200
    assert target_bytes_per_item(BENCH_CELLS, SKEW_NORMAL_B, N_TRT) * bounded <= 4e9


def test_no_approximation_collapses_the_chunk_to_one():
    assert target_chunk_size_for_budget(REAL_CELLS, int(NO_APPROX_B), N_TRT, 2875, 4.0) == 1


def test_real_scale_leaves_the_gene_stage_unchunked():
    """The default budget should not chunk a real analysis: 244 genes at 131k
    cells is well inside 4 GB."""
    assert gene_chunk_size_for_budget(REAL_CELLS, 244, 4.0) == 244


# --- the guarantee: chunk size is an upper bound, not a mandate -----------


@pytest.mark.parametrize("asked", [200, 5_000, 100_000])
def test_any_requested_chunk_size_is_clamped_to_the_budget(asked):
    with pytest.warns(UserWarning, match="stay within chunk_memory_gb"):
        got = _resolve_target_chunk_size(BENCH_CELLS, SKEW_NORMAL_B, N_TRT, 3026, asked, 4.0)
    assert got < asked
    assert target_bytes_per_item(BENCH_CELLS, SKEW_NORMAL_B, N_TRT) * got <= 4e9


def test_a_chunk_already_within_budget_is_left_alone_and_silent():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _resolve_target_chunk_size(1_000, 499, [10], 50, 8, 4.0) == 8


def test_severity_note_appears_only_when_the_chunk_collapses_to_one():
    with pytest.warns(UserWarning, match="may still exhaust memory") as severe:
        _resolve_target_chunk_size(REAL_CELLS, int(NO_APPROX_B), N_TRT, 2875, 200, 4.0)
    assert "no_approximation" in str(severe[0].message)

    with pytest.warns(UserWarning, match="stay within chunk_memory_gb") as mild:
        got = _resolve_target_chunk_size(BENCH_CELLS, SKEW_NORMAL_B, N_TRT, 3026, 200, 4.0)
    assert got > 1
    assert "may still exhaust memory" not in str(mild[0].message)


def test_the_warning_breaks_the_cost_into_fit_and_draws():
    """A user should be able to see which part is expensive."""
    with pytest.warns(UserWarning) as rec:
        _resolve_target_chunk_size(BENCH_CELLS, SKEW_NORMAL_B, N_TRT, 3026, 200, 4.0)
    message = str(rec[0].message)
    assert "dense binomial-fit arrays" in message
    assert "CRT draws" in message


# --- results must not depend on any of this -------------------------------


def _inputs(n_cells=600, n_genes=3, n_targets=8, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.column_stack([np.ones(n_cells), rng.normal(size=n_cells)])
    mu = np.exp(cov @ np.array([1.6, 0.2]))
    resp = rng.negative_binomial(8.0, 8.0 / (8.0 + mu), size=(n_genes, n_cells)).astype(float)
    gene_ids = [f"gene_{i}" for i in range(n_genes)]
    cells = {f"target_{t}": rng.choice(n_cells, size=50, replace=False) for t in range(n_targets)}
    pairs = pd.DataFrame([{"response_id": g, "grna_target": t} for g in gene_ids for t in cells])
    return resp, gene_ids, cov, cells, pairs


@pytest.mark.parametrize("chunk_size", [1, 2, 100])
def test_results_are_identical_across_chunk_sizes(chunk_size):
    """The correctness argument for clamping. Chunking changes peak memory and
    batching width, not which draws are taken. If this fails, the auto-shrink
    is unsafe.

    Exact frame equality holds because target chunking does not touch gene
    fitting, and the ~1e-15 spread in the batched logistic solve is far too
    small to move an integer binomial draw count.
    """
    resp, gene_ids, cov, cells, pairs = _inputs()
    common = dict(
        response_matrix=resp,
        gene_ids=gene_ids,
        covariate_matrix=cov,
        grna_target_cells=cells,
        pairs=pairs,
        side="left",
        seed=7,
    )
    reference = run_discovery_analysis(**common, target_chunk_size=3)
    other = run_discovery_analysis(**common, target_chunk_size=chunk_size)
    pd.testing.assert_frame_equal(reference, other)


def test_a_tiny_budget_does_not_change_results():
    resp, gene_ids, cov, cells, pairs = _inputs()
    common = dict(
        response_matrix=resp,
        gene_ids=gene_ids,
        covariate_matrix=cov,
        grna_target_cells=cells,
        pairs=pairs,
        side="left",
        seed=7,
    )
    reference = run_discovery_analysis(**common)
    with pytest.warns(UserWarning):
        squeezed = run_discovery_analysis(**common, chunk_memory_gb=1e-7)
    pd.testing.assert_frame_equal(reference, squeezed)
