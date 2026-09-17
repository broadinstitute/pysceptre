"""Tests for the CRT-draw memory guard in pipeline/discovery.py.

`no_approximation` sizes B3 as `mult * n_pairs / alpha`, which at moi5 scale
(33,066 pairs, one-sided, alpha=0.1) is 1,653,300 draws per target -- about
5.2 GB of index arrays per target, or ~1 TB at the default chunk size of 200.
The guard shrinks the chunk rather than letting the process be OOM-killed.
"""

import numpy as np
import pandas as pd
import pytest

from pysceptre.pipeline.api import run_discovery_analysis
from pysceptre.pipeline.discovery import (
    _fit_chunk_size_to_budget,
    estimate_draw_memory_bytes,
)

MOI5_N_TRT = [396] * 100  # median treated-cell count at moi5 scale


def test_estimate_matches_hand_calculation():
    # 5498 draws x 400 cells x 8 bytes x 10 targets
    assert estimate_draw_memory_bytes(5498, [400], 10) == 5498 * 400 * 8 * 10


def test_estimate_uses_the_median_treated_cell_count():
    # median of [10, 100, 1000] is 100, not the mean (370)
    assert estimate_draw_memory_bytes(1, [10, 100, 1000], 1) == 100 * 8


def test_estimate_is_zero_when_there_are_no_draws():
    assert estimate_draw_memory_bytes(0, [400], 10) == 0.0
    assert estimate_draw_memory_bytes(5498, [], 10) == 0.0


def test_skew_normal_at_moi5_scale_is_not_shrunk():
    """The budget default must leave the normal path alone: B1+B2 = 5498 draws
    over 200 targets is about 3.5 GB, under the 8 GB default."""
    B_total = 499 + 4999
    assert estimate_draw_memory_bytes(B_total, MOI5_N_TRT, 200) < 8e9
    assert _fit_chunk_size_to_budget(B_total, MOI5_N_TRT, 200, 8.0) == 200


def test_no_approximation_at_moi5_scale_is_shrunk_to_one_with_a_warning():
    B_total = 499 + 5 * 33_066 // 0.1  # one-sided, alpha=0.1
    per_target = estimate_draw_memory_bytes(int(B_total), MOI5_N_TRT, 1)
    assert per_target > 5e9, "one target alone should exceed 5 GB"

    with pytest.warns(UserWarning, match="may still exhaust memory"):
        fitted = _fit_chunk_size_to_budget(int(B_total), MOI5_N_TRT, 200, 8.0)
    assert fitted == 1


def test_intermediate_budget_trims_the_chunk_without_the_severity_note():
    """When the chunk shrinks but not all the way to 1, the warning states the
    numbers and stops there -- no "may exhaust memory" alarm."""
    B_total = 499 + 4999
    with pytest.warns(UserWarning, match="not results") as record:
        fitted = _fit_chunk_size_to_budget(B_total, MOI5_N_TRT, 200, 1.0)
    assert 1 < fitted < 200
    assert estimate_draw_memory_bytes(B_total, MOI5_N_TRT, fitted) <= 1e9
    assert "may still exhaust memory" not in str(record[0].message)


def test_guard_never_raises_chunk_size_above_what_was_asked():
    assert _fit_chunk_size_to_budget(499, [10], 4, 1000.0) == 4


def _inputs(n_cells=500, n_genes=2, n_targets=6, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.column_stack([np.ones(n_cells), rng.normal(size=n_cells)])
    gene_ids = [f"gene_{i}" for i in range(n_genes)]
    resp = rng.poisson(5.0, size=(n_genes, n_cells)).astype(float)
    cells = {f"target_{t}": rng.choice(n_cells, size=50, replace=False) for t in range(n_targets)}
    pairs = pd.DataFrame([{"response_id": g, "grna_target": t} for g in gene_ids for t in cells])
    return resp, gene_ids, cov, cells, pairs


@pytest.mark.parametrize("chunk_size", [1, 2, 100])
def test_results_are_identical_across_chunk_sizes(chunk_size):
    """The guard's correctness argument: chunking changes peak memory and
    batching width, never which draws are taken. If this ever fails, the
    auto-shrink is not safe."""
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
