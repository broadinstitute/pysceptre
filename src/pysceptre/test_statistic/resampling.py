"""Port of `run_low_level_test_full_v4` (low_level_full_test.cpp): the staged
B1 -> B2 -> B3 escalation that decides, per pair, whether the empirical
p-value from the first B1=499 draws is already trustworthy, or whether a
skew-normal tail extrapolation (or a further empirical batch) is needed.

Verified against the pinned upstream commit's source directly; this is a
straight port of ~35 lines of control flow, not a reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .empirical_p import compute_empirical_p_value
from .fold_change import estimate_log_fold_change
from .score_stat import (
    compute_null_statistics_from_draws,
    compute_observed_full_statistic,
    draws_to_matrix,
    row_slice,
    stack_pieces,
)
from .skew_normal import fit_and_evaluate_skew_normal

P_THRESH = 0.02


@dataclass
class PairResult:
    p_value: float
    z_orig: float
    fold_change: float
    se_fold_change: float
    stage: int
    sn_params: (
        tuple[float, float, float] | None
    )  # (xi, omega, alpha), None if SN was never fit/used
    resampling_dist: np.ndarray | None = None


def run_low_level_test_full(
    y: np.ndarray,
    mu: np.ndarray,
    a: np.ndarray,
    w: np.ndarray,
    D: np.ndarray,
    trt_idxs: np.ndarray,
    synthetic_idxs,
    *,
    stacked: np.ndarray | None = None,
    B1: int = 499,
    B2: int = 4999,
    B3: int = 0,
    fit_parametric_curve: bool = True,
    side_code: int = 0,
    p_thresh: float = P_THRESH,
    return_resampling_dist: bool = False,
) -> PairResult:
    """trt_idxs: 0-based observed treated-cell indices.

    synthetic_idxs: the B1+B2+B3 CRT draws, consumed in three consecutive
        slices [0:B1], [B1:B1+B2], [B1+B2:B1+B2+B3]. Either a `(B, n_cells)`
        0/1 CSR matrix from `draws_to_matrix` -- the fast path, built once per
        target and shared across that target's genes -- or a list of ragged
        index arrays, which is converted here. The list form is kept because
        it is the natural thing to write by hand in a test; the analysis
        always passes the matrix.
    stacked: `stack_pieces(a, w, D)`, built once per gene and reused across
        that gene's targets. Computed here when omitted.
    """
    if not sparse.issparse(synthetic_idxs):
        synthetic_idxs = draws_to_matrix(list(synthetic_idxs), a.shape[0])
    if stacked is None:
        stacked = stack_pieces(a, w, D)

    fc, se = estimate_log_fold_change(y, mu, trt_idxs)
    z_orig = compute_observed_full_statistic(a, w, D, trt_idxs)

    sn_params: tuple[float, float, float] | None = None
    stage = 1
    null_statistics = compute_null_statistics_from_draws(stacked, row_slice(synthetic_idxs, 0, B1))
    p = compute_empirical_p_value(null_statistics, z_orig, side_code)

    if p <= p_thresh:
        sn_fit_used = False
        if fit_parametric_curve:
            null_statistics = compute_null_statistics_from_draws(
                stacked, row_slice(synthetic_idxs, B1, B1 + B2)
            )
            sn_result = fit_and_evaluate_skew_normal(z_orig, null_statistics, side_code)
            p = sn_result.p
            sn_fit_used = sn_result.used
            if sn_fit_used:
                sn_params = (sn_result.xi, sn_result.omega, sn_result.alpha)
            stage = 2

        if not fit_parametric_curve or not sn_fit_used:
            if B3 > 0:
                null_statistics = compute_null_statistics_from_draws(
                    stacked, row_slice(synthetic_idxs, B1 + B2, B1 + B2 + B3)
                )
            p = compute_empirical_p_value(null_statistics, z_orig, side_code)
            stage = 3

    return PairResult(
        p_value=p,
        z_orig=z_orig,
        fold_change=fc,
        se_fold_change=se,
        stage=stage,
        sn_params=sn_params,
        resampling_dist=null_statistics if return_resampling_dist else None,
    )
