"""Port of `estimate_log_fold_change_v2` (shared_low_level_functions.cpp)."""

from __future__ import annotations

import numpy as np


def estimate_log_fold_change(
    y: np.ndarray, mu: np.ndarray, trt_idxs: np.ndarray, trt_weights: np.ndarray | None = None
) -> tuple[float, float]:
    """trt_idxs: 0-based indices of treated cells. Returns (fold_change, se_fold_change).

    With `trt_weights` (the dose test), the fold change at full weight, with the effect
    taken as linear in the weight: `1 + sum t (y - mu) / sum t^2 mu`. Weights all equal
    to 1 give the unweighted estimate exactly. `docs/design.md`, "The dose test".
    """
    if trt_weights is not None and not np.all(trt_weights == 1.0):
        t = np.asarray(trt_weights, dtype=np.float64)
        y_t, mu_t = y[trt_idxs], mu[trt_idxs]
        info = np.sum(t * t * mu_t)
        fc = 1.0 + np.sum(t * (y_t - mu_t)) / info
        se = np.sqrt(np.sum(t * t * (y_t - mu_t * (1.0 + t * (fc - 1.0))) ** 2)) / info
        return float(fc), float(se)
    y_trt = y[trt_idxs]
    mu_trt = mu[trt_idxs]
    sum_y = y_trt.sum()
    sum_mu = mu_trt.sum()
    fc = sum_y / sum_mu
    se = np.sqrt(np.sum((y_trt - fc * mu_trt) ** 2) / sum_mu**2)
    return float(fc), float(se)
