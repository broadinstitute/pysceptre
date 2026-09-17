"""Port of `estimate_log_fold_change_v2` (shared_low_level_functions.cpp)."""

from __future__ import annotations

import numpy as np


def estimate_log_fold_change(y: np.ndarray, mu: np.ndarray, trt_idxs: np.ndarray) -> tuple[float, float]:
    """trt_idxs: 0-based indices of treated cells. Returns (fold_change, se_fold_change)."""
    y_trt = y[trt_idxs]
    mu_trt = mu[trt_idxs]
    sum_y = y_trt.sum()
    sum_mu = mu_trt.sum()
    fc = sum_y / sum_mu
    se = np.sqrt(np.sum((y_trt - fc * mu_trt) ** 2) / sum_mu**2)
    return float(fc), float(se)
