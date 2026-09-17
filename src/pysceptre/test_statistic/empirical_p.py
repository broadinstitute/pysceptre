"""Port of `compute_empirical_p_value` (shared_low_level_functions.cpp).

side: -1 = left, 0 = both, 1 = right (matches sceptre's side_code convention).
"""

from __future__ import annotations

import numpy as np


def compute_empirical_p_value(null_statistics: np.ndarray, z_orig: float, side: int) -> float:
    null_statistics = np.asarray(null_statistics)
    B = null_statistics.size
    if side == -1:
        counter = np.sum(z_orig >= null_statistics)
        return (1.0 + counter) / (1.0 + B)
    if side == 1:
        counter = np.sum(z_orig <= null_statistics)
        return (1.0 + counter) / (1.0 + B)
    # two-sided
    return 2.0 * min(
        compute_empirical_p_value(null_statistics, z_orig, -1),
        compute_empirical_p_value(null_statistics, z_orig, 1),
    )
