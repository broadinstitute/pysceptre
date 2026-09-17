"""Exact port of `fit_skew_normal_funct` / `check_sn_tail` / `check_for_outliers`
/ `fit_and_evaluate_skew_normal` (shared_low_level_functions.cpp), verified
against the pinned upstream commit source directly.

Uses moment-matching (Azzalini's closed-form reparameterization from sample
skewness), NOT `scipy.stats.skewnorm.fit` (which does MLE and would not
reproduce R's numbers) -- `scipy.stats.skewnorm` is only used here to
evaluate the fitted distribution's CDF/SF, which uses the identical Azzalini
parameterization (`a`=alpha, `loc`=xi, `scale`=omega) as boost's
`skew_normal(xi, omega, alpha)`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import skewnorm

_MAX_GAMMA_1 = 0.995
_TAIL_RATIO_THRESH = 2.0
_OUTLIER_RATIO_THRESH = 1.5
_MIN_P = 1.0e-250


@dataclass
class SkewNormalFit:
    xi: float
    omega: float
    alpha: float
    mean: float
    sd: float


def fit_skew_normal_funct(y: np.ndarray) -> SkewNormalFit:
    y = np.asarray(y, dtype=float)
    n = y.size
    m_y = y.mean()
    sd_y = math.sqrt((y**2).mean() - m_y**2)

    gamma1 = np.sum((y - m_y) ** 3) / (n * sd_y**3)
    if gamma1 > _MAX_GAMMA_1:
        gamma1 = 0.9 * _MAX_GAMMA_1

    b = math.sqrt(2.0 / math.pi)
    r = math.copysign(1.0, gamma1) * (2 * abs(gamma1) / (4 - math.pi)) ** (1.0 / 3.0)
    delta = r / (b * math.sqrt(1 + r * r))
    alpha = delta / math.sqrt(1 - delta * delta)
    mu_z = b * delta
    sd_z = math.sqrt(1 - mu_z * mu_z)
    omega = sd_y / sd_z
    xi = m_y - omega * mu_z

    return SkewNormalFit(xi=xi, omega=omega, alpha=alpha, mean=float(m_y), sd=float(sd_y))


def check_sn_tail(
    y_sorted_ascending: np.ndarray, xi_hat: float, omega_hat: float, alpha_hat: float
) -> bool:
    n = y_sorted_ascending.size
    for i in range(180, 199):
        p = i / 200.0
        idx = math.ceil(n * p)
        quantile = y_sorted_ascending[idx]
        sn_tail_prob = skewnorm.sf(quantile, a=alpha_hat, loc=xi_hat, scale=omega_hat)
        if sn_tail_prob <= 0:
            return False
        ratio = (1.0 - p) / sn_tail_prob
        if ratio > _TAIL_RATIO_THRESH:
            return False
    return True


def check_for_outliers(null_statistics_sorted_ascending: np.ndarray, mu: float, sd: float) -> bool:
    min_z = null_statistics_sorted_ascending[0]
    max_z = null_statistics_sorted_ascending[-1]
    B = null_statistics_sorted_ascending.size
    R_max = max_z / (mu + sd * math.sqrt(2 * math.log(B)))
    R_min = min_z / (mu - sd * math.sqrt(2 * math.log(B)))
    return R_max <= _OUTLIER_RATIO_THRESH and R_min <= _OUTLIER_RATIO_THRESH


@dataclass
class SkewNormalEvalResult:
    xi: float
    omega: float
    alpha: float
    p: float  # -1.0 if the SN fit was rejected (caller should fall back to empirical p)

    @property
    def used(self) -> bool:
        return self.p > -0.5


def fit_and_evaluate_skew_normal(
    z_orig: float, null_statistics: np.ndarray, side_code: int
) -> SkewNormalEvalResult:
    null_statistics = np.asarray(null_statistics, dtype=float)
    fit = fit_skew_normal_funct(null_statistics)
    p = -1.0

    finite_params = all(np.isfinite(v) for v in (fit.xi, fit.omega, fit.alpha, fit.mean, fit.sd))
    if finite_params:
        sorted_null = np.sort(null_statistics)
        n = sorted_null.size
        median_idx = (n - 1) // 2
        median = sorted_null[median_idx]
        check_right_tail = z_orig >= median

        outlier_ok = check_for_outliers(sorted_null, fit.mean, fit.sd)
        fit_ok = False
        if outlier_ok:
            if check_right_tail:
                fit_ok = check_sn_tail(sorted_null, fit.xi, fit.omega, fit.alpha)
            else:
                negated_ascending = -sorted_null[::-1]
                fit_ok = check_sn_tail(negated_ascending, -fit.xi, fit.omega, -fit.alpha)

        if outlier_ok and fit_ok:
            if side_code == 0:
                p = 2.0 * (
                    skewnorm.sf(z_orig, a=fit.alpha, loc=fit.xi, scale=fit.omega)
                    if check_right_tail
                    else skewnorm.cdf(z_orig, a=fit.alpha, loc=fit.xi, scale=fit.omega)
                )
            elif side_code == 1:
                p = skewnorm.sf(z_orig, a=fit.alpha, loc=fit.xi, scale=fit.omega)
            else:
                p = skewnorm.cdf(z_orig, a=fit.alpha, loc=fit.xi, scale=fit.omega)
            p = max(p, _MIN_P)

    return SkewNormalEvalResult(xi=fit.xi, omega=fit.omega, alpha=fit.alpha, p=p)
