"""R's hypergeometric distribution function, on the log scale.

Port of `phyper` and the functions it calls -- `pdhyper`, `dhyper`, `dbinom_raw`, `stirlerr`
and `bd0` -- from R's `src/nmath` (R 4.5; GPL-2-or-later, see `THIRD_PARTY_LICENSES`). fishash
computes every p-value with `phyper(..., log.p = TRUE)`, and after its first pass the margins
it passes are not whole numbers, which R rounds inside `phyper`. Reproducing that rounding and
R's tail algorithm is what lets the port agree with R entry for entry. See `docs/design.md`,
"The hypergeometric tail is R's phyper".

Two implementations compute the same IEEE operations per element: a numba kernel, used when
numba is installed, and a vectorised numpy fallback. Which one runs is decided at call time
from `_HAVE_NUMBA`.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["log_phyper"]

_NEG_INF = -math.inf
_DBL_EPSILON = 2.220446049250313e-16
_DBL_MIN = 2.2250738585072014e-308
_M_LN2 = 0.693147180559945309417232121458
_M_LN_2PI = 1.837877066409345483560659472811
_M_LN_SQRT_2PI = 0.918938533204672741780329736406

# stirlerr's series coefficients and its table of exact values at 0, 0.5, ..., 15 (nmath/stirlerr.c).
_S = np.array(
    [
        0.083333333333333333333,  # 1/12
        0.00277777777777777777778,  # 1/360
        0.00079365079365079365079365,  # 1/1260
        0.000595238095238095238095238,  # 1/1680
        0.0008417508417508417508417508,  # 1/1188
        0.0019175269175269175269175262,  # 691/360360
        0.0064102564102564102564102561,  # 1/156
        0.029550653594771241830065352,  # 3617/122400
        0.17964437236883057316493850,  # 43867/244188
        1.3924322169059011164274315,  # 174611/125400
        13.402864044168391994478957,  # 77683/5796
        156.84828462600201730636509,  # 236364091/1506960
        2193.1033333333333333333333,  # 657931/300
        36108.771253724989357173269,  # 3392780147/93960
        691472.26885131306710839498,  # 1723168255201/2492028
        15238221.539407416192283370,  # 7709321041217/505920
        382900751.39141414141414141,  # 151628697551/396
    ]
)
_SFERR_HALVES = np.array(
    [
        0.0,  # n = 0: a placeholder, never used
        0.1534264097200273452913848,
        0.0810614667953272582196702,
        0.0548141210519176538961390,
        0.0413406959554092940938221,
        0.03316287351993628748511048,
        0.02767792568499833914878929,
        0.02374616365629749597132920,
        0.02079067210376509311152277,
        0.01848845053267318523077934,
        0.01664469118982119216319487,
        0.01513497322191737887351255,
        0.01387612882307074799874573,
        0.01281046524292022692424986,
        0.01189670994589177009505572,
        0.01110455975820691732662991,
        0.010411265261972096497478567,
        0.009799416126158803298389475,
        0.009255462182712732917728637,
        0.008768700134139385462952823,
        0.008330563433362871256469318,
        0.007934114564314020547248100,
        0.007573675487951840794972024,
        0.007244554301320383179543912,
        0.006942840107209529865664152,
        0.006665247032707682442354394,
        0.006408994188004207068439631,
        0.006171712263039457647532867,
        0.005951370112758847735624416,
        0.005746216513010115682023589,
        0.005554733551962801371038690,
    ]
)


# ---------------------------------------------------------------------------------------------
# Scalar functions, written so numba can compile them unchanged.
# ---------------------------------------------------------------------------------------------


def _stirlerr_series(n, k):
    # (S0 - (S1 - (... - S_k/nn)/nn ...)/nn)/n, evaluated innermost first as R writes it.
    nn = n * n
    t = _S[k]
    for j in range(k - 1, -1, -1):
        t = _S[j] - t / nn
    return t / n


def _stirlerr_py(n):
    if n <= 23.5:
        nn = n + n
        if n <= 15.0 and nn == int(nn):
            return _SFERR_HALVES[int(nn)]
        if n <= 5.25:
            if n >= 1.0:
                l_n = math.log(n)
                return math.lgamma(n) + n * (1.0 - l_n) + (l_n - _M_LN_2PI) * 0.5
            # R calls lgamma1p(n) here; phyper only passes integers, so this is unreachable.
            return math.lgamma(n + 1.0) - (n + 0.5) * math.log(n) + n - _M_LN_SQRT_2PI
        if n > 12.8:
            return _stirlerr_series(n, 6)
        if n > 12.3:
            return _stirlerr_series(n, 7)
        if n > 8.9:
            return _stirlerr_series(n, 8)
        if n > 7.3:
            return _stirlerr_series(n, 10)
        if n > 6.6:
            return _stirlerr_series(n, 12)
        if n > 6.1:
            return _stirlerr_series(n, 14)
        return _stirlerr_series(n, 16)
    if n > 15.7e6:
        return _stirlerr_series(n, 0)
    if n > 6180.0:
        return _stirlerr_series(n, 1)
    if n > 205.0:
        return _stirlerr_series(n, 2)
    if n > 86.0:
        return _stirlerr_series(n, 3)
    if n > 27.0:
        return _stirlerr_series(n, 4)
    return _stirlerr_series(n, 5)


def _bd0_py(x, np_):
    if not math.isfinite(x) or not math.isfinite(np_) or np_ == 0.0:
        return math.nan
    if abs(x - np_) < 0.1 * (x + np_):
        d = x - np_
        v = d / (x + np_)
        if d != 0.0 and v == 0.0:
            x_ = x * 0.25
            n_ = np_ * 0.25
            v = (x_ - n_) / (x_ + n_)
        s = (d * 0.5) * v
        if abs(s * 2.0) < _DBL_MIN:
            return s * 2.0
        ej = x * v
        v = v * v
        for j in range(1, 1000):
            ej = ej * v
            s_ = s
            s = s + ej / ((j << 1) + 1)
            if s == s_:
                return s * 2.0
        # R warns here ("should never happen") and falls through to the log form below.
    lg = math.log(x / np_) if math.isfinite(x / np_) else math.log(x) - math.log(np_)
    if x > np_:
        return x * (lg - 1.0) + np_
    return x * lg + np_ - x


def _dbinom_raw_log_py(x, n, p, q):
    if p == 0.0:
        return 0.0 if x == 0.0 else _NEG_INF
    if q == 0.0:
        return 0.0 if x == n else _NEG_INF
    if x == 0.0:
        if n == 0.0:
            return 0.0
        if p > q:
            return n * math.log(q)
        return n * math.log1p(-p)
    if x == n:
        if p > q:
            return n * math.log1p(-q)
        return n * math.log(p)
    if x < 0.0 or x > n:
        return _NEG_INF
    lc = _stirlerr(n) - _stirlerr(x) - _stirlerr(n - x) - _bd0(x, n * p) - _bd0(n - x, n * q)
    lf = _M_LN_2PI + math.log(x) + math.log1p(-(x / n))
    return lc - 0.5 * lf


def _dhyper_log_py(x, r, b, n):
    if n < x or r < x or n - x > b:
        return _NEG_INF
    if n == 0.0:
        return 0.0 if x == 0.0 else _NEG_INF
    p = n / (r + b)
    q = (r + b - n) / (r + b)
    p1 = _dbinom_raw_log(x, r, p, q)
    p2 = _dbinom_raw_log(n - x, b, p, q)
    p3 = _dbinom_raw_log(n, r + b, p, q)
    return p1 + p2 - p3


def _pdhyper_log_py(x, NR, NB, n):
    s = 0.0
    term = 1.0
    while x > 0.0 and term >= _DBL_EPSILON * s:
        term *= x * (NB - n + x) / (n + 1.0 - x) / (NR + 1.0 - x)
        s += term
        x -= 1.0
    return math.log1p(s)


def _log_phyper_one_py(x, NR, NB, n, lower_tail):
    # Arguments arrive already floored (x) and rounded (NR, NB, n) by the caller.
    if math.isnan(x) or math.isnan(NR) or math.isnan(NB) or math.isnan(n):
        return x + NR + NB + n
    if NR < 0.0 or NB < 0.0 or not math.isfinite(NR + NB) or n < 0.0 or n > NR + NB:
        return math.nan
    if x * (NR + NB) > n * NR:
        NR, NB = NB, NR
        x = n - x - 1.0
        lower_tail = not lower_tail
    if x < 0.0 or x < n - NB:
        return _NEG_INF if lower_tail else 0.0
    if x >= NR or x >= n:
        return 0.0 if lower_tail else _NEG_INF
    d = _dhyper_log(x, NR, NB, n)
    if d == _NEG_INF:
        return _NEG_INF if lower_tail else 0.0
    v = d + _pdhyper_log(x, NR, NB, n)
    if lower_tail:
        return v
    if v > -_M_LN2:
        return math.log(-math.expm1(v))
    return math.log1p(-math.exp(v))


def _log_phyper_loop_py(x, NR, NB, n, lower_tail, out):
    for i in range(x.shape[0]):
        out[i] = _log_phyper_one(x[i], NR[i], NB[i], n[i], lower_tail)


try:
    from numba import njit

    _stirlerr_series = njit(cache=True)(_stirlerr_series)
    _stirlerr = njit(cache=True)(_stirlerr_py)
    _bd0 = njit(cache=True)(_bd0_py)
    _dbinom_raw_log = njit(cache=True)(_dbinom_raw_log_py)
    _dhyper_log = njit(cache=True)(_dhyper_log_py)
    _pdhyper_log = njit(cache=True)(_pdhyper_log_py)
    _log_phyper_one = njit(cache=True)(_log_phyper_one_py)
    _log_phyper_loop = njit(cache=True)(_log_phyper_loop_py)
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised by the no-numba CI check
    _stirlerr = _stirlerr_py
    _bd0 = _bd0_py
    _dbinom_raw_log = _dbinom_raw_log_py
    _dhyper_log = _dhyper_log_py
    _pdhyper_log = _pdhyper_log_py
    _log_phyper_one = _log_phyper_one_py
    _log_phyper_loop = _log_phyper_loop_py
    _HAVE_NUMBA = False


# ---------------------------------------------------------------------------------------------
# The numpy fallback: the same operations, element by element, with branches as masks.
# ---------------------------------------------------------------------------------------------


def _stirlerr_numpy(n):
    out = np.empty_like(n)
    nn = n + n
    table = (n <= 15.0) & (nn == np.floor(nn))
    out[table] = _SFERR_HALVES[nn[table].astype(np.int64)]
    rest = ~table
    if not rest.any():
        return out
    # Cut points of the series, in the order R tests them, each with its number of terms.
    cuts = (
        (n > 15.7e6, 0),
        (n > 6180.0, 1),
        (n > 205.0, 2),
        (n > 86.0, 3),
        (n > 27.0, 4),
        (n > 23.5, 5),
        (n > 12.8, 6),
    )
    todo = rest.copy()
    for cond, k in cuts:
        sel = todo & cond
        if sel.any():
            out[sel] = _stirlerr_series_numpy(n[sel], k)
            todo &= ~sel
    if todo.any():
        # Non-integers at or below 12.8: never passed by phyper; computed by the scalar code.
        idx = np.flatnonzero(todo)
        out[idx] = [_stirlerr_py(float(v)) for v in n[idx]]
    return out


def _stirlerr_series_numpy(n, k):
    nn = n * n
    t = np.full_like(n, _S[k])
    for j in range(k - 1, -1, -1):
        t = _S[j] - t / nn
    return t / n


def _bd0_numpy(x, np_):
    out = np.empty_like(x)
    bad = ~np.isfinite(x) | ~np.isfinite(np_) | (np_ == 0.0)
    out[bad] = np.nan
    taylor = ~bad & (np.abs(x - np_) < 0.1 * (x + np_))
    other = ~bad & ~taylor
    if taylor.any():
        t_idx = np.flatnonzero(taylor)
        values, converged = _bd0_taylor_numpy(x[t_idx], np_[t_idx])
        out[t_idx[converged]] = values[converged]
        # R falls through to the log form when the series has not converged.
        other[t_idx[~converged]] = True
    if other.any():
        xo = x[other]
        no = np_[other]
        ratio = xo / no
        lg = np.where(np.isfinite(ratio), np.log(ratio), np.log(xo) - np.log(no))
        out[other] = np.where(xo > no, xo * (lg - 1.0) + no, xo * lg + no - xo)
    return out


def _bd0_taylor_numpy(x, np_):
    d = x - np_
    v = d / (x + np_)
    under = (d != 0.0) & (v == 0.0)
    if under.any():
        x_ = x[under] * 0.25
        n_ = np_[under] * 0.25
        v[under] = (x_ - n_) / (x_ + n_)
    s = (d * 0.5) * v
    out = np.empty_like(x)
    converged = np.ones(x.shape, dtype=bool)
    tiny = np.abs(s * 2.0) < _DBL_MIN
    out[tiny] = s[tiny] * 2.0
    ej = x * v
    v2 = v * v
    idx = np.flatnonzero(~tiny)
    j = 1
    while idx.size and j < 1000:
        ej[idx] = ej[idx] * v2[idx]
        s_old = s[idx]
        s_new = s_old + ej[idx] / ((j << 1) + 1)
        s[idx] = s_new
        done = s_new == s_old
        out[idx[done]] = s_new[done] * 2.0
        idx = idx[~done]
        j += 1
    converged[idx] = False
    return out, converged


def _dbinom_raw_log_numpy(x, n, p, q):
    out = np.empty_like(x)
    todo = np.ones(x.shape, dtype=bool)

    def put(sel, values):
        nonlocal todo
        sel = sel & todo
        out[sel] = values[sel] if isinstance(values, np.ndarray) else values
        todo &= ~sel

    with np.errstate(divide="ignore", invalid="ignore"):
        put(p == 0.0, np.where(x == 0.0, 0.0, _NEG_INF))
        put(q == 0.0, np.where(x == n, 0.0, _NEG_INF))
        put((x == 0.0) & (n == 0.0), 0.0)
        put(x == 0.0, np.where(p > q, n * np.log(q), n * np.log1p(-p)))
        put(x == n, np.where(p > q, n * np.log1p(-q), n * np.log(p)))
        put((x < 0.0) | (x > n), _NEG_INF)
    if todo.any():
        xs, ns, ps, qs = x[todo], n[todo], p[todo], q[todo]
        lc = (
            _stirlerr_numpy(ns)
            - _stirlerr_numpy(xs)
            - _stirlerr_numpy(ns - xs)
            - _bd0_numpy(xs, ns * ps)
            - _bd0_numpy(ns - xs, ns * qs)
        )
        lf = _M_LN_2PI + np.log(xs) + np.log1p(-(xs / ns))
        out[todo] = lc - 0.5 * lf
    return out


def _dhyper_log_numpy(x, r, b, n):
    # Only called after phyper's support checks, which make dhyper's own checks pass.
    p = n / (r + b)
    q = (r + b - n) / (r + b)
    p1 = _dbinom_raw_log_numpy(x, r, p, q)
    p2 = _dbinom_raw_log_numpy(n - x, b, p, q)
    p3 = _dbinom_raw_log_numpy(n, r + b, p, q)
    return p1 + p2 - p3


def _pdhyper_log_numpy(x, NR, NB, n):
    x = x.copy()
    s = np.zeros_like(x)
    term = np.ones_like(x)
    idx = np.flatnonzero(x > 0.0)
    while idx.size:
        xi = x[idx]
        t = term[idx] * (xi * (NB[idx] - n[idx] + xi) / (n[idx] + 1.0 - xi) / (NR[idx] + 1.0 - xi))
        term[idx] = t
        s_new = s[idx] + t
        s[idx] = s_new
        xi = xi - 1.0
        x[idx] = xi
        idx = idx[(xi > 0.0) & (t >= _DBL_EPSILON * s_new)]
    return np.log1p(s)


def _log_phyper_numpy(x, NR, NB, n, lower_tail):
    out = np.full(x.shape, np.nan)
    nan_in = np.isnan(x) | np.isnan(NR) | np.isnan(NB) | np.isnan(n)
    with np.errstate(invalid="ignore"):
        domain = (
            ~nan_in & (NR >= 0.0) & (NB >= 0.0) & np.isfinite(NR + NB) & (n >= 0.0) & (n <= NR + NB)
        )
    swap = domain & (x * (NR + NB) > n * NR)
    nr = np.where(swap, NB, NR)
    nb = np.where(swap, NR, NB)
    xx = np.where(swap, n - x - 1.0, x)
    lower = np.where(swap, not lower_tail, lower_tail)
    dt0 = np.where(lower, _NEG_INF, 0.0)
    dt1 = np.where(lower, 0.0, _NEG_INF)
    zero = domain & ((xx < 0.0) | (xx < n - nb))
    one = domain & ~zero & ((xx >= nr) | (xx >= n))
    out[zero] = dt0[zero]
    out[one] = dt1[one]
    idx = np.flatnonzero(domain & ~zero & ~one)
    if idx.size:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            d = _dhyper_log_numpy(xx[idx], nr[idx], nb[idx], n[idx])
            v = d + _pdhyper_log_numpy(xx[idx], nr[idx], nb[idx], n[idx])
            upper = np.where(v > -_M_LN2, np.log(-np.expm1(v)), np.log1p(-np.exp(v)))
        res = np.where(lower[idx], v, upper)
        res = np.where(d == _NEG_INF, dt0[idx], res)
        out[idx] = res
    return out


# ---------------------------------------------------------------------------------------------
# Entry points.
# ---------------------------------------------------------------------------------------------


def _log_phyper_rounded(x, NR, NB, n, lower_tail):
    """`log_phyper` on arguments already floored (x) and rounded (NR, NB, n), as 1-D arrays."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    NR = np.ascontiguousarray(NR, dtype=np.float64)
    NB = np.ascontiguousarray(NB, dtype=np.float64)
    n = np.ascontiguousarray(n, dtype=np.float64)
    if _HAVE_NUMBA:
        out = np.empty_like(x)
        _log_phyper_loop(x, NR, NB, n, bool(lower_tail), out)
        return out
    return _log_phyper_numpy(x, NR, NB, n, bool(lower_tail))


def round_phyper_args(q, m, n, k):
    """R's rounding of `phyper`'s arguments: `floor(q + 1e-7)`; `m`, `n`, `k` to nearest, half to even.

    Each of `m`, `n` and `k` is rounded separately, so `m + n` after rounding need not equal
    the rounded population size.
    """
    q, m, n, k = np.broadcast_arrays(*(np.asarray(a, dtype=np.float64) for a in (q, m, n, k)))
    return np.floor(q + 1e-7), np.rint(m), np.rint(n), np.rint(k)


def log_phyper(
    q: ArrayLike, m: ArrayLike, n: ArrayLike, k: ArrayLike, *, lower_tail: bool = True
) -> np.ndarray:
    """R's `phyper(q, m, n, k, lower.tail, log.p = TRUE)`, elementwise.

    `X` is the number of white balls in `k` draws without replacement from an urn with `m`
    white and `n` black balls (R's argument names).

    Args:
        q: Quantiles; floored after adding 1e-7, as R does.
        m: Number of white balls. Rounded to the nearest integer, half to even.
        n: Number of black balls. Rounded separately from `m`.
        k: Number of draws. Rounded.
        lower_tail: If True, `log P(X <= q)`; otherwise `log P(X > q)`.

    Returns:
        Log probabilities, broadcast to the arguments' common shape. NaN where R returns NaN
        (a NaN argument, or rounded arguments outside the domain).
    """
    x, nr, nb, kk = round_phyper_args(q, m, n, k)
    shape = x.shape
    out = _log_phyper_rounded(x.ravel(), nr.ravel(), nb.ravel(), kk.ravel(), lower_tail)
    return out.reshape(shape)
