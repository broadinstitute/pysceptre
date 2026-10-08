"""R's phyper on the log scale (`pysceptre.assignment.hypergeom`), against R itself.

The fixture's "phyper" section holds `stats::phyper(q, m, n, k, lower.tail, log.p = TRUE)` at
hand-picked branch points and a systematic sweep in which m and n are shifted by 0, 0.37 and
0.5, so R's rounding of non-integer margins (half to even, m and n separately) is exercised.
The port runs the same algorithm on the same rounded arguments; what can differ is libm's last
bits and the multiply-adds R's compiler contracts. So values must agree to 1e-12 relative to
max(1, |R|), and the 0, -inf and NaN patterns exactly. Both the numba kernel and the numpy
fallback are run, through the `assignment_kernel` fixture.
"""

import numpy as np
import pytest

from pysceptre.assignment import hypergeom

RTOL = 1e-12

_SPECIAL = {"Inf": np.inf, "-Inf": -np.inf, "NaN": np.nan, "NA": np.nan}


def _num(values) -> np.ndarray:
    return np.array([_SPECIAL[v] if isinstance(v, str) else v for v in values], dtype=np.float64)


@pytest.fixture(scope="module")
def grid(fishash_ground_truth):
    g = fishash_ground_truth["phyper"]
    return {key: _num(g[key]) for key in ("q", "m", "n", "k", "upper", "lower")}


def _assert_same(py, r, label):
    np.testing.assert_array_equal(np.isnan(py), np.isnan(r), err_msg=f"{label}: NaN pattern")
    np.testing.assert_array_equal(np.isneginf(py), np.isneginf(r), err_msg=f"{label}: -inf pattern")
    np.testing.assert_array_equal(py == 0, r == 0, err_msg=f"{label}: zero pattern")
    fin = np.isfinite(r) & (r != 0)
    err = np.abs(py[fin] - r[fin]) / np.maximum(1.0, np.abs(r[fin]))
    worst = int(np.argmax(err))
    assert err.max() <= RTOL, (
        f"{label}: max relative error {err.max():.3g} at point {np.flatnonzero(fin)[worst]}"
    )


@pytest.mark.parametrize("lower_tail", [False, True], ids=["upper", "lower"])
def test_log_phyper_matches_r(grid, assignment_kernel, lower_tail):
    r = grid["lower" if lower_tail else "upper"]
    py = hypergeom.log_phyper(grid["q"], grid["m"], grid["n"], grid["k"], lower_tail=lower_tail)
    _assert_same(py, r, f"{'lower' if lower_tail else 'upper'} tail, {assignment_kernel}")


def test_numba_and_numpy_agree(grid, monkeypatch):
    if not hypergeom._HAVE_NUMBA:
        pytest.skip("numba is not installed")
    for lower_tail in (False, True):
        args = (grid["q"], grid["m"], grid["n"], grid["k"])
        a = hypergeom.log_phyper(*args, lower_tail=lower_tail)
        with monkeypatch.context() as mp:
            mp.setattr(hypergeom, "_HAVE_NUMBA", False)
            b = hypergeom.log_phyper(*args, lower_tail=lower_tail)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_array_equal(np.isinf(a), np.isinf(b))
        fin = np.isfinite(a)
        # Not bit for bit on every platform: docs/design.md, "The hypergeometric tail is R's phyper".
        np.testing.assert_array_max_ulp(a[fin], b[fin], maxulp=64)


def _stirlerr_bin(n: np.ndarray) -> list[str]:
    edges = [
        (n <= 15, "stirlerr_table"),
        ((n > 15) & (n <= 23.5), "stirlerr_16_23"),
        ((n > 23.5) & (n <= 27), "stirlerr_24_27"),
        ((n > 27) & (n <= 86), "stirlerr_28_86"),
        ((n > 86) & (n <= 205), "stirlerr_87_205"),
        ((n > 205) & (n <= 6180), "stirlerr_206_6180"),
        ((n > 6180) & (n <= 15.7e6), "stirlerr_6181_15.7e6"),
        (n > 15.7e6, "stirlerr_above_15.7e6"),
    ]
    return [label for cond, label in edges if cond.any()]


def _dbinom_tags(x, n, p, q) -> set[str]:
    tags = set()
    x0 = x == 0
    xn = (x == n) & ~x0
    gen = ~x0 & ~xn
    if (x0 & (p > q)).any():
        tags.add("dbinom_x0_p_above_q")
    if (x0 & (p <= q)).any():
        tags.add("dbinom_x0_p_at_most_q")
    if (xn & (p > q)).any():
        tags.add("dbinom_xn_p_above_q")
    if (xn & (p <= q)).any():
        tags.add("dbinom_xn_p_at_most_q")
    if gen.any():
        tags.add("dbinom_general")
        xs, ns, ps, qs = x[gen], n[gen], p[gen], q[gen]
        for arg in (ns, xs, ns - xs):
            tags.update(_stirlerr_bin(arg))
        for a, b in ((xs, ns * ps), (ns - xs, ns * qs)):
            taylor = np.abs(a - b) < 0.1 * (a + b)
            if taylor.any():
                tags.add("bd0_taylor")
            if (~taylor).any():
                tags.add("bd0_log")
    return tags


def test_grid_reaches_every_branch(grid):
    """A narrower grid would quietly test less; this keeps every branch of the port reached."""
    x, nr, nb, n = hypergeom.round_phyper_args(grid["q"], grid["m"], grid["n"], grid["k"])
    tags = set()
    nan_in = np.isnan(x) | np.isnan(nr) | np.isnan(nb) | np.isnan(n)
    if nan_in.any():
        tags.add("nan_argument")
    with np.errstate(invalid="ignore"):
        domain = ~nan_in & (nr >= 0) & (nb >= 0) & (n >= 0) & (n <= nr + nb)
    if (~nan_in & ~domain).any():
        tags.add("domain_error")
    x, nr, nb, n = x[domain], nr[domain], nb[domain], n[domain]
    swap = x * (nr + nb) > n * nr
    tags.update({"swap"} if swap.any() else set())
    tags.update({"no_swap"} if (~swap).any() else set())
    r = np.where(swap, nb, nr)
    b = np.where(swap, nr, nb)
    xx = np.where(swap, n - x - 1, x)
    zero = (xx < 0) | (xx < n - b)
    one = ~zero & ((xx >= r) | (xx >= n))
    tags.update({"support_zero"} if zero.any() else set())
    tags.update({"support_one"} if one.any() else set())
    gen = ~zero & ~one
    xx, r, b, n = xx[gen], r[gen], b[gen], n[gen]
    if (xx > 0).any():
        tags.add("pdhyper_series")
    p = n / (r + b)
    q = (r + b - n) / (r + b)
    for xa, na in ((xx, r), (n - xx, b), (n, r + b)):
        tags |= _dbinom_tags(xa, na, p, q)
    expected = {
        "nan_argument", "domain_error", "swap", "no_swap", "support_zero", "support_one",
        "pdhyper_series", "dbinom_general", "dbinom_x0_p_above_q", "dbinom_x0_p_at_most_q",
        "dbinom_xn_p_above_q", "dbinom_xn_p_at_most_q", "bd0_taylor", "bd0_log",
        "stirlerr_table", "stirlerr_16_23", "stirlerr_24_27", "stirlerr_28_86",
        "stirlerr_87_205", "stirlerr_206_6180", "stirlerr_6181_15.7e6", "stirlerr_above_15.7e6",
    }  # fmt: skip
    missing = expected - tags
    assert not missing, f"the grid no longer reaches: {sorted(missing)}"


def test_rounds_half_to_even_and_floors_q():
    x, nr, nb, n = hypergeom.round_phyper_args([2.9999999, 2.5], [2.5, 3.5], [6.5, 4.5], [5.5, 4.5])
    np.testing.assert_array_equal(x, [3.0, 2.0])
    np.testing.assert_array_equal(nr, [2.0, 4.0])
    np.testing.assert_array_equal(nb, [6.0, 4.0])
    np.testing.assert_array_equal(n, [6.0, 4.0])
