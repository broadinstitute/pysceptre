"""A test-only, exact replica of sceptre's `run_reduced_em_algo_cpp` (src/mixture_functs.cpp).

Dense over every cell, with sums accumulated left to right as `Rcpp::sum` does. The package's
EM (`pysceptre.assignment.mixture`) groups the zero-count cells instead; this replica is the
yardstick between the two. Like `r_samplers.py`, it stays out of `src/`.
"""

from __future__ import annotations

import math

import numpy as np


def _seq_sum(x: np.ndarray) -> float:
    return float(np.cumsum(x)[-1]) if x.size else 0.0


def _tolerance(curr: float, prev: float) -> float:
    if curr == -math.inf or prev == -math.inf:
        return 1.0
    return abs(curr - prev) / min(abs(curr), abs(prev))


def run_reduced_em(pi_guesses, g_pert_guesses, g, mu0, log_g_factorial) -> dict:
    """`run_reduced_em_algo_cpp(pi_guesses, g_pert_guesses, g, g_mus_pert0, log_g_factorial)`.

    Returns the C++ list's fields (`outer_Ti1s`, `outer_i` 0-based, `outer_converged`,
    `outer_log_lik`) plus each start's converged flag and log-likelihood.
    """
    g = np.asarray(g, dtype=np.float64)
    mu0 = np.asarray(mu0, dtype=np.float64)
    lgf = np.asarray(log_g_factorial, dtype=np.float64)
    n = float(g.size)
    outer = {
        "outer_Ti1s": np.zeros(g.size),
        "outer_i": 0,
        "outer_converged": False,
        "outer_log_lik": -math.inf,
    }
    starts = []
    with np.errstate(all="ignore"):
        for b, (pi, g_pert) in enumerate(zip(pi_guesses, g_pert_guesses, strict=True)):
            converged = False
            prev_ll = -math.inf
            iteration = 1
            ti = np.zeros(g.size)
            while True:
                mu1 = mu0 + g_pert
                p0 = np.exp(np.log(1 - pi) + g * np.log(mu0) - mu0 - lgf)
                p1 = np.exp(np.log(pi) + g * np.log(mu1) - mu1 - lgf)
                s = p0 + p1
                s[s < 1e-100] = 1e-100
                curr_ll = _seq_sum(np.log(s))
                quotient = np.log(1 - pi) - np.log(pi) + g * (np.log(mu0) - np.log(mu1)) + mu1 - mu0
                ti = 1 / (np.exp(quotient) + 1)
                if np.all(ti <= 1e-100) or not np.all(np.isfinite(ti)):
                    curr_ll = -math.inf
                    break
                pi = _seq_sum(ti) / n
                if pi > 0.5:
                    ti = 1 - ti
                    pi = 1 - pi
                if _tolerance(curr_ll, prev_ll) < 0.5 * 1e-4 and iteration >= 3:
                    converged = True
                    break
                prev_ll = curr_ll
                iteration += 1
                if iteration >= 50:
                    break
                e1 = _seq_sum(ti * g)
                e2 = _seq_sum(ti * mu0)
                g_pert = float(np.log(e1) - np.log(e2))
            starts.append({"converged": converged, "log_lik": curr_ll})
            if curr_ll > outer["outer_log_lik"] and converged:
                outer = {
                    "outer_Ti1s": ti.copy(),
                    "outer_i": b,
                    "outer_converged": True,
                    "outer_log_lik": curr_ll,
                }
    outer["starts"] = starts
    return outer
