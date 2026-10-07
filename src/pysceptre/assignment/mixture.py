"""sceptre's mixture assignment: a Poisson GLM per gRNA, then a two-component EM.

Port of sceptre 0.10.3's `assign_grnas(method = "mixture")` -- `obtain_em_assignments` and
`run_reduced_em_algo_cpp` (Katsevich-Lab/sceptre, GPL-3; see `THIRD_PARTY_LICENSES`). For each
gRNA with at least `n_nonzero_cells_cutoff` cells holding one UMI or more, its counts are fitted
by a Poisson GLM on the covariates; an EM then splits the cells into two components, the second
adding `g_pert` to the fitted mean, from five fixed starts; cells whose posterior for the second
component reaches `probability_threshold` are assigned. A gRNA with too few cells, or whose EM
never converges, is assigned where its count reaches `backup_threshold`.

The EM is sceptre's exactly, including that `g_pert`, updated as a log ratio, is added to the
mean on the count scale. Cells with a count of zero and a fitted mean up to 200 share one
posterior and are handled as a group, which makes each EM step cost the gRNA's nonzero cells
rather than all cells. See `docs/design.md`, "sceptre's mixture assignment".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import gammaln

from ..glm.design import validate_design_matrix
from ..glm.irls import fit_poisson_glm_batch, x_outer_flat
from ._common import as_count_matrix, check_ids

__all__ = ["G_PERT_GUESSES", "PI_GUESSES", "MixtureResult", "assign_grnas_mixture"]

# The five starting values sceptre 0.10.3 draws in get_random_starting_guesses: set.seed(4);
# runif(5, 1e-5, 0.1); runif(5, log(10), log(5000)). Fixed constants of the method.
PI_GUESSES = tuple(
    float.fromhex(h)
    for h in (
        "0x1.dfebea1e2046dp-5",
        "0x1.da369ba1b5c7ep-11",
        "0x1.e160f21692792p-6",
        "0x1.c691cc73ca6cbp-6",
        "0x1.4d3f64f86c8b5p-4",
    )
)
G_PERT_GUESSES = tuple(
    float.fromhex(h)
    for h in (
        "0x1.f5e4b12f8c326p+1",
        "0x1.b37ca9b453a98p+2",
        "0x1.fbbffba804914p+2",
        "0x1.066a7b1160713p+3",
        "0x1.60ea44b776f71p+1",
    )
)

_EP_TOL = 0.5 * 1e-4
_MIN_IT = 3
_MAX_IT = 50
_S_FLOOR = 1e-100
# A zero-count cell keeps its own term above this fitted mean: there sceptre's 1e-100 floor on
# a cell's likelihood can bind, which the grouped formula does not reproduce.
_FLAT_MU_MAX = 200.0
# glm.fit bounds a Poisson fitted mean below at .Machine$double.eps.
_R_MU_FLOOR = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class MixtureResult:
    """What `assign_grnas_mixture` returns.

    Attributes:
        assigned: `(n_grnas, n_cells)` boolean CSR matrix of assignments.
        posterior: `(n_grnas, n_cells)` CSR matrix of the chosen start's posterior for the
            second component, at each mixture-path gRNA's cells with a nonzero count (and any
            zero-count cell with a fitted mean above 200).
        posterior_zero: `(n_grnas,)` the posterior shared by a gRNA's other zero-count cells;
            NaN for gRNAs on the backup rule.
        fits: One row per gRNA: grna_id, n_nonzero, method ("mixture" or "backup"),
            backup_reason ("", "n_nonzero" or "em"), em_converged, log_lik, best_start
            (0-based), pi, g_pert (the chosen start's last estimates), glm_converged,
            glm_n_iter, n_assigned.
        grna_ids: Row names.
    """

    assigned: sparse.csr_matrix
    posterior: sparse.csr_matrix
    posterior_zero: np.ndarray
    fits: pd.DataFrame
    grna_ids: tuple[str, ...]


def _tolerance(curr, prev):
    if curr == -math.inf or prev == -math.inf:
        return 1.0
    return abs(curr - prev) / min(abs(curr), abs(prev))


def _seq_sum(x):
    s = 0.0
    for v in x:
        s += v
    return s


def _reduced_em_py(pi_guesses, g_pert_guesses, g, mu0, lgf, n_flat, s0_flat, m0min_flat, n_cells):
    """sceptre's `run_reduced_em_algo_cpp` with the flat zero-count cells handled as a group.

    `g`, `mu0`, `lgf` hold the explicit cells; the flat group has `n_flat` cells with count 0,
    fitted means summing to `s0_flat` and with minimum `m0min_flat`.
    """
    n_starts = pi_guesses.shape[0]
    n_exp = g.shape[0]
    log_mu0 = np.log(mu0)
    outer_ti = np.zeros(n_exp)
    outer_ti_flat = 0.0
    outer_i = 0
    outer_ll = -math.inf
    outer_converged = False
    outer_pi = math.nan
    outer_g_pert = math.nan
    start_ll = np.full(n_starts, -math.inf)
    start_converged = np.zeros(n_starts, dtype=np.bool_)
    for b in range(n_starts):
        converged = False
        prev_ll = -math.inf
        g_pert = g_pert_guesses[b]
        pi = pi_guesses[b]
        iteration = 1
        ti = np.zeros(n_exp)
        ti_flat = 0.0
        curr_ll = -math.inf
        while True:
            mu1 = mu0 + g_pert
            log_mu1 = np.log(mu1)
            l1mp = np.log(1.0 - pi)
            lp = np.log(pi)
            p0 = np.exp(l1mp + g * log_mu0 - mu0 - lgf)
            p1 = np.exp(lp + g * log_mu1 - mu1 - lgf)
            s = p0 + p1
            for j in range(n_exp):
                if s[j] < _S_FLOOR:
                    s[j] = _S_FLOOR
            curr_ll = _seq_sum(np.log(s))
            flat_nan = False
            if n_flat > 0:
                if not np.isfinite(g_pert) or m0min_flat + g_pert <= 0.0:
                    flat_nan = True
                    curr_ll = math.nan
                else:
                    curr_ll += n_flat * np.log((1.0 - pi) + pi * np.exp(-g_pert)) - s0_flat
            quotient = l1mp - lp + g * (log_mu0 - log_mu1) + mu1 - mu0
            ti = 1.0 / (np.exp(quotient) + 1.0)
            ti_flat = 1.0 / (np.exp(l1mp - lp + g_pert) + 1.0)
            all_zero = True
            any_na = False
            for j in range(n_exp):
                if ti[j] > 1e-100:
                    all_zero = False
                if not np.isfinite(ti[j]):
                    any_na = True
            if n_flat > 0:
                if ti_flat > 1e-100:
                    all_zero = False
                if flat_nan or not np.isfinite(ti_flat):
                    any_na = True
            if all_zero or any_na:
                curr_ll = -math.inf
                break
            pi = (_seq_sum(ti) + n_flat * ti_flat) / n_cells
            if pi > 0.5:
                ti = 1.0 - ti
                ti_flat = 1.0 - ti_flat
                pi = 1.0 - pi
            tol = _tolerance(curr_ll, prev_ll)
            if tol < _EP_TOL and iteration >= _MIN_IT:
                converged = True
                break
            prev_ll = curr_ll
            iteration += 1
            if iteration >= _MAX_IT:
                break
            e1 = _seq_sum(ti * g)
            e2 = _seq_sum(ti * mu0) + ti_flat * s0_flat
            g_pert = np.log(e1) - np.log(e2)
        start_ll[b] = curr_ll
        start_converged[b] = converged
        if curr_ll > outer_ll and converged:
            outer_ti = ti.copy()
            outer_ti_flat = ti_flat
            outer_i = b
            outer_ll = curr_ll
            outer_converged = True
            outer_pi = pi
            outer_g_pert = g_pert
    return (
        outer_ti,
        outer_ti_flat,
        outer_i,
        outer_converged,
        outer_ll,
        outer_pi,
        outer_g_pert,
        start_ll,
        start_converged,
    )


try:
    from numba import njit

    _tolerance = njit(cache=True)(_tolerance)
    _seq_sum = njit(cache=True)(_seq_sum)
    _reduced_em_jit = njit(cache=True)(_reduced_em_py)
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised by the no-numba CI check
    _reduced_em_jit = None
    _HAVE_NUMBA = False


def _reduced_em(*args):
    if _HAVE_NUMBA:
        return _reduced_em_jit(*args)
    with np.errstate(all="ignore"):
        return _reduced_em_py(*args)


def _run_em(cells, g_nz, mu0, pi_guesses, g_pert_guesses) -> dict:
    """The EM for one gRNA, from its nonzero cells and counts and its fitted means.

    Splits the cells into the explicit ones (nonzero counts, and zero counts with a fitted mean
    above 200) and the flat group, runs the EM, and returns the chosen start's posterior for
    every cell (`ti_all`), with the explicit cells, the flat posterior and the EM's flags.
    """
    n_cells = mu0.size
    big = mu0 > _FLAT_MU_MAX
    big[cells] = False
    exp_cells = np.union1d(cells, np.flatnonzero(big))
    g_exp = np.zeros(exp_cells.size)
    g_exp[np.searchsorted(exp_cells, cells)] = g_nz
    flat = np.ones(n_cells, dtype=bool)
    flat[exp_cells] = False
    n_flat = int(np.count_nonzero(flat))
    s0_flat = float(np.cumsum(mu0[flat])[-1]) if n_flat else 0.0
    m0min_flat = float(mu0[flat].min()) if n_flat else math.inf
    ti, ti_flat, best, converged, ll, pi_hat, g_pert_hat, start_ll, start_conv = _reduced_em(
        np.asarray(pi_guesses, dtype=np.float64),
        np.asarray(g_pert_guesses, dtype=np.float64),
        g_exp,
        mu0[exp_cells],
        gammaln(g_exp + 1.0),
        n_flat,
        s0_flat,
        m0min_flat,
        float(n_cells),
    )
    ti_all = np.full(n_cells, ti_flat)
    ti_all[exp_cells] = ti
    return {
        "exp_cells": exp_cells,
        "flat": flat,
        "ti": ti,
        "ti_flat": float(ti_flat),
        "ti_all": ti_all,
        "best_start": int(best),
        "converged": bool(converged),
        "log_lik": float(ll),
        "pi": float(pi_hat),
        "g_pert": float(g_pert_hat),
        "start_log_lik": np.asarray(start_ll),
        "start_converged": np.asarray(start_conv),
    }


def assign_grnas_mixture(
    grna_matrix: sparse.spmatrix | np.ndarray,
    grna_ids: Sequence[str],
    covariate_matrix: np.ndarray,
    *,
    probability_threshold: float = 0.8,
    n_nonzero_cells_cutoff: int = 10,
    backup_threshold: float = 5,
    pi_guesses: Sequence[float] = PI_GUESSES,
    g_pert_guesses: Sequence[float] = G_PERT_GUESSES,
) -> MixtureResult:
    """Assign gRNAs to cells with sceptre's mixture method.

    Args:
        grna_matrix: `(n_grnas, n_cells)` raw integer gRNA UMI counts, scipy.sparse or dense.
        grna_ids: The `n_grnas` row names, distinct.
        covariate_matrix: `(n_cells, p)` design, intercept included -- sceptre's default is
            built by `mixture_design_matrix`.
        probability_threshold: Posterior a cell needs to be assigned.
        n_nonzero_cells_cutoff: A gRNA with fewer cells of count >= 1 uses the backup rule.
        backup_threshold: The backup rule assigns cells whose count reaches this.
        pi_guesses: Starting mixing proportions, one per EM start.
        g_pert_guesses: Starting perturbation effects, one per start.

    Returns:
        A `MixtureResult`.

    Raises:
        ValueError: invalid inputs, or a GLM fit with a non-finite fitted mean (the gRNA is
            named; sceptre's runners retry with a smaller design then).
    """
    counts = as_count_matrix(grna_matrix).tocsr()
    counts.sort_indices()
    n_grnas, n_cells = counts.shape
    ids = check_ids(grna_ids, n_grnas, "grna_ids")
    X = np.asarray(covariate_matrix, dtype=np.float64)
    validate_design_matrix(X, n_cells=n_cells)
    pi_guesses = np.asarray(pi_guesses, dtype=np.float64)
    g_pert_guesses = np.asarray(g_pert_guesses, dtype=np.float64)
    if pi_guesses.shape != g_pert_guesses.shape or pi_guesses.ndim != 1 or pi_guesses.size == 0:
        raise ValueError("pi_guesses and g_pert_guesses must be 1-D, equally long and non-empty")
    xo = x_outer_flat(X)
    y = np.zeros(n_cells)

    assigned_rows, assigned_cols = [], []
    post_rows, post_cols, post_vals = [], [], []
    posterior_zero = np.full(n_grnas, np.nan)
    records = []
    for r in range(n_grnas):
        lo, hi = counts.indptr[r], counts.indptr[r + 1]
        cells = counts.indices[lo:hi].astype(np.int64)
        g_nz = counts.data[lo:hi]
        n_nonzero = int(np.count_nonzero(g_nz >= 1))
        rec = {
            "grna_id": ids[r],
            "n_nonzero": n_nonzero,
            "method": "backup",
            "backup_reason": "",
            "em_converged": False,
            "log_lik": np.nan,
            "best_start": -1,
            "pi": np.nan,
            "g_pert": np.nan,
            "glm_converged": False,
            "glm_n_iter": 0,
        }
        chosen = None
        if n_nonzero < n_nonzero_cells_cutoff:
            rec["backup_reason"] = "n_nonzero"
        else:
            y[cells] = g_nz
            fit = fit_poisson_glm_batch(X, y, X_outer_flat=xo, mu_floor=_R_MU_FLOOR)
            y[cells] = 0.0
            mu0 = np.asarray(fit.fitted_values, dtype=np.float64)
            rec["glm_converged"] = bool(fit.converged)
            rec["glm_n_iter"] = int(fit.n_iter)
            if not np.all(np.isfinite(mu0)):
                raise ValueError(f"the Poisson GLM for gRNA {ids[r]!r} has non-finite fitted means")
            em = _run_em(cells, g_nz, mu0, pi_guesses, g_pert_guesses)
            rec["em_converged"] = em["converged"]
            if em["converged"] and em["log_lik"] != -math.inf:
                rec.update(
                    method="mixture",
                    log_lik=em["log_lik"],
                    best_start=em["best_start"],
                    pi=em["pi"],
                    g_pert=em["g_pert"],
                )
                exp_cells = em["exp_cells"]
                posterior_zero[r] = em["ti_flat"]
                post_rows.append(np.full(exp_cells.size, r, dtype=np.int64))
                post_cols.append(exp_cells)
                post_vals.append(em["ti"])
                chosen = exp_cells[em["ti"] >= probability_threshold]
                if em["flat"].any() and em["ti_flat"] >= probability_threshold:
                    chosen = np.union1d(chosen, np.flatnonzero(em["flat"]))
            else:
                rec["backup_reason"] = "em"
        if chosen is None:
            chosen = cells[g_nz >= backup_threshold]
        rec["n_assigned"] = int(chosen.size)
        records.append(rec)
        assigned_rows.append(np.full(chosen.size, r, dtype=np.int64))
        assigned_cols.append(np.asarray(chosen, dtype=np.int64))

    def _stack(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    ar, ac = _stack(assigned_rows, np.int64), _stack(assigned_cols, np.int64)
    assigned = sparse.csr_matrix((np.ones(ar.size, dtype=bool), (ar, ac)), shape=(n_grnas, n_cells))
    posterior = sparse.csr_matrix(
        (_stack(post_vals, np.float64), (_stack(post_rows, np.int64), _stack(post_cols, np.int64))),
        shape=(n_grnas, n_cells),
    )
    return MixtureResult(
        assigned=assigned,
        posterior=posterior,
        posterior_zero=posterior_zero,
        fits=pd.DataFrame.from_records(records),
        grna_ids=ids,
    )
