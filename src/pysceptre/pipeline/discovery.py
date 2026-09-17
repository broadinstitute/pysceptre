"""Complement-mode discovery-analysis orchestration.

Verified against `run_crt_in_memory_v2`: for high-MOI/complement,
`run_outer_regression` is always true and `subset_to_nt_cells` is always
false, so each gene's Poisson/NB precomputation is fit exactly ONCE, against
the *full* covariate matrix (all cells), and cached/reused across every gRNA
target it's paired with (this is the `crt_glm_factored_out` code path) --
there is no per-pair GLM refit. Similarly each target's logistic
precomputation + CRT draw happens once and is reused across every gene
paired with it.

This is the actual batching opportunity for the "quickly run" performance
goal: all genes share one design matrix (the full covariate matrix), so the
per-gene Poisson fit is one batched `fit_poisson_glm_batch` call, not a
per-gene loop; likewise all targets share it for the logistic fit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..crt.sampler import crt_index_sampler_fast
from ..glm.irls import fit_binomial_glm_batch, fit_poisson_glm_batch
from ..glm.nb_theta import estimate_theta
from ..precompute.pieces import PrecomputationPieces, compute_precomputation_pieces
from ..test_statistic.resampling import run_low_level_test_full

# Budget for the transient dense arrays the gene IRLS needs. 2 GB holds
# ~480 genes at 131k cells, well above the few hundred a real analysis
# tests, so normal runs are a single chunk as before.
_DEFAULT_MAX_FIT_MEMORY_GB = 2.0


# theta estimation methods reported by glm/nb_theta.py::estimate_theta.
# Anything other than MLE means the MLE failed and a fallback was used.
_THETA_METHOD_NAMES = {1: "MLE", 2: "method of moments", 3: "pilot estimate"}
_THETA_BOUNDS = (0.01, 1000.0)


@dataclass
class GenePrecomputation:
    y: np.ndarray
    fitted_coefs: np.ndarray
    theta: float
    pieces: PrecomputationPieces
    # Diagnostics, so a run can tell when a fit was degenerate rather than
    # silently returning a statistic built on it.
    theta_method: int = 1
    theta_clamped: bool = False
    glm_converged: bool = True


@dataclass
class TargetPrecomputation:
    trt_idxs: np.ndarray  # 0-based
    fitted_probabilities: np.ndarray
    synthetic_idxs: list[np.ndarray]


def _get_row(response_matrix, i: int) -> np.ndarray:
    """Returns gene row i as a dense 1D float array, for either a dense
    ndarray (n_genes, n_cells) or a scipy.sparse matrix in CSR-like format."""
    if hasattr(response_matrix, "toarray"):
        return np.asarray(response_matrix[i].toarray()).ravel().astype(float)
    return np.asarray(response_matrix[i], dtype=float)


def gene_chunk_size_for_budget(n_cells: int, n_genes: int, max_memory_gb: float) -> int:
    """How many genes to densify and fit at once.

    The Poisson IRLS needs dense `(n_cells, k)` arrays -- the responses plus
    mu, weights and working response -- so the transient cost of fitting k
    genes together is about `4 * n_cells * k * 8` bytes. Densifying every gene
    at once is what made a genome-wide run unaffordable: 38,606 genes over
    131k cells is a 40 GB response array and ~162 GB peak.

    Chunking does not change which genes get which fit: each column's normal
    equations are solved independently. Agreement is to floating-point
    tolerance rather than bit-for-bit, because the solve is one batched
    `np.linalg.solve` whose blocking depends on the batch width -- measured
    spread ~1e-15, which cannot move a rank-based p-value.
    """
    per_gene = 4 * n_cells * 8
    if per_gene <= 0:
        return n_genes
    return max(1, min(n_genes, int(max_memory_gb * 1e9 // per_gene)))


def fit_all_genes(
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    *,
    max_fit_memory_gb: float = _DEFAULT_MAX_FIT_MEMORY_GB,
) -> dict[str, GenePrecomputation]:
    """Batched Poisson IRLS across genes (they share the full covariate
    matrix), then per-gene theta estimation and precomputation pieces.

    Genes are fit in chunks sized by `max_fit_memory_gb` rather than all at
    once; see `gene_chunk_size_for_budget`. Note this bounds only the
    *transient* fitting cost -- the returned precomputations are retained for
    the whole run at about `(4 + p) * n_cells * 8` bytes per gene, which is
    why callers should pass only the genes that appear in `pairs`.
    """
    n_cells = covariate_matrix.shape[0]
    dfr = covariate_matrix.shape[0] - covariate_matrix.shape[1]
    chunk = gene_chunk_size_for_budget(n_cells, len(gene_ids), max_fit_memory_gb)

    out: dict[str, GenePrecomputation] = {}
    for start in range(0, len(gene_ids), chunk):
        chunk_ids = gene_ids[start : start + chunk]
        Y = np.empty((n_cells, len(chunk_ids)))
        for j in range(len(chunk_ids)):
            Y[:, j] = _get_row(response_matrix, start + j)

        fit = fit_poisson_glm_batch(covariate_matrix, Y)

        for j, gene_id in enumerate(chunk_ids):
            y_j = Y[:, j]
            theta_est, method = estimate_theta(y=y_j, mu=fit.fitted_values[:, j], dfr=dfr)
            lo, hi = _THETA_BOUNDS
            theta = max(min(theta_est, hi), lo)
            pieces = compute_precomputation_pieces(y_j, covariate_matrix, fit.coefs[:, j], theta)
            out[gene_id] = GenePrecomputation(
                y=y_j,
                fitted_coefs=fit.coefs[:, j],
                theta=theta,
                pieces=pieces,
                theta_method=method,
                theta_clamped=not (lo <= theta_est <= hi),
                glm_converged=bool(fit.converged[j]),
            )

    _warn_about_degenerate_gene_fits(out)
    return out


def _design_is_rank_deficient(pieces: PrecomputationPieces) -> bool:
    """Whether Zt_wZ is numerically rank deficient.

    Uses the same relative tolerance convention as `np.linalg.matrix_rank`:
    an eigenvalue counts as zero below `max_eigenvalue * p * eps`. The test
    must be relative -- exactly collinear covariates produce a smallest
    eigenvalue near 1e-14, not 0, and leave D finite, so an absolute
    `<= 0` check or an `isfinite(D)` check both miss them.
    """
    min_eig, max_eig = pieces.min_eigenvalue, pieces.max_eigenvalue
    if not np.isfinite(min_eig) or not np.isfinite(max_eig):
        return True
    if max_eig <= 0.0:
        return True
    p = pieces.D.shape[0]
    return min_eig <= max_eig * p * np.finfo(float).eps


def summarize_gene_fits(gene_precomps: dict[str, GenePrecomputation]) -> dict[str, list[str]]:
    """Group gene ids by the ways their fit was degenerate.

    Returns a dict with keys `glm_not_converged`, `theta_fallback`,
    `theta_clamped` and `singular_design`, each holding the ids affected.
    All are empty for a healthy fit.
    """
    summary: dict[str, list[str]] = {
        "glm_not_converged": [],
        "theta_fallback": [],
        "theta_clamped": [],
        "singular_design": [],
    }
    for gene_id, gp in gene_precomps.items():
        if not gp.glm_converged:
            summary["glm_not_converged"].append(gene_id)
        if gp.theta_method != 1:
            summary["theta_fallback"].append(gene_id)
        if gp.theta_clamped:
            summary["theta_clamped"].append(gene_id)
        if _design_is_rank_deficient(gp.pieces):
            summary["singular_design"].append(gene_id)
    return summary


def _warn_about_degenerate_gene_fits(gene_precomps: dict[str, GenePrecomputation]) -> None:
    """One summary warning for the whole gene set rather than per-gene spam.

    These conditions were previously computed and thrown away -- `estimate_theta`
    returns a method code that says whether the MLE succeeded, and the smallest
    eigenvalue of Zt_wZ says whether D is meaningful -- so a run could report a
    statistic built on a degenerate fit with no indication.
    """
    summary = summarize_gene_fits(gene_precomps)
    n = len(gene_precomps)
    parts = []
    if summary["singular_design"]:
        parts.append(
            f"{len(summary['singular_design'])} with a numerically rank-deficient "
            f"design, usually collinear covariates (rows of their D matrix are "
            f"amplified roundoff, so statistics built from it are unreliable)"
        )
    if summary["glm_not_converged"]:
        parts.append(f"{len(summary['glm_not_converged'])} whose Poisson GLM did not converge")
    if summary["theta_clamped"]:
        lo, hi = _THETA_BOUNDS
        parts.append(
            f"{len(summary['theta_clamped'])} with the dispersion estimate "
            f"clamped to [{lo}, {hi}], which usually means counts close to "
            f"equidispersed, leaving theta unidentifiable"
        )
    if summary["theta_fallback"]:
        methods = sorted(
            {
                _THETA_METHOD_NAMES.get(gene_precomps[g].theta_method, "unknown")
                for g in summary["theta_fallback"]
            }
        )
        parts.append(
            f"{len(summary['theta_fallback'])} where the dispersion MLE failed "
            f"and fell back to {' / '.join(methods)}"
        )
    if parts:
        warnings.warn(
            f"degenerate gene fits out of {n}: " + "; ".join(parts) + ". "
            "Call pysceptre.pipeline.discovery.summarize_gene_fits for the "
            "affected gene ids.",
            stacklevel=3,
        )


def fit_all_targets(
    grna_target_cells: dict[str, np.ndarray],
    covariate_matrix: np.ndarray,
    *,
    B1: int,
    B2: int,
    B3: int,
    rng: np.random.Generator,
) -> dict[str, TargetPrecomputation]:
    """One batched binomial IRLS call across all targets (indicator columns
    share the full covariate matrix), then a CRT draw per target."""
    n_cells = covariate_matrix.shape[0]
    target_ids = list(grna_target_cells.keys())
    Y = np.zeros((n_cells, len(target_ids)))
    for k, target_id in enumerate(target_ids):
        Y[grna_target_cells[target_id], k] = 1.0

    fit = fit_binomial_glm_batch(covariate_matrix, Y)

    out: dict[str, TargetPrecomputation] = {}
    B_total = B1 + B2 + B3
    for k, target_id in enumerate(target_ids):
        fitted_probabilities = fit.fitted_values[:, k]
        synthetic_idxs = crt_index_sampler_fast(fitted_probabilities, B_total, rng)
        out[target_id] = TargetPrecomputation(
            trt_idxs=grna_target_cells[target_id],
            fitted_probabilities=fitted_probabilities,
            synthetic_idxs=synthetic_idxs,
        )
    return out


_DEFAULT_TARGET_CHUNK_SIZE = 200

# Budget for the CRT draws held in memory at once. Chosen so the skew_normal
# path at real scale is never auto-shrunk (moi5: B_total=5498 x ~400 treated
# cells x 8 B x 200 targets ~= 3.5 GB), while still catching the
# no_approximation blow-up, which is three orders of magnitude larger.
_DEFAULT_MAX_DRAW_MEMORY_GB = 8.0


_BYTES_PER_INDEX = 8  # int64 cell index


def estimate_draw_memory_bytes(B_total: int, n_trt_values, chunk_size: int) -> float:
    """Bytes of CRT draws held at once: one chunk of targets, each holding
    B_total ragged index arrays of about n_trt entries.

    Uses the *median* treated-cell count as the per-target size, matching how
    `crt_index_sampler_fast` draws (expected inclusions per draw equals the
    observed treated-cell count, by the intercept property of the logistic
    MLE -- see crt/sampler.py).
    """
    if B_total <= 0 or len(n_trt_values) == 0:
        return 0.0
    median_n_trt = float(np.median(np.asarray(n_trt_values, dtype=float)))
    return B_total * median_n_trt * _BYTES_PER_INDEX * chunk_size


def _fit_chunk_size_to_budget(
    B_total: int, n_trt_values, chunk_size: int, max_memory_gb: float
) -> int:
    """Shrink `chunk_size` until the estimated draw memory fits the budget.

    Chunk size affects only peak memory and batching width, not which draws
    are taken: `fit_all_targets` draws per target in a fixed order, so the RNG
    stream is unchanged by where chunk boundaries fall
    (`test_memory_guard.py` pins this).

    As with gene chunking, the batched logistic solve is not bit-for-bit
    across batch widths (~1e-15), so fitted probabilities can differ in their
    last bits. That is far too small to change a binomial draw count in
    practice, but "identical" here means to floating-point tolerance, not
    exactly.
    """
    budget = max_memory_gb * 1e9
    if estimate_draw_memory_bytes(B_total, n_trt_values, chunk_size) <= budget:
        return chunk_size

    per_target = estimate_draw_memory_bytes(B_total, n_trt_values, 1)
    if per_target <= 0:
        return chunk_size
    fitted = min(chunk_size, max(1, int(budget // per_target)))

    message = (
        f"reducing target_chunk_size from {chunk_size} to {fitted}: CRT draws "
        f"need about {per_target / 1e9:.2f} GB per target "
        f"(B1+B2+B3 = {B_total:,} draws), against "
        f"max_draw_memory_gb={max_memory_gb}. Chunk size affects only peak "
        f"memory and batching width, not results."
    )
    if fitted == 1:
        # Either a single target already blows the budget, or it only just
        # fits -- both mean every target is processed alone, and both are
        # overwhelmingly likely to be the no_approximation blow-up.
        message += (
            " At target_chunk_size=1 each target is drawn on its own, which is"
            " slow, and peak memory runs several times the figure above while"
            " a draw is being built, so this run may still exhaust memory."
            " This is usually resampling_approximation='no_approximation',"
            " whose B3 grows as n_pairs / multiple_testing_alpha;"
            " 'skew_normal' needs a few MB per target instead."
        )
    warnings.warn(message, stacklevel=3)
    return fitted


def run_discovery_ntcells_complement(
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    grna_target_cells: dict[str, np.ndarray],
    pairs: pd.DataFrame,
    *,
    B1: int = 499,
    B2: int = 4999,
    B3: int = 0,
    fit_parametric_curve: bool = True,
    side_code: int = 0,
    seed: int | None = None,
    target_chunk_size: int = _DEFAULT_TARGET_CHUNK_SIZE,
    max_draw_memory_gb: float = _DEFAULT_MAX_DRAW_MEMORY_GB,
    max_fit_memory_gb: float = _DEFAULT_MAX_FIT_MEMORY_GB,
) -> pd.DataFrame:
    """pairs: DataFrame with columns 'response_id', 'grna_target' -- the
    QC-passed pairs to test. Returns a DataFrame with one row per pair:
    response_id, grna_target, p_value, fold_change, log_2_fold_change, z_orig, stage.

    Targets are fit and CRT-drawn in chunks of `target_chunk_size` rather than
    all at once: each target's B1+B2+B3 synthetic index sets are individually
    small, but holding *every* target's draws in memory simultaneously does
    not scale -- at real dataset sizes (~3,000 targets x ~5,500 resamples)
    this was measured to OOM-kill the process. Chunking keeps peak memory
    bounded to O(chunk_size) while still batching the (bulk of the) per-target
    logistic fit and CRT draw across many targets at once for speed, not
    falling back to a slow one-target-at-a-time loop.

    `target_chunk_size` is automatically reduced if the resulting draws would
    exceed `max_draw_memory_gb`, which matters for
    `resampling_approximation="no_approximation"`: its B3 grows as
    `n_pairs / multiple_testing_alpha`, reaching 1.65M draws (5.2 GB) per
    target at moi5 scale, or ~1 TB at the default chunk size. Shrinking the
    chunk does not change results -- see `_fit_chunk_size_to_budget`.
    """
    rng = np.random.default_rng(seed)

    gene_precomps = fit_all_genes(
        response_matrix, gene_ids, covariate_matrix, max_fit_memory_gb=max_fit_memory_gb
    )

    pairs_by_target = {target_id: group for target_id, group in pairs.groupby("grna_target")}
    target_ids_needed = [t for t in grna_target_cells if t in pairs_by_target]

    target_chunk_size = _fit_chunk_size_to_budget(
        B1 + B2 + B3,
        [len(grna_target_cells[t]) for t in target_ids_needed],
        target_chunk_size,
        max_draw_memory_gb,
    )

    rows = []
    for chunk_start in range(0, len(target_ids_needed), target_chunk_size):
        chunk_ids = target_ids_needed[chunk_start : chunk_start + target_chunk_size]
        chunk_cells = {t: grna_target_cells[t] for t in chunk_ids}
        target_precomps = fit_all_targets(
            chunk_cells, covariate_matrix, B1=B1, B2=B2, B3=B3, rng=rng
        )

        for target_id in chunk_ids:
            target = target_precomps[target_id]
            for row in pairs_by_target[target_id].itertuples(index=False):
                gene = gene_precomps[row.response_id]

                result = run_low_level_test_full(
                    y=gene.y,
                    mu=gene.pieces.mu,
                    a=gene.pieces.a,
                    w=gene.pieces.w,
                    D=gene.pieces.D,
                    trt_idxs=target.trt_idxs,
                    synthetic_idxs=target.synthetic_idxs,
                    B1=B1,
                    B2=B2,
                    B3=B3,
                    fit_parametric_curve=fit_parametric_curve,
                    side_code=side_code,
                )
                rows.append(
                    {
                        "response_id": row.response_id,
                        "grna_target": target_id,
                        "p_value": result.p_value,
                        "fold_change": result.fold_change,
                        "log_2_fold_change": np.log2(result.fold_change),
                        "z_orig": result.z_orig,
                        "stage": result.stage,
                    }
                )
        del target_precomps  # free this chunk's synthetic_idxs before the next one

    return pd.DataFrame(rows)
