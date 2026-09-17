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

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..crt.sampler import crt_index_sampler_fast
from ..glm.irls import fit_binomial_glm_batch, fit_poisson_glm_batch
from ..glm.nb_theta import estimate_theta
from ..precompute.pieces import PrecomputationPieces, compute_precomputation_pieces
from ..test_statistic.resampling import run_low_level_test_full


@dataclass
class GenePrecomputation:
    y: np.ndarray
    fitted_coefs: np.ndarray
    theta: float
    pieces: PrecomputationPieces


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


def fit_all_genes(response_matrix, gene_ids: list[str], covariate_matrix: np.ndarray) -> dict[str, GenePrecomputation]:
    """One batched Poisson IRLS call across all genes (they share the full
    covariate matrix), then per-gene theta estimation and precomputation pieces."""
    n_cells = covariate_matrix.shape[0]
    Y = np.empty((n_cells, len(gene_ids)))
    for k, _gene_id in enumerate(gene_ids):
        Y[:, k] = _get_row(response_matrix, k)

    fit = fit_poisson_glm_batch(covariate_matrix, Y)
    dfr = covariate_matrix.shape[0] - covariate_matrix.shape[1]

    out: dict[str, GenePrecomputation] = {}
    for k, gene_id in enumerate(gene_ids):
        y_k = Y[:, k]
        theta_est, _method = estimate_theta(y=y_k, mu=fit.fitted_values[:, k], dfr=dfr)
        theta = max(min(theta_est, 1000.0), 0.01)
        pieces = compute_precomputation_pieces(y_k, covariate_matrix, fit.coefs[:, k], theta)
        out[gene_id] = GenePrecomputation(y=y_k, fitted_coefs=fit.coefs[:, k], theta=theta, pieces=pieces)
    return out


def fit_all_targets(
    grna_target_cells: dict[str, np.ndarray],
    covariate_matrix: np.ndarray,
    *, B1: int, B2: int, B3: int, rng: np.random.Generator,
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
    """
    rng = np.random.default_rng(seed)

    gene_precomps = fit_all_genes(response_matrix, gene_ids, covariate_matrix)

    pairs_by_target = {target_id: group for target_id, group in pairs.groupby("grna_target")}
    target_ids_needed = [t for t in grna_target_cells if t in pairs_by_target]

    rows = []
    for chunk_start in range(0, len(target_ids_needed), target_chunk_size):
        chunk_ids = target_ids_needed[chunk_start : chunk_start + target_chunk_size]
        chunk_cells = {t: grna_target_cells[t] for t in chunk_ids}
        target_precomps = fit_all_targets(chunk_cells, covariate_matrix, B1=B1, B2=B2, B3=B3, rng=rng)

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
                    B1=B1, B2=B2, B3=B3,
                    fit_parametric_curve=fit_parametric_curve,
                    side_code=side_code,
                )
                rows.append({
                    "response_id": row.response_id,
                    "grna_target": target_id,
                    "p_value": result.p_value,
                    "fold_change": result.fold_change,
                    "log_2_fold_change": np.log2(result.fold_change),
                    "z_orig": result.z_orig,
                    "stage": result.stage,
                })
        del target_precomps  # free this chunk's synthetic_idxs before the next one

    return pd.DataFrame(rows)
