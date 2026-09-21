"""Building the analytical power estimator's inputs from a pysceptre analysis.

`compute_power_posthoc` takes three things pysceptre does not otherwise
produce: per-gRNA cell counts, per-gene baseline expression statistics, and
the screen's own nominal p-value threshold. These helpers derive each one.

They take arrays and frames, never a loaded export. Reading an `.h5mu`
requires the `io` extra and nothing under `src/` imports it, so the glue that
turns a file into these arguments belongs in `scripts/`. See `docs/design.md`,
"Analytical per-pair power".

Every function here can be got subtly wrong in a way that still returns
plausible numbers, which is why each says what it is reproducing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

__all__ = [
    "bh_nominal_cutoff",
    "cells_per_grna_from_assignments",
    "poscounts_size_factors",
    "baseline_expression_stats",
    "baseline_expression_stats_from_fits",
]


def cells_per_grna_from_assignments(
    grna_target_data_frame: pd.DataFrame,
    targeting_grna_cells: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Per-gRNA cell counts in the shape `compute_power_posthoc` wants.

    Args:
        grna_target_data_frame: the screen's *design*, one row per
            (`grna_id`, `grna_target`). A guide inside several overlapping
            candidate elements appears once per target, and every one of those
            rows is kept.
        targeting_grna_cells: `grna_id` -> that guide's cell indices. Guides
            absent from it are guides with no cells.

    Returns:
        `grna_id`, `grna_target`, `num_cells`, one row per design row.

    Two things this does that a shorter version would not, both of which
    change the answer rather than the style:

    **It iterates the design, not the guides that have cells.** A designed
    guide that ended up with none is a `num_cells = 0` row, not an absent one.
    Dropped instead, `num_trt_cells_sq` would be computed over a smaller guide
    set than the screen used, which is the term no target-level total can
    supply in the first place.

    **It keeps a guide under every target it belongs to.** The map is
    many-to-many, so a guide in two overlapping elements contributes its cells
    to both sums; that is what R does. Deduplicating by `grna_id` looks like
    hygiene and silently shrinks exactly those targets.
    """
    needed = ("grna_id", "grna_target")
    absent = [c for c in needed if c not in grna_target_data_frame.columns]
    if absent:
        raise KeyError(f"grna_target_data_frame is missing column(s): {absent}")

    design = grna_target_data_frame.loc[:, list(needed)].astype(str)
    dup = design.duplicated()
    if dup.any():
        offenders = sorted(design.loc[dup].itertuples(index=False, name=None))
        raise ValueError(
            f"grna_target_data_frame repeats {len(offenders)} (grna_id, grna_target) pair(s), "
            f"including: {offenders[:5]}"
        )

    counts = {str(g): int(np.asarray(idx).size) for g, idx in targeting_grna_cells.items()}
    out = design.reset_index(drop=True)
    out["num_cells"] = out["grna_id"].map(counts).fillna(0).astype(int)
    return out


def poscounts_size_factors(counts: sparse.spmatrix) -> np.ndarray:
    """DESeq2 "poscounts" size factors, one per cell, over the nonzeros only.

    Args:
        counts: `(n_genes, n_cells)` sparse counts. **All** genes and **all**
            cells, not a subset -- see the note below.

    Returns:
        `(n_cells,)` positive size factors.

    Reproduces the R this estimator's `expression_mean` was validated against
    (WattEG `prepare_sim_input.R:157-223`): a per-gene geometric mean over the
    nonzero counts, then each cell's factor is the median log-ratio of its own
    nonzeros against those means.

    **The gene and cell sets are part of the definition.** The geometric mean
    runs over every gene and the median over every cell, so computing this on
    a subset gives different factors for the cells that remain, and therefore
    different normalised means for every gene. Both results look equally
    complete. Pass the whole matrix.

    DESeq2 itself is not used because `DESeqDataSetFromMatrix` densifies; this
    walks the nonzeros.

    Raises:
        ValueError: a negative count, or a cell whose factor is undefined
            because it has no nonzero in any gene with a finite geometric
            mean.
    """
    csc = sparse.csc_matrix(counts)
    csc.eliminate_zeros()
    n_genes, n_cells = csc.shape
    if csc.data.size and csc.data.min() < 0:
        raise ValueError("counts contain negative values")

    log_x = np.log(csc.data.astype(float))
    gene_of_nonzero = csc.indices

    # Geometric mean per gene, over all cells: the log-sum of a gene's nonzeros divided by the
    # cell count, matching R's `tapply(..., sum, default = 0) / n_cell`. A gene that is zero
    # everywhere is excluded rather than given a mean of 1.
    log_geomeans = np.bincount(gene_of_nonzero, weights=log_x, minlength=n_genes) / n_cells
    row_totals = np.bincount(gene_of_nonzero, minlength=n_genes)
    usable_gene = row_totals > 0

    ratio = log_x - log_geomeans[gene_of_nonzero]
    valid = usable_gene[gene_of_nonzero]

    # The nonzeros are already grouped by cell, which is what CSC means, so each cell's median is
    # taken over a contiguous slice and nothing has to be sorted by cell.
    factors = np.empty(n_cells, dtype=float)
    starts, ends = csc.indptr[:-1], csc.indptr[1:]
    for k in range(n_cells):
        sl = slice(starts[k], ends[k])
        v = ratio[sl][valid[sl]]
        factors[k] = np.median(v) if v.size else np.nan
    factors = np.exp(factors)

    bad = ~np.isfinite(factors) | (factors <= 0)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} of {n_cells} cells have no usable size factor (no nonzero count in "
            "any gene with a finite geometric mean). Filter those cells out first."
        )
    return factors


def baseline_expression_stats(
    counts: sparse.spmatrix,
    gene_ids: list[str],
    thetas: dict[str, float] | np.ndarray,
    *,
    size_factors: np.ndarray | None = None,
    gene_subset: list[str] | None = None,
) -> pd.DataFrame:
    """Per-gene `expression_mean` and `expression_size` for `compute_power_posthoc`.

    Args:
        counts: `(n_genes, n_cells)` sparse counts for **all** genes and cells.
        gene_ids: row labels of `counts`, in order.
        thetas: per-gene NB size. A mapping keyed by gene id, or an array
            aligned to `gene_ids`. `discovery.py::fit_all_genes` returns
            objects carrying exactly this as `.theta`, so a completed analysis
            already has them and nothing needs refitting.
        size_factors: precomputed per-cell factors, to skip a pass over the
            matrix. Computed by `poscounts_size_factors` when omitted.
        gene_subset: return only these genes. The statistics are still
            computed over the full matrix first, which is the point: the
            normalisation depends on the whole gene and cell set, so
            subsetting the *input* and subsetting the *output* are different
            operations with different answers.

    Returns:
        `response_id`, `expression_mean`, `expression_size`.

    `expression_mean` is the size-factor-normalised mean: each cell's counts
    divided by its size factor, then averaged over cells. `expression_size` is
    the NB size, theta, not the dispersion.

    **Prefer `baseline_expression_stats_from_fits` unless you are reproducing
    the published comparison.** This function's mean comes from a
    normalisation scheme sceptre does not use, and on day0 it sits about 16%
    below the mean sceptre's own model implies.

    **Two caveats on theta**, because a number that transfers badly here is
    invisible in the output. `fit_all_genes` clamps theta to `(0.01, 1000.0)`,
    so a gene at a bound carries the bound rather than its estimate; and theta
    is fitted under whatever `covariate_matrix` was passed, so it matches
    another implementation's only if the design matrices match.
    """
    csc = sparse.csc_matrix(counts)
    n_genes, n_cells = csc.shape
    if len(gene_ids) != n_genes:
        raise ValueError(f"gene_ids has {len(gene_ids)} entries for {n_genes} matrix rows")

    if size_factors is None:
        size_factors = poscounts_size_factors(csc)
    size_factors = np.asarray(size_factors, dtype=float)
    if size_factors.shape != (n_cells,):
        raise ValueError(f"size_factors has shape {size_factors.shape}, expected ({n_cells},)")
    if np.any(size_factors <= 0) or not np.all(np.isfinite(size_factors)):
        raise ValueError("size_factors must all be finite and positive")

    # Divide each column by its factor, then row-average. Done on the nonzeros: a dense
    # normalisation of a real screen is tens of GB and is what the first port died of.
    per_nonzero_factor = np.repeat(size_factors, np.diff(csc.indptr))
    normalised = csc.data.astype(float) / per_nonzero_factor
    means = np.bincount(csc.indices, weights=normalised, minlength=n_genes) / n_cells

    if isinstance(thetas, dict):
        missing = [g for g in gene_ids if g not in thetas]
        if missing and gene_subset is None:
            raise KeyError(f"thetas is missing {len(missing)} gene(s), including: {missing[:5]}")
        theta_arr = np.array([thetas.get(g, np.nan) for g in gene_ids], dtype=float)
    else:
        theta_arr = np.asarray(thetas, dtype=float)
        if theta_arr.shape != (n_genes,):
            raise ValueError(f"thetas has shape {theta_arr.shape}, expected ({n_genes},)")

    out = pd.DataFrame(
        {"response_id": list(gene_ids), "expression_mean": means, "expression_size": theta_arr}
    )
    if gene_subset is not None:
        wanted = list(dict.fromkeys(gene_subset))
        unknown = [g for g in wanted if g not in set(gene_ids)]
        if unknown:
            raise KeyError(f"gene_subset names {len(unknown)} unknown gene(s): {unknown[:5]}")
        out = out.set_index("response_id").loc[wanted].reset_index()
    bad = ~np.isfinite(out["expression_size"].to_numpy())
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} returned gene(s) have no finite theta, including: "
            f"{out.loc[bad, 'response_id'].tolist()[:5]}"
        )
    return out


def bh_nominal_cutoff(p_values: np.ndarray, alpha: float) -> float:
    """The nominal p-value threshold a BH-corrected analysis actually applied.

    Args:
        p_values: the discovery p-values. NaN entries -- the pairs that failed
            pairwise QC and were never tested -- are dropped before the
            correction, as R does, rather than inflating its denominator.
        alpha: the analysis's `multiple_testing_alpha`.

    Returns:
        The largest p-value BH calls significant, which is what
        `compute_power_posthoc`'s `cutoff` wants (halved, for the default
        `side="left"`).

    Ports WattEG's `discovery_threshold()`, including its refusal to return a
    number when nothing is significant. The code that preceded it returned
    `-Inf` there and every pair silently got zero power, so an empty result is
    raised rather than propagated.

    pysceptre applies no multiple-testing correction itself, so this does both
    halves: the BH step and then reading the threshold off it.

    Raises:
        ValueError: no usable p-value, or none significant at `alpha`.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must lie in (0, 1], got {alpha!r}")
    p = np.asarray(p_values, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        raise ValueError("no finite p-values, so no threshold can be derived")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")

    order = np.argsort(p)
    ranked = p[order]
    m = ranked.size
    # BH: the largest k with p_(k) <= k/m * alpha.
    below = ranked <= (np.arange(1, m + 1) / m) * alpha
    if not below.any():
        raise ValueError(
            f"no pair is significant at alpha={alpha}, so the largest significant p-value is "
            "undefined. Pass `cutoff` explicitly instead of deriving one."
        )
    return float(ranked[np.flatnonzero(below)[-1]])


def baseline_expression_stats_from_fits(
    covariate_matrix: np.ndarray,
    gene_fits,
    *,
    gene_subset: list[str] | None = None,
) -> pd.DataFrame:
    """Baseline statistics on the scale sceptre's own model works on.

    **This is the recommended way to build them.** Both numbers come out of
    the same negative-binomial fit the discovery test itself uses, so the
    power estimate and the test it predicts are on one expression scale by
    construction rather than by coincidence.

    Args:
        covariate_matrix: `(n_cells, p)`, the same design matrix the fits were
            produced with. Using a different one silently answers a different
            question.
        gene_fits: `gene_id` -> an object with `.fitted_coefs` and `.theta`,
            which is exactly what `discovery.py::fit_all_genes` returns. A
            completed analysis already has them.
        gene_subset: return only these genes, in this order.

    Returns:
        `response_id`, `expression_mean`, `expression_size`.

    `expression_mean` is `mean(exp(Z @ coefs))`, the average expected count
    the fitted model gives the gene. That is the quantity the score
    statistic's variance is built from, so it is what the closed form wants.
    `expression_size` is the same fit's theta.

    **Why not the size-factor-normalised mean.** `baseline_expression_stats`
    computes that instead, because it is what the published comparison used.
    Measured on day0, it sits about 16% *below* this one, near enough a
    constant across genes (correlation of logs 0.9999), so it makes the power
    estimate conservative rather than wrong-shaped. It also drags in a
    normalisation convention that has nothing to do with sceptre. Prefer this
    function unless you are reproducing those published numbers; see
    `docs/design.md`, "Analytical per-pair power".
    """
    Z = np.asarray(covariate_matrix, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"covariate_matrix must be 2-D, got shape {Z.shape}")

    ids = list(gene_fits) if gene_subset is None else list(dict.fromkeys(gene_subset))
    unknown = [g for g in ids if g not in gene_fits]
    if unknown:
        raise KeyError(f"gene_fits has no entry for {len(unknown)} gene(s): {unknown[:5]}")

    means = np.empty(len(ids), dtype=float)
    thetas = np.empty(len(ids), dtype=float)
    for k, gene in enumerate(ids):
        fit = gene_fits[gene]
        coefs = np.asarray(fit.fitted_coefs, dtype=float)
        if coefs.shape != (Z.shape[1],):
            raise ValueError(
                f"gene {gene!r} was fitted with {coefs.size} coefficients but covariate_matrix "
                f"has {Z.shape[1]} columns; these are not the same design"
            )
        # One gene at a time: the per-cell mu of a real screen is tens of MB, and holding one
        # per gene would be the densify this package refuses everywhere else.
        means[k] = float(np.mean(np.exp(Z @ coefs)))
        thetas[k] = float(fit.theta)

    out = pd.DataFrame({"response_id": ids, "expression_mean": means, "expression_size": thetas})
    bad = ~np.isfinite(out["expression_mean"].to_numpy()) | (out["expression_mean"].to_numpy() <= 0)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} gene(s) have a non-positive or non-finite fitted mean, including: "
            f"{out.loc[bad, 'response_id'].tolist()[:5]}"
        )
    return out
