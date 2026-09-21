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

import hashlib
import itertools
import os
import sys
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from ..crt.permutations import permutation_draws
from ..crt.sampler import crt_index_sampler_fast
from ..glm.irls import fit_binomial_glm_batch, fit_poisson_glm_batch, x_outer_flat
from ..glm.nb_theta import estimate_theta
from ..precompute.pieces import compute_precomputation_pieces
from ..test_statistic.resampling import run_low_level_test_full
from ..test_statistic.score_stat import (
    ListDraws,
    PermutationPrefixSums,
    PermutationSliceDraws,
    StagedDraws,
    stack_pieces,
)

# Budget for the arrays a *chunk* holds, in GB. It sizes how many genes or
# targets are processed together; it is NOT a cap on the process's memory,
# which also carries the input, retained state and allocator overhead.
#
# Users are not expected to set this. 1.0 is both the fastest and the leanest
# setting measured -- a larger budget produces chunks past the point
# where batching still pays, so it costs memory for no throughput:
#
#   budget   wall     peak RSS
#    1 GB    315.7 s   3.78 GB
#    2 GB    400.2 s   6.87 GB
#    3 GB    430.1 s   7.36 GB
#    4 GB    314.4 s   8.42 GB
#
# It exists as a parameter for datasets far outside that shape, and to bound
# the pathological cases: an unbounded target fit measured 13.1 GB, and
# no_approximation draws reach ~1 TB.
_DEFAULT_CHUNK_MEMORY_GB = 1.0

# Genes are fitted ONE AT A TIME, not batched.
#
# Batching the gene GLM means a gene's fit depends on which other genes share
# its BLAS call: floating-point addition is not associative, so the normal
# equations accumulate slightly differently, and for genes whose likelihood is
# flat -- those with near-zero expression, whose coefficients are large and
# poorly determined -- that surfaces as ~2e-8 in the coefficients with the
# deviance unchanged. Both answers solve the GLM to the requested precision;
# neither is more correct. But it means "the fit for gene X" is not a
# well-defined quantity until you say what else was in the run.
#
# Fitting each gene alone makes it well-defined, and it is not even a
# trade-off once the fits are parallelized -- which they can be precisely
# because they are independent. Measured on day0 (237 genes, 567,690 cells,
# p=11):
#
#     batched, serial      24.96 s
#     per-gene, serial     29.34 s
#     per-gene, 8 workers  10.38 s
#
# So the reproducible form is 2.4x faster than the batched one it replaced.
# Batching traded a well-defined answer for 17% on one stage, and gave that
# back the moment the stage was parallelized.
_GENE_BATCH_WIDTH = 1

# Retained as the memory bound for the per-gene path and for callers of
# `fit_all_genes` that still batch deliberately. Not the caller's
# `chunk_memory_gb`: a memory knob must not be a numerical one.
# `chunk_memory_gb`. Batching the gene GLM across a chunk means the chunk's
# composition can perturb a fit -- sporadically, only for genes with
# near-zero expression, and bounded at ~2e-8 with the deviance unchanged, but
# it happens. Driving that from a user-tunable memory budget made a memory
# knob into a numerical one: on day0, raising the budget from 1 GB to 4 GB
# moved 2 of 237 gene fits, which was enough to flip a discrete p-value
# comparison. Pinning it here means the budget cannot change a result, and
# the gene batching depends only on the dataset's cell count.
#
# The value matches the previous default, so results are unchanged for anyone
# who was already on it.
_GENE_CHUNK_MEMORY_GB = 1.0

# Dense (n_cells,)-sized arrays one IRLS column costs, measured rather than
# counted from the source. Counting the obvious ones -- the responses plus mu,
# weights and working response -- gives 4, which under-predicts peak RSS
# badly: the IRLS loop also holds eta and per-iteration temporaries, and
# `fit_all_genes` additionally churns a precomputation per gene.
#
# Measured at 131,055 cells, p=12, growth attributable to
# `fit_all_genes`:
#
#   budget   chunk   predicted at 4   actual   implied arrays/column
#    0.2 GB     47          0.20 GB   1.23 GB                   25.0
#    0.5 GB    119          0.50 GB   2.38 GB                   19.1
#    1.0 GB    238          1.00 GB   2.59 GB                   10.4
#
# 10 is the asymptotic figure at useful chunk sizes. Note the actual cost also
# carries a roughly 1 GB floor from the per-gene precomputation loop, which
# runs once per gene regardless of chunking and so cannot be reduced by
# shrinking the chunk -- which is why the implied factor rises at small
# budgets. The budget therefore bounds the part that scales with the chunk,
# not total RSS. See ROADMAP T1.4.
_IRLS_ARRAYS_PER_COLUMN = 10
_BYTES_PER_FLOAT = 8


# theta estimation methods reported by glm/nb_theta.py::estimate_theta.
# Anything other than MLE means the MLE failed and a fallback was used.
_THETA_METHOD_NAMES = {1: "MLE", 2: "method of moments", 3: "pilot estimate"}
_THETA_BOUNDS = (0.01, 1000.0)


@dataclass
class GenePrecomputation:
    """What is retained per gene for the whole run.

    Deliberately just the fitted coefficients and theta -- the same two things
    upstream sceptre's `perform_response_precomputation` returns. `mu`, `w`,
    `a` and `D` are all recoverable from these plus the response vector, and
    recomputing them costs less than reading them back from anywhere:
    measured at 131k cells and p=6, `compute_precomputation_pieces` takes
    1.59 ms, while memory-mapping `D` alone off local disk takes 1.04 ms. So
    there is nothing to be gained by spilling them to parquet, zarr or a
    memmap -- recomputation wins outright, and holding them cost 10.5 MB per
    gene (2.56 GB for a 244-gene analysis, 405 GB genome-wide).
    """

    fitted_coefs: np.ndarray
    theta: float
    # Diagnostics, so a run can tell when a fit was degenerate rather than
    # silently returning a statistic built on it.
    theta_method: int = 1
    theta_clamped: bool = False
    glm_converged: bool = True
    # Conditioning of Zt_wZ, kept from the fit so degeneracy can still be
    # reported without retaining D itself.
    min_eigenvalue: float = float("nan")
    max_eigenvalue: float = float("nan")
    n_covariates: int = 0


@dataclass
class TargetPrecomputation:
    trt_idxs: np.ndarray  # 0-based
    # `None` under permutations, where no logistic fit is performed: the
    # probabilities exist only to draw from, and permutation draws do not
    # come from them. Kept on the CRT path because the sampler needs them.
    fitted_probabilities: np.ndarray | None
    # The target's B1+B2+B3 resamples, as `(B, n_cells)` 0/1 CSR rows built
    # one stage at a time. Held per target rather than per pair because every
    # gene paired with this target needs exactly the same flattening, which
    # used to be redone once per *pair*: 2,451 rebuilds of a target-fixed
    # structure in one profiled run.
    draws: StagedDraws


# 97.5th percentile of the standard normal, for two-sided 95% intervals.
_CI_Z = 1.959963984540054


def _as_pct(fold_change: float) -> float:
    """Fold change (1 = no effect) as a percent change from baseline."""
    return (fold_change - 1.0) * 100.0


def _get_row(response_matrix, i: int) -> np.ndarray:
    """Returns gene row i as a dense 1D float array, for either a dense
    ndarray (n_genes, n_cells) or a scipy.sparse matrix in CSR-like format."""
    if hasattr(response_matrix, "toarray"):
        return np.asarray(response_matrix[i].toarray()).ravel().astype(float)
    return np.asarray(response_matrix[i], dtype=float)


def _format_bytes(n: float) -> str:
    """Human-readable size, so a warning does not report "0.000 GB" for a
    small dataset or an unreadable digit count for a large one."""
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n:.0f} B"


def irls_bytes_per_column(n_cells: int) -> int:
    """Dense bytes one IRLS column costs: the response plus mu, weights and
    working response, all `(n_cells,)` float64.

    Densification is unavoidable here -- IRLS is defined on the dense working
    response -- so the only lever is how many columns share a chunk.
    """
    return _IRLS_ARRAYS_PER_COLUMN * n_cells * _BYTES_PER_FLOAT


def chunk_size_for_budget(bytes_per_item: float, n_items: int, chunk_memory_gb: float) -> int:
    """Largest chunk whose working arrays fit `chunk_memory_gb`, at least 1.

    Every per-chunk cost in this module is linear in the chunk size, so one
    budget and one division covers all of them -- there is no need for
    separate knobs per stage.
    """
    if bytes_per_item <= 0:
        return max(1, n_items)
    return max(1, min(n_items, int(chunk_memory_gb * 1e9 // bytes_per_item)))


def gene_chunk_size_for_budget(n_cells: int, n_genes: int, chunk_memory_gb: float) -> int:
    """How many genes to densify and fit at once.

    Densifying every gene at once is what made a genome-wide run
    unaffordable: 38,606 genes over 131k cells is a 40 GB response array and
    ~162 GB peak.

    Chunking does not change which genes get which fit: each column's normal
    equations are solved independently. Agreement is to floating-point
    tolerance rather than bit-for-bit, because the solve is one batched
    `np.linalg.solve` whose blocking depends on the batch width -- measured
    spread ~1e-15, which cannot move a rank-based p-value.
    """
    return chunk_size_for_budget(irls_bytes_per_column(n_cells), n_genes, chunk_memory_gb)


_GENE_FIT_STATE: dict = {}


def _gene_fit_job(job: tuple[int, list[str], list[int]]) -> dict[str, GenePrecomputation]:
    """Fit one slab of genes. With `_GENE_BATCH_WIDTH = 1` that is one gene.

    Independent by construction -- a gene's fit uses only its own counts and
    the shared covariate matrix -- which is precisely what makes fitting them
    individually parallelizable, and is the other half of that decision. Only
    the gene ids and their row indices cross a process boundary; the matrix
    and covariates are inherited.
    """
    _, chunk_ids, chunk_rows = job
    st = _GENE_FIT_STATE
    response_matrix = st["response_matrix"]
    covariate_matrix = st["covariate_matrix"]
    dfr = st["dfr"]
    xo = st["x_outer_flat"]
    n_cells = covariate_matrix.shape[0]

    contiguous = chunk_rows == list(range(chunk_rows[0], chunk_rows[0] + len(chunk_rows)))
    if contiguous and hasattr(response_matrix, "rows"):
        # Backed input over a contiguous gene range: one contiguous slice of
        # the stored CSC buffers, so a single read instead of k.
        block = response_matrix.rows(chunk_rows[0], chunk_rows[0] + len(chunk_rows))
        Y = np.asarray(block.toarray(), dtype=float)
    else:
        Y = np.empty((len(chunk_ids), n_cells))
        for j, row in enumerate(chunk_rows):
            Y[j] = _get_row(response_matrix, row)

    fit = fit_poisson_glm_batch(covariate_matrix, Y, X_outer_flat=xo)

    out: dict[str, GenePrecomputation] = {}
    for j, gene_id in enumerate(chunk_ids):
        y_j = Y[j]
        theta_est, method = estimate_theta(y=y_j, mu=fit.fitted_values[j], dfr=dfr)
        lo, hi = _THETA_BOUNDS
        theta = max(min(theta_est, hi), lo)
        # Built here only to read off the conditioning diagnostics, then
        # dropped -- see GenePrecomputation. It is rebuilt per gene per target
        # chunk in run_discovery_ntcells_complement.
        pieces = compute_precomputation_pieces(y_j, covariate_matrix, fit.coefs[j], theta)
        out[gene_id] = GenePrecomputation(
            fitted_coefs=fit.coefs[j],
            theta=theta,
            theta_method=method,
            theta_clamped=not (lo <= theta_est <= hi),
            glm_converged=bool(fit.converged[j]),
            min_eigenvalue=pieces.min_eigenvalue,
            max_eigenvalue=pieces.max_eigenvalue,
            n_covariates=pieces.D.shape[0],
        )
    return out


def fit_all_genes(
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    *,
    chunk_memory_gb: float = _DEFAULT_CHUNK_MEMORY_GB,
    gene_rows: list[int] | None = None,
    batch_width: int | None = None,
    n_jobs: int = 1,
    x_outer_flat_shared: np.ndarray | None = None,
) -> dict[str, GenePrecomputation]:
    """Batched Poisson IRLS across genes (they share the full covariate
    matrix), then per-gene theta estimation and precomputation pieces.

    Genes are fit in chunks that fit `chunk_memory_gb` rather than all at once;
    see `gene_chunk_size_for_budget`. Only the fit is transient -- the
    returned precomputations are just coefficients and theta (80 bytes per
    gene), so nothing large is retained.

    `n_jobs` distributes the fits. They are independent -- a gene's fit uses
    only its own counts and the shared covariate matrix -- which is what
    fitting them individually buys, so this is free parallelism and cannot
    change a result.

    `batch_width` pins how many genes share one BLAS call, overriding the
    budget. The pipeline passes 1, so each fit depends only on its own gene --
    see `_GENE_BATCH_WIDTH` for why. Larger values are faster by a few percent
    and make a fit depend on its batch companions.

    `gene_rows` gives each gene's row in `response_matrix`, for when `gene_ids`
    is a *subset* of the matrix's rows rather than all of them in order. The
    fits are per-column independent, so fitting a subset gives the same result
    for the genes fit -- which is why the caller can safely skip genes no pair
    mentions.

    "The same" to floating-point precision, not bit-for-bit: a different
    subset means a different batch width, so `(k, n) @ (n, p*p)` is a
    differently-shaped matmul that BLAS may block and accumulate differently.
    Measured exactly 0.0 on macOS/Accelerate and nonzero in the last bits on
    Linux/OpenBLAS. Don't assert bitwise equality across batch widths.
    """
    n_cells = covariate_matrix.shape[0]
    dfr = covariate_matrix.shape[0] - covariate_matrix.shape[1]
    chunk = (
        max(1, int(batch_width))
        if batch_width is not None
        else gene_chunk_size_for_budget(n_cells, len(gene_ids), chunk_memory_gb)
    )
    rows = list(range(len(gene_ids))) if gene_rows is None else list(gene_rows)
    if len(rows) != len(gene_ids):
        raise ValueError(f"gene_rows has {len(rows)} entries for {len(gene_ids)} gene_ids")

    jobs = [
        (start, gene_ids[start : start + chunk], rows[start : start + chunk])
        for start in range(0, len(gene_ids), chunk)
    ]
    _GENE_FIT_STATE.update(
        response_matrix=response_matrix,
        covariate_matrix=covariate_matrix,
        dfr=dfr,
        # Built once by the caller and shared by every fit; see
        # `glm.irls.x_outer_flat`. Inherited by forked workers, shared by
        # threads, so it crosses no boundary either way.
        x_outer_flat=x_outer_flat_shared,
    )
    out: dict[str, GenePrecomputation] = {}
    try:
        for produced in _map_jobs(_gene_fit_job, jobs, n_jobs):
            out.update(produced)
    finally:
        _GENE_FIT_STATE.clear()

    _warn_about_degenerate_gene_fits(out)
    return out


def _design_is_rank_deficient(pieces) -> bool:
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
    # Accepts anything carrying min_eigenvalue/max_eigenvalue plus a covariate
    # count: a PrecomputationPieces (which exposes p as D.shape[0]) or a
    # GenePrecomputation (which keeps n_covariates so D need not be retained).
    p = getattr(pieces, "n_covariates", 0) or pieces.D.shape[0]
    # bool(), not the numpy scalar the comparison yields -- callers should get
    # a plain Python bool.
    return bool(min_eig <= max_eig * p * np.finfo(float).eps)


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
        if _design_is_rank_deficient(gp):
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


def target_seed_sequence(seed, target_id: str) -> np.random.SeedSequence:
    """A stream keyed by the target's *name*, not its position.

    Every target draws from its own independent substream, derived from
    `(seed, target_id)`. This is what makes a result reproducible in the sense
    that matters in practice: **adding or removing targets or pairs does not
    move any other target's numbers.**

    The alternative, spawning by position, is enough to make results
    independent of worker count and chunk size but not of composition -- insert
    one target at the front and every later target draws different numbers.
    Keying on identity removes that.

    `hashlib` rather than `hash()`: Python salts string hashing per process
    unless PYTHONHASHSEED is fixed, so `hash("target_1")` differs between runs
    and would make the seeding irreproducible in exactly the situation this
    function exists to prevent.
    """
    key = int.from_bytes(
        hashlib.blake2b(str(target_id).encode("utf-8"), digest_size=8).digest(), "big"
    )
    # `seed` is expected to be already-resolved entropy (see `resolve_entropy`),
    # so that `seed=None` reads OS entropy once per run rather than once per
    # target -- which would be thousands of reads, and would muddle "one
    # unseeded run" with "one unseeded target".
    return np.random.SeedSequence(entropy=seed, spawn_key=(key,))


def resolve_entropy(seed) -> int:
    """Turn `seed` into concrete entropy, once per run.

    `seed=None` means "pick something"; picking it here rather than per target
    keeps an unseeded run internally coherent and makes the chosen value a
    single thing that could be reported or logged.
    """
    return np.random.SeedSequence(seed).entropy


# Keyed by a per-call token, not a single shared slot. `fit_all_targets` runs
# concurrently now -- `_PREFETCH_DEPTH` chunks are prepared at once -- and a
# single slot meant one call's `finally: clear()` deleted another's inputs
# mid-flight, which surfaced as `KeyError` on a target id. The dict still
# lives at module scope so a forked child inherits it without copying.
_TARGET_STATE: dict[int, dict] = {}
_TARGET_STATE_SEQ = itertools.count()


def _target_draw_job(job: tuple[int, int, str]) -> tuple[str, TargetPrecomputation]:
    """One target's resamples.

    Independent of every other target: the CRT seeds from the target's own
    name, and permutations read a prefix of one shared array.
    """
    token, k, target_id = job
    st = _TARGET_STATE[token]
    fitted_values = st["fitted_values"]
    perms = st["permutations"]
    fitted_probabilities = None if fitted_values is None else fitted_values[k]
    trt_idxs = st["grna_target_cells"][target_id]
    if perms is None:
        rng = np.random.default_rng(target_seed_sequence(st["seed"], target_id))
        synthetic_idxs = crt_index_sampler_fast(fitted_probabilities, st["B_total"], rng)
        draws = ListDraws(synthetic_idxs, st["n_cells"])
    else:
        # Permutations: every target reads the same draws, taking a prefix
        # the size of its own treated set. Held by reference rather than
        # copied -- a target costs nothing until a stage is actually reached,
        # which is the point of sharing the draws and was being thrown away
        # by materializing them per target. See crt/permutations.py.
        draws = PermutationSliceDraws(perms, len(trt_idxs), st["n_cells"])
    return target_id, TargetPrecomputation(
        trt_idxs=trt_idxs,
        fitted_probabilities=fitted_probabilities,
        draws=draws,
    )


def fit_all_targets(
    grna_target_cells: dict[str, np.ndarray],
    covariate_matrix: np.ndarray,
    *,
    B1: int,
    B2: int,
    B3: int,
    seed,
    n_jobs: int = 1,
    permutations: np.ndarray | None = None,
    x_outer_flat_shared: np.ndarray | None = None,
) -> dict[str, TargetPrecomputation]:
    """Per-target resamples, plus the logistic fit the CRT draws them from.

    On the CRT path that is one batched binomial IRLS call across all
    targets (indicator columns share the full covariate matrix), then a draw
    per target. On the permutation path the fit is skipped -- nothing reads
    it there -- and each target takes a prefix of the shared draws.

    Each target's draw comes from its own stream, keyed by its name rather
    than drawn from one shared generator in sequence -- see
    `target_seed_sequence`. That makes the draws independent of the order
    targets are processed in and of which other targets are present, and is
    what lets them be computed in parallel at all.
    """
    n_cells = covariate_matrix.shape[0]
    target_ids = list(grna_target_cells.keys())

    # **Skipped entirely under permutations, matching R.** The logistic fit
    # exists to give `crt_index_sampler_fast` the per-cell probabilities to
    # draw from; permutation draws are uniform subsets and never consult it,
    # and `fitted_probabilities` has no other reader. R makes the same
    # split -- `perform_grna_precomputation` is called only from
    # `crt_glm_factored_out` and `discovery_ntcells_crt`, never from
    # `perm_test_glm_factored_out` -- and doing it anyway was the largest
    # single term in a permutation profile at **36% of runtime**, for a
    # result that was then thrown away.
    #
    # It is also the expensive shape: the responses are indicator columns,
    # ~0.1% dense, held as float64 because the IRLS needs them dense, so the
    # discarded work is a dense fit over all 3,026 tested targets x 567,690
    # cells, taken a chunk at a time.
    fitted_values = None
    if permutations is None:
        # Target-major, matching the gene fit and glm/irls.py's convention.
        Y = np.zeros((len(target_ids), n_cells))
        for k, target_id in enumerate(target_ids):
            Y[k, grna_target_cells[target_id]] = 1.0
        fitted_values = fit_binomial_glm_batch(
            covariate_matrix, Y, X_outer_flat=x_outer_flat_shared
        ).fitted_values

    # The draws parallelize now that each target seeds from its own name --
    # under a shared generator the order of consumption was the answer, so
    # this could not have been done before that change.
    #
    # **Threads, not processes, and against the platform default.** A target's
    # draw matrix is large -- 2.7M nonzeros, ~33 MB, at day0 scale -- so
    # returning it across a process boundary would cost more than the draw.
    # Threads pay nothing to return it, and numpy releases the GIL for the
    # bulk `binomial` and `integers` calls that dominate here, which the
    # gather in the per-pair statistic does not. Measured on 16 day0 targets:
    # 1.98x at 2 threads, 2.90x at 4, and 2.40x at 8 -- the turnover is
    # allocation and bandwidth contention on those 33 MB buffers.
    token = next(_TARGET_STATE_SEQ)
    _TARGET_STATE[token] = dict(
        permutations=permutations,
        fitted_values=fitted_values,
        grna_target_cells=grna_target_cells,
        seed=seed,
        B_total=B1 + B2 + B3,
        n_cells=n_cells,
    )
    out: dict[str, TargetPrecomputation] = {}
    try:
        for target_id, precomp in _map_jobs(
            _target_draw_job,
            [(token, k, t) for k, t in enumerate(target_ids)],
            n_jobs,
            backend="thread",
        ):
            out[target_id] = precomp
    finally:
        _TARGET_STATE.pop(token, None)
    return out


_DEFAULT_TARGET_CHUNK_SIZE = 200

# How many chunks of target state to prepare ahead of the one being tested.
#
# The logistic fit does not parallelize *within* a batch: splitting one
# chunk's targets across threads caps at 1.59x however many it gets, because
# the work is bandwidth-bound and the per-iteration Python overhead holds the
# GIL. Different chunks' fits are independent, though, and each stays a full
# efficient batched call, so running several concurrently does scale --
# measured on eight chunk-fits: 1.78x at two, 2.24x at three, 2.71x at four.
#
# Three recovers most of that while holding three chunks of draws instead of
# one. Depth 1 is the previous behaviour, and 0 disables prefetching.
_PREFETCH_DEPTH = 3

# Above this many chunks, the per-chunk gene pool uses threads even where the
# platform default is processes.
#
# The cost being traded is a worker pool built **per chunk** against the GIL.
# Permutations run as a single chunk and pay the fork once, so processes win
# there and win big. The CRT at the default budget has 217 chunks, so fork
# cost dominates and threads win -- and the statistic is a sparse matmul now,
# which releases the GIL where the old gather did not.
#
# Placed between two measured points, not fitted. See docs/design.md,
# "Choosing a parallel backend".
_THREAD_ABOVE_N_CHUNKS = 8

_BYTES_PER_INDEX = 8  # int64 cell index


def target_bytes_per_item(
    n_cells: int, B_total: int, n_trt_values, *, include_fit: bool = True
) -> float:
    """Bytes one target costs a chunk: its share of the dense binomial fit
    plus the resamples it holds.

    The fit term is the wasteful one -- the responses are indicators with
    roughly `n_trt` ones per column (396 of 586,309 in the benchmark,
    0.07% dense) yet held as float64 because the IRLS needs them dense.
    Measured unbounded peak was 13.1 GB at `target_chunk_size=200` over 586k
    cells. The draw term is what explodes under `no_approximation`, where
    `B_total` reaches 1.65M and a single target needs 5.2 GB.

    `include_fit=False` for permutations, which never build those arrays at
    all. Charging for absent memory is not merely conservative: on day0 the
    fit term is 45.4 MB per target over 567,690 cells against 135.9 MB of
    draws, so dropping it took the chunk from 5 to 7 at
    `chunk_memory_gb=1.0`. Since each gene's precomputation pieces are
    rebuilt once per chunk, a wider chunk is fewer rebuilds rather than
    merely spare headroom.
    """
    fit = irls_bytes_per_column(n_cells) if include_fit else 0.0
    return fit + _draw_bytes_per_target(B_total, n_trt_values)


def _draw_bytes_per_target(B_total: int, n_trt_values) -> float:
    """Uses the *median* treated-cell count, matching how
    `crt_index_sampler_fast` draws: expected inclusions per draw equal the
    observed treated-cell count, by the intercept property of the logistic
    MLE (see crt/sampler.py).
    """
    if B_total <= 0 or len(n_trt_values) == 0:
        return 0.0
    median_n_trt = float(np.median(np.asarray(n_trt_values, dtype=float)))
    return B_total * median_n_trt * _BYTES_PER_INDEX


def estimate_draw_memory_bytes(B_total: int, n_trt_values, chunk_size: int) -> float:
    """Bytes of resamples held at once, for one chunk of targets."""
    return _draw_bytes_per_target(B_total, n_trt_values) * chunk_size


def target_chunk_size_for_budget(
    n_cells: int,
    B_total: int,
    n_trt_values,
    n_targets: int,
    chunk_memory_gb: float,
    *,
    include_fit: bool = True,
) -> int:
    """Largest target chunk whose fit arrays *and* resamples fit the budget."""
    return chunk_size_for_budget(
        target_bytes_per_item(n_cells, B_total, n_trt_values, include_fit=include_fit),
        n_targets,
        chunk_memory_gb,
    )


def _resolve_target_chunk_size(
    n_cells: int,
    B_total: int,
    n_trt_values,
    n_targets: int,
    chunk_size: int,
    chunk_memory_gb: float,
    *,
    include_fit: bool = True,
) -> int:
    """Clamp the requested chunk size to the memory budget, warning if it moves.

    `chunk_size` is an upper bound, not a mandate: whatever a caller asks for,
    the working arrays stay under `chunk_memory_gb`. Chunk size affects only
    peak memory and batching width, not which draws are taken --
    `fit_all_targets` draws per target in a fixed order, so the RNG stream is
    unchanged by where boundaries fall (`test_memory_guard.py` pins this).

    The batched logistic solve is not bit-for-bit across batch widths
    (~1e-15), so fitted probabilities can differ in their last bits. That is
    far too small to change an integer binomial draw count in practice, but
    "identical" means to floating-point tolerance, not exactly.
    """
    fitted = min(
        chunk_size,
        target_chunk_size_for_budget(
            n_cells, B_total, n_trt_values, n_targets, chunk_memory_gb, include_fit=include_fit
        ),
    )
    if fitted >= chunk_size:
        return chunk_size

    per_target = target_bytes_per_item(n_cells, B_total, n_trt_values, include_fit=include_fit)
    draw_part = _draw_bytes_per_target(B_total, n_trt_values)
    fit_clause = (
        f"{_format_bytes(irls_bytes_per_column(n_cells))} of dense "
        f"binomial-fit arrays over {n_cells:,} cells, plus "
        if include_fit
        else "no binomial fit on the permutation path, only "
    )
    message = (
        f"reducing target_chunk_size from {chunk_size} to {fitted} to stay "
        f"within chunk_memory_gb={chunk_memory_gb}: each target needs about "
        f"{_format_bytes(per_target)} "
        f"({fit_clause}{_format_bytes(draw_part)} of resamples for "
        f"B1+B2+B3 = {B_total:,}). Chunk size affects only peak memory and "
        f"batching width, not results."
    )
    if fitted == 1:
        message += (
            " At target_chunk_size=1 each target is processed alone, which is"
            " slow, and peak memory runs several times the figure above while"
            " a draw is being built, so this run may still exhaust memory."
            " This is usually resampling_approximation='no_approximation',"
            " whose B3 grows as n_pairs / multiple_testing_alpha;"
            " 'skew_normal' needs a few MB of draws per target instead."
        )
    warnings.warn(message, stacklevel=3)
    return fitted


# Populated by the orchestrator immediately before a chunk's genes are mapped,
# and read by `_gene_job` in whichever process or thread runs it. A module
# global rather than a closure because a forked child inherits it for free
# (copy-on-write, so the response matrix is not copied) while a closure over
# these arrays would have to be pickled per task.
_WORKER_STATE: dict = {}


def _prefix_nulls(prefix: PermutationPrefixSums, n_trt: int, lo: int, hi: int):
    """Adapter so the per-target closure stays picklable under the process
    backend, where a lambda over the loop variable would not be."""
    return prefix.statistics(lo, hi, n_trt)


def _gene_job(job: tuple[str, list[str]]) -> dict[tuple[str, str], dict]:
    """One gene's pairs against the current chunk's targets.

    Reads its inputs from `_WORKER_STATE` rather than taking them as
    arguments, so that under a process backend nothing large crosses the
    process boundary -- only the gene id and its target list go in, and only
    the finished result rows come back.
    """
    gene_id, targets_here = job
    st = _WORKER_STATE
    gene = st["gene_precomps"][gene_id]
    y = _get_row(st["response_matrix"], st["gene_row_index"][gene_id])
    pieces = compute_precomputation_pieces(y, st["covariate_matrix"], gene.fitted_coefs, gene.theta)
    # a, w and D all need the same per-resample segment sums, so stacking them
    # once per gene lets each of that gene's pairs get all three from a single
    # matmul.
    stacked = stack_pieces(pieces.a, pieces.w, pieces.D)

    # Permutations share their draws across targets, so this gene's segment
    # sums for *every* target are one prefix scan along those shared rows --
    # see `PermutationPrefixSums`. Built once here and read by each target,
    # in place of a matmul per target. The CRT has no shared ordering to scan
    # along, so it keeps the matmul.
    perms = st["permutations"]
    prefix = PermutationPrefixSums(stacked, perms) if perms is not None else None

    out: dict[tuple[str, str], dict] = {}
    for target_id in targets_here:
        target = st["target_precomps"][target_id]
        null_fn = None
        if prefix is not None:
            n_trt = len(target.trt_idxs)
            null_fn = partial(_prefix_nulls, prefix, n_trt)
        result = run_low_level_test_full(
            y=y,
            mu=pieces.mu,
            a=pieces.a,
            w=pieces.w,
            D=pieces.D,
            trt_idxs=target.trt_idxs,
            synthetic_idxs=target.draws,
            stacked=stacked,
            B1=st["B1"],
            B2=st["B2"],
            B3=st["B3"],
            fit_parametric_curve=st["fit_parametric_curve"],
            side_code=st["side_code"],
            null_statistics_fn=null_fn,
        )
        half_width = _CI_Z * result.se_fold_change
        out[(gene_id, target_id)] = {
            "response_id": gene_id,
            "grna_target": target_id,
            "p_value": result.p_value,
            "fold_change": result.fold_change,
            "se_fold_change": result.se_fold_change,
            # The effect size people actually read, with its interval.
            # `log2(fold_change)` is not returned: it is a pure transform of a
            # column already present, so it would be bytes rather than
            # information.
            #
            # `_es` because `DataFrame.pct_change` is a pandas method. Named
            # `pct_change`, attribute access returns the method rather than
            # the column, and arithmetic on it raises a TypeError about
            # 'method' rather than a KeyError -- which cost time twice here
            # before the column was renamed.
            "pct_change_es": _as_pct(result.fold_change),
            "pct_change_es_ci_low": _as_pct(result.fold_change - half_width),
            "pct_change_es_ci_high": _as_pct(result.fold_change + half_width),
            "z_orig": result.z_orig,
            "stage": result.stage,
        }
    return out


def _limit_blas_threads() -> None:
    """One BLAS thread per worker, so W workers do not spawn W x N threads."""
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1, user_api="blas")
    except Exception:  # pragma: no cover - threadpoolctl is a hard dependency
        pass


def resolve_n_jobs(n_jobs: int) -> int:
    """Worker count: `n_jobs <= 0` means every core."""
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return os.cpu_count() or 1
    return n_jobs


def parallel_backend() -> str:
    """The platform default backend for a worker pool: "fork" or "thread".

    `fork` on Linux, `thread` everywhere else. `PYSCEPTRE_BACKEND` overrides
    it to either value; a request for `fork` off Linux is refused with a
    warning, because forking after Apple's Accelerate BLAS can deadlock.

    Callers that build one pool per chunk should ask `gene_job_backend`
    instead, which may override this. See docs/design.md, "Choosing a
    parallel backend".
    """
    override = os.environ.get("PYSCEPTRE_BACKEND", "").strip().lower()
    if override in ("fork", "thread"):
        if override == "fork" and not (sys.platform.startswith("linux") and hasattr(os, "fork")):
            warnings.warn(
                "PYSCEPTRE_BACKEND=fork ignored: fork is only used on Linux, because "
                "forking after Apple's Accelerate BLAS can deadlock.",
                stacklevel=2,
            )
        else:
            return override
    if sys.platform.startswith("linux") and hasattr(os, "fork"):
        return "fork"
    return "thread"


def gene_job_backend(n_chunks: int) -> str | None:
    """Backend for the per-chunk gene pool, or `None` for the platform default.

    Returns "thread" when the platform default is `fork` and the run has more
    than `_THREAD_ABOVE_N_CHUNKS` chunks, since one fork per chunk then costs
    more than the GIL does. `None` everywhere else, including on platforms
    whose default is already threads. See docs/design.md, "Choosing a
    parallel backend".
    """
    if parallel_backend() != "fork":
        return None
    return "thread" if n_chunks > _THREAD_ABOVE_N_CHUNKS else None


def _map_jobs(fn, jobs: list, n_jobs: int, backend: str | None = None):
    """Run `fn` over `jobs`, sequentially or in parallel.

    `fn` must be a module-level function that reads its bulk inputs from a
    module global, so that under the process backend only the small job
    descriptor crosses the boundary.

    `backend` overrides the platform default, for stages whose shape makes
    the other choice wrong -- see `fit_all_targets`, where the results are far
    too large to send between processes.
    """
    workers = min(resolve_n_jobs(n_jobs), len(jobs))
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            yield fn(job)
        return

    backend = backend or parallel_backend()
    if backend == "fork":
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("fork")
        # The pool is created per chunk, after that chunk's draws exist, so a
        # forked child inherits them without copying. On Linux a fork is tens
        # of milliseconds against a chunk that takes seconds.
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=ctx, initializer=_limit_blas_threads
        ) as ex:
            yield from ex.map(fn, jobs)
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            yield from ex.map(fn, jobs)


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
    chunk_memory_gb: float = _DEFAULT_CHUNK_MEMORY_GB,
    n_jobs: int = 1,
    resampling_mechanism: str = "crt",
) -> pd.DataFrame:
    """pairs: DataFrame with columns 'response_id', 'grna_target' -- the
    QC-passed pairs to test. Returns a DataFrame with one row per pair:
    response_id, grna_target, p_value, fold_change, se_fold_change, pct_change_es,
    pct_change_es_ci_low, pct_change_es_ci_high, z_orig, stage.

    Targets are fit and CRT-drawn in chunks of `target_chunk_size` rather than
    all at once: each target's B1+B2+B3 synthetic index sets are individually
    small, but holding *every* target's draws in memory simultaneously does
    not scale -- at real dataset sizes (~3,000 targets x ~5,500 resamples)
    this was measured to OOM-kill the process. Chunking keeps peak memory
    bounded to O(chunk_size) while still batching the (bulk of the) per-target
    logistic fit and CRT draw across many targets at once for speed, not
    falling back to a slow one-target-at-a-time loop.

    `target_chunk_size` is an upper bound, not a mandate: it is reduced
    automatically so the working arrays stay under `chunk_memory_gb`, so no
    chunk size a caller passes can blow up memory. Both per-chunk costs are
    linear in the chunk size -- the dense binomial-fit arrays and the CRT
    draws -- which is why one budget covers both. That matters most for
    `resampling_approximation="no_approximation"`, whose B3 grows as
    `n_pairs / multiple_testing_alpha` and reaches 1.65M draws (5.2 GB) per
    target at real dataset scale.
    """
    # No single shared generator: each target seeds its own from its name, so
    # results do not depend on target order or on which other targets are in
    # the run. See `target_seed_sequence`. Entropy is resolved once here so an
    # unseeded run still draws it a single time.
    entropy = resolve_entropy(seed)

    # Fit only the genes some pair mentions. `gene_ids` labels every row of
    # `response_matrix`, which for an all-genes dataset is far more than the
    # analysis touches: on a transcriptome-wide screen a calibration check
    # tested 9,045 of 38,606
    # genes, so fitting all of them is 4.3x the necessary work. The fits are
    # per-column independent, so this changes no result. It was invisible while
    # the export carried only the genes under test -- the export was doing the
    # filtering, and widening it exposed the omission.
    tested_genes = set(pairs["response_id"])
    needed_gene_ids = [g for g in gene_ids if g in tested_genes]
    needed_gene_rows = [i for i, g in enumerate(gene_ids) if g in tested_genes]
    # One per run. Every fit in the analysis -- 237 gene fits and one per
    # target chunk, ~454 calls on day0 -- reads the same design matrix, and
    # this is the only part of the IRLS setup that depends on nothing else.
    # See `glm.irls.x_outer_flat`.
    shared_x_outer = x_outer_flat(covariate_matrix)

    gene_precomps = fit_all_genes(
        response_matrix,
        needed_gene_ids,
        covariate_matrix,
        x_outer_flat_shared=shared_x_outer,
        # One gene per BLAS call, so a fit depends on nothing but that gene.
        # See `_GENE_BATCH_WIDTH`. The caller's `chunk_memory_gb` governs
        # target chunking only, which is result-neutral.
        chunk_memory_gb=_GENE_CHUNK_MEMORY_GB,
        gene_rows=needed_gene_rows,
        batch_width=_GENE_BATCH_WIDTH,
        n_jobs=n_jobs,
    )

    pairs_by_target = {target_id: group for target_id, group in pairs.groupby("grna_target")}
    target_ids_needed = [t for t in grna_target_cells if t in pairs_by_target]

    # Permutation draws are generated **once for the whole analysis**, not
    # per chunk, matching R and keeping results independent of the chunking.
    #
    # M is the largest cell count over **every** supplied target, not only the
    # tested ones, which is R's rule:
    #
    #     max(vapply(grna_assignments$grna_group_idxs, length, integer(1)))
    #
    # It matters for more than parity. Taking the maximum over the whole
    # target set makes M independent of the *pair list*, so adding pairs for
    # targets already present cannot move a result. Only adding a larger
    # target to the dataset can.
    permutations = None
    if resampling_mechanism == "permutations":
        m = max(len(v) for v in grna_target_cells.values())
        permutations = permutation_draws(
            covariate_matrix.shape[0],
            m,
            B1 + B2 + B3,
            np.random.default_rng(entropy),
        )

    # **Permutations are not chunked at all**, which makes the loop below
    # gene-major in effect and matches R's structure: `run_perm_test_in_memory`
    # is gene-outer and builds each gene's precomputation pieces once, where
    # chunking rebuilds them once per gene per chunk -- 237 against ~5,600 on
    # day0.
    #
    # Chunking exists to bound what a live target costs, and for permutations
    # that is now nothing. There is no logistic fit (`fit_all_targets`), the
    # draws are one shared array held by reference
    # (`PermutationSliceDraws`), stage 1 is served from a per-gene scan
    # without materializing anything per target (`PermutationPrefixSums`),
    # and the escalation stages are rebuilt rather than cached. A target
    # costs its `trt_idxs` and two pointers.
    #
    # The CRT is chunked as before: its draws are genuinely per-target and
    # large, which is what the budget is for.
    if permutations is None:
        target_chunk_size = _resolve_target_chunk_size(
            covariate_matrix.shape[0],
            B1 + B2 + B3,
            [len(grna_target_cells[t]) for t in target_ids_needed],
            len(target_ids_needed),
            target_chunk_size,
            chunk_memory_gb,
        )
    elif target_chunk_size == _DEFAULT_TARGET_CHUNK_SIZE:
        target_chunk_size = max(1, len(target_ids_needed))
    # An explicitly requested chunk size is still honoured on this path, even
    # though nothing here needs bounding. It costs only speed, results are
    # invariant to it either way, and silently ignoring a caller's memory
    # knob is worse than being slower than necessary -- it is also what
    # `test_permutations_do_not_depend_on_chunking_or_workers` exercises, and
    # a knob that no longer moves anything makes that test vacuous.

    gene_row_index = {gene_id: i for i, gene_id in enumerate(gene_ids)}
    pairs_by_gene: dict[str, list[str]] = {}
    for gene_id, group in pairs.groupby("response_id"):
        pairs_by_gene[str(gene_id)] = list(group["grna_target"])

    # **A chunk's targets are prepared while the previous chunk's genes are
    # still being tested.** `fit_all_targets` runs the batched logistic fit in
    # this process, not in the worker pool, so with it inline every chunk
    # boundary is a stretch where one core works and the rest wait. Amdahl on
    # the measured CRT scaling puts that at **46% of runtime serial** -- 8
    # workers returned 1.88x -- and it is the same fit whose removal took the
    # permutation path's serial fraction to 14%.
    #
    # Prefetching hides it instead of dividing it, and that choice is about
    # results rather than speed. Splitting the batch across workers would make
    # the sub-batch width a function of `n_jobs`, and the batched solve is not
    # bit-for-bit across widths: measured identical when 40 rows are split at
    # width 20 or 10, but 8.5e-16 at width 5 and 2.6e-15 at width 1, the
    # stability above 10 being a BLAS blocking artefact rather than a promise.
    # At a chunk of 14 an 8-way split is width 2, so every p-value would
    # depend on the worker count. Prefetching leaves the batch exactly as it
    # was -- same rows, same order, same arithmetic -- so results are
    # bit-identical and `n_jobs` still cannot move them.
    #
    # Draws are unaffected for the same reason they parallelize at all: each
    # target seeds from its own name (`target_seed_sequence`), so preparing a
    # chunk earlier cannot change what it draws.
    #
    # The cost is one extra chunk of target state alive at a time, so peak
    # memory carries two chunks rather than one.
    prep_draw_jobs = max(1, resolve_n_jobs(n_jobs) // max(1, _PREFETCH_DEPTH))

    def _prepare(ids: list[str]) -> dict[str, TargetPrecomputation]:
        return fit_all_targets(
            {t: grna_target_cells[t] for t in ids},
            covariate_matrix,
            B1=B1,
            B2=B2,
            B3=B3,
            seed=entropy,
            # Divided by the pipeline depth: `_PREFETCH_DEPTH` prepares run
            # at once, so a full `n_jobs` each would ask for that multiple of
            # the machine. The draws inside a prepare are the only part that
            # maps, and they are a small share of it.
            n_jobs=prep_draw_jobs,
            permutations=permutations,
            x_outer_flat_shared=shared_x_outer,
        )

    starts = list(range(0, len(target_ids_needed), target_chunk_size))
    chunks = [target_ids_needed[i : i + target_chunk_size] for i in starts]

    rows: dict[tuple[str, str], dict] = {}
    # **Only when the caller asked for parallelism.** The prefetch runs on a
    # thread of its own, so enabling it at `n_jobs=1` would quietly make a
    # single-worker run use two cores -- measured 831.0 s against 579.1 s on
    # day0, a 1.43x "speedup" that is just the second thread. `n_jobs` has to
    # mean what it says: a single-core benchmark, a cgroup-limited container
    # and every matched-core comparison against R depend on it.
    # Chosen once, from the chunk count: a run that builds a pool per chunk
    # pays fork repeatedly, one that builds a single pool does not.
    gene_backend = gene_job_backend(len(chunks))
    # PYSCEPTRE_PREFETCH_DEPTH overrides the default, for tuning on a
    # machine whose balance differs.
    _d = os.environ.get("PYSCEPTRE_PREFETCH_DEPTH", "")
    want_depth = int(_d) if _d.isdigit() and int(_d) > 0 else _PREFETCH_DEPTH
    depth = min(want_depth, len(chunks) - 1) if resolve_n_jobs(n_jobs) > 1 else 0
    prefetch = ThreadPoolExecutor(max_workers=depth) if depth > 0 else None
    pending: deque = deque()
    try:
        # Prime the pipeline so `depth` fits are in flight before the first
        # chunk's genes are tested, rather than one.
        for i in range(depth):
            pending.append(prefetch.submit(_prepare, chunks[i]))

        for chunk_i, chunk_ids in enumerate(chunks):
            target_precomps = pending.popleft().result() if pending else _prepare(chunk_ids)
            ahead = chunk_i + depth
            if prefetch is not None and ahead < len(chunks):
                pending.append(prefetch.submit(_prepare, chunks[ahead]))

            # Gene-outer inside the chunk so each gene's pieces are rebuilt once
            # per chunk rather than once per pair: 244 genes x ~15 chunks is ~3,660
            # rebuilds (~0.1 min) against 33,066 rebuilds (~0.9 min), while
            # holding one gene's pieces (10.5 MB) instead of every gene's
            # (2.56 GB).
            #
            # **That comparison is against R's CRT, and only its CRT.**
            # `crt_glm_factored_out` is target-outer and rebuilds a gene's pieces
            # inside its gene loop, so it pays one per pair -- this layout beats
            # it. R's *permutation* workhorse is the other way up:
            # `run_perm_test_in_memory` is gene-outer and
            # `perm_test_glm_factored_out` loops targets inside, so R pays one
            # rebuild per gene, 237 on day0, against roughly 5,600 here. R can
            # invert the loop because permutations have no per-target state to
            # hold; the CRT does, which is what forces target chunking, and this
            # path uses one layout for both mechanisms. Fixing it means not
            # materializing per-target draws for permutations at all.
            chunk_target_set = set(chunk_ids)
            gene_jobs = [
                (gene_id, [t for t in gene_pairs if t in chunk_target_set])
                for gene_id, gene_pairs in pairs_by_gene.items()
            ]
            # Longest first. A gene's job costs roughly one piece rebuild plus one
            # test per pair, so pair count is a good proxy for duration, and they
            # vary a lot -- day0 has a median of 154 targets per gene and a max of
            # 299. A pool pulling tasks in arrival order would hand a worker
            # several heavy genes at the end of a chunk and leave the rest idle.
            #
            # Sorting descending is enough; no explicit bin packing is needed,
            # because the executor already pulls dynamically. That makes this
            # Longest-Processing-Time-first scheduling, which is within 4/3 of
            # optimal makespan. Ordering only affects scheduling: each gene job is
            # independent and results are collected into a dict, so this cannot
            # change a number.
            gene_jobs = sorted((job for job in gene_jobs if job[1]), key=lambda j: -len(j[1]))

            # Genes within a chunk are independent -- each rebuilds its own pieces
            # from this chunk's already-drawn synthetic index sets -- so this is
            # where the work parallelizes. Deliberately NOT over chunks: the RNG is
            # consumed target-by-target in `fit_all_targets`, so running chunks
            # concurrently would change every p-value, and chunk-parallelism would
            # multiply `chunk_memory_gb` by the worker count instead of sharing one
            # chunk's draws.
            _WORKER_STATE.update(
                response_matrix=response_matrix,
                covariate_matrix=covariate_matrix,
                gene_precomps=gene_precomps,
                permutations=permutations,
                gene_row_index=gene_row_index,
                target_precomps=target_precomps,
                B1=B1,
                B2=B2,
                B3=B3,
                fit_parametric_curve=fit_parametric_curve,
                side_code=side_code,
            )
            for produced in _map_jobs(_gene_job, gene_jobs, n_jobs, backend=gene_backend):
                rows.update(produced)
            _WORKER_STATE.clear()

            del target_precomps  # free this chunk's synthetic_idxs before the next one

    finally:
        if prefetch is not None:
            # Cancel any in-flight prepare on the way out, including on an
            # exception, so a failed run does not wait on a chunk nobody
            # will consume.
            prefetch.shutdown(wait=False, cancel_futures=True)
    # Emit in the original target-major order, so inverting the loops above is
    # not observable in the output.
    ordered = [
        rows[(row.response_id, target_id)]
        for target_id in target_ids_needed
        for row in pairs_by_target[target_id].itertuples(index=False)
    ]
    return pd.DataFrame(ordered)
