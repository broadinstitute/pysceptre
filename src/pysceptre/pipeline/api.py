"""Top-level public entry point.

Assumes gRNA assignment and QC have already happened upstream (out of
scope -- this package is the statistical engine only, not sceptre's
`assign_grnas`/`run_qc`), and that the covariate matrix is already a plain
numeric design matrix (no `model.matrix`-equivalent formula DSL -- exact
factor-contrast parity with R across languages is its own project, and
orthogonal to the statistical engine targeted here).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .calibration import (
    build_negative_control_pairs,
    negative_control_pairs_from_names,
)
from .discovery import (
    _DEFAULT_CHUNK_MEMORY_GB,
    _DEFAULT_TARGET_CHUNK_SIZE,
    run_discovery_ntcells_complement,
)
from .grouping import aggregate_bonferroni, singleton_pairs, sort_like_r
from .power import (
    annotate_pairwise_qc,
    construct_positive_control_pairs,
    merge_qc_failures,
)

_SIDE_CODES = {"left": -1, "both": 0, "right": 1}
_RESAMPLING_APPROXIMATIONS = ("skew_normal", "no_approximation")
_RESAMPLING_MECHANISMS = ("crt", "permutations")
_GRNA_INTEGRATION_STRATEGIES = ("union", "singleton", "bonferroni")


def _validate_covariate_matrix(covariate_matrix: np.ndarray, n_cells: int | None = None) -> None:
    """Refuse a design matrix the GLM cannot fit, and say which columns are at fault.

    A rank-deficient design fails inside the batched weighted least squares as
    `numpy.linalg.LinAlgError: Singular matrix`, eight frames deep and with no
    mention of covariates. Since `covariate_matrix` is built by the caller --
    there is no formula DSL to catch an aliased contrast -- that is a likely
    mistake with an unhelpful symptom, so it is caught here instead.

    Rank comes from the singular values of `R` in a thin QR of the matrix,
    with the same relative tolerance `numpy.linalg.matrix_rank` uses. The QR
    is what makes it both cheap and correct: `R` is p x p, so every rank
    question after it is tiny, and because `Q` has orthonormal columns the
    rank of any column subset of `R` equals that of the same subset of the
    matrix.

    **Not the Gram matrix.** `X.T @ X` looks like the natural p x p route and
    is the thing the fit depends on -- `Zt_wZ` is it reweighted, and positive
    weights cannot restore rank -- but its eigenvalues are the *squares* of
    the singular values, so it squares the condition number. A design whose
    columns span fourteen orders of magnitude, raw UMI counts beside a small
    covariate say, is full rank and the Gram route rejected it.

    Raises:
        ValueError: not 2-D, empty, non-finite entries, a row count that
            disagrees with the data, or rank deficiency.
    """
    X = np.asarray(covariate_matrix)
    if X.ndim != 2:
        raise ValueError(f"covariate_matrix must be 2-D (n_cells, p), got shape {X.shape}")
    n, p = X.shape
    if n == 0 or p == 0:
        raise ValueError(f"covariate_matrix is empty, shape {X.shape}")
    if n_cells is not None and n != n_cells:
        raise ValueError(
            f"covariate_matrix has {n} rows but the response matrix has {n_cells} cells"
        )
    if p > n:
        raise ValueError(
            f"covariate_matrix has shape {X.shape}, more columns ({p}) than rows ({n}), so it "
            "cannot be full rank. It is expected as (n_cells, p); this looks transposed."
        )
    X = X.astype(float, copy=False)
    if not np.all(np.isfinite(X)):
        bad = np.flatnonzero(~np.all(np.isfinite(X), axis=0)).tolist()
        raise ValueError(f"covariate_matrix has non-finite values in column(s) {bad[:10]}")

    eps = np.finfo(float).eps
    r = np.linalg.qr(X, mode="r")

    def _rank(block: np.ndarray) -> int:
        sv = np.linalg.svd(block, compute_uv=False)
        if sv.size == 0 or sv[0] <= 0:
            return 0
        return int(np.sum(sv > sv[0] * max(n, block.shape[1]) * eps))

    rank = _rank(r)
    if rank == 0:
        raise ValueError("covariate_matrix is all zeros")
    if rank == p:
        return

    # Which columns are redundant, reported left to right so the first of a collinear set is
    # kept and the later ones are named -- the same convention R's `lm` follows when it drops
    # aliased terms, and the more useful one, since column 0 is usually the intercept.
    kept: list[int] = []
    for j in range(p):
        trial = kept + [j]
        if _rank(r[:, trial]) == len(trial):
            kept.append(j)
    redundant = [j for j in range(p) if j not in kept]
    raise ValueError(
        f"covariate_matrix is rank deficient: rank {rank} of {p} columns. "
        f"Column(s) {redundant} are linear combinations of the ones before them, so the "
        "weighted least squares inside the GLM fit is singular. This is usually an aliased "
        "contrast (a factor's full dummy set alongside an intercept), a duplicated column, or "
        "a covariate that is constant within another's levels. Drop the named column(s), or "
        "build the design with one reference level held out."
    )


def run_discovery_analysis(
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    grna_target_cells: dict[str, np.ndarray],
    pairs: pd.DataFrame,
    *,
    side: str = "both",
    grna_integration_strategy: str = "union",
    grna_target_data_frame: pd.DataFrame | None = None,
    resampling_approximation: str = "skew_normal",
    multiple_testing_alpha: float = 0.1,
    seed: int | None = None,
    target_chunk_size: int = _DEFAULT_TARGET_CHUNK_SIZE,
    chunk_memory_gb: float = _DEFAULT_CHUNK_MEMORY_GB,
    n_jobs: int = 1,
    resampling_mechanism: str = "crt",
) -> pd.DataFrame:
    """response_matrix: (n_genes, n_cells) dense ndarray or scipy.sparse matrix.
    gene_ids: row labels for response_matrix, in order.
    covariate_matrix: (n_cells, p) already formula-expanded design matrix.
    grna_target_cells: dict[target -> 0-based treated-cell indices].
    pairs: DataFrame['response_id', 'grna_target'] -- QC-passed pairs to test.
    multiple_testing_alpha: only used to size the `no_approximation` resampling
        budget, exactly as R's `run_qc` does. pysceptre does *not* apply any
        multiple-testing correction to the returned p-values.
    target_chunk_size: how many gRNA targets' logistic fits + CRT draws to
        batch/hold in memory at once (see pipeline/discovery.py) -- lower this
        if you hit memory pressure, raise it for a modest speed gain if you
        have memory to spare.
    resampling_mechanism: `"crt"` (default) or `"permutations"`, matching
        sceptre's own option for high-MOI data. The CRT draws each target's
        synthetic treated set from that target's own fitted probabilities;
        permutations draw one set of random subsets, sized by the largest
        target, and reuse it for every target.

        The choice is a real trade and is left to the caller.
        **Permutations cannot be reproducible across a change of pair list**:
        the shared draws are sized by the largest target present, so adding a
        target bigger than the current largest moves every result in the run.
        The CRT path has no such dependence -- each target seeds its own
        stream from its own name -- so a subset can be analysed, checked and
        extended with the earlier pairs reused unchanged. Permutations also
        pair with `B3 = 24999` in R against the CRT's `0`, so sampling is
        cheaper but the escalation batch is five times larger.
    n_jobs: worker processes (Linux) or threads (elsewhere) used for the
        per-pair tests, which are ~80% of the runtime. 1 disables
        parallelism, a negative value uses every core. Results do not depend
        on it: only the genes within an already-drawn target chunk are
        distributed, so the resampling draws are made in the same order
        whatever the worker count. Memory grows by roughly one gene's working
        arrays per worker, not by `chunk_memory_gb` per worker.
    chunk_memory_gb: budget for the gRNA-target chunk's arrays. **It cannot
        change a result**: it sizes target chunking only, and the binomial fit
        is bitwise identical across chunk widths (verified at 14 against 114
        on 567,690 cells). Gene fitting uses a fixed internal budget precisely
        so that this knob stays numerical-free -- driving both from it meant
        raising the budget moved 2 of 237 gene fits on day0. Budget for the
        arrays a chunk holds, which sizes how many
        genes or targets are processed together. Not a cap on the process's
        memory -- the input, retained state and allocator overhead sit outside
        it. You should not normally need to change this; the default is the
        fastest and leanest setting measured. Reductions to
        `target_chunk_size` are warned about and do not change results.

    Targets sceptre's complement-control-group + CRT discovery-analysis path
    (the only valid combination for high-MOI data -- see pipeline/discovery.py).
    """
    if side not in _SIDE_CODES:
        raise ValueError(f"side must be one of {sorted(_SIDE_CODES)}, got {side!r}")
    # Before any fitting: a rank-deficient design otherwise surfaces as a bare
    # LinAlgError from deep inside the batched solve. The calibration and power checks
    # reach this function too, so one call covers all three entry points.
    _validate_covariate_matrix(covariate_matrix, n_cells=response_matrix.shape[1])
    if resampling_approximation not in _RESAMPLING_APPROXIMATIONS:
        raise ValueError(
            f"resampling_approximation must be one of "
            f"{list(_RESAMPLING_APPROXIMATIONS)}, got {resampling_approximation!r}"
        )
    if resampling_mechanism not in _RESAMPLING_MECHANISMS:
        raise ValueError(
            f"resampling_mechanism must be one of {list(_RESAMPLING_MECHANISMS)}, "
            f"got {resampling_mechanism!r}"
        )
    if grna_integration_strategy not in _GRNA_INTEGRATION_STRATEGIES:
        raise ValueError(
            f"grna_integration_strategy must be one of "
            f"{list(_GRNA_INTEGRATION_STRATEGIES)}, got {grna_integration_strategy!r}"
        )

    per_guide = grna_integration_strategy in ("singleton", "bonferroni")
    if per_guide:
        if grna_target_data_frame is None:
            raise ValueError(
                f"grna_integration_strategy={grna_integration_strategy!r} needs "
                "grna_target_data_frame, the (grna_id, grna_target) design, to expand each pair "
                "to its target's guides. grna_target_cells must then be keyed by guide."
            )
        expanded = singleton_pairs(pairs, grna_target_data_frame)
        # The engine looks its treated cells up by the pair frame's `grna_target` column, so the
        # guide id goes there and the real target is restored afterwards. That keeps
        # discovery.py entirely unaware of which strategy is in play, exactly as sceptre's
        # engine is: it only ever sees a `grna_group`.
        #
        # **Tested once per (gene, guide), reported once per (gene, guide, target).** A guide in
        # two overlapping elements yields two rows in R, and they are necessarily identical:
        # same gene, same guide, so the same treated cells and the same test. Running it twice
        # would only cost time and risk the two copies disagreeing, so the engine is handed the
        # distinct tests and the target column is restored by a fan-out afterwards.
        engine_pairs = (
            expanded.loc[:, ["response_id", "grna_id"]]
            .drop_duplicates()
            .rename(columns={"grna_id": "grna_target"})
            .reset_index(drop=True)
        )
    else:
        if grna_target_data_frame is not None:
            raise ValueError(
                "grna_target_data_frame is only used by the singleton and bonferroni "
                "strategies; under 'union' a pair already names the unit that is tested."
            )
        engine_pairs = pairs

    side_code = _SIDE_CODES[side]
    fit_parametric_curve = resampling_approximation == "skew_normal"
    B2, B3 = _resampling_budget(
        resampling_approximation,
        side_code,
        len(engine_pairs),
        multiple_testing_alpha,
        resampling_mechanism,
    )

    result = run_discovery_ntcells_complement(
        response_matrix=response_matrix,
        gene_ids=gene_ids,
        covariate_matrix=covariate_matrix,
        grna_target_cells=grna_target_cells,
        pairs=engine_pairs,
        B2=B2,
        B3=B3,
        fit_parametric_curve=fit_parametric_curve,
        side_code=side_code,
        seed=seed,
        target_chunk_size=target_chunk_size,
        chunk_memory_gb=chunk_memory_gb,
        n_jobs=n_jobs,
        resampling_mechanism=resampling_mechanism,
    )
    if not per_guide:
        return result

    # Restore the two-column identity R reports: the guide that was tested, and the target it
    # belongs to. Joining would be ambiguous for a guide in several targets, so the mapping is
    # carried positionally from the expansion instead.
    result = result.rename(columns={"grna_target": "grna_id"})
    result = expanded.merge(result, on=["response_id", "grna_id"], how="left")
    front = ["response_id", "grna_id", "grna_target"]
    result = result.loc[:, front + [c for c in result.columns if c not in front]]
    if grna_integration_strategy == "bonferroni":
        result = aggregate_bonferroni(result)
    return sort_like_r(result)


def run_calibration_check(
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    ntc_grna_cells: dict[str, np.ndarray],
    *,
    n_calibration_pairs: int,
    calibration_group_size: int,
    n_nonzero_trt_thresh: int = 7,
    n_nonzero_cntrl_thresh: int = 7,
    pass_qc_rate: float = 1.0,
    negative_control_pairs: pd.DataFrame | None = None,
    side: str = "both",
    resampling_approximation: str = "skew_normal",
    multiple_testing_alpha: float = 0.1,
    seed: int | None = None,
    target_chunk_size: int = _DEFAULT_TARGET_CHUNK_SIZE,
    chunk_memory_gb: float = _DEFAULT_CHUNK_MEMORY_GB,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run sceptre's calibration check: the discovery test over negative controls.

    Synthetic negative-control targets are built by regrouping individual
    non-targeting gRNAs, then tested with the *same* engine as
    `run_discovery_analysis`. Because no target is real, a correctly calibrated
    method returns p-values that are uniform on (0, 1); that uniformity, not
    agreement with any other implementation, is what the check measures.

    ntc_grna_cells: dict[NTC gRNA id -> 0-based cell indices]. This must be
        keyed by individual gRNA, not by target -- a target-keyed mapping
        collapses every non-targeting gRNA into one entry (and in sceptre's own
        object, omits them entirely), leaving nothing to regroup.
    n_calibration_pairs: how many pairs to test. R defaults this to the number
        of discovery pairs that passed QC.
    calibration_group_size: how many NTC gRNAs per synthetic target. R's
        default is the median number of gRNAs per real target, capped at the
        number of NTC gRNAs; that median is not derivable from this function's
        arguments, so it is required here and the dataset export records it.
    n_nonzero_trt_thresh / n_nonzero_cntrl_thresh: pairwise QC thresholds. Note
        these *filter construction* rather than being reported: every returned
        pair passes, so there is no `pass_qc` column, unlike a discovery result.
    negative_control_pairs: skip construction and test exactly these pairs,
        whose `grna_target` entries must be "&"-joined NTC gRNA ids. This is
        the validation path -- R's own pair selection is unseeded and varies
        run to run, so a pair-by-pair comparison is only meaningful when both
        sides are given the same pairs.

    Returns the same columns as `run_discovery_analysis`.
    """
    if negative_control_pairs is None:
        rng = np.random.default_rng(seed)
        synthetic_target_cells, pairs = build_negative_control_pairs(
            response_matrix,
            gene_ids,
            ntc_grna_cells,
            covariate_matrix.shape[0],
            n_calibration_pairs=n_calibration_pairs,
            calibration_group_size=calibration_group_size,
            n_nonzero_trt_thresh=n_nonzero_trt_thresh,
            n_nonzero_cntrl_thresh=n_nonzero_cntrl_thresh,
            pass_qc_rate=pass_qc_rate,
            rng=rng,
        )
    else:
        pairs = negative_control_pairs.reset_index(drop=True)
        synthetic_target_cells = negative_control_pairs_from_names(
            pairs, ntc_grna_cells, covariate_matrix.shape[0]
        )

    return run_discovery_analysis(
        response_matrix=response_matrix,
        gene_ids=gene_ids,
        covariate_matrix=covariate_matrix,
        grna_target_cells=synthetic_target_cells,
        pairs=pairs,
        side=side,
        resampling_approximation=resampling_approximation,
        multiple_testing_alpha=multiple_testing_alpha,
        seed=seed,
        target_chunk_size=target_chunk_size,
        chunk_memory_gb=chunk_memory_gb,
        n_jobs=n_jobs,
    )


def run_power_check(
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    grna_target_cells: dict[str, np.ndarray],
    *,
    positive_control_pairs: pd.DataFrame | None = None,
    n_nonzero_trt_thresh: int = 7,
    n_nonzero_cntrl_thresh: int = 7,
    side: str = "both",
    resampling_approximation: str = "skew_normal",
    multiple_testing_alpha: float = 0.1,
    seed: int | None = None,
    target_chunk_size: int = _DEFAULT_TARGET_CHUNK_SIZE,
    chunk_memory_gb: float = _DEFAULT_CHUNK_MEMORY_GB,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run sceptre's power check: the discovery test over positive controls.

    A positive control is a (gene, target) pair where an effect is expected
    -- usually a gRNA against the gene's own TSS. The check asks whether the
    pipeline recovers effects it should, and is read next to
    `run_calibration_check`, which asks whether it invents effects it
    should not.

    positive_control_pairs: the pairs to test, as
        `DataFrame['response_id', 'grna_target']`. **Supply these.** They are
        a biological claim about which target perturbs which gene, and only
        the experiment knows it. When omitted, R's name-matching rule is used
        as a fallback -- a target that is itself a gene id pairs with that
        gene -- which works when targets are named after genes and finds
        nothing when they are named after genomic intervals. A screen of the
        latter kind raises rather than silently returning an empty result.
    n_nonzero_trt_thresh / n_nonzero_cntrl_thresh: pairwise QC thresholds.
        Unlike the calibration check, failures are **reported, not
        filtered**: the returned frame has a `pass_qc` column and NaN
        results for pairs that did not meet them. Dropping them would
        overstate power by hiding the controls the screen had too few cells
        to test.

    Returns one row per supplied pair, with `pass_qc`, `n_nonzero_trt` and
    `n_nonzero_cntrl` alongside the usual columns. No multiple-testing
    correction is applied, matching R: these are a diagnostic, not
    discoveries, and adjusting them against each other answers no question.
    """
    if positive_control_pairs is None:
        positive_control_pairs = construct_positive_control_pairs(gene_ids, list(grna_target_cells))
        if positive_control_pairs.empty:
            raise ValueError(
                "no positive control pairs: no gRNA target is named after a gene, so "
                "R's name-matching rule finds nothing. This is normal for a screen "
                "targeting genomic intervals -- pass positive_control_pairs explicitly."
            )

    annotated = annotate_pairwise_qc(
        positive_control_pairs[["response_id", "grna_target"]],
        response_matrix,
        gene_ids,
        grna_target_cells,
        covariate_matrix.shape[0],
        n_nonzero_trt_thresh=n_nonzero_trt_thresh,
        n_nonzero_cntrl_thresh=n_nonzero_cntrl_thresh,
    )
    testable = annotated[annotated["pass_qc"]][["response_id", "grna_target"]]
    if testable.empty:
        raise ValueError(
            f"no positive control pair passes pairwise QC (thresholds "
            f"trt >= {n_nonzero_trt_thresh}, cntrl >= {n_nonzero_cntrl_thresh})"
        )

    tested = run_discovery_analysis(
        response_matrix=response_matrix,
        gene_ids=gene_ids,
        covariate_matrix=covariate_matrix,
        grna_target_cells={t: grna_target_cells[t] for t in testable["grna_target"].unique()},
        pairs=testable.reset_index(drop=True),
        side=side,
        resampling_approximation=resampling_approximation,
        multiple_testing_alpha=multiple_testing_alpha,
        seed=seed,
        target_chunk_size=target_chunk_size,
        chunk_memory_gb=chunk_memory_gb,
        n_jobs=n_jobs,
    )
    return merge_qc_failures(tested, annotated)


def _resampling_budget(
    resampling_approximation: str,
    side_code: int,
    n_pairs: int,
    multiple_testing_alpha: float,
    resampling_mechanism: str = "crt",
) -> tuple[int, int]:
    """Port of R's B2/B3 sizing (`run_discovery_analysis` + `run_qc_pt_2` in
    `s4_analysis_functs_1.R`). B1 is always 499 and is left at its default.

    `skew_normal` -> (4999, 0) for the CRT, and (4999, 24999) for
    permutations, which is R's rule verbatim:

        B3 <- if (resampling_mechanism == "permutations") 24999L else 0L

    The asymmetry is structural rather than arbitrary. Under the CRT a
    rejected skew-normal fit falls back to the B2 statistics already drawn,
    because 24,999 more draws *per target* would be ruinous. Permutation
    draws are made once and shared by every target, so a large third batch
    is nearly free -- and it buys a p-value floor of 1/25000 = 4e-5 instead
    of 1/5000 = 2e-4 for pairs whose fit is rejected.

    `no_approximation` -> (0, ceil(mult * n_pairs / alpha)), with mult = 10
    two-sided and 5 one-sided. B2 is 0 because no curve is fit. `n_pairs` is
    the analog of R's `n_ok_discovery_pairs`: `pairs` is documented as already
    QC-passed, and pysceptre has no positive-control set, so R's
    `max(discovery, positive_control)` collapses to just this count.

    Note the scale: this makes B3 grow linearly in the number of pairs. A
    33,066-pair one-sided analysis gives B3 = 1,653,300 draws *per target*, so
    `no_approximation` needs a very small `target_chunk_size` and is slow. That
    cost is inherent to the method -- R pays it too, which is why
    `skew_normal` is the default there and here.

    The formula cuts the other way for small analyses: 4 pairs one-sided at
    alpha=0.1 gives B3 = 200, *below* B1 = 499, so `no_approximation` is
    actually coarser than `skew_normal` there. That is R's behavior, not a
    correction applied here.
    """
    if resampling_approximation == "skew_normal":
        return (4999, 24999) if resampling_mechanism == "permutations" else (4999, 0)
    mult_fact = 10 if side_code == 0 else 5
    return 0, math.ceil(mult_fact * n_pairs / multiple_testing_alpha)
