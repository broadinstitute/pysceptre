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

_SIDE_CODES = {"left": -1, "both": 0, "right": 1}
_RESAMPLING_APPROXIMATIONS = ("skew_normal", "no_approximation")


def run_discovery_analysis(
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    grna_target_cells: dict[str, np.ndarray],
    pairs: pd.DataFrame,
    *,
    side: str = "both",
    resampling_approximation: str = "skew_normal",
    multiple_testing_alpha: float = 0.1,
    seed: int | None = None,
    target_chunk_size: int = _DEFAULT_TARGET_CHUNK_SIZE,
    chunk_memory_gb: float = _DEFAULT_CHUNK_MEMORY_GB,
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
    chunk_memory_gb: budget for the arrays a chunk holds, which sizes how many
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
    if resampling_approximation not in _RESAMPLING_APPROXIMATIONS:
        raise ValueError(
            f"resampling_approximation must be one of "
            f"{list(_RESAMPLING_APPROXIMATIONS)}, got {resampling_approximation!r}"
        )

    side_code = _SIDE_CODES[side]
    fit_parametric_curve = resampling_approximation == "skew_normal"
    B2, B3 = _resampling_budget(
        resampling_approximation, side_code, len(pairs), multiple_testing_alpha
    )

    return run_discovery_ntcells_complement(
        response_matrix=response_matrix,
        gene_ids=gene_ids,
        covariate_matrix=covariate_matrix,
        grna_target_cells=grna_target_cells,
        pairs=pairs,
        B2=B2,
        B3=B3,
        fit_parametric_curve=fit_parametric_curve,
        side_code=side_code,
        seed=seed,
        target_chunk_size=target_chunk_size,
        chunk_memory_gb=chunk_memory_gb,
    )


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
    )


def _resampling_budget(
    resampling_approximation: str,
    side_code: int,
    n_pairs: int,
    multiple_testing_alpha: float,
) -> tuple[int, int]:
    """Port of R's B2/B3 sizing (`run_discovery_analysis` + `run_qc_pt_2` in
    `s4_analysis_functs_1.R`). B1 is always 499 and is left at its default.

    `skew_normal` -> (4999, 0). B3 is 0 because this package only implements
    the CRT resampling mechanism; R uses B3=24999 only for `permutations`.

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
        return 4999, 0
    mult_fact = 10 if side_code == 0 else 5
    return 0, math.ceil(mult_fact * n_pairs / multiple_testing_alpha)
