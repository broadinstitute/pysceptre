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
from scipy import sparse

from ..glm.design import redundant_columns
from ..glm.design import validate_design_matrix as _validate_covariate_matrix
from .calibration import (
    GROUP_NAME_SEPARATOR,
    build_negative_control_pairs,
    group_weights,
    negative_control_pairs_from_names,
)
from .discovery import (
    _DEFAULT_CHUNK_MEMORY_GB,
    _DEFAULT_TARGET_CHUNK_SIZE,
    run_discovery_nt_cells,
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
_MOIS = ("high", "low")
_CONTROL_GROUPS = ("complement", "nt_cells")


def resolve_analysis_settings(
    moi: str, control_group: str | None, resampling_mechanism: str | None
) -> tuple[str, str]:
    """`(control_group, resampling_mechanism)` with R's defaults filled in.

    Ports the defaults of sceptre's `set_analysis_parameters`: high MOI uses the
    complement control group and the CRT; low MOI uses the NT cells and
    permutations. Either can be overridden, except that the NT cells are refused
    in high MOI. R silently replaces any high-MOI control group with the
    complement; refusing is stricter, so an explicit setting is never ignored.

    Raises:
        ValueError: an unknown value, or `control_group="nt_cells"` with
            `moi="high"`.
    """
    if moi not in _MOIS:
        raise ValueError(f"moi must be one of {list(_MOIS)}, got {moi!r}")
    low = moi == "low"
    if control_group is None:
        control_group = "nt_cells" if low else "complement"
    if control_group not in _CONTROL_GROUPS:
        raise ValueError(
            f"control_group must be one of {list(_CONTROL_GROUPS)}, got {control_group!r}"
        )
    if control_group == "nt_cells" and not low:
        raise ValueError(
            "control_group='nt_cells' needs moi='low'. In high MOI a cell carries several "
            "gRNAs, so the cells with an NT gRNA are not untreated; sceptre quietly uses the "
            "complement there instead, and pysceptre refuses rather than ignore the setting."
        )
    if resampling_mechanism is None:
        resampling_mechanism = "permutations" if low else "crt"
    if resampling_mechanism not in _RESAMPLING_MECHANISMS:
        raise ValueError(
            f"resampling_mechanism must be one of {list(_RESAMPLING_MECHANISMS)}, "
            f"got {resampling_mechanism!r}"
        )
    return control_group, resampling_mechanism


def nt_cell_pool(ntc_grna_cells: dict[str, np.ndarray], n_cells: int) -> np.ndarray:
    """The NT cells, as R's `all_nt_idxs`: each NT gRNA's cells in iteration order.

    Each gRNA's cells are sorted and concatenated in the dict's order, which is
    R's order when the dict is built in R's gRNA order. The order decides which
    cell each resample position refers to, so a different order is a different
    Monte Carlo draw: every p-value moves at the resampling resolution, and
    fitted values in their last bits. Build the dict in a fixed order.

    Raises:
        ValueError: no NT cells, an index out of range, or a cell under two NT
            gRNAs -- sceptre's low-MOI QC removes every cell with more than one
            gRNA, and the NT-cells analysis assumes it.
    """
    if not ntc_grna_cells:
        raise ValueError(
            "the NT-cells control group needs ntc_grna_cells, the cells of each individual "
            "non-targeting gRNA"
        )
    parts = [np.unique(np.asarray(c, dtype=np.int64)) for c in ntc_grna_cells.values()]
    pool = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    if pool.size == 0:
        raise ValueError("ntc_grna_cells holds no cells")
    if pool.min() < 0 or pool.max() >= n_cells:
        raise ValueError(
            f"ntc_grna_cells indices out of range for n_cells={n_cells}: "
            f"[{pool.min()}, {pool.max()}]"
        )
    n_shared = pool.size - np.unique(pool).size
    if n_shared:
        raise ValueError(
            f"{n_shared} cell(s) are listed under more than one non-targeting gRNA. In low MOI "
            "each cell carries at most one gRNA: sceptre's QC removes cells with zero or two "
            "or more before the analysis, and the NT-cells control group relies on it. Remove "
            "those cells upstream."
        )
    return pool


def _nt_pool_for(control_group: str, ntc_grna_cells, n_cells: int) -> np.ndarray | None:
    """The NT pool when the control group needs one; refuses NT cells it would ignore."""
    if control_group == "nt_cells":
        return nt_cell_pool(ntc_grna_cells, n_cells)
    if ntc_grna_cells is not None:
        raise ValueError(
            "ntc_grna_cells is only used with control_group='nt_cells'; the complement "
            "control group does not single out the NT cells."
        )
    return None


class _CellSubset:
    """Rows of a backed response matrix, restricted to some cells, read on demand."""

    def __init__(self, base, cells: np.ndarray):
        self._base = base
        self._cells = np.asarray(cells, dtype=np.int64)
        self.shape = (base.shape[0], self._cells.size)

    def rows(self, start: int, stop: int) -> sparse.csr_matrix:
        return sparse.csr_matrix(self._base.rows(start, stop))[:, self._cells]

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.rows(i.start or 0, self.shape[0] if i.stop is None else i.stop)
        return sparse.csr_matrix(self._base[int(i)])[:, self._cells]

    def toarray(self):
        raise NotImplementedError("refusing to densify a backed response matrix")


def _restrict_cells(response_matrix, cells: np.ndarray):
    """`response_matrix` with only `cells`, in that order, as its columns."""
    if hasattr(response_matrix, "rows"):
        return _CellSubset(response_matrix, cells)
    if sparse.issparse(response_matrix):
        return sparse.csr_matrix(response_matrix)[:, cells]
    return np.asarray(response_matrix)[:, cells]


def _check_dose_weights(
    weights: dict[str, np.ndarray],
    cells_of: dict[str, np.ndarray],
    name: str = "grna_target_weights",
) -> dict[str, np.ndarray]:
    """The dose test's weights as float64 arrays, one finite positive weight per listed cell."""
    out = {}
    for t, cells in cells_of.items():
        if t not in weights:
            raise ValueError(f"{name} has no weights for {t!r}")
        wts = np.asarray(weights[t], dtype=np.float64)
        if wts.shape != cells.shape:
            raise ValueError(f"{name}[{t!r}]: {wts.size} weights for {cells.size} cells")
        if wts.size and (not np.all(np.isfinite(wts)) or wts.min() <= 0):
            raise ValueError(f"{name}[{t!r}]: weights must be finite and positive")
        out[t] = wts
    return out


def run_discovery_analysis(
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    grna_target_cells: dict[str, np.ndarray],
    pairs: pd.DataFrame,
    *,
    side: str = "both",
    moi: str = "high",
    control_group: str | None = None,
    ntc_grna_cells: dict[str, np.ndarray] | None = None,
    grna_integration_strategy: str = "union",
    grna_target_data_frame: pd.DataFrame | None = None,
    drop_duplicate_design_rows: bool = False,
    resampling_approximation: str = "skew_normal",
    multiple_testing_alpha: float = 0.1,
    seed: int | None = None,
    target_chunk_size: int = _DEFAULT_TARGET_CHUNK_SIZE,
    chunk_memory_gb: float = _DEFAULT_CHUNK_MEMORY_GB,
    n_jobs: int = 1,
    resampling_mechanism: str | None = None,
    grna_target_weights: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """response_matrix: (n_genes, n_cells) dense ndarray or scipy.sparse matrix.
    gene_ids: row labels for response_matrix, in order.
    covariate_matrix: (n_cells, p) already formula-expanded design matrix.
    grna_target_cells: dict[target -> 0-based treated-cell indices].
    pairs: DataFrame['response_id', 'grna_target'] -- QC-passed pairs to test.
    moi: `"high"` (default) or `"low"`, sceptre's `import_data(moi=)`. It sets
        the defaults of `control_group` and `resampling_mechanism`, as R's
        `set_analysis_parameters` does, and is the only way to reach the NT
        cells.
    control_group: `"complement"` or `"nt_cells"`; `None` takes the MOI's
        default, `"complement"` in high MOI and `"nt_cells"` in low MOI. With
        `"nt_cells"` each pair is tested on its target's cells against the NT
        cells only, and the gene's GLM is refit for every pair, as in R.
    ntc_grna_cells: dict[NTC gRNA id -> 0-based cell indices], required with
        `control_group="nt_cells"` and refused otherwise. The NT cells are their
        union, which must share no cell with a tested target, and no cell may
        sit under two NT gRNAs: sceptre's low-MOI QC removes every cell with
        more than one gRNA, and this path relies on that.
    multiple_testing_alpha: only used to size the `no_approximation` resampling
        budget, exactly as R's `run_qc` does. pysceptre does *not* apply any
        multiple-testing correction to the returned p-values.
    target_chunk_size: how many gRNA targets' logistic fits + CRT draws to
        batch/hold in memory at once (see pipeline/discovery.py) -- lower this
        if you hit memory pressure, raise it for a modest speed gain if you
        have memory to spare.
    resampling_mechanism: `"crt"` or `"permutations"`; `None` takes the MOI's
        default, `"crt"` in high MOI and `"permutations"` in low MOI, as in R.
        The CRT draws each target's synthetic treated set from that target's
        own fitted probabilities; permutations draw one set of random subsets,
        sized by the targets present, and reuse it for every target.

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
    grna_target_weights: `{target: weights}`, aligned with `grna_target_cells`, runs the dose
        test instead of sceptre's: each treated cell enters the score statistic with its weight
        in place of 1 (`dose_weights` builds both dicts from gRNA counts). Weights must be
        finite and positive; all-ones weights give sceptre's result exactly. Complement control
        group, CRT and `grna_integration_strategy="union"` only. Not from sceptre:
        `docs/design.md`, "The dose test".
    """
    control_group, resampling_mechanism = resolve_analysis_settings(
        moi, control_group, resampling_mechanism
    )
    nt_cells = _nt_pool_for(control_group, ntc_grna_cells, response_matrix.shape[1])
    return _run_discovery(
        response_matrix,
        gene_ids,
        covariate_matrix,
        grna_target_cells,
        pairs,
        side=side,
        nt_cells=nt_cells,
        grna_integration_strategy=grna_integration_strategy,
        grna_target_data_frame=grna_target_data_frame,
        drop_duplicate_design_rows=drop_duplicate_design_rows,
        resampling_approximation=resampling_approximation,
        multiple_testing_alpha=multiple_testing_alpha,
        seed=seed,
        target_chunk_size=target_chunk_size,
        chunk_memory_gb=chunk_memory_gb,
        n_jobs=n_jobs,
        resampling_mechanism=resampling_mechanism,
        grna_target_weights=grna_target_weights,
    )


def _run_discovery(
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    grna_target_cells: dict[str, np.ndarray],
    pairs: pd.DataFrame,
    *,
    side: str,
    nt_cells: np.ndarray | None,
    grna_integration_strategy: str,
    grna_target_data_frame: pd.DataFrame | None,
    drop_duplicate_design_rows: bool,
    resampling_approximation: str,
    multiple_testing_alpha: float,
    seed: int | None,
    target_chunk_size: int,
    chunk_memory_gb: float,
    n_jobs: int,
    resampling_mechanism: str,
    permutation_width: int | None = None,
    grna_target_weights: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """`run_discovery_analysis` once the settings are resolved.

    `nt_cells` is the NT pool for the NT-cells control group and `None` for the
    complement. `permutation_width` sets the shared permutation width on the
    complement path (see `run_discovery_ntcells_complement`).
    """
    if side not in _SIDE_CODES:
        raise ValueError(f"side must be one of {sorted(_SIDE_CODES)}, got {side!r}")
    # Before any fitting: a rank-deficient design otherwise surfaces as a bare
    # LinAlgError from deep inside the batched solve. The calibration and power checks
    # reach this function too, so one call covers all three entry points.
    _validate_covariate_matrix(covariate_matrix, n_cells=response_matrix.shape[1])
    covariate_matrix = np.asarray(covariate_matrix, dtype=float)
    grna_target_cells = {t: np.asarray(c, dtype=np.int64) for t, c in grna_target_cells.items()}
    if grna_target_weights is not None:
        grna_target_weights = _check_dose_weights(grna_target_weights, grna_target_cells)
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
        expanded = singleton_pairs(
            pairs,
            grna_target_data_frame,
            drop_duplicate_design_rows=drop_duplicate_design_rows,
        )
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
        if drop_duplicate_design_rows:
            raise ValueError(
                "drop_duplicate_design_rows only applies to the singleton and bonferroni "
                "strategies. Under 'union' a duplicated design row cannot change anything: "
                "a target's treated cells are the set union over its guides, so listing one "
                "twice contributes the same cells once."
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

    engine_args = dict(
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
    if grna_target_weights is not None and (
        nt_cells is not None or per_guide or resampling_mechanism != "crt"
    ):
        raise ValueError(
            "the dose test (grna_target_weights) runs with the complement control group, the CRT "
            "and grna_integration_strategy='union' only"
        )
    if nt_cells is None:
        result = run_discovery_ntcells_complement(
            **engine_args,
            permutation_width=permutation_width,
            grna_target_weights=grna_target_weights,
        )
    else:
        result = run_discovery_nt_cells(**engine_args, nt_cells=nt_cells)
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
    moi: str = "high",
    control_group: str | None = None,
    resampling_approximation: str = "skew_normal",
    multiple_testing_alpha: float = 0.1,
    seed: int | None = None,
    target_chunk_size: int = _DEFAULT_TARGET_CHUNK_SIZE,
    chunk_memory_gb: float = _DEFAULT_CHUNK_MEMORY_GB,
    n_jobs: int = 1,
    resampling_mechanism: str | None = None,
    ntc_grna_weights: dict[str, np.ndarray] | None = None,
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
    moi, control_group, resampling_mechanism: as in `run_discovery_analysis`,
        with the same MOI-dependent defaults. With `control_group="nt_cells"`
        the whole check runs on the NT cells alone, as in R: the synthetic
        targets are tested against the remaining NT cells, so it needs at
        least two NT gRNAs and no cell under two of them. Under permutations
        the shared draws are as wide as the `calibration_group_size` largest
        NT gRNAs together, R's rule.
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
    ntc_grna_weights: `{NTC gRNA id: dose weights}`, aligned with `ntc_grna_cells`, to check the
        dose test instead of sceptre's: a synthetic target's cell takes its largest weight over
        the group's gRNAs (`DoseWeights.ntc_grna_weights`). Complement control group and CRT only,
        as in `run_discovery_analysis`.

    Returns the same columns as `run_discovery_analysis`.
    """
    control_group, resampling_mechanism = resolve_analysis_settings(
        moi, control_group, resampling_mechanism
    )
    if ntc_grna_weights is not None and control_group == "nt_cells":
        raise ValueError(
            "the dose test (ntc_grna_weights) runs with the complement control group only"
        )
    # Checked here, before the NT-cells restriction below gives both matrices the
    # same width and hides a mismatch.
    _validate_covariate_matrix(covariate_matrix, n_cells=response_matrix.shape[1])
    covariate_matrix = np.asarray(covariate_matrix, dtype=float)
    n_cells = covariate_matrix.shape[0]
    ntc_grna_cells = {g: np.asarray(c, dtype=np.int64) for g, c in ntc_grna_cells.items()}
    if ntc_grna_weights is not None:
        ntc_grna_weights = _check_dose_weights(ntc_grna_weights, ntc_grna_cells, "ntc_grna_weights")
    # R drops the NT gRNAs QC left without cells before it builds any group; a
    # supplied pair may still name one, and it then contributes no cells.
    with_cells = {g: c for g, c in ntc_grna_cells.items() if len(c)}
    if control_group == "nt_cells":
        if len(with_cells) < 2:
            raise ValueError(
                "a calibration check against the NT cells needs at least two non-targeting "
                "gRNAs, one to stand in for a target and one to leave as the control group"
            )
        # R's `subset_to_nt_cells`: the NT cells become the whole universe, so the
        # complement of a synthetic target is the rest of the NT cells.
        pool = nt_cell_pool(with_cells, n_cells)
        position = np.full(n_cells, -1, dtype=np.int64)
        position[pool] = np.arange(pool.size)
        ntc_grna_cells = {g: position[np.unique(c)] for g, c in ntc_grna_cells.items()}
        with_cells = {g: c for g, c in ntc_grna_cells.items() if len(c)}
        response_matrix = _restrict_cells(response_matrix, pool)
        covariate_matrix = covariate_matrix[pool]
        n_cells = pool.size
        redundant = redundant_columns(np.asarray(covariate_matrix, dtype=float))[1]
        if redundant:
            raise ValueError(
                f"on the NT cells alone, covariate_matrix column(s) {redundant} are linear "
                "combinations of the ones before them, so the calibration check's GLM cannot be "
                "fit there; sceptre stops here too. Drop the column(s), or use "
                "control_group='complement'."
            )

    if negative_control_pairs is None:
        rng = np.random.default_rng(seed)
        synthetic_target_cells, pairs = build_negative_control_pairs(
            response_matrix,
            gene_ids,
            with_cells,
            n_cells,
            n_calibration_pairs=n_calibration_pairs,
            calibration_group_size=calibration_group_size,
            n_nonzero_trt_thresh=n_nonzero_trt_thresh,
            n_nonzero_cntrl_thresh=n_nonzero_cntrl_thresh,
            pass_qc_rate=pass_qc_rate,
            rng=rng,
        )
    else:
        pairs = negative_control_pairs.reset_index(drop=True)
        synthetic_target_cells = negative_control_pairs_from_names(pairs, ntc_grna_cells, n_cells)
        if control_group == "nt_cells":
            # Construction never builds one, but a supplied group can leave the NT
            # cells, here the whole universe, with no control cells at all.
            whole = [t for t, c in synthetic_target_cells.items() if len(c) == n_cells]
            if whole:
                raise ValueError(
                    f"{len(whole)} synthetic target(s) hold every NT cell, e.g. {whole[0]!r}, "
                    "which leaves no control cells against the NT cells"
                )

    synthetic_target_weights = None
    if ntc_grna_weights is not None:
        synthetic_target_weights = {
            t: group_weights(t.split(GROUP_NAME_SEPARATOR), ntc_grna_cells, ntc_grna_weights)
            for t in synthetic_target_cells
        }

    permutation_width = None
    engine_seed = seed
    if resampling_mechanism == "permutations":
        sizes = sorted((len(c) for c in with_cells.values()), reverse=True)
        permutation_width = max(
            sum(sizes[:calibration_group_size]),
            max(len(c) for c in synthetic_target_cells.values()),
        )
        # The shared draws get a stream of their own: `rng` above chose the pairs they test.
        if seed is not None:
            engine_seed = int(np.random.SeedSequence(seed).spawn(1)[0].generate_state(1)[0])

    return _run_discovery(
        response_matrix,
        gene_ids,
        covariate_matrix,
        synthetic_target_cells,
        pairs,
        side=side,
        nt_cells=None,
        grna_integration_strategy="union",
        grna_target_data_frame=None,
        drop_duplicate_design_rows=False,
        resampling_approximation=resampling_approximation,
        multiple_testing_alpha=multiple_testing_alpha,
        seed=engine_seed,
        target_chunk_size=target_chunk_size,
        chunk_memory_gb=chunk_memory_gb,
        n_jobs=n_jobs,
        resampling_mechanism=resampling_mechanism,
        permutation_width=permutation_width,
        grna_target_weights=synthetic_target_weights,
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
    moi: str = "high",
    control_group: str | None = None,
    ntc_grna_cells: dict[str, np.ndarray] | None = None,
    resampling_approximation: str = "skew_normal",
    multiple_testing_alpha: float = 0.1,
    seed: int | None = None,
    target_chunk_size: int = _DEFAULT_TARGET_CHUNK_SIZE,
    chunk_memory_gb: float = _DEFAULT_CHUNK_MEMORY_GB,
    n_jobs: int = 1,
    resampling_mechanism: str | None = None,
    grna_target_weights: dict[str, np.ndarray] | None = None,
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
        to test. With `control_group="nt_cells"` the control count is the
        gene's nonzero NT cells, as in R.
    moi, control_group, ntc_grna_cells, resampling_mechanism, grna_target_weights: as in
        `run_discovery_analysis`, with the same MOI-dependent defaults.

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

    control_group, resampling_mechanism = resolve_analysis_settings(
        moi, control_group, resampling_mechanism
    )
    covariate_matrix = np.asarray(covariate_matrix, dtype=float)
    grna_target_cells = {t: np.asarray(c, dtype=np.int64) for t, c in grna_target_cells.items()}
    nt_cells = _nt_pool_for(control_group, ntc_grna_cells, covariate_matrix.shape[0])
    annotated = annotate_pairwise_qc(
        positive_control_pairs[["response_id", "grna_target"]],
        response_matrix,
        gene_ids,
        grna_target_cells,
        covariate_matrix.shape[0],
        n_nonzero_trt_thresh=n_nonzero_trt_thresh,
        n_nonzero_cntrl_thresh=n_nonzero_cntrl_thresh,
        control_cells=nt_cells,
    )
    testable = annotated[annotated["pass_qc"]][["response_id", "grna_target"]]
    if testable.empty:
        raise ValueError(
            f"no positive control pair passes pairwise QC (thresholds "
            f"trt >= {n_nonzero_trt_thresh}, cntrl >= {n_nonzero_cntrl_thresh})"
        )

    # Every target is passed, not only the tested ones: the engine tests only
    # the targets in `pairs`, and permutations size their shared draws over all
    # of them, as R does.
    tested = _run_discovery(
        response_matrix,
        gene_ids,
        covariate_matrix,
        grna_target_cells,
        testable.reset_index(drop=True),
        side=side,
        nt_cells=nt_cells,
        grna_integration_strategy="union",
        grna_target_data_frame=None,
        drop_duplicate_design_rows=False,
        resampling_approximation=resampling_approximation,
        multiple_testing_alpha=multiple_testing_alpha,
        seed=seed,
        target_chunk_size=target_chunk_size,
        chunk_memory_gb=chunk_memory_gb,
        n_jobs=n_jobs,
        resampling_mechanism=resampling_mechanism,
        grna_target_weights=grna_target_weights,
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
