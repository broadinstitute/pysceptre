"""The trans background: a screen's targets against genes, with the cis run's test.

`iter_trans_analysis` tests (gene, target) pairs exactly as a reference
discovery run tests them: the same CRT, seed and resampling budget, and every
target fit in the reference run's own chunk, because under the default
`glm_solver="outer"` the batched logistic fit depends on its chunk-mates in
the last bits (under `"structured"` it does not, and the chunks change
nothing). A pair tested in full gets the result `run_discovery_analysis`
gives it, bit for bit on the same CPU class and BLAS, with numba, and with the
covariate matrix in the same memory layout.

Two screens skip work that no call of the specificity check can see: a pair
whose fold change is at least 1 is not resampled, and the first stage stops
once the pair can no longer escalate nor reach the nominal level. Both are
exact for every table `run_specificity_check` returns. Each gene's pieces are
built once per call rather than once per target chunk, which is what makes a
background of 1e8 pairs affordable. See docs/design.md, "The trans background".
"""

from __future__ import annotations

import gc
import itertools
import warnings
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..crt import sampler as _sampler
from ..glm.design import targets_in_span, validate_design_matrix
from ..glm.irls import x_outer_flat
from ..precompute.pieces import compute_precomputation_pieces
from ..test_statistic.resampling import run_low_level_test_full, run_low_level_test_screened
from ..test_statistic.score_stat import FlatDraws, stack_pieces
from .discovery import (
    _ENGINE_LOCK,
    _GENE_BATCH_WIDTH,
    _GENE_CHUNK_MEMORY_GB,
    GenePrecomputation,
    TargetPrecomputation,
    _design_for,
    _engine_blas_limit,
    _get_row,
    _map_jobs,
    fit_all_genes,
    fit_all_targets,
    reference_fit_chunks,
    resolve_entropy,
    resolve_n_jobs,
)
from .gene_cache import _check_data

__all__ = [
    "PAIR_MODES",
    "SCREEN_CODES",
    "TransGeneResult",
    "iter_trans_analysis",
    "trans_fit_chunks",
    "trans_results_frame",
]

# `screen` values, in code order.
SCREEN_CODES = ("none", "fold_change", "early_stop")
# Per-pair `mode` values: screened (the default), full (sceptre's test, no
# screen), audit (both, the full result kept beside the screened one).
PAIR_MODES = ("screened", "full", "audit")
_SCREENED, _FULL, _AUDIT = range(3)

_SIDE_CODES = {"left": -1, "both": 0, "right": 1}
_B1 = 499
_SKEW_NORMAL_B2 = 4999

_FLOAT_COLUMNS = ("p_value", "fold_change", "se_fold_change", "z_orig", "stage", "p_value_floor")
_AUDIT_COLUMNS = (
    "audit_p_value",
    "audit_fold_change",
    "audit_se_fold_change",
    "audit_z_orig",
    "audit_stage",
)


@dataclass(frozen=True)
class TransGeneResult:
    """One gene's pairs, as columns aligned with `rows`.

    `rows` are positions in the `pairs` frame the call was given. `columns`
    holds `p_value`, `fold_change`, `se_fold_change`, `z_orig`, `stage`,
    `p_value_floor` (float64), `screen` (int8 codes into `SCREEN_CODES`),
    `n_null_evaluated` (int32) and the `audit_*` columns, which are NaN except
    on rows whose mode was "audit", where they hold that pair's full result.
    """

    gene_id: str
    rows: np.ndarray
    columns: dict


def trans_fit_chunks(
    grna_target_cells: dict[str, np.ndarray],
    reference_pairs: pd.DataFrame,
    n_cells: int,
    *,
    target_chunk_size: int,
    chunk_memory_gb: float,
    expected_fit_chunks: tuple[int, int] | None = None,
) -> list[list[str]]:
    """The reference run's target fit chunks, as `run_discovery_analysis` built them.

    `reference_pairs` are the pairs the reference (cis) run was given, and
    `target_chunk_size` and `chunk_memory_gb` its settings. `expected_fit_chunks`
    is `(width, count)`; a layout of a different width or count is refused,
    because it would fit every target with different chunk-mates.
    """
    with warnings.catch_warnings():
        # The reference run already warned when it reduced its chunk width.
        warnings.filterwarnings("ignore", message="reducing target_chunk_size")
        chunks = reference_fit_chunks(
            grna_target_cells,
            reference_pairs,
            n_cells,
            B_total=_B1 + _SKEW_NORMAL_B2,
            target_chunk_size=target_chunk_size,
            chunk_memory_gb=chunk_memory_gb,
            resampling_mechanism="crt",
        )
    if expected_fit_chunks is not None:
        got = (len(chunks[0]) if chunks else 0, len(chunks))
        if got != tuple(expected_fit_chunks):
            raise ValueError(
                f"the reference pairs give fit chunks of width {got[0]} and count {got[1]}, "
                f"not the expected {tuple(expected_fit_chunks)}: check reference_pairs, "
                "grna_target_cells order, target_chunk_size and chunk_memory_gb against the "
                "reference run"
            )
    return chunks


def _validated(
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    grna_target_cells: dict,
    pairs: pd.DataFrame,
    seed,
    side: str,
    resampling_approximation: str,
    p_exact_below: float,
    block_rows: int,
):
    """Refuse what cannot reproduce the reference run, and normalize the inputs as it did."""
    if seed is None:
        raise ValueError("seed is required: a trans run must use the reference run's seed")
    if side not in _SIDE_CODES:
        raise ValueError(f"side must be one of {sorted(_SIDE_CODES)}, got {side!r}")
    if resampling_approximation != "skew_normal":
        raise ValueError(
            "the trans background runs with resampling_approximation='skew_normal' only: "
            "'no_approximation' sizes its resampling budget by the number of pairs, so no "
            "two pair sets draw the same resamples"
        )
    if not _sampler._HAVE_NUMBA:
        raise RuntimeError(
            "the trans background needs numba: without it the CRT sampler orders each "
            "draw differently, so results would not reproduce a run made with numba"
        )
    if not 0.0 < p_exact_below < 1.0:
        raise ValueError(f"p_exact_below must lie in (0, 1), got {p_exact_below!r}")
    if int(block_rows) < 1:
        raise ValueError(f"block_rows must be positive, got {block_rows!r}")
    for column in ("response_id", "grna_target"):
        if column not in pairs.columns:
            raise KeyError(f"pairs is missing column {column!r}")
    validate_design_matrix(covariate_matrix, n_cells=response_matrix.shape[1])
    covariate_matrix = np.asarray(covariate_matrix, dtype=float)
    grna_target_cells = {t: np.asarray(c, dtype=np.int64) for t, c in grna_target_cells.items()}
    unknown_genes = sorted(set(pairs["response_id"]) - set(gene_ids))
    if unknown_genes:
        raise ValueError(
            f"pairs name {len(unknown_genes)} gene(s) not in gene_ids, e.g. {unknown_genes[:5]}"
        )
    if "mode" in pairs.columns:
        bad = sorted(set(pairs["mode"]) - set(PAIR_MODES))
        if bad:
            raise ValueError(f"pairs['mode'] must be one of {PAIR_MODES}, got {bad[:5]}")
    return covariate_matrix, grna_target_cells


def _selected_chunks(
    chunks: list[list[str]], pairs: pd.DataFrame, fit_chunk_ids: Sequence[int] | None
) -> list[list[str]]:
    """The reference chunks a call fits: the given ids, or every chunk a pair needs."""
    chunk_of = {t: i for i, chunk in enumerate(chunks) for t in chunk}
    targets = pd.unique(pairs["grna_target"])
    outside = [t for t in targets if t not in chunk_of]
    if outside:
        raise ValueError(
            f"{len(outside)} target(s) of pairs are in no reference fit chunk, e.g. "
            f"{outside[:5]}: a target can only be tested with the chunk-mates it was fit "
            "with in the reference run"
        )
    if fit_chunk_ids is None:
        ids = sorted({chunk_of[t] for t in targets})
    else:
        ids = [int(i) for i in fit_chunk_ids]
        if len(set(ids)) != len(ids) or any(not 0 <= i < len(chunks) for i in ids):
            raise ValueError(
                f"fit_chunk_ids must be distinct positions in [0, {len(chunks)}), got some "
                "repeated or out of range"
            )
        chosen = set(ids)
        missing = [t for t in targets if chunk_of[t] not in chosen]
        if missing:
            raise ValueError(
                f"{len(missing)} target(s) of pairs lie outside fit_chunk_ids, e.g. {missing[:5]}"
            )
    return [chunks[i] for i in ids]


def _prepare_targets(
    chunks: list[list[str]],
    grna_target_cells: dict[str, np.ndarray],
    covariate_matrix: np.ndarray,
    *,
    B2: int,
    B3: int,
    entropy: int,
    block_rows: int,
    xo: np.ndarray | None,
    design,
    n_jobs: int,
    fit_concurrency: int,
) -> dict[str, TargetPrecomputation]:
    """Fit and draw every target of `chunks`, one `fit_all_targets` call per chunk.

    Each chunk is fit with exactly its reference targets, in their order, so
    the batched solve and every draw are the reference run's. Chunks run on a
    few threads at once, as the reference run's prefetch did; that changes no
    chunk's arithmetic. Each target keeps its draws as `FlatDraws` with the
    first stage's row blocks prebuilt, and drops its fitted probabilities.
    """
    n_cells = covariate_matrix.shape[0]
    concurrency = max(1, min(int(fit_concurrency), len(chunks)))
    draw_jobs = max(1, resolve_n_jobs(n_jobs) // concurrency)

    def prepare(chunk: list[str]) -> dict[str, TargetPrecomputation]:
        fitted = fit_all_targets(
            {t: grna_target_cells[t] for t in chunk},
            covariate_matrix,
            B1=_B1,
            B2=B2,
            B3=B3,
            seed=entropy,
            n_jobs=draw_jobs,
            x_outer_flat_shared=xo,
            design=design,
        )
        out = {}
        for t in chunk:
            draws = FlatDraws(fitted[t].draws._idxs, n_cells)
            draws.prebuild_blocks(_B1, block_rows)
            out[t] = TargetPrecomputation(
                trt_idxs=fitted[t].trt_idxs, fitted_probabilities=None, draws=draws
            )
        return out

    targets: dict[str, TargetPrecomputation] = {}
    if concurrency == 1:
        for chunk in chunks:
            targets.update(prepare(chunk))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for out in ex.map(prepare, chunks):
                targets.update(out)
    return targets


def _gene_fits_for(
    genes: list[str],
    gene_fits,
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    xo: np.ndarray | None,
    design,
    n_jobs: int,
) -> dict[str, GenePrecomputation]:
    """Width-1 fits for `genes`: from `gene_fits` where present, else fit as the engine fits."""
    have = {} if gene_fits is None else dict(getattr(gene_fits, "fits", gene_fits))
    missing = [g for g in genes if g not in have]
    if missing:
        if gene_fits is not None:
            warnings.warn(
                f"gene_fits lacks {len(missing)} of {len(genes)} gene(s); fitting them now",
                stacklevel=3,
            )
        row_of = {g: i for i, g in enumerate(gene_ids)}
        have.update(
            fit_all_genes(
                response_matrix,
                missing,
                covariate_matrix,
                x_outer_flat_shared=xo,
                design=design,
                chunk_memory_gb=_GENE_CHUNK_MEMORY_GB,
                gene_rows=[row_of[g] for g in missing],
                batch_width=_GENE_BATCH_WIDTH,
                n_jobs=n_jobs,
            )
        )
    return {g: have[g] for g in genes}


# Read by `_trans_gene_job` in whichever process or thread runs it; a module
# global so a forked worker inherits it without copying. Keyed by a per-call
# token, as `discovery._TARGET_STATE` is, so that two generators interleaved
# in one thread each read their own inputs.
_TRANS_STATE: dict[int, dict] = {}
_TRANS_STATE_SEQ = itertools.count()


def _new_columns(n: int) -> dict:
    cols = {name: np.full(n, np.nan) for name in _FLOAT_COLUMNS + _AUDIT_COLUMNS}
    cols["screen"] = np.zeros(n, dtype=np.int8)
    cols["n_null_evaluated"] = np.zeros(n, dtype=np.int32)
    return cols


def _n_null_full(stage: float, B2: int, B3: int) -> int:
    """Null statistics a full test computed, from the stage it ended at."""
    if not np.isfinite(stage):
        return 0
    return _B1 + (B2 if stage >= 2 else 0) + (B3 if stage == 3 else 0)


def _trans_gene_job(job: tuple[int, str, np.ndarray, np.ndarray, np.ndarray]) -> TransGeneResult:
    """One gene's pairs: its pieces built once, then each pair tested in its mode."""
    token, gene_id, rows, target_codes, modes = job
    st = _TRANS_STATE[token]
    gene = st["gene_fits"][gene_id]
    y = _get_row(st["response_matrix"], st["gene_row_index"][gene_id])
    pieces = compute_precomputation_pieces(y, st["covariate_matrix"], gene.fitted_coefs, gene.theta)
    stacked = stack_pieces(pieces.a, pieces.w, pieces.D)
    B2, B3 = st["B2"], st["B3"]
    cols = _new_columns(len(rows))
    for j in range(len(rows)):
        target_id = st["target_ids"][target_codes[j]]
        if target_id in st["spanned"]:
            continue  # NaN, as the reference run reports it
        target = st["targets"][target_id]
        common = dict(
            y=y,
            mu=pieces.mu,
            a=pieces.a,
            w=pieces.w,
            D=pieces.D,
            trt_idxs=target.trt_idxs,
            synthetic_idxs=target.draws,
            stacked=stacked,
            B1=_B1,
            B2=B2,
            B3=B3,
            fit_parametric_curve=True,
            side_code=st["side_code"],
        )
        mode = modes[j]
        if mode != _SCREENED:
            full = run_low_level_test_full(**common)
            prefix = "" if mode == _FULL else "audit_"
            cols[prefix + "p_value"][j] = full.p_value
            cols[prefix + "fold_change"][j] = full.fold_change
            cols[prefix + "se_fold_change"][j] = full.se_fold_change
            cols[prefix + "z_orig"][j] = full.z_orig
            cols[prefix + "stage"][j] = full.stage
            if mode == _FULL:
                cols["n_null_evaluated"][j] = _n_null_full(full.stage, B2, B3)
                continue
        r = run_low_level_test_screened(
            **common,
            screen_fold_change=True,
            p_exact_below=st["p_exact_below"],
            block_rows=st["block_rows"],
        )
        cols["p_value"][j] = r.p_value
        cols["fold_change"][j] = r.fold_change
        cols["se_fold_change"][j] = r.se_fold_change
        cols["z_orig"][j] = r.z_orig
        cols["stage"][j] = r.stage
        cols["p_value_floor"][j] = r.p_value_floor
        cols["screen"][j] = SCREEN_CODES.index(r.screen)
        cols["n_null_evaluated"][j] = r.n_null_evaluated
    return TransGeneResult(gene_id=gene_id, rows=rows, columns=cols)


def iter_trans_analysis(
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    grna_target_cells: dict[str, np.ndarray],
    pairs: pd.DataFrame,
    *,
    reference_pairs: pd.DataFrame,
    seed: int,
    side: str = "both",
    resampling_approximation: str = "skew_normal",
    target_chunk_size: int = 200,
    chunk_memory_gb: float = 1.0,
    expected_fit_chunks: tuple[int, int] | None = None,
    fit_chunk_ids: Sequence[int] | None = None,
    gene_fits=None,
    p_exact_below: float = 0.05,
    block_rows: int = 32,
    n_jobs: int = 1,
    backend: str | None = None,
    fit_concurrency: int = 3,
    glm_solver: str = "outer",
) -> Iterator[TransGeneResult]:
    """Test `pairs` with a reference run's test, yielding one `TransGeneResult` per gene.

    From the first `next()` until it is exhausted or closed, the generator
    holds the engine lock, so an analysis started in another thread waits for
    it, and the process-wide limit of one BLAS thread. Generators interleaved
    in one thread keep their own inputs and may finish in any order; the
    BLAS limit goes back to what it was before the first of them started
    only when the last one finishes. The engine lock belongs to the thread
    that started the generator: advance it, and exhaust or close it, in that
    thread.

    Args:
        response_matrix, gene_ids, covariate_matrix, grna_target_cells: as for
            `run_discovery_analysis`, and the same objects the reference run
            was given (complement control group, CRT, union strategy).
        pairs: `response_id`, `grna_target`, and optionally `mode`, one of
            `PAIR_MODES` per pair ("screened" when absent). Pairs must already
            pass pairwise QC, as for `run_discovery_analysis`.
        reference_pairs: the pairs the reference run was given. They fix its
            target fit chunks; see `trans_fit_chunks`.
        seed, side, resampling_approximation, target_chunk_size,
            chunk_memory_gb: the reference run's settings. `seed` is required
            and the approximation must be "skew_normal". The screens are
            exact for calls with a negative effect, the only ones the
            specificity check makes, whatever the side; under side "right" a
            pair called with a fold change of at least 1 is screened.
        expected_fit_chunks: `(width, count)` the reference layout must have.
        fit_chunk_ids: positions of the reference chunks this call fits, for a
            run split into shards; by default every chunk a pair needs. All
            targets of a chosen chunk are fit, tested or not.
        gene_fits: width-1 gene fits to reuse (a `GeneFitCache` or a dict of
            `GenePrecomputation`); genes it lacks are fit here, with a warning.
            A `GeneFitCache` is checked against this call's data as
            `load_gene_cache` checks it: the covariate matrix's shape, values
            and memory layout, the cell count, `glm_solver`, and the counts of
            every gene of `pairs` it supplies. A plain dict is not checked.
        p_exact_below: the first stage stops only once a pair's p-value is
            known to be at least this, and above the escalation threshold.
        block_rows: rows of first-stage resamples evaluated between stop
            checks. It decides where a pair can stop early, so whether it is
            screened "early_stop" and its `p_value_floor` and
            `n_null_evaluated`; a pair that is not screened gets the full
            test's result whatever its value.
        n_jobs, backend: workers for the per-gene tests and their backend
            ("fork" or "thread"; the platform default when None). Neither can
            change a result.
        fit_concurrency: target chunks fit at once.
        glm_solver: the reference run's `glm_solver` (see
            `run_discovery_analysis`).

    Yields:
        One `TransGeneResult` per gene of `pairs`, in `gene_ids` order. A pair
        tested in full matches `run_discovery_analysis` on the same inputs, bit
        for bit on the same CPU class and BLAS, with numba, and with the
        covariate matrix in the same memory layout. A screened pair has
        `p_value` NaN and
        `screen` "fold_change" (its fold change is at least 1) or "early_stop"
        (its p-value is at least `p_value_floor`).

    Raises:
        ValueError: a setting that cannot reproduce the reference run, a pair
            outside the reference chunks, an unknown gene or mode, or a
            `GeneFitCache` that does not match the data or `glm_solver`.
        RuntimeError: numba is not available.
    """
    covariate_matrix, grna_target_cells = _validated(
        response_matrix,
        gene_ids,
        covariate_matrix,
        grna_target_cells,
        pairs,
        seed,
        side,
        resampling_approximation,
        p_exact_below,
        block_rows,
    )
    n_cells = covariate_matrix.shape[0]
    chunks = trans_fit_chunks(
        grna_target_cells,
        reference_pairs,
        n_cells,
        target_chunk_size=target_chunk_size,
        chunk_memory_gb=chunk_memory_gb,
        expected_fit_chunks=expected_fit_chunks,
    )
    chosen = _selected_chunks(chunks, pairs, fit_chunk_ids)
    target_ids = [t for chunk in chosen for t in chunk]
    target_code = {t: i for i, t in enumerate(target_ids)}
    # As the reference run: such a target's pairs are NaN, and it stays in its chunk.
    spanned = targets_in_span(
        covariate_matrix, {t: grna_target_cells[t] for t in pd.unique(pairs["grna_target"])}
    )

    gene_order = {g: i for i, g in enumerate(gene_ids)}
    pair_genes = pairs["response_id"].to_numpy()
    gene_rank = np.fromiter((gene_order[g] for g in pair_genes), dtype=np.int64, count=len(pairs))
    codes = np.fromiter(
        (target_code[t] for t in pairs["grna_target"].to_numpy()), dtype=np.int32, count=len(pairs)
    )
    modes = (
        np.fromiter((PAIR_MODES.index(m) for m in pairs["mode"]), dtype=np.int8, count=len(pairs))
        if "mode" in pairs.columns
        else np.zeros(len(pairs), dtype=np.int8)
    )
    order = np.argsort(gene_rank, kind="stable")
    bounds = np.flatnonzero(np.diff(gene_rank[order])) + 1
    groups = np.split(order, bounds) if len(order) else []
    genes = [gene_ids[gene_rank[g[0]]] for g in groups]

    entropy = resolve_entropy(seed)
    side_code = _SIDE_CODES[side]
    B2, B3 = _SKEW_NORMAL_B2, 0

    design = _design_for(glm_solver, covariate_matrix)
    if hasattr(gene_fits, "fingerprint"):
        _check_data(
            gene_fits.fingerprint,
            response_matrix=response_matrix,
            gene_ids=gene_ids,
            covariate_matrix=covariate_matrix,
            glm_solver=glm_solver,
            genes=[g for g in genes if g in gene_fits.fits],
        )
    with _ENGINE_LOCK, _engine_blas_limit():
        xo = None if design is not None else x_outer_flat(covariate_matrix)
        fits = _gene_fits_for(
            genes, gene_fits, response_matrix, gene_ids, covariate_matrix, xo, design, n_jobs
        )
        targets = _prepare_targets(
            chosen,
            grna_target_cells,
            covariate_matrix,
            B2=B2,
            B3=B3,
            entropy=entropy,
            block_rows=int(block_rows),
            xo=xo,
            design=design,
            n_jobs=n_jobs,
            fit_concurrency=fit_concurrency,
        )
        # Every fit is done; the (n, p^2) array is the largest object alive and
        # forked workers would otherwise carry it.
        del xo
        gc.collect()

        token = next(_TRANS_STATE_SEQ)
        jobs = [
            (token, gene, rows, codes[rows], modes[rows])
            for gene, rows in zip(genes, groups, strict=True)
        ]
        _TRANS_STATE[token] = dict(
            response_matrix=response_matrix,
            covariate_matrix=covariate_matrix,
            gene_fits=fits,
            gene_row_index={g: gene_order[g] for g in genes},
            targets=targets,
            target_ids=target_ids,
            spanned=spanned,
            side_code=side_code,
            B2=B2,
            B3=B3,
            p_exact_below=float(p_exact_below),
            block_rows=int(block_rows),
        )
        try:
            yield from _map_jobs(_trans_gene_job, jobs, n_jobs, backend=backend)
        finally:
            _TRANS_STATE.pop(token, None)


def trans_results_frame(pairs: pd.DataFrame, results: Iterable[TransGeneResult]) -> pd.DataFrame:
    """The yielded results as one table, in `pairs` order.

    Columns: `response_id`, `grna_target`, `p_value`, `fold_change`,
    `se_fold_change`, `z_orig`, `stage`, `screen`, `p_value_floor`,
    `n_null_evaluated`, and the `audit_*` columns when any pair was audited.
    """
    results = list(results)
    n = len(pairs)
    cols = _new_columns(n)
    seen = np.zeros(n, dtype=bool)
    for r in results:
        seen[r.rows] = True
        for name, values in r.columns.items():
            cols[name][r.rows] = values
    if not seen.all():
        raise ValueError(f"{int((~seen).sum())} pair(s) have no result")
    out = pairs.loc[:, ["response_id", "grna_target"]].reset_index(drop=True)
    for name in _FLOAT_COLUMNS[:5]:
        out[name] = cols[name]
    out["screen"] = np.asarray(SCREEN_CODES, dtype=object)[cols["screen"]]
    out["p_value_floor"] = cols["p_value_floor"]
    out["n_null_evaluated"] = cols["n_null_evaluated"]
    audited = "mode" in pairs.columns and (pairs["mode"].to_numpy() == "audit").any()
    if audited:
        for name in _AUDIT_COLUMNS:
            out[name] = cols[name]
    return out
