"""From gRNA UMI counts to the cells and weights the dose test takes.

Not from sceptre. See `docs/design.md`, "The dose test".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from ._common import as_count_matrix, check_ids

__all__ = ["DoseFloor", "DoseWeights", "dose_ramp", "dose_weights", "estimate_dose_floor"]

_NON_TARGETING = "non-targeting"


@dataclass(frozen=True)
class DoseFloor:
    """The noise share of gRNA entries at each count, and the floor it gives.

    Attributes:
        floor: The largest count whose entries are at least `threshold` noise; 0 if none is.
        noise_share: Indexed by count `k`, the share of the entries with exactly `k` UMIs whose
            spread over gRNAs matches the single-UMI entries rather than the real ones.
        n_entries: Indexed by count `k`, how many (gRNA, cell) entries have exactly `k` UMIs.
    """

    floor: int
    noise_share: pd.Series
    n_entries: pd.Series


def _mixture_share(n: np.ndarray, noise: np.ndarray, real: np.ndarray) -> float:
    """Maximum-likelihood `a` in `n ~ Poisson(N (a noise + (1 - a) real))`, by bisection on the
    score, which is decreasing in `a`."""
    use = (n > 0) & ((noise > 0) | (real > 0))
    n, noise, real = n[use], noise[use], real[use]
    if n.size == 0:
        return float("nan")

    def score(a):
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(np.sum(n * (noise - real) / (a * noise + (1 - a) * real)))

    if score(1.0) >= 0:
        return 1.0
    if score(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if score(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def estimate_dose_floor(
    grna_matrix: sparse.spmatrix | np.ndarray,
    *,
    real_min: int = 50,
    max_count: int = 30,
    threshold: float = 0.5,
) -> DoseFloor:
    """The dose test's floor from the gRNA counts alone: where single-UMI-like noise ends.

    For each count `k` up to `max_count`, the entries with exactly `k` UMIs are spread over the
    gRNAs as a mixture of the single-UMI entries' spread and the spread of entries with at least
    `real_min` UMIs; the mixing share is fitted by maximum likelihood. The floor is the largest
    `k` whose share is at least `threshold`.

    Args:
        grna_matrix: `(n_grnas, n_cells)` raw integer UMI counts, scipy.sparse or dense.
        real_min: Entries with at least this many UMIs stand for real guides.
        max_count: The largest count examined.
        threshold: The noise share a count's entries must reach to fall at or below the floor.

    Raises:
        ValueError: non-integer or negative counts, no single-UMI entries, or none with at least
            `real_min` UMIs.
    """
    if real_min <= 1 or max_count < 2:
        raise ValueError("need real_min > 1 and max_count >= 2")
    c = as_count_matrix(grna_matrix)
    rows, values = c.indices, c.data.astype(np.int64)
    n_grnas = c.shape[0]
    noise = np.bincount(rows[values == 1], minlength=n_grnas).astype(np.float64)
    real = np.bincount(rows[values >= real_min], minlength=n_grnas).astype(np.float64)
    if noise.sum() == 0 or real.sum() == 0:
        raise ValueError(
            f"need entries with 1 UMI and with at least {real_min} UMIs to estimate the floor"
        )
    noise, real = noise / noise.sum(), real / real.sum()
    ks = np.arange(1, max_count + 1)
    shares, counts = [], []
    for k in ks:
        n_k = np.bincount(rows[values == k], minlength=n_grnas).astype(np.float64)
        counts.append(int(n_k.sum()))
        shares.append(_mixture_share(n_k, noise, real))
    share = pd.Series(shares, index=pd.Index(ks, name="count"), name="noise_share")
    above = share.index[share.to_numpy() >= threshold]
    return DoseFloor(
        floor=int(above.max()) if above.size else 0,
        noise_share=share,
        n_entries=pd.Series(counts, index=share.index, name="n_entries"),
    )


@dataclass(frozen=True)
class DoseWeights:
    """Cells and weights for `run_discovery_analysis`, `run_power_check` and `run_calibration_check`.

    Attributes:
        grna_target_cells: `{target: ascending cell indices}`, the cells whose largest count over
            the target's gRNAs is above the floor.
        grna_target_weights: `{target: weights}`, aligned with `grna_target_cells`.
        ntc_grna_cells: `{non-targeting gRNA: ascending cell indices}`, each gRNA on its own.
        ntc_grna_weights: `{non-targeting gRNA: weights}`, aligned with `ntc_grna_cells`.
        floor: Counts at or below this get no weight.
        ceiling: Counts at or above this get weight 1.
    """

    grna_target_cells: dict[str, np.ndarray]
    grna_target_weights: dict[str, np.ndarray]
    ntc_grna_cells: dict[str, np.ndarray]
    ntc_grna_weights: dict[str, np.ndarray]
    floor: float
    ceiling: float


def dose_ramp(counts: np.ndarray, *, floor: float, ceiling: float) -> np.ndarray:
    """The dose weight of each count: `clip(log(c / floor) / log(ceiling / floor), 0, 1)`.

    Zero at or below `floor`, one at or above `ceiling`, linear in `log(c)` between.

    Raises:
        ValueError: unless `0 < floor < ceiling`.
    """
    if not 0 < floor < ceiling:
        raise ValueError(f"need 0 < floor < ceiling, got floor={floor}, ceiling={ceiling}")
    c = np.asarray(counts, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.clip(np.log(c / floor) / np.log(ceiling / floor), 0.0, 1.0)


def _max_over_rows(csr: sparse.csr_matrix, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For the cells with a count in any of `rows`: ascending cell indices, largest count."""
    parts = [slice(csr.indptr[r], csr.indptr[r + 1]) for r in rows]
    if not parts:
        return np.empty(0, dtype=np.int64), np.empty(0)
    cells = np.concatenate([csr.indices[p] for p in parts]).astype(np.int64)
    counts = np.concatenate([csr.data[p] for p in parts])
    if cells.size == 0:
        return cells, counts
    order = np.argsort(cells, kind="stable")
    cells, counts = cells[order], counts[order]
    first = np.flatnonzero(np.r_[True, cells[1:] != cells[:-1]])
    return cells[first], np.maximum.reduceat(counts, first)


def dose_weights(
    grna_matrix: sparse.spmatrix | np.ndarray,
    grna_ids: Sequence[str],
    grna_target_data_frame: pd.DataFrame,
    *,
    floor: float | str = "auto",
    ceiling: float = 500.0,
) -> DoseWeights:
    """Each target's cells and dose weights, from raw gRNA UMI counts.

    A cell's count for a target is its largest count over the target's gRNAs; the cell is
    kept when that count is above `floor` and weighted by `dose_ramp`. A non-targeting gRNA
    is kept on its own, as `run_calibration_check` regroups them.

    Args:
        grna_matrix: `(n_grnas, n_cells)` raw integer UMI counts, scipy.sparse or dense.
        grna_ids: The row names.
        grna_target_data_frame: Columns `grna_id` and `grna_target`; a target of
            "non-targeting" marks a non-targeting gRNA. A gRNA listed here but absent from
            `grna_ids` contributes no cells.
        floor: Counts at or below this get no weight. `"auto"` (default) estimates it from
            `grna_matrix` with `estimate_dose_floor`'s defaults, and at least 1.
        ceiling: Counts at or above this get weight 1.

    Returns:
        A `DoseWeights`, whose `floor` is the one used. Targets keep their order of first
        appearance in the data frame.

    Raises:
        KeyError: a required column is missing.
        ValueError: non-integer or negative counts, or not `0 < floor < ceiling`.
    """
    if isinstance(floor, str):
        if floor != "auto":
            raise ValueError(f"floor must be a number or 'auto', got {floor!r}")
        floor = float(max(1, estimate_dose_floor(grna_matrix).floor))
    dose_ramp(np.empty(0), floor=floor, ceiling=ceiling)  # validates floor and ceiling
    missing = [c for c in ("grna_id", "grna_target") if c not in grna_target_data_frame.columns]
    if missing:
        raise KeyError(f"grna_target_data_frame is missing column(s) {missing}")
    csr = as_count_matrix(grna_matrix).tocsr()
    csr.sort_indices()
    row_of = {g: r for r, g in enumerate(check_ids(grna_ids, csr.shape[0], "grna_ids"))}

    def cells_and_weights(rows):
        cells, counts = _max_over_rows(csr, np.asarray(rows, dtype=np.int64))
        keep = counts > floor
        return cells[keep], dose_ramp(counts[keep], floor=floor, ceiling=ceiling)

    design = grna_target_data_frame[["grna_id", "grna_target"]].astype(str)
    target_cells, target_weights, ntc_cells, ntc_weights = {}, {}, {}, {}
    for target, group in design.groupby("grna_target", sort=False):
        rows = [row_of[g] for g in group["grna_id"] if g in row_of]
        if target == _NON_TARGETING:
            for g in dict.fromkeys(group["grna_id"]):
                ntc_cells[g], ntc_weights[g] = cells_and_weights([row_of[g]] if g in row_of else [])
        else:
            target_cells[target], target_weights[target] = cells_and_weights(rows)
    return DoseWeights(target_cells, target_weights, ntc_cells, ntc_weights, floor, ceiling)
