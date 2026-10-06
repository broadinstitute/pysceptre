"""Port of `crt_index_sampler_fast` / `crt_index_sampler` (generate_samples_functions.cpp).

For each cell j, the number of the B synthetic (resampled) treatment sets
that include it, M_j, is Binomial(B, fitted_probabilities[j])-distributed,
and conditional on M_j, cell j is placed into a uniformly random size-M_j
subset of the B sets. R's `crt_index_sampler_fast` implements this via a
Binomial draw + Fisher-Yates without-replacement placement per cell; that is
*exactly* distributionally equivalent to the much simpler "naive" algorithm
of drawing B independent Bernoulli(p_j) coin flips per cell
(`crt_index_sampler`, also in the source) -- R only bothers with the
WOR-based algorithm because it's faster in a per-cell C++ loop when p_j is
small.

That performance argument doesn't transfer to a Python port: a per-cell
Python loop (whether doing the naive Bernoulli-per-draw or the WOR-based
"fast" algorithm) is far too slow at real scale (~3,000 gRNA targets x
~590,000 cells x ~5,500 resamples -- measured in the hours here). But
naively vectorizing the *naive* algorithm (materializing a dense
(n_cells, B) boolean matrix) is no better: that's O(n_cells * B) random
draws regardless of how few cells actually end up included anywhere, which
at real scale is billions of draws per target x thousands of targets.

The actual fix is to generate only the sparse set of (cell, draw) inclusions
that occur, exploiting that E[inclusions] = B * sum(fitted_probabilities) =
B * n_trt (by the intercept property of logistic-regression MLE fits,
sum(fitted_probabilities) exactly equals the observed treated-cell count)
-- a few million entries per target, not billions:
  1. Draw M_j ~ Binomial(B, p_j) for every cell at once (vectorized, O(n_cells)).
  2. Generate sum(M_j) candidate (cell, slot) pairs by sampling each cell's
     M_j slots *with* replacement from {0,...,B-1}, an approximation to the
     WOR placement R's algorithm uses. **Nothing is de-duplicated**: a cell
     that lands on the same slot twice is listed, and counted, twice in that
     resample. About half the inclusion probability of listings repeat, so
     the approximation is only used where targets are a tiny share of the
     cells; `crt_index_sampler` chooses (see docs/design.md, "The CRT sampler
     draws with replacement").
  3. Group the resulting (cell, slot) pairs by slot. Since slot values are
     bounded integers in [0, B), this is a counting-sort problem (O(n + B)),
     not a general comparison-sort one -- profiling showed a numpy
     `argsort`-based grouping was the dominant cost of this whole module at
     real dataset scale, so the grouping step is instead a `numba`-jitted
     two-pass counting sort (count, then scatter) when numba is available,
     falling back to the `argsort`-based version otherwise.

R seeds `boost::mt19937` with a fixed literal seed (4) on every call, making
its draws exactly reproducible *within R*. Reproducing that exact bit stream
from Python is not attempted (different RNG algorithm and seeding scheme) --
validate the distribution of draws instead (per-cell inclusion rate,
resulting null-statistic distribution), not bit-for-bit equality. See
tests/validation/test_crt_sampler.py.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit

    @njit(cache=True)
    def _counting_sort_group(
        cell_ids: np.ndarray, positions: np.ndarray, B: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = cell_ids.shape[0]
        counts = np.zeros(B, dtype=np.int64)
        for i in range(n):
            counts[positions[i]] += 1
        boundaries = np.cumsum(counts)
        starts = boundaries - counts
        write_pos = starts.copy()
        sorted_cells = np.empty(n, dtype=np.int64)
        for i in range(n):
            b = positions[i]
            sorted_cells[write_pos[b]] = cell_ids[i]
            write_pos[b] += 1
        return sorted_cells, starts, boundaries

    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


def _group_by_position_numpy(
    cell_ids: np.ndarray, positions: np.ndarray, B: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Stability is not needed: downstream consumers only ever sum over each
    # bucket's cells (order-invariant), so the default (unstable, much
    # faster) sort algorithm is used rather than "stable" (mergesort).
    order = np.argsort(positions)
    sorted_cells = cell_ids[order]
    sorted_positions = positions[order]
    counts = np.bincount(sorted_positions, minlength=B)
    boundaries = np.cumsum(counts)
    starts = boundaries - counts
    return sorted_cells, starts, boundaries


def crt_index_sampler_fast(
    fitted_probabilities: np.ndarray, B: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Returns a length-B list of 0-based integer arrays: synthetic_idxs[b] holds
    the cells treated in synthetic draw b, and may list a cell more than once.
    See the module docstring for the sparse-generation strategy (O(B * n_trt)
    rather than O(B * n_cells)) and for when that repeat matters."""
    n_cells = fitted_probabilities.size
    M = rng.binomial(B, fitted_probabilities)
    total = int(M.sum())
    if total == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(B)]

    cell_ids = np.repeat(np.arange(n_cells, dtype=np.int64), M)
    positions = rng.integers(0, B, size=total)
    # No de-duplication of (cell, position) pairs: a collision (the same cell
    # landing on the same slot twice via this with-replacement approximation
    # to WOR) is already rare given M_j << B in practice (see module
    # docstring), and an exact global de-dup via np.unique on the combined
    # (cell, position) key is the dominant cost at real dataset scale for a
    # negligible correctness gain -- skip it.

    if _HAVE_NUMBA:
        sorted_cells, starts, boundaries = _counting_sort_group(
            cell_ids, positions.astype(np.int64), B
        )
    else:
        sorted_cells, starts, boundaries = _group_by_position_numpy(cell_ids, positions, B)

    return [sorted_cells[starts[b] : boundaries[b]] for b in range(B)]


def crt_index_sampler_exact(
    fitted_probabilities: np.ndarray, B: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """`crt_index_sampler_fast` without its approximation, for large probabilities.

    Same return shape: a length-B list of 0-based cell-index arrays, each
    ascending. Cell j is in each draw independently with probability
    `fitted_probabilities[j]`, which is the law of R's binomial count plus
    without-replacement placement, and no cell is ever listed twice in a draw.
    Work stays proportional to the number of inclusions: each cell's draw
    positions are the partial sums of geometric gaps.

    For the NT-cells control group, where a target is a large share of its
    combined cells. See docs/design.md, "The CRT against the NT cells".
    """
    p = np.clip(np.asarray(fitted_probabilities, dtype=float), 0.0, 1.0)
    if B <= 0:
        return []
    live = np.flatnonzero(p > 0)
    cells_out: list[np.ndarray] = []
    draws_out: list[np.ndarray] = []
    # Every cell starts before draw 0, then walks forward until it passes B - 1.
    cursor = np.full(live.size, -1, dtype=np.int64)

    def walk(todo: np.ndarray) -> None:
        pj = p[live[todo]]
        mean = B * pj
        n_gaps = np.ceil(mean + 6.0 * np.sqrt(mean * (1.0 - pj)) + 10.0).astype(np.int64)
        owner = np.repeat(todo, n_gaps)
        # Capped at B + 1, which already overshoots from the -1 start.
        gaps = np.minimum(rng.geometric(np.repeat(pj, n_gaps)), B + 1)
        ends = np.cumsum(n_gaps)
        steps = np.cumsum(gaps)
        steps -= np.repeat(steps[ends - n_gaps] - gaps[ends - n_gaps], n_gaps)
        position = np.repeat(cursor[todo], n_gaps) + steps
        keep = position < B
        cells_out.append(live[owner[keep]])
        draws_out.append(position[keep])
        cursor[todo] = position[ends - 1]

    # The first pass in blocks of cells, to bound the gap arrays; consecutive
    # blocks consume the stream exactly as one call would.
    first = np.ceil(B * p[live] + 6.0 * np.sqrt(B * p[live] * (1.0 - p[live])) + 10.0)
    bounds = np.searchsorted(
        np.cumsum(first), np.arange(1, 1 + int(first.sum() // _GAPS_PER_BLOCK)) * _GAPS_PER_BLOCK
    )
    for block in np.split(np.arange(live.size), bounds):
        if block.size:
            walk(block)
    todo = np.flatnonzero(cursor < B)
    passes = 1
    while todo.size:
        walk(todo)
        todo = todo[cursor[todo] < B]
        passes += 1

    cell_ids = np.concatenate(cells_out) if cells_out else np.empty(0, dtype=np.int64)
    positions = np.concatenate(draws_out) if draws_out else np.empty(0, dtype=np.int64)
    if cell_ids.size == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(B)]
    # The first pass leaves cells ascending; only later passes can disorder them.
    if passes > 1:
        order = np.argsort(cell_ids, kind="stable")
        cell_ids, positions = cell_ids[order], positions[order]
    # Input ordered by cell, so the counting sort leaves each draw ascending.
    if _HAVE_NUMBA:
        sorted_cells, starts, boundaries = _counting_sort_group(cell_ids, positions, B)
    else:
        order = np.argsort(positions, kind="stable")
        sorted_cells = cell_ids[order]
        counts = np.bincount(positions, minlength=B)
        boundaries = np.cumsum(counts)
        starts = boundaries - counts
    return [sorted_cells[starts[b] : boundaries[b]] for b in range(B)]


# A target above this share of its cells gets the exact sampler: the fast one
# repeats about half the share of its listings. See docs/design.md, "The CRT
# sampler draws with replacement".
EXACT_SAMPLER_ABOVE_SHARE = 2e-3

# Geometric gaps drawn per block of cells in the exact sampler's first pass.
_GAPS_PER_BLOCK = 4_000_000


def crt_index_sampler(
    fitted_probabilities: np.ndarray, B: int, rng: np.random.Generator, n_trt: int
) -> list[np.ndarray]:
    """The CRT draw for a target with `n_trt` treated cells among `fitted_probabilities`.

    `crt_index_sampler_exact` when the target is more than
    `EXACT_SAMPLER_ABOVE_SHARE` of the cells, `crt_index_sampler_fast`
    otherwise. The choice depends only on the two counts, never on fitted
    values, so it cannot change with how a run is chunked.
    """
    if n_trt > EXACT_SAMPLER_ABOVE_SHARE * fitted_probabilities.size:
        return crt_index_sampler_exact(fitted_probabilities, B, rng)
    return crt_index_sampler_fast(fitted_probabilities, B, rng)


def crt_index_sampler_naive(
    fitted_probabilities: np.ndarray, B: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Reference implementation used only for cross-checking (see
    test_crt_sampler.py): draws the full (B, n_cells) boolean matrix in one
    shot rather than chunking it -- simpler, but O(B * n_cells) memory, so
    not suitable at real dataset scale."""
    n_cells = fitted_probabilities.size
    draws = rng.random((B, n_cells)) < fitted_probabilities[None, :]
    return [np.flatnonzero(draws[b]) for b in range(B)]
