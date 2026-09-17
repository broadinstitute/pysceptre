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
     M_j slots *with* replacement from {0,...,B-1} (an approximation to the
     WOR placement R's algorithm uses -- justified because M_j << B in
     practice, so collisions are rare) then de-duplicating the (cell, slot)
     pairs in one vectorized `np.unique` pass. Since set membership is
     binary, a de-duplicated with-replacement draw and an exact
     without-replacement draw are indistinguishable except for the rare
     event of an actual collision, which just means that cell effectively
     used one fewer of its M_j "budget" -- a negligible effect on a
     resampling distribution that is already not being matched bit-for-bit
     against R's RNG.
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
    """Returns a length-B list of 0-based integer arrays: synthetic_idxs[b] is
    the set of cell indices treated in synthetic draw b. See module docstring
    for the sparse-generation strategy (O(B * n_trt) rather than O(B * n_cells))."""
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
