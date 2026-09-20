"""Permutation resampling, sceptre's alternative to the CRT.

Where the CRT draws each target's synthetic treated set from *that target's*
fitted probabilities, permutation resampling draws one set of random subsets
and reuses it for every target. For the complement control group, sceptre's
rule is::

    M <- max(sapply(grna_group_idxs, length))
    fisher_yates_samlper(n_tot = n_cells, M = M, B = B)

One sampler call for the whole analysis: B subsets of size M, where M is the
largest target's cell count. A target with `n_trt` cells uses the **first
`n_trt`** entries of each subset, which is valid because each subset is a
uniformly random ordering of distinct cells, so any prefix is a uniformly
random subset of that size.

**This cannot be composition-invariant, and the reason is structural.** The
shared draws are sized by `M`, the largest target in the dataset. Add a
target larger than all the others and `M` grows, every subset changes, and
every result in the run moves. Add a smaller one, or add pairs for targets
already present, and nothing changes -- R takes the maximum over the whole
target set rather than over the tested pairs, which at least keeps the pair
list out of it.

Contrast the CRT path, where each target seeds its own stream from its own
name and is invariant to everything: the pair list, the target set, the
chunking and the worker count (see
`pipeline/discovery.py::target_seed_sequence`).

That is not a defect of this implementation; it follows from sharing one
draw set across targets, which is what makes permutations cheap. Callers who
need results that survive extending an analysis should use the CRT. The
tradeoff belongs to the caller, so it is documented rather than decided
here.

The cost profile is the mirror of the CRT's. Sampling is nearly free -- one
call instead of one per target -- and, less obviously, **the per-target
logistic fit disappears with it**: those fitted probabilities exist only for
`crt_index_sampler_fast` to draw from, and permutation draws never consult
them, so `fit_all_targets` skips the fit entirely. R draws the same line,
calling `perform_grna_precomputation` from `crt_glm_factored_out` and
`discovery_ntcells_crt` but not from `perm_test_glm_factored_out`.

Against that, R pairs permutations with `B3 = 24,999` where the CRT gets
`0`, so the escalation batch is far larger -- but only for pairs that
escalate, which on a real run is 1,265 of 34,886.
"""

from __future__ import annotations

import numpy as np


def permutation_draws(n_cells: int, m: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """`b` random subsets of `m` distinct cells, as a `(b, m)` index array.

    One `Generator.choice(..., replace=False)` per draw. That looks like the
    naive option and is in fact the fast one: numpy samples without
    replacement in C, so the Python loop costs one call per draw rather than
    one per cell. Measured 0.28 s for 30,497 draws of 692 cells from 567,690.

    Two cleverer approaches were tried and rejected. Rejection sampling --
    draw with replacement, redraw rows containing a repeat -- is elegant when
    `m` is tiny relative to `n_cells`, and collapses when it is not: at
    `m = 120` from 1,500 cells the expected collision count per row is 4.8
    and fewer than 1% of rows survive, so it cannot serve as a general
    sampler. Ranking `rng.random((b, n_cells))` is `O(b * n_cells)`, which is
    17 billion values at real scale for 21 million wanted.

    **Each row is a random *ordering*, deliberately not sorted.** That is
    what lets every target share these draws: a prefix of a uniformly random
    ordering of distinct cells is a uniformly random subset of that length,
    so a target with `n_trt` cells takes `row[:n_trt]`. Sorting here would
    break that silently -- prefixes would be the lowest-numbered cells rather
    than random ones -- so a test asserts the rows are not sorted.

    Reimplemented rather than ported: sceptre's `fisher_yates_samlper` is C++
    reached through `.Call`, and the draws could not match bit-for-bit
    anyway, since the two languages seed different generators.
    """
    if m > n_cells:
        raise ValueError(
            f"cannot draw {m} distinct cells from {n_cells}; the largest gRNA target "
            "has more cells than the dataset has"
        )
    if m < 0 or b < 0:
        raise ValueError(f"m and b must be non-negative, got m={m}, b={b}")
    if b == 0 or m == 0:
        return np.empty((b, m), dtype=np.int64)

    out = np.empty((b, m), dtype=np.int64)
    for i in range(b):
        out[i] = rng.choice(n_cells, size=m, replace=False)
    return out


def draws_for_target(perms: np.ndarray, n_trt: int) -> list[np.ndarray]:
    """Each permutation's first `n_trt` cells, as one index array per draw.

    Matches the shape the CRT sampler returns, so everything downstream --
    `draws_to_matrix`, the staged test, the statistic -- is shared between
    the two mechanisms rather than duplicated.

    **Deliberately not sorted**, matching R, which reads `curr_vect[j]` for
    `j < n_trt` straight off the shared array and sorts nothing. The prefix
    of a random ordering is already a uniformly random subset; sorting it
    only fixes the order the statistic sums in. An earlier version sorted so
    that the CSR would be canonical, which cost 1.3M `np.sort` calls -- 7.9 s
    of a 180 s profile, about 5% -- for no change to the mathematics.

    scipy does not require sorted indices: `csr_matvecs` sums a row's
    nonzeros in stored order. It does mean the sum is taken in a different
    order, so permutation results moved once by ~1e-14 when this changed.
    That is floating-point associativity, not a different estimate.
    """
    if n_trt > perms.shape[1]:
        raise ValueError(
            f"target needs {n_trt} cells but the shared draws hold only "
            f"{perms.shape[1]}; they are sized by the largest target, so this "
            "means the draws were built for a different target set"
        )
    return [row[:n_trt] for row in perms]
