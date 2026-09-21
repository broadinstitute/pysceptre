# Design decisions

Why the port differs from R where it does, and what was measured to decide.

This page exists so the source does not have to carry it. A docstring says
what a function does; the reasoning behind a choice, and the numbers that
settled it, live here. Measurements are a property of a machine and a
dataset, not of the code — keeping them in source means an unchanged
function accumulates commits that only edit prose, and `git log -p` stops
answering "when did this last change?".

Where a choice in the source would otherwise look arbitrary, the code
carries a one-line pointer to a section below rather than the argument
itself.

!!! info "Benchmark figures"
    Numbers here name the machine they were taken on. The full tables, the
    R-versus-pysceptre comparison and the methodology live with the
    manuscript, in a separate repository — this one is strictly the tool.
    Until the preprint is out that repository is private, so this page
    carries the figures a reader needs to judge the decision rather than
    deep-linking to a page they cannot open.

## Choosing a parallel backend

`n_jobs` distributes the per-pair tests over the genes of an
already-drawn target chunk — not over chunks. Two constraints force that.
The RNG is consumed target-by-target in the parent, so draws happen in the
same order at any worker count and **every p-value is independent of
`n_jobs`**; chunk-level parallelism would have moved all of them and forced
a re-validation against R for no scientific gain. And a chunk's draws are
shared rather than duplicated, so memory grows by roughly one gene's working
arrays per worker instead of by `chunk_memory_gb` per worker.

The backend itself follows the **mechanism, not the platform**.
`parallel_backend()` gives the platform default — `fork` on Linux, threads
elsewhere — and `gene_job_backend()` overrides it to threads for the
per-chunk gene pool once a run exceeds eight chunks.

Measured on an `n2-standard-16` at 8 workers, day0 (34,886 pairs,
567,690 cells):

| | fork | thread |
|---|---|---|
| CRT | 679.7 s, 9.46 GB | **601.9 s, 6.93 GB** |
| permutations | **70.3 s, 4.61 GB** | 143.8 s, 4.55 GB |

Threads win the CRT and lose permutations, and the reason is structural.
The CRT builds a worker pool **per chunk** — 217 of them at the default
budget — so fork cost dominates. Permutations run as a single chunk and pay
that cost once, which leaves the GIL as their binding constraint. Hence the
chunk-count threshold rather than a platform check.

Threads win the CRT while showing *lower* occupancy, 4.56 cores against
5.10. Occupancy is a diagnostic, not a goal: fork spends cores on work that
threads never do.

The threshold sits at eight chunks. Only the two points above are measured,
so it is **placed in the gap between them rather than fitted** — anything
from a handful of chunks to a hundred classifies both cases identically, and
pretending to more precision than two points support would be false.

`PYSCEPTRE_BACKEND` overrides the platform default, to `fork` or `thread`,
so a caller who has measured their own machine can act on it. It cannot make
macOS safe for `fork`: that restriction is about deadlocking after
Accelerate, not about speed, and the override is refused with a warning.

macOS keeps threads throughout regardless of chunk count, because `fork`
after Apple's Accelerate BLAS has run can deadlock — the Grand Central
Dispatch pools it relies on are not fork-safe. `spawn` is not an
alternative: a chunk's draws run to hundreds of MB and would be pickled per
worker per chunk, costing more than the parallelism saves.

!!! warning "A superseded measurement"
    The rule above replaces an earlier one that chose processes on Linux
    unconditionally, on the strength of `1.85x with threads, 3.54x with
    processes`. That was measured on a per-pair gather (`D[:, flat_idxs]`
    plus `reduceat`) which held the GIL and **no longer exists** — the
    statistic is now a sparse matmul, and permutations reach it through a
    prefix scan. Both release the GIL. The old figure survived in a
    docstring for months across a change that reversed it, which is the
    reason this page exists.

## BLAS threading in the IRLS loop

`glm/irls.py` runs its fits inside
`threadpool_limits(limits=1, user_api="blas")`. This is deliberate. The IRLS
matmuls are thin — `p` is a handful of covariates — and called in a tight
loop, so multi-threaded OpenBLAS loses to the single-threaded path:
**68.7 s to 17.7 s** for a 150-column batch over 100k cells, on x86_64 Linux
with OpenBLAS. The limit is scoped to the module so it does not clobber BLAS
threading process-wide.

It is a **silent no-op on macOS**, where numpy is built against Apple
Accelerate, which `threadpoolctl` cannot introspect: `threadpool_info()`
returns `[]` and a matmul takes the same time inside and outside the context
manager. Local benchmarks on a Mac therefore do not exercise this path while
CI does, so any measurement of it has to name the BLAS it was taken on.

## The CRT sampler draws with replacement

R places treated cells by a without-replacement Fisher-Yates shuffle. The
port draws **with** replacement and deduplicates with `np.unique`. This is an
approximation, and an intentional one: the number of treated cells for a
target is far smaller than the number of resamples, so collisions are rare.

The alternative — a dense `(n_cells, B)` matrix, the straightforward
translation — is what made the first port unusably slow at real scale. The
sparse draw is the reason the CRT path finishes at all.

## Resampling budgets, and why `B3` is zero

`B1`, `B2` and `B3` are derived in `api.py::_resampling_budget`, porting R's
own sizing from `s4_analysis_functs_1.R`. `B1 = 499` always; `skew_normal`
gives `(4999, 0)`; `no_approximation` gives
`(0, ceil(mult * n_pairs / alpha))`.

`B3 = 0` on the default `skew_normal` path is **parity with R**, not a stub.
R only uses `B3 = 24999` for permutations.

A related trap: `stage == 3` is still reachable on the default path. It is
entered whenever the skew-normal fit is *rejected*, regardless of `B3`, and
the p-value then comes from the already-drawn `B2 = 4999` statistics. Only
`stage == 2` means a skew-normal fit was actually used.

## Pairwise QC appears in two different shapes

Cell-level and gRNA-level QC are out of scope, but the calibration and power
checks construct or receive their own pairs, so both have to decide which
pairs are testable at all. Pairwise nonzero-count filtering therefore lives
in `pipeline/pairwise_qc.py`, shared by both — and the two use it in
opposite directions.

The calibration check **samples** its pairs, so it avoids ones that would
fail; every returned row passed by construction and there is no `pass_qc`
column. A positive control is a specific claim about a specific pair, so the
power check **reports** failures with a NaN result: dropping them would
overstate power by hiding exactly the controls the screen had too few cells
to test. Discovery behaves like the power check.

On one real screen the difference is visible in the shape of the output:
a discovery result of 34,256 rows of which 1,121 fail, against a calibration
result of 33,135 rows with no `pass_qc` column at all.

## Randomness does not reproduce R bit-for-bit

`sceptre` seeds `boost::mt19937`; this uses `numpy.random.Generator`.
Validation matches **distributions**, not draws, and chasing exact agreement
is not a goal.

One consequence is easy to mistake for a bug. R's calibration *pair
selection* is not reproducible even against itself: nothing in that path
calls `set.seed` and `sceptre_object` has no seed slot. Two R runs on the
same object produced the identical 100 gRNA groups but shared only about 20
of each group's 331 genes. Pair-by-pair comparison against R therefore has
to feed R's own `grna_target` column back through
`calibration.negative_control_pairs_from_names`; the constructor itself is
validated distributionally, which is what a calibration check measures.
