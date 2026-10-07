# Design decisions

Why the port differs from R where it does, and what was measured to decide.

This page exists so the source does not have to carry it. A docstring says
what a function does; the reasoning behind a choice, and the numbers that
settled it, live here. Measurements are a property of a machine and a
dataset, not of the code -- keeping them in source means an unchanged
function accumulates commits that only edit prose, and `git log -p` stops
answering "when did this last change?".

Where a choice in the source would otherwise look arbitrary, the code
carries a one-line pointer to a section below rather than the argument
itself.

!!! info "Benchmark figures"
    Numbers here name the machine they were taken on. The full tables, the
    R-versus-pysceptre comparison and the methodology live with the
    manuscript, in a separate repository -- this one is strictly the tool.
    Until the preprint is out that repository is private, so this page
    carries the figures a reader needs to judge the decision rather than
    deep-linking to a page they cannot open.

## Choosing a parallel backend

`n_jobs` distributes the per-pair tests over the genes of an
already-drawn target chunk -- not over chunks. Two constraints force that.
The RNG is consumed target-by-target in the parent, so draws happen in the
same order at any worker count and **every p-value is independent of
`n_jobs`**; chunk-level parallelism would have moved all of them and forced
a re-validation against R for no scientific gain. And a chunk's draws are
shared rather than duplicated, so memory grows by roughly one gene's working
arrays per worker instead of by `chunk_memory_gb` per worker.

The backend itself follows the **mechanism, not the platform**.
`parallel_backend()` gives the platform default -- `fork` on Linux, threads
elsewhere -- and `gene_job_backend()` overrides it to threads for the
per-chunk gene pool once a run exceeds eight chunks.

Measured on an `n2-standard-16` at 8 workers, day0 (34,886 pairs,
567,690 cells):

| | fork | thread |
|---|---|---|
| CRT | 679.7 s, 9.46 GB | **601.9 s, 6.93 GB** |
| permutations | **70.3 s, 4.61 GB** | 143.8 s, 4.55 GB |

Threads win the CRT and lose permutations, and the reason is structural.
The CRT builds a worker pool **per chunk** -- 217 of them at the default
budget -- so fork cost dominates. Permutations run as a single chunk and pay
that cost once, which leaves the GIL as their binding constraint. Hence the
chunk-count threshold rather than a platform check.

Threads win the CRT while showing *lower* occupancy, 4.56 cores against
5.10. Occupancy is a diagnostic, not a goal: fork spends cores on work that
threads never do.

The threshold sits at eight chunks. Only the two points above are measured,
so it is **placed in the gap between them rather than fitted** -- anything
from a handful of chunks to a hundred classifies both cases identically, and
pretending to more precision than two points support would be false.

`PYSCEPTRE_BACKEND` overrides the platform default, to `fork` or `thread`,
so a caller who has measured their own machine can act on it. It cannot make
macOS safe for `fork`: that restriction is about deadlocking after
Accelerate, not about speed, and the override is refused with a warning.

macOS keeps threads throughout regardless of chunk count, because `fork`
after Apple's Accelerate BLAS has run can deadlock -- the Grand Central
Dispatch pools it relies on are not fork-safe. `spawn` is not an
alternative: a chunk's draws run to hundreds of MB and would be pickled per
worker per chunk, costing more than the parallelism saves.

!!! warning "A superseded measurement"
    The rule above replaces an earlier one that chose processes on Linux
    unconditionally, on the strength of `1.85x with threads, 3.54x with
    processes`. That was measured on a per-pair gather (`D[:, flat_idxs]`
    plus `reduceat`) which held the GIL and **no longer exists** -- the
    statistic is now a sparse matmul, which permutations reach through a
    prefix scan where that is the cheaper route. Both release the GIL. The old figure survived in a
    docstring for months across a change that reversed it, which is the
    reason this page exists.

## BLAS threading in the IRLS loop

`glm/irls.py` runs its fits inside
`threadpool_limits(limits=1, user_api="blas")`. This is deliberate. The IRLS
matmuls are thin -- `p` is a handful of covariates -- and called in a tight
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

R decides how many of the `B` resamples include cell `j` with a binomial draw,
`M_j ~ Binomial(B, p_j)`, then places the cell into `M_j` distinct resamples
by a without-replacement Fisher-Yates shuffle. `crt_index_sampler_fast` keeps
the binomial count but places the cell **with** replacement, and does not
deduplicate: a cell that lands on the same resample twice is listed, and
counted, twice in it. Deduplicating the `(cell, resample)` pairs with
`np.unique` was the dominant cost of the sampler at real scale, so it was
dropped.

This is an approximation, and it is only accurate where inclusion
probabilities are tiny. The expected number of repeats in a resample is about
`sum(p_j ** 2) / 2` against `sum(p_j)` cells listed, so the repeated fraction
is about half the typical inclusion probability, and `p_j` averages a target's
share of the cells it is tested on. `scripts/measure_nt_cells_crt_sampler.py`
measured the repeated share at the dimensions of four screens:

| screen and control group | target share of the cells | listings repeated |
|---|---|---|
| day0, complement | 0.07% | 0.035% |
| Gasperini (sceptredata), complement | 2.7% | 1.6% |
| Papalexi (sceptredata), complement | 3.8% | 2.3% |
| Papalexi (sceptredata), NT cells | 25% | 14.0% |

A repeated cell inflates the null statistics' variance, so the test turns
conservative, and at the larger shares it shows: on a synthetic low-MOI screen
with no effects the fast sampler put 5.6% of null p-values below 0.1 where
10% belong (see [The CRT against the NT cells](#the-crt-against-the-nt-cells)).

**So the sampler is chosen per target, from counts.** `crt_index_sampler`
takes `crt_index_sampler_exact` when a target is more than 0.2% of the cells it
is tested on (`EXACT_SAMPLER_ABOVE_SHARE`), where repeats would pass 0.1%, and
the fast sampler below that. The exact sampler costs about five times as much
per target, which is why it is not used everywhere. The rule reads the treated
count and the number of cells, never the fitted probabilities: those are not
bit-identical across chunk widths, and a target near the boundary could
otherwise change sampler with `target_chunk_size`.

day0 sits entirely below the boundary -- its largest target is 692 of
567,690 cells, and its calibration check's synthetic targets, 15
non-targeting gRNAs of a median 32 cells each, reach at most 0.12% -- so its
results are bit-identical to earlier releases. A screen's CRT p-values move
for every target above 0.2% of its cells, which on sceptredata is every target
of both screens.

The alternative -- a dense `(n_cells, B)` matrix, the straightforward
translation -- is what made the first port unusably slow at real scale. The
sparse draw is the reason the CRT path finishes at all, and the exact sampler
keeps it sparse.

## A rank check that does not square the condition number

`run_discovery_analysis` refuses a rank-deficient `covariate_matrix` before
any fitting, naming the redundant columns; without the check it fails deep
inside the batched weighted least squares as a bare `LinAlgError`. The same
test, `glm.design.redundant_columns`, runs again on each target's cells
together with the NT cells under the NT-cells control group, where R stops on
an NA coefficient.

Rank comes from the singular values of `R` in a thin QR of the matrix, with the
relative tolerance `numpy.linalg.matrix_rank` uses. The QR is what makes it both
cheap and correct: `R` is `p x p`, so every rank question after it is tiny, and
because `Q` has orthonormal columns the rank of any column subset of `R` equals
that of the same subset of the matrix. Redundant columns are named left to
right, so the first of a collinear set is kept, which is R's convention.

**Not the Gram matrix.** `X.T @ X` looks like the natural `p x p` route and is
the thing the fit depends on -- `Zt_wZ` is it reweighted, and positive weights
cannot restore rank -- but its eigenvalues are the *squares* of the singular
values, so it squares the condition number. A design whose columns span
fourteen orders of magnitude, raw UMI counts beside a small covariate say, is
full rank and the Gram route rejected it.

## Resampling budgets, and why `B3` is zero

`B1`, `B2` and `B3` are derived in `api.py::_resampling_budget`, porting R's
own sizing from `s4_analysis_functs_1.R`. `B1 = 499` always; `skew_normal`
gives `(4999, 0)`; `no_approximation` gives
`(0, ceil(mult * n_pairs / alpha))`.

`B3 = 0` on the default `skew_normal` path is **parity with R**, not a stub.
R only uses `B3 = 24999` for permutations.

Under `no_approximation` the count that sizes `B3` is the tested pairs of the
call. R sizes the calibration check from the requested `n_calibration_pairs`,
where pysceptre uses the pairs actually built, which differ only when fewer
pass QC than were asked for. R sizes the power check from the larger of the
discovery and positive-control counts, because both live on one
`sceptre_object`; `run_power_check` sees only its positive controls, so its
`B3`, and with it the smallest p-value it can report, is smaller.

A related trap: `stage == 3` is still reachable on the default path. It is
entered whenever the skew-normal fit is *rejected*, regardless of `B3`, and
the p-value then comes from the already-drawn `B2 = 4999` statistics. Only
`stage == 2` means a skew-normal fit was actually used.

## When the permutation prefix scan pays, and when it does not

Under permutations every target reads the same draw rows and differs only in
how far along each row it reads, so one gene's segment sums for *every*
target are columns of a single cumulative sum along those rows. That is
`PermutationPrefixSums`: one gather-and-scan of `B * m * (p + 2)`, where `m`
is the width of the shared rows, in place of one sparse matmul of
`B * n_trt * (p + 2)` per target.

**Counting elements makes the scan look unconditionally better, and it is
not.** An element costs far more on the scan than on the matmul, so the two
counts are not comparable as they stand. Measured at one real screen's
dimensions (`n_cells = 131,055`, `m = 406`, `p + 2 = 14`), one target, on an
Apple M4 Max:

| | scan (gather + `cumsum`) | matmul (`draws_to_matrix` + `@`) | ratio |
|---|---|---|---|
| stage 1, `B = 499`   | 13.6 ms  | 1.7 ms + 0.2 ms  | 7.2 |
| stage 2, `B = 4,999` | 111.2 ms | 12.8 ms + 1.7 ms | 7.7 |

The CSR build is counted in because `PermutationSliceDraws` deliberately does
not memoize it. Unlike the IRLS threading figures this ratio is not a BLAS
measurement -- the sparse matmul goes through scipy's sparsetools -- so the
Accelerate caveat above does not apply to it.

So the scan pays only when a gene's targets together demand more prefix than
the scan computes, by that factor: `sum(n_trt) >= SCAN_BREAK_EVEN * m`, which
is `prefix_scan_pays`, applied in `_gene_job` before the scan object is built
at all. `SCAN_BREAK_EVEN` is 8.0, rounded up from the measurements above so
that a borderline gene takes the route that is never much worse.

The two regimes are far apart, which is why the exact constant matters little.
A single-target call sits at a ratio of 1 and can never pay, while day0 pairs
each gene with about 147 targets (34,886 pairs over 237 genes), whose cell
counts put its demand at tens of times `m` however `m` is measured. It keeps
the scan. Reported as
[issue #2](https://github.com/broadinstitute/pysceptre/issues/2), where a
single-target caller measured 182 ms to 87 ms per pair, 2.1x, for taking the
matmul route instead.

**The routes are bit-identical**, so this is a pure cost decision and nothing
about it is user-visible. `test_prefix_sums_match_the_per_target_matmul_bit_for_bit`
asserts the sums agree exactly, and
`test_the_prefix_route_and_the_matmul_route_agree_end_to_end` asserts a whole
analysis does; that second test needs enough targets to clear the gate, or it
compares the matmul route against itself.

Two alternatives were rejected. Replacing `cumsum` with `sum` when only one
prefix is wanted drops one of the two large arrays but keeps the gather, and
still loses: 35.6 ms against the matmul's 14.5 ms at stage 2 above. Exposing
the route as a parameter would put a bit-identical implementation detail in
the API for callers to get wrong.

One thing the gate does not know: **the demand it sees is stage 1's.** Later
stages are read only by the pairs that escalate, so a gene with many targets
of which few escalate keeps a scan those few cannot cover. At real `m` the
memory gate usually declines those stages anyway (`max_bytes` refuses day0's
stage 2 by a factor of 3.6), and the escalating fraction is not knowable
before the pairs are tested, so the over-estimate is left rather than guessed
at.

## Pairwise QC appears in two different shapes

Cell-level and gRNA-level QC are out of scope, but the calibration and power
checks construct or receive their own pairs, so both have to decide which
pairs are testable at all. Pairwise nonzero-count filtering therefore lives
in `pipeline/pairwise_qc.py`, shared by both -- and the two use it in
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

## Low MOI and the NT-cells control group

In a low-MOI screen each cell carries at most one gRNA, so the cells carrying
a non-targeting (NT) gRNA are untreated and can serve as the control group on
their own. sceptre calls that the NT-cells control group. In its statistical
engine MOI sets the defaults of the control group and the resampling mechanism,
and allows the NT cells at all. `moi` does the same here, the way R's
`import_data(moi=)` does, and it is the only way in.

| `moi` | default `control_group` | default `resampling_mechanism` | `"nt_cells"` allowed |
|---|---|---|---|
| `"high"` | `"complement"` | `"crt"` | no, refused; R quietly uses the complement |
| `"low"` | `"nt_cells"` | `"permutations"` | yes |

Low MOI with `control_group="complement"` runs the high-MOI engine unchanged,
and `test_low_moi_complement_is_the_high_moi_engine` asserts it bit for bit.
R's own code has two more low-MOI branches, both in the calibration check: it
skips `unique()` on a synthetic target's cells, and it counts a synthetic
target's nonzero cells as a sum over its NT gRNAs rather than over their union.
Neither can change anything once each cell carries at most one gRNA.

R's low MOI does two things before the analysis: by default it assigns each
cell its single strongest gRNA, `assign_grnas()`'s maximum method, and its QC
removes cells with zero or two or more gRNAs. The first is ported. The second
is cell-level QC and out of scope: `assign_grnas_maximum` and
`cells_w_zero_or_twoplus_grnas` report the cells sceptre would remove (see
[Thresholding and maximum](#thresholding-and-maximum)), but pysceptre never
removes them. The NT-cells analysis **relies** on that removal, and R's engine
never checks it, since its QC guarantees it. pysceptre takes post-QC inputs
and cannot, so it checks what it can see and refuses rather than guessing: a
cell listed under two NT gRNAs, and a tested target sharing a cell with the NT
cells. A cell in two targets is **not** refused. With a guide in two
overlapping elements one gRNA legitimately puts a cell in both, and that
changes nothing here, since each pair is tested on its own target's cells.

### One fit per pair

Against the complement a gene's GLM is fit once, on every cell, and reused for
all its targets. Against the NT cells it cannot be: R's
`discovery_ntcells_perm_test` and `discovery_ntcells_crt` test each pair on
`c(trt_idxs, all_nt_idxs)`, its target's cells followed by the NT cells, and
fit the gene's Poisson GLM and dispersion on exactly those cells. Every pair
has its own design rows, so nothing is shared across a gene's targets.

`run_discovery_nt_cells` keeps the complement path's structure -- targets in
memory-budgeted chunks, prepared ahead of the chunk being tested, genes
distributed across workers within a chunk -- and changes what a pair costs.
Each pair is fit **on its own**, never batched with the other genes of its
target, for the reason `_GENE_BATCH_WIDTH = 1` gives on the complement path:
a batched solve makes a fit depend on what else shared its BLAS call. So a
pair's result depends on its gene, its target, the NT cells and the seed, and
on nothing else in the run, except under `no_approximation`, whose third batch
is sized by the pair count.

The workers are chosen for that cost. A pair's fit is many small numpy calls
that hold the GIL, so threads contend where processes would not: on
sceptredata's Papalexi screen against the NT cells (6,205 pairs, Apple M4 Max,
threads) a run took 22.3 s at `n_jobs=1`, 14.7 s at 2, 14.1 s at 4 and 20.5 s
at 8. Off Linux the gene pool is therefore capped at four threads
(`_NT_THREAD_CAP`); on Linux it always forks, whatever the chunk count,
because a chunk's per-pair fits dwarf the cost of a fork. Results do not
depend on either choice.

The combined cells are R's order exactly: the target's cells ascending, then
the NT cells in pool order, each NT gRNA's cells in turn. Order moves fitted
values only in their last bits, but a resample is a set of positions, so a
different order is a different Monte Carlo draw; matching R's order is what
lets R's own draws be fed in and compared value for value.

R stops when a coefficient cannot be estimated on a pair's cells, which a
covariate constant on one target's cells and on every NT cell can cause even
though the full design is full rank. pysceptre checks every tested target's
combined design before fitting anything, and names the target. Adding rows
cannot lower a matrix's rank, so a full-rank design over the NT cells alone
clears every target at once, and the per-target check runs only when it is
not.

### Permutations against the NT cells

A target with `n_trt` cells is tested against `N` NT cells, so its resamples
are random `n_trt`-subsets of `N + n_trt` positions. R draws them once for
every target, with `hybrid_fisher_iwor_sampler(N, m, M, B)`, `m` and `M`
being the smallest and largest nonempty target: `B` rows of `M` positions
each, such that for every `k` in `[m, M]` the first `k` entries of a row are a
uniformly random `k`-subset of `{0, ..., N + k - 1}`. One array then serves
every target, each reading the prefix as long as itself.

R builds each row forwards: a Fisher-Yates shuffle for the first `m` entries,
then an inductive step that grows the universe by one element at a time and
takes the new element with probability `k / (N + k)`.
`nested_permutation_draws` builds the same thing **backwards**, which needs
nothing but a uniform ordering and a swap per step, vectorized across rows:
start from a uniformly random ordering of `M` distinct elements of
`{0, ..., N + M - 1}`, then for `k = M` down to `m + 1`, if element
`N + k - 1` sits in the first `k` positions swap it to position `k - 1`,
otherwise swap a uniformly chosen one of the first `k` there.

The two constructions have the same joint law over the whole chain of
prefixes, not merely the same marginals. Writing `S_k` for the set of the
first `k` entries, both give each `S_k` the uniform law over `k`-subsets of
`{0, ..., N + k - 1}`, and the transition between consecutive prefixes agrees:

| `S_k` | forward, `P(S_{k-1}) P(S_k given S_{k-1})` | backward, `P(S_k) P(S_{k-1} given S_k)` |
|---|---|---|
| contains `N + k - 1` | `k / ((N + k) C(N + k - 1, k - 1))` | `1 / C(N + k, k)`, the same number |
| does not | `1 / ((N + k) C(N + k - 1, k - 1))` | `(1 / C(N + k, k)) (1 / k)`, the same number |

`test_nt_cells_sampler.py` checks the marginal law of each prefix, the
inclusion rate `k / (N + k)` of the newest element, and the joint law of the
chain against a forward implementation of R's algorithm.

Like the complement's permutations, these draws are shared, so they cannot be
invariant to a change of target set: a new target larger than `M` or smaller
than `m` moves every result. `m` and `M` are taken over every target supplied,
not only the tested ones, which is R's rule and keeps the pair list out of it.

### The CRT against the NT cells

Under the CRT each target's resamples come from a logistic fit on its own
combined cells, as in R's `discovery_ntcells_crt`, and are drawn from a stream
keyed by the target's name, so the CRT keeps its invariance to the rest of the
run here too.

The sampler is different, and has to be. `crt_index_sampler_fast` is accurate
only while inclusion probabilities are tiny (see
[The CRT sampler draws with replacement](#the-crt-sampler-draws-with-replacement)),
and against the NT cells they are not: a target is 10 to 18% of its combined
cells on sceptre's simulated example (20 targets of 21 to 41 cells against
185 NT cells) and 23 to 28% in the screen measured below. At an inclusion probability of 0.25 the fast sampler repeats
about one listing in nine (11.5% in a simulation of the sampler), and the
extra variance makes the test conservative. On a synthetic screen with no
effects and every cell carrying one gRNA -- 6,000 cells, 12 targets of 343 to
444 cells, 1,157 NT cells, 480 pairs -- `scripts/measure_nt_cells_crt_sampler.py`
measured:

| test | null p-values below 0.1 | mean null p-value |
|---|---|---|
| permutations, exact by construction | 0.100 | 0.534 |
| CRT, `crt_index_sampler_exact` | 0.096 | 0.533 |
| CRT, `crt_index_sampler_fast` | **0.056** | **0.576** |

`crt_index_sampler_exact` removes the approximation without giving up the
sparse cost. Each cell's inclusions are a Bernoulli process along the
resamples, so their positions are partial sums of geometric gaps: no cell can
be listed twice, the law is R's binomial count plus without-replacement
placement exactly, and the work is still proportional to the number of
inclusions.

Against the NT cells every target is far above the 0.2% boundary, so the rule
in [The CRT sampler draws with replacement](#the-crt-sampler-draws-with-replacement)
always picks the exact sampler there, and so does it for the calibration check,
which runs on the NT cells alone.

### The calibration check on the NT cells

R's calibration check against the NT cells does not use the NT-cells test at
all. It restricts the whole analysis to the NT cells (`subset_to_nt_cells`):
the GLM is fit once per gene on the NT cells, and a synthetic target is tested
against the rest of them. pysceptre does the same restriction up front and then
runs the complement check unchanged, and
`test_nt_cells_calibration_is_the_complement_check_on_the_nt_cells` asserts the
two agree bit for bit. It therefore needs at least two NT gRNAs, as R does.

Under permutations both control groups size the shared draws as R does for a
calibration check: as wide as the `calibration_group_size` largest NT gRNAs
together. Unlike the width of the largest synthetic target, that does not
depend on which groups were sampled.

### A dispersion estimate can stop on a rounding accident

For a gene with no overdispersion on a pair's cells the dispersion MLE is at
infinity, and Newton's iteration in `estimate_theta` walks there, theta
growing about 1.5 times a step. It stops by floating-point accident: when the
score's cancelling terms sum to exactly zero, which reads as converged, or
when it runs out of iterations, which R and pysceptre both answer with a
method-of-moments estimate. R's boost special functions and scipy's round
differently, so the two can stop on different branches. Both keep the
algorithm; neither is more correct.

Measured on sceptredata's Papalexi screen against the NT cells: one pair of
6,205, `CTD-2044J15.2` against `CAV1` (83 counts over 2,552 cells), where
pysceptre's iteration stopped on an exact zero at theta = 676,341, clamped to
1,000, and R's ran out at 50 iterations and took 10.99. The statistic moved
by 1e-3 relative, because a gene that sparse barely depends on theta. Every
other pair's statistic agrees with R to 4e-9 relative, and 99% of them to
3e-11.

### Pairwise QC against the NT cells

The control count changes meaning. Against the complement it is the gene's
nonzero cells minus the target's; against the NT cells it is the gene's
nonzero NT cells, the same for every target of that gene, as in R's
`compute_nt_nonzero_matrix_and_n_ok_pairs_v3`. A calibration pair's control
count is the NT total minus its synthetic target's, since its universe is the
NT cells.

## Analytical per-pair power

`analytical_power/` answers a different question from the rest of the package:
not "is this pair significant?" but "if this element really did reduce this
gene by X%, would this screen have detected it?" It is a closed form -- one
pass of arithmetic, no resampling, no fitted parameters -- and it is a port of
PerturbPlan's `compute_power_posthoc()` rather than of sceptre. The MIT notice
is in `THIRD_PARTY_LICENSES`.

### Why a port, and not the formula this project derived

The obvious alternative was a two-parameter model,
`power = Phi(k * beta * n^b / sqrt(1/mu + dispersion) - z)`, fitted on
simulations. It was built, fitted and scored against per-pair simulation at
screen scale, and read as the decision a user actually takes -- is this pair's
power at least 0.8? -- the unfitted closed form won, on both error directions
at once. So this ports the closed form rather than shipping the fitted model.

Two further findings shaped what is offered here. The theory did not need
fitting: held at their theoretical values the constants are already right, so
recovering them from simulation was the unnecessary step. And a calibrated
error bar around the closed form was measured and is **not** offered, because
it does not transfer between screens and makes the decision slightly worse
overall. The point estimate is reported as it comes.

The measurements behind all three are in the manuscript repository and are
deliberately not restated here: a number that lives in two places drifts, and
these are the manuscript's results rather than the tool's.

### Does it agree with the truth? Measured on day0, under the CRT

Matching PerturbPlan's R to 1e-9 says the port computes what it ports. Whether
that is worth computing is a separate question, and the only available truth
is simulation. Scored against WattEG's day0 sweep, 100 simulations per pair
over **34,886 pairs**, at the bar a reader acts on rather than as a distance:

| effect size | powered by simulation | sensitivity | specificity | wrong claims | missed | MCC |
|---|---:|---:|---:|---:|---:|---:|
| 0.05 | 274 | 96.4 % | 98.25 % | 606 | 10 | 0.536 |
| 0.10 | 6,204 | 94.1 % | 98.08 % | 551 | 369 | 0.911 |
| 0.15 | 16,050 | 96.4 % | 97.66 % | 440 | 580 | **0.941** |
| 0.20 | 22,131 | 97.9 % | 98.63 % | 175 | 459 | 0.961 |
| 0.25 | 25,563 | 98.2 % | 99.07 % | 87 | 472 | 0.960 |
| 0.50 | 31,925 | 98.6 % | 99.97 % | 1 | 447 | 0.925 |

`scripts/score_power_against_simulation.py`, guarded by
`tests/validation/test_power_against_simulation_day0.py`.

**What this corroborates, and whose result it is.** The finding that
PerturbPlan's closed form agrees with per-pair simulation is not this
package's; it was established by scoring their R against a large simulation
sweep, and it is why the closed form is worth porting at all. This is a second
instance of that same comparison, so it tests the estimator rather than
proposing one. It is not a re-run: the original scored a different screen
analysed under sceptre's *permutation* test, which an ondisc-backed response
matrix forced, while day0 ran under the **CRT**, which that work explicitly
lists as untested.

The agreement holds: MCC between 0.91 and 0.96 from a 10 % to a 50 %
knockdown, and it does **not** degrade as the signal grows, which is the
failure mode the fitted alternative in the original comparison did show.

**The es = 0.05 row is not a result.** 274 of 34,886 pairs are powered at a
5 % knockdown, so every rate on that line is a few hundred pairs wide against
a 34,000-pair negative class, which is why its MCC is 0.536 while its
sensitivity is 96.4 %. It is kept to show the row exists.

### Which mean is correct: PerturbPlan's own definition settles it

Not an argument from sceptre, and not an empirical question. **PerturbPlan
defines what its input means**, in `power_function`:

```r
expression_mean = avg_library_size * relative_expression
```

with `relative_expression = rowSums(counts) / sum(counts)`
(`obtain_expression_information`). So `expression_mean` is the gene's share of
all counts times the library size: **the expected raw UMI count for that gene
in a cell of that depth.** Nothing normalised about it.

That decides it. The fitted mean is the gene's average observed count over the
cells the fit ran on -- exactly, since a GLM with an intercept satisfies
`sum(fitted) == sum(observed)` -- which is the quantity PerturbPlan asks for.
The size-factor-normalised mean is a DESeq2 poscounts quantity answering a
different question, and on day0 it sits 16 % below what the formula wants.

It is a happy coincidence rather than a second argument that the fitted mean
is also on the scale sceptre's test operates on: `exp(Z b)` is both. The two
routes agree because the expected count is the expected count.

**So the comparison below is corroboration, not adjudication**, which the
first version of this section got wrong by presenting it as evidence for the
default. It cannot adjudicate in any case: the ground truth is a simulation
that drew counts as `mean_i * sf_j`, realising about 0.959 of the raw mean, so
the truth's own scale sits 4.3 % from the fitted mean and 12.2 % from the
normalised one. The convention nearer the truth scores better partly for that
reason. The same proximity makes the headline table mildly *pessimistic*,
since the estimator is scored against genes about 4 % dimmer than the screen
has.

With that read, on the same pairs and the same ground truth and only the
expression input changed:

| effect size | MCC, model mean | MCC, normalised mean |
|---|---:|---:|
| 0.10 | **0.911** | 0.901 |
| 0.15 | **0.941** | 0.910 |
| 0.20 | **0.961** | 0.920 |
| 0.25 | **0.960** | 0.914 |
| 0.50 | **0.925** | 0.856 |

And it errs in the predicted direction: at a 15 % knockdown the normalised
mean misses 1,497 powered pairs against 580, while making 112 wrong claims
against 440. So the 16 % shortfall buys specificity at a larger cost in
sensitivity, which is what a conservative input does, and the gap widens as
the effect size grows. That directional result is not affected by the
proximity problem above -- a lower expression input gives a lower power
estimate whatever the ground truth is.

### Five things that look like simplifications and are not

Each of these turns the estimator into one that nothing has scored.

**`num_trt_cells` is the sum over a target's gRNAs, not the union of its
perturbed cells.** `compute_power` groups `cells_per_grna` by target
and sums, and that is the input the estimator was validated with. pysceptre's
own data model carries the union instead (`grna_target_cells`, one index array
per target), and a cell can carry several of a target's gRNAs, so union <= sum.

**How far apart are they? Measured, and the answer is awkward: almost never,
but not never.** `scripts/measure_union_vs_sum.R` reads the two slots off a
sceptre object and compares them. On two real screens the union/sum ratio has
median **1.0000** and mean **0.9985**; the two agree exactly for 57.6% and
63.2% of targets, and the 1st percentile is 0.9927 and 0.9900. The worst
single target on one screen sits at **0.4990**, its gRNAs overlapping enough
to halve the count.

So substituting the union would look harmless on almost every target and be
badly wrong on a handful, which is the least useful shape an error can have:
too small to notice in aggregate, large enough to move a specific pair's
answer. That is a reason to keep the sum, not a reason to relax about it.

**The stronger reason is that the union cannot supply the other term at all.**
`num_trt_cells_sq`, the sum of *squared* per-gRNA counts, is the across-gRNA
variance term, and no target-level total contains it: a target's 15 gRNAs
could be equal or wildly unequal at the same total. `target_cell_counts`
therefore requires per-gRNA granularity outright rather than deriving it.

One visible consequence is kept visible rather than smoothed over: the sum can
exceed the number of distinct cells, which would empty the complement group,
and `compute_power` raises with that explanation rather than returning
a NaN. Note the trap `measure_union_vs_sum.R` documents, because it bites
anyone recomputing this: `@initial_grna_assignment_list` is indexed against
*all* cells while `grna_group_idxs` is indexed against `cells_in_use` (586,309
against 567,690 on one screen), so the per-gRNA counts have to be restricted
to the cells in use before the two are comparable.

**`cutoff` is required.** PerturbPlan will derive one from the predicted
distributions when it is `NULL`; that is its design-planning mode, and only
the explicit-cutoff path is ported. The threshold is a screen's own
multiple-testing correction, and it is load-bearing rather than a nuisance
parameter: a design testing an order of magnitude more pairs carries an order
of magnitude stricter threshold, and scoring one design's pairs against
another's was measured to more than double the error and to do so in the
optimistic direction. A default would be the easiest available way to get a
confidently wrong answer. A corollary worth knowing: a good deal of any
apparent detectability gap between two designs is testing burden rather than
biology.

**`side="left"` at `cutoff = alpha/2` is the validated configuration.** A
simulation counts as a success when `p < alpha` *and* the fold change is
negative; sceptre's p-value is two-sided, so the matching analytical call is
the left tail at half the threshold. `side="both"` at `cutoff = alpha` is
reachable because a user will reach for it, but it is not what was scored.

**`fold_change_sd` has no default.** The `0.13` behind every published number
came from PerturbPlan's documentation example, not from data. It enters the
variance through the across-gRNA term, so if accuracy moves when it moves,
"no parameters fitted" is too generous a description of the estimator. The
sensitivity has not been measured. A default would turn a stated open question
into an invisible one.

**The nonzero-count thresholds default to 0, not to R's 7.** Power is
multiplied by `1 - P(the pair fails pairwise QC)`. pysceptre is fed pairs that
already passed QC in the real analysis, so at a threshold of 7 their power
would be discounted a second time for a risk that did not materialise. They
are exposed for the case the discount is meant for: pairs that were never
tested.

### Power is not monotone in the effect size, and that is the estimator

On a 3-cell target and a near-zero-expression gene, PerturbPlan's own R
reports power *falling* as the knockdown deepens: 2.43e-4 at a 5% knockdown,
1.19e-4 at 15%, 1.09e-6 at 50%. A deeper knockdown lowers the treated group's
negative-binomial variance, and when the statistic's mean is still ~3.3 sd
short of a threshold that far into the tail, concentrating the statistic moves
mass *away* from the rejection region instead of into it.

It is confined to pairs whose power is ~0 under every effect size, so it
changes no decision, and it is reproduced rather than corrected:
`tests/validation/test_analytical_power.py` pins it deliberately. A "fix"
would be a different estimator.

### Why the covariates can come from anywhere

On the default inputs the estimator's inputs are not properties of the
*pair*: expression mean and dispersion belong to the gene, cell counts to the
element. The information-matched mean below is per pair, but it is still
composed from the gene's fit and the element's cells. That is measured
rather than assumed: one screen processed through sceptre twice, under two
designs sharing genes and elements but almost no pairs, gives covariates that
match exactly across the entities the two runs share. So a pair's inputs
can be composed from its gene and its element measured separately, in any
pairing, which is what makes the estimate available for pairs no screen
tested. sceptre's requirement that a pair carry its own QC before it can be
tested constrains *simulating* that pair, not estimating its detectability.

### What the estimate is not good enough for

The closed form misses a few percent of the pairs the simulations call
powered, and makes wrong claims on well under one percent of the rest. Those
rates are good enough to plan a screen and to triage its negatives; they are
**not** good enough to close a question about one element-gene pair, and the
estimate should not be used that way.

The evidence behind them is also one dataset: one platform, one MOI, one cell
line, and sceptre's *permutation* test rather than the CRT, which an
ondisc-backed response matrix forced. It held across a tenfold change in
threshold and a tenfold range of effect size, which is real evidence it is not
tuned to a corner, but whether it holds on another lab and protocol is
untested.

### Covariates through the information-matched mean

`compute_power` summarises a gene by one `expression_mean` and treats every
cell as if it had that mean. sceptre's score statistic does not: it is built
cell by cell from the fitted means `mu_j = exp(Z_j b)`, and a cell contributes
information `w_j = mu_j / (1 + mu_j / theta)`, the `w` of
`precompute/pieces.py`. Two effects follow, in opposite directions.

- **`w` is concave in `mu`**, so when `mu_j` varies a lot between cells (a
  batch effect, library size) the mean of `mu` overstates the information the
  test gets. DC-TAP K562's CYBA (`ENSG00000051523`), nearly absent from six of
  eleven batches, is the worst case: one mean per gene reads its pairs too
  high.
- **An element's perturbed cells are not the average cell.** They have larger
  libraries and carry more information than one mean per gene gives them. The
  median of `info_ratio_pert` over the trans pairs, the perturbed cells'
  average information over the average cell's, is 1.06 in K562 and 1.11 in
  WTC11. That is where the estimate's slight conservatism on those screens
  comes from.

`inputs.matched_expression_stats` corrects both without a new fit and without
touching the closed form. Per pair, with `P` the target's perturbed cells and
`C` the rest, it averages `w` over each (`f_p`, `f_c`) and hands PerturbPlan
the single mean that gives its statistic the same variance:

    f               = (1/n_p + 1/n_c) / (1/(n_p f_p) + 1/(n_c f_c))
    expression_mean = f / (1 - f / theta)

Every `w_j < theta`, so `f` lies below theta too and the inversion cannot
fail; there is no clamp to add. `expression_size` stays the fit's theta. The
result has one row per pair, which is why `compute_power` accepts a
`grna_target` column in `baseline_expression_stats`. This is the one input
that *is* a property of the pair, so it exists only for pairs whose target
has cells.

**The averages and the counts use different cell sets, on purpose.** `f_p`
and `f_c` average over the *union* of the target's cells and its complement,
because that is what the test sees. `n_p` and `n_c` are PerturbPlan's own
per-gRNA *sum* and `num_total_cells - n_p`, the first of the "Five things
that look like simplifications and are not" above. That combination produced
every number below. Do not tidy it into union/union or
sum/sum without re-running the comparison.

Agreement of the 0.8 call with WattEG's simulation at a 15% knockdown
(`fold_change_mean=0.85`, `fold_change_sd=0`, `side="both"`, each screen's
own discovery threshold as `cutoff`). "Over" is `compute_power >= 0.8` where
the simulation is not, "under" the reverse. `test_matched_expression_dctap.py`
reproduces the per-element column from the same inputs.

| screen, pairs | pairs | one mean per gene | gene-level matched | per-element matched |
|---|---:|---|---|---|
| K562 cis    | 7,493   | 98.02% (41 over, 107 under)      | 97.68% (8 over, 166 under)     | **98.56%** (32 over, 76 under)       |
| K562 trans  | 297,707 | 97.70% (1,617 over, 5,232 under) | 96.94% (300 over, 8,824 under) | **98.26%** (1,518 over, 3,648 under) |
| WTC11 cis   | 6,574   | 97.61% (1 over, 156 under)       | 97.54% (1 over, 161 under)     | **99.04%** (16 over, 47 under)       |
| WTC11 trans | 196,615 | 97.38% (28 over, 5,122 under)    | 97.17% (13 over, 5,558 under)  | **99.07%** (493 over, 1,331 under)   |

The per-element mean improves all four, mostly by removing too-low calls,
and it gives back some too-high ones. That is the trade to watch: an
over-call is the error that closes a question wrongly.

**The gene-level version is not offered.** Dropping the element split gives
one matched mean per gene, `f = mean over all cells of w_j`. It fixes CYBA and
nothing else, and it lowers agreement on all four tables, because it removes
the too-high calls and adds more too-low ones by ignoring that perturbed cells
are bigger.

**What this does not establish.** One lab, two DC-TAP screens, one effect
size, `fold_change_sd = 0` and `side="both"` only. The day0 comparison of
`compute_power` has not been repeated with the matched mean, which is why it
sits beside `baseline_expression_stats_from_fits` rather than replacing it.

**What it still leaves out.** sceptre's statistic also subtracts the treated
cells' projection onto the covariates (`lower_right` in
`test_statistic/score_stat.py`). Matching `w` captures how much information
the perturbed and control cells carry, not how much of the treatment
indicator the covariates explain. For an element whose perturbed cells sit
mostly in one batch that term matters, and the matched mean does not see it.

### Building the estimator's inputs

`analytical_power/inputs.py` derives the three things `compute_power`
needs that no other part of pysceptre produces. Each can be got subtly wrong
in a way that still returns plausible numbers, so each is validated against
someone else's code rather than against itself.

**The helpers take arrays and frames, never a loaded export.** Reading an
`.h5mu` needs the `io` extra and nothing under `src/` imports it, so the glue
from a file to these arguments lives in `scripts/`. There is deliberately no
convenience wrapper that takes an export and returns a power table: the
estimator's signature is the validated one, and a wrapper is where the
sum-for-union and all-cells-for-cells-in-use substitutions would creep back in
unnoticed.

#### `expression_mean` has to be on the scale sceptre's own model works on

This is the constraint that matters, and it is easy to state wrongly. The
estimator exists to predict **sceptre's** test, so the gene expression it is
handed has to be the expression that test operates on. Any other
normalisation, however respectable, answers a different question.

sceptre caches `fitted_coefs` and `theta` per gene, so the mean its model
implies is `mean(exp(Z b))`, the average fitted value.
`baseline_expression_stats_from_fits` returns exactly that, taken from the
same negative-binomial fit that produces theta, so the power estimate and the
test it predicts are on one scale by construction rather than by coincidence.
It is validated against sceptre's own cached fit over day0's 237 genes, and
the two claims are worth separating because they are not the same size. The
**fitted values** -- `exp(Z b)`, which is what the mean is taken over -- agree
to a relative **9.2e-10** across all 134,542,530 gene x cell values. The
**coefficients** themselves agree less tightly, as correlated design columns
will: 2.3e-9 on the intercept but **4.5e-7** at worst, on
`replicate_factorRep 4`. Every one of the eleven columns is compared
separately, dummies included, because a factor-contrast ordering mismatch
would shift the baseline by batch while a pooled maximum looked fine.
pysceptre's fitted values already match sceptre's at 1e-6
(`test_glm_fits.py`), so nothing new is being trusted here.

**The published comparison did not use that mean, and the difference is
measurable.** It used a size-factor-normalised mean instead, which
`baseline_expression_stats` reproduces. On day0, over the 237 genes both
cover:

| | | cell set |
|---|---|---|
| sceptre's model mean / the normalised mean | median **1.1613**, sd 0.0218, range 1.0948 to 1.2235 | mixed |
| correlation of the logs | **0.99995** | mixed |
| the raw mean / the normalised mean | **1.1875** | all cells, both |
| sceptre's model mean / the raw mean | 0.9786 | mixed |

**Two of those rows mix cell sets, and the mix is the whole of their
residual.** The model mean is over `cells_in_use`, which is what the fit was
computed on; the normalised and raw means come from `row_data`, which is over
every cell in the object. Measured on **one** cell set the model reproduces
the raw mean *exactly* -- `mean(exp(Z b)) / raw = 1.0000000000`, maximum
deviation 8.1e-9 -- because a Poisson GLM with an intercept satisfies
`sum(fitted) == sum(observed)`. The 2.14 % in the last row is entirely
`raw(all cells) / raw(cells_in_use) = 1.0219`, a property of which cells the
normalisation ran over and not of the model. Read as a model residual it would
be simply wrong.

So the clean statement is the **third** row, which compares like with like:
the normalised mean sits **15.8 % below** the raw mean, and sceptre's scale is
the raw mean. About 16 %, and the 1.1613 in the first row reads low only
because its numerator is over a cell set whose mean is 2.2 % smaller. The
offset is a single factor rather than a reshuffling either way: the two agree
almost perfectly on which genes are expressed more than which. Raising
`expression_mean` by that factor raises the closed form's power on 17 of the
fixture's 18 pairs, the exception being a pair whose power is ~0 under either
(the non-monotone corner below). So feeding it the normalised mean makes the
estimate **conservative** rather than wrong-shaped, which is the better
direction to err in and still the wrong number.

**Which to use.** `baseline_expression_stats_from_fits` for any new analysis,
because it is the only one consistent with the test being predicted.
`baseline_expression_stats` only to reproduce the published comparison, whose
sensitivity and specificity were measured with the normalised mean on both
sides -- the closed form and the simulation that served as its ground truth
both read it, so that comparison is internally consistent and its conclusion
stands.

**The two sides of that comparison were not on the same scale, which is
worth knowing before its error profile is taken at face value.** The
simulation that produced its ground truth multiplies each cell's size factor
back in when it draws counts; the closed form has no size factors and applies
no such scaling. So being fed the same column, the closed form was predicting
for genes about **13.9 %** dimmer than the simulation actually made them,
which is the mean size factor on day0, 1.1389.

**What that does to the reported residuals is not established, and it would
be easy to over-read.** The comparison found the formula "leans low", a
positive median residual in every bin of predicted power, and a 13.9 % dim
input would produce that signature. But the same baseline also *understated
the count variance* by about 13.5 %, and less variance inflates power where
less expression deflates it. Two errors of opposite sign, neither measured
against the other, so the net direction of the simulated power is unknown.
The *ranking* result is scale-insensitive and unaffected either way; the
*calibration* half should not be read as characterising the estimator until
the two are separated.

None of this is pysceptre's to fix and nothing here depends on it: the helpers
are validated against R output value for value, and which mean to feed the
estimator is the caller's. It is recorded so a later comparison against those
sweeps does not inherit the mismatch silently. It describes the **published**
sweeps: the pipeline that produced them has since been changed to draw from
`exp(X b)`, sceptre's own model, which removes the mismatch at the source.

(Two earlier versions of this note were wrong. The first said the simulation
ran 16 % low, missing that it restores the size factors. The second said its
power was therefore biased low, which the variance error contradicts.)

#### A note on the poscounts convention, for the reproduction path only

The normalised mean's size factors are DESeq2-style "poscounts", and they are
**not** centred the way DESeq2 centres them: DESeq2 divides its factors by
their geometric mean and the implementation here does not. Measured over
day0's 586,309 cells that geometric mean is 1.0639, so the factors are 6.4 %
larger and the resulting gene means 6.4 % smaller than DESeq2's convention
would give. (This is a *different* 6 % from the 16 % above, and it is a
subset of it: both push the same way.)

This is recorded for the reproduction path and for anyone comparing against
DESeq2 output, not as a live decision. sceptre computes no size factors and is
not a party to it. The validation splits accordingly: every ratio in the
factor vector is checked against DESeq2 itself, and the absolute scale against
real output from the run that produced the published numbers, because neither
check alone constrains both.

#### Why this is 40 lines here rather than a library call

Reimplementing a published normalisation is the wrong default, so two
candidates were tried against the fixture before it was written. Neither
supplies the quantity this needs, and the reason is the same for both: they
implement the estimator DESeq2 and edgeR use for **bulk** data, which assumes
genes that are nonzero in every sample.

| candidate | what it offers | result on the fixture |
|---|---|---|
| `edgepython` 0.2.6, an edgeR port | TMM, RLE, upperquartile | a different quantity: TMM correlates **-0.05** with poscounts factors and RLE **-0.07**. `upperquartile` returns `inf` |
| `pydeseq2` 0.5.4 | DESeq2's default median-of-ratios | does not match, and not by a constant either: the ratio to R spreads by **1.9** |

On the sparse case both fail outright, `pydeseq2` returning all `NaN`. The
cause is visible directly: at 33 % nonzero, **0 of 300** genes are nonzero in
every cell, and both estimators need the geometric mean of a gene across all
of them. "poscounts" exists precisely because that assumption does not hold
for single-cell data, and neither package implements it.

Two further reasons, either of which would matter on its own. `pydeseq2`
rejects a `scipy.sparse` matrix (`TypeError` on `log`) and `edgepython` wants
dense as well, while **never densify** is a standing constraint here. And
between them they would add `anndata`, `formulaic`, `matplotlib`, `patsy`,
`scikit-learn`, `statsmodels` and `numba` to a package whose runtime
dependencies are four.

So the arithmetic is implemented, on the nonzeros, and validated against
DESeq2 for every ratio and against real R output for the absolute scale. That
is a better trade than a dependency that computes something else.

#### The gene and cell sets are part of the definition

The geometric mean runs over every gene and the median over every cell, so
computing either on a subset gives different factors for the cells that
remain, and a different mean for every gene. Both results look equally
complete, which is what makes it dangerous. On day0 the validated factors were
computed over all 292 genes and all 586,309 cells, the 18,619 that QC removed
included, while the gene statistics were then reported for the 237 genes
appearing in pairs.

Two consequences. `baseline_expression_stats` computes over the full matrix
and has a `gene_subset` argument for the reporting step, so subsetting the
*output* is available and subsetting the *input* is not disguised as the same
thing. And **an export now carries every gene and every cell by default**,
because that choice is not recoverable afterwards: exporting everything costs
disk and nothing else, since an analysis reads only the genes in the discovery
pairs and the loader subsets back to `cells_in_use`, so the extra rows and
columns never reach a per-gene fit or a worker process.

#### Theta is reused, not refitted, and it carries two caveats

`expression_size` is the NB size, theta, which `discovery.py::fit_all_genes`
already returns, so a completed analysis has it and
`baseline_expression_stats` accepts it rather than refitting. Two things about
it that are invisible in the output: it is **clamped** to `(0.01, 1000.0)`, so
a gene at a bound carries the bound instead of its estimate; and it is fitted
under whatever `covariate_matrix` was passed, so it matches another
implementation's only if the design matrices match. On day0 every one of R's
237 cached values sits inside the clamp, median 16.84, so the clamp is not
biting there. The day0 check reports the comparison rather than asserting a
tolerance, because two different estimators agreeing to a tolerance is not
something a test should demand.

#### The gRNA-to-target map is many-to-many

Overlapping candidate elements share guides. On day0, 1,673 of 43,736 guides
sit inside two or three of them, so the design table carries 45,463 rows for
43,736 distinct ids and the same guide appears under several targets. R sums
such a guide's cells into every target it belongs to, and
`cells_per_grna_from_assignments` does the same. Deduplicating by `grna_id`
looks like hygiene and would silently shrink exactly those targets;
`compute_power` originally refused a repeated `grna_id` for that
reason and was wrong to, which the fixture now pins against R.

A designed guide that ended up with no cells is a `num_cells = 0` row rather
than an absent one, for the same family of reason: dropped, `num_trt_cells_sq`
would be computed over a smaller guide set than the screen actually used.

#### The cutoff helper does the correction pysceptre otherwise skips

`bh_nominal_cutoff` applies BH and returns the largest p-value it calls
significant, which is what `cutoff` wants. It ports WattEG's
`discovery_threshold()` including its refusal to return anything when nothing
is significant: the code that preceded it returned `-Inf` there, and every
pair silently got zero power. NaN p-values, the pairs that failed pairwise QC
and were never tested, are dropped before the correction rather than inflating
its denominator. On day0 it reproduces R's derived threshold,
`0.00072628634455531758`, exactly.

## Specificity check

`specificity/` answers the question sceptre's two other checks leave open. The
calibration check asks whether the pipeline invents effects, the power check
whether it recovers effects it should. Neither says how many of the links a
screen *did* discover are real. Like `analytical_power/`, this is not a port of
sceptre and has no R counterpart; unlike it, there is no external
implementation to validate against either. Its reference is the analysis it
was ported from, WattEG-paper's `analysis/direct_indirect.py`, which itself
started from EngreitzLab/CRISPR_indirect_effects (DC-TAP Fig. 4 and S3).

A **link** is an element-gene pair on the same chromosome, called at the
screen's cutoff with a negative effect. "Direct" and "indirect", the original
names, are not used: the background is mostly noise and genes that shift with
any perturbation, not indirect regulation.

### The background, matched gene for gene

An element cannot regulate a gene on another chromosome directly, so how often
those tests are called at the **cis** cutoff, with a negative effect, is how
often a test is called without regulation. `background_pairs` takes the
screen's elements against its cis genes on other chromosomes, drops chrY, and
keeps tested pairs only.

**One pooled rate is biased**, because well-expressed genes are called more
often and the mix of genes differs between distance bins. So each gene gets
its own rate `r_g`, the share of its background tests called, and a bin's
background is the mean of `r_g` over the bin's tested pairs: a gene counts as
often as it is tested there. Links above background is
`(cis rate - background) / cis rate`.

**The bootstrap draws elements, not pairs.** An element's cis and background
tests stay together, and every `r_g` is recomputed in each draw. The notebook
built a dense elements by bins by genes tensor for this; the port sums over
pairs in blocks of draws instead, which gives the same numbers (to 1e-13 on
three screens) and does not grow with the square of the panel.
`test_specificity.py` keeps the dense formulation as its reference.

**Per-gene rates are noisy where the background is low.** A gene with no
background call gets a rate of zero, which is an underestimate rather than a
measurement. Shrinkage toward the pooled rate is open, and not done.

### Broad-effect elements

Some elements lower many genes on other chromosomes, far beyond the
background: a one-sided binomial test of the element's background calls
against the pooled rate, flagged at `p < 1e-3`. They make their own
background, which a per-gene rate cannot see, so the check reports them and
repeats the by-distance table without them.

**Remove them from both sides.** Removing their cis links while keeping their
background calls inflates the background. `above_background_by_distance` only
counts the background tests of elements present in `links`, so dropping an
element from `links` drops it from both sides; do not loosen that filter.

### The TSS lookup and the detour check

Every target within 1 kb of a measured gene's TSS knocks that gene down,
whether or not the design labelled it a control. Its effects on every other
measured gene give a gene-to-gene table (`gene_lookup`). A far link E to H is
a **detour** when E also lowers some G, either as a link or by sitting on G's
TSS, whose TSS knockdown lowers H. `detour_check` reports `detour`,
`detour (nominal)` (only at `p < 0.05`), `no detour` (such G were tested
against H and none lowers it) and `can't check` (no G was tested against H).

A target equidistant from two measured TSSs, a bidirectional promoter, is
assigned the alphabetically first. The notebook's sort was not stable and
picked one arbitrarily; on day4 that is the one place the two differ
(`chr17:28357330-28357839`, 57.5 bp from both POLDIP2 and TMEM199), and it
changes no detour status. A promoter knocks down both genes, which one gene
per target cannot express.

### Measured on three screens

day0, day2 and day4 of an endothelial differentiation, cis and background both
tested with pysceptre's permutation test, cutoff BH at 10% over each screen's
cis pairs, 2,000 draws. `test_specificity_days.py` reproduces these from the
same inputs.

| | day0 | day2 | day4 |
|---|---|---|---|
| links within 50 kb above background, lowest bin | 0.98 | 0.97 | 0.98 |
| far links (> 100 kb) above background | 0.20 (-0.30 to 0.44) | 0.45 (0.28 to 0.57) | 0.55 (0.37 to 0.66) |
| same, without broad-effect elements | 0.18 (-0.33 to 0.43) | 0.50 (0.28 to 0.63) | 0.66 (0.53 to 0.74) |
| background rate | 0.084% | 0.142% | 0.087% |
| broad-effect elements | 4 | 40 | 16 |
| far links: detour, nominal detour, no detour, can't check | 0, 0, 1, 25 | 12, 5, 9, 41 | 0, 0, 5, 47 |

On day2 the elements with the most background calls sit at the TSSs of FOXH1
(74 bp for one of them), KDR (0 bp) and HAND1 (81 bp).

### What it does not establish, and what is left out

- **No ground truth outside the notebook.** The pooled version (one rate, no
  matching) could be checked value for value against CRISPR_indirect_effects'
  R on DC-TAP; that has not been done. The matched rate, the broad-effect flag
  and the detour check have no external reference at all.
- **The fingerprint check is not implemented.** The notebook's fourth part
  asks whether a far link E to G also lowers G's own downstream genes, scored
  with `compute_power` on the matched mean. Near links, which the background
  says are real, recover far less of their expected fingerprint on day4 than
  on day2, and until that is understood it is not shipped.
- **Positions are the caller's.** Distances come from `element_positions` and
  `gene_positions`, never from target names. On DC-TAP the target names are
  hg19 coordinates and the TSSs hg38, which put near links into the far bins
  in the notebook's first pass.
- **Same test on both sides.** The cis and background results should come
  from the same test. Mixing R's CRT on one side with pysceptre on the other
  is an untested assumption.

## gRNA integration strategies

sceptre tests whatever it calls a `grna_group`, and
`grna_integration_strategy` only decides what a group is: the target
(`union`), the individual guide (`singleton`), or the guide followed by an
aggregation back to the target (`bonferroni`). **Nothing statistical differs
between them.** The same CRT, the same score statistic, the same skew-normal
escalation; only the treated cell set changes, and in `bonferroni`'s case
what happens to the results afterwards. So `pipeline/grouping.py` holds pair
bookkeeping and one aggregation, and `discovery.py` never learns which
strategy is in play, exactly as sceptre's engine does not.

The expansion is validated against `update_dfs_based_on_grouping_strategy`
itself rather than against a reading of it, on four cached cases and then on a
real screen: day0's 36,450 pairs become 515,972, row for row and in order.
That comparison earned its keep twice over, as both of the following came out
of it rather than out of reasoning.

### A guide shared by two targets is one test reported twice

Overlapping candidate elements share guides, and on day0 1,673 of 43,736 sit
in two or three. R expands the pair list many-to-many and so tests such a
guide once per target it belongs to. Those tests are necessarily identical:
the same gene and the same guide means the same treated cells. pysceptre
therefore runs the distinct tests and fans the target column out afterwards,
which is cheaper and removes the possibility of two copies disagreeing. The
reported rows are the same rows.

### A duplicated design row is kept, and that is a choice

day0's design lists 36 `(grna_id, grna_target)` pairs twice: the same guide
against the same target, which carries no information the single row does not.
Deduplicating them was the first implementation, and it disagreed with R by
exactly the 288 expansion rows they fan out to.

Whether that matters depends entirely on the strategy, which is why the
default is what it is:

| path | effect of a duplicated design row |
|---|---|
| `union` | **none, provably.** R builds a target's cells as `unique(unlist(...))`, so listing a guide twice contributes the same set once. The union path does not read the design table at all. |
| `singleton` | the pair is expanded twice, so the guide appears twice in the result and is double counted by any correction applied across rows |
| `bonferroni` | the correction factor is `sum(pass_qc)`, so the guide is counted twice and the corrected p-value is **inflated** |
| `compute_power` | the guide's cells enter `num_trt_cells` and its squared sum twice, which is simply wrong |

So the discovery path **keeps** them by default, because matching R is how
correctness is established here and the error direction is conservative, and
warns with the one-line alternative. `drop_duplicate_design_rows=True`
collapses them, at the stated cost of no longer reproducing R's row count.
Asking for it under `union` is refused rather than ignored, since it could not
change anything there.

`cells_per_grna_from_assignments` **raises** instead of warning, and takes the
same option to collapse. The asymmetry is deliberate: there is no parity
argument on that path, because PerturbPlan never reads a design table, and
double counting cells into a variance term is not a conservative error but a
wrong one.

### Two deliberate deviations from R

**An orphan target raises.** A pair naming a target with no guides in the
design leaves `grna_group` as `NA` in R's left join, and the row survives to
ask for the cells of a group called `NA`. An untestable row reaching a result
is the failure this package refuses everywhere else.

**`pass_qc` is optional in the aggregation.** R reads it from
`discovery_pairs_with_info`. `run_discovery_analysis` is handed pairs already
judged testable and does not produce that column, so when it is absent every
row counts as passing and the correction factor is the group size. Supplying
it -- `pipeline/pairwise_qc.py` computes it at guide resolution -- recovers
R's behaviour for a target whose guides individually fail, which is the case
the default cannot see.

## gRNA assignment

`pysceptre.assignment` holds four ways of deciding which cells carry which
gRNA, all behind one entry point, `assign_grnas`, like sceptre's.
`assign_grnas_mixture`, `assign_grnas_thresholding` and `assign_grnas_maximum`
are sceptre's own three methods, ported from sceptre 0.10.3.
`assign_grnas_fishash` is not sceptre's: it ports fishash (jackkamm/fishash
0.99.5, commit 5eabd3c; MIT; Kamm, Yeung and Forrest, bioRxiv
10.64898/2026.01.22.701179), a one-sided Fisher test per (gRNA, cell). All
four take raw integer UMI counts, gRNAs as rows and cells as columns, and all
four are validated against their R originals value for value on synthetic
data. The notices for sceptre and fishash, and for the part of R's
mathematical library fishash needs, are in `THIRD_PARTY_LICENSES`.

fishash 0.99.5 is ported rather than 0.3.0, the version in the preprint's
Table 1. The algorithm is the same except for commit ff4de6b, which changed
which guides and cells the noise fit skips from "fully masked" to "no unmasked
count"; the two differ only where 0.3.0 would divide 0 by 0. The other changes
after 0.3.0 touch output structure only (`drop0` on the calls, how the
assignment strings are built) or the simulator's defaults, which the
preprint's `simulate_guidebender2` calls override.

### What fishash normalizes, and what it does not

**The counts are never transformed.** There is no log, no CLR, no size factor.
Each nonzero count is tested on its 2x2 table -- this gRNA or another, this
cell or another -- and the test conditions on the table's margins: under the
null the expected count is the cell's total times the gRNA's share of the
counts in other cells. That accounts for each cell's depth and each gRNA's
abundance at once, and leaves the cut to an FDR procedure (Guo and Sarkar's
block procedure by default) rather than to a per-gRNA threshold.

**With `refit > 0` the reference is the noise, not the total.** Each later
pass masks the previous pass's calls and refits the masked entries from the
rest (below), so the other cells' margins describe the gRNA's ambient level.
That is fishash's correction for Simpson's paradox.

**The cell side includes the cell's own signal.** A cell's total counts every
gRNA it carries, so each gRNA's share of the cell falls as the number of gRNAs
per cell rises. Nothing in the method offsets that.

**Only integer counts are accepted.** R's `fishash()` accepts any numbers and
rounds inside `phyper`; the port raises instead, which also keeps a normalized
matrix out of a count test.

### fishash on the nonzero entries

The mask, the noise estimate, the calls and the p-values all live on the same
entries: the nonzero counts. A call needs a p-value, which only a nonzero
count has, and the mask is the previous calls. So every pass works on 1-D
arrays aligned with the counts' column-major entries, which is also the order
R's `TsparseMatrix` gives them, and nothing is densified.

Sums follow R's order where it decides a last bit: margins by `bincount`
(column-major, as Matrix accumulates), totals left to right rather than by
numpy's pairwise sum. Block p-values are BH-adjusted over every cell, empty
ones included, and a cell with no entries counts as `log p = 0`, as
`sparseMatrixStats::colMins` reads an implicit zero.

When nothing passes, the cut is `log(0) = -Inf`, as in R. When the noise
estimate outside a cell sums below one count, R's `phyper` returns NaN and the
run cannot continue (R stops with "missing value where TRUE/FALSE needed"); the
port raises a ValueError naming the cause.

### The hypergeometric tail is R's phyper

scipy's `hypergeom` cannot stand in. It refuses non-integer arguments, and the
refit passes produce non-integer margins that R's `phyper` rounds; and its log
upper tail loops over elements in Python. `hypergeom.py` therefore ports
`phyper` and what it calls -- `pdhyper`, `dhyper`, `dbinom_raw`, `stirlerr`,
`bd0` -- from R's `src/nmath` (R 4.5 branch). Three details decide agreement:

- `m`, `n` and `k` are rounded separately, half to even (`nearbyint`), and `q`
  is floored after adding 1e-7. The preprint's equation (7) rounds the noise
  margins; the R code relies on `phyper` doing it.
- `pdhyper` accumulates in `long double`, which is plain `double` on arm64
  macOS, where the fixtures are made, and 80-bit on x86-64 Linux.
- Apple's compiler contracts multiply-adds in R's build, so bitwise agreement
  is not a goal; agreement is to 1e-12 relative to max(1, |log p|).

The numba kernel and the numpy fallback perform the same operations per
element and agree bit for bit on the fixture's grid.

### Masked counts are imputed by a rank-one Poisson fit

`impute_masked_counts` fits the unmasked counts as `guide_freqs[g] *
cell_sizes[c]` by alternating closed-form updates (at most ten, stopping when
both factors move less than 1e-4 on the log scale) and replaces the masked
entries by the fit. fishash's loop recomputes the mask from scratch for the
first three passes and keeps every earlier mask afterwards, which stops
borderline calls from alternating; the deep run of the fixture reaches five
passes so both rules are exercised.

### sceptre's mixture assignment

Per gRNA with at least ten cells of count one or more: a Poisson GLM of its
counts on the cell covariates, then a two-component EM from five fixed starts,
assigning the cells whose posterior for the second component reaches 0.8.
Fewer cells, or an EM that never converges, falls back to `count >= 5`.

Two of sceptre's behaviours are ported as they are, not corrected:

- **`g_pert` is updated as a log ratio and added to the mean on the count
  scale**: `g_mus_pert1 = g_mus_pert0 + g_pert`.
- **The EM keeps the smaller component as the perturbed one.** When more than
  half the cells look perturbed it swaps the components, so the fixture's row
  with 70% of cells perturbed assigns no cell at all.

The five starting values are the draws of sceptre 0.10.3's
`get_random_starting_guesses` (`set.seed(4)`), stored as constants: the
package does not reproduce R's random number generator.

### The EM groups the zero-count cells

A cell with a count of zero contributes to the EM only through its fitted
mean, and its posterior does not depend on that mean, so all such cells share
one posterior. They are handled as a group -- their number, the sum of their
fitted means and the smallest of them -- which makes each EM step cost the
gRNA's nonzero cells instead of every cell. The algebra is sceptre's; the
summation order differs, which moves the log-likelihood in the last bits. A
zero-count cell whose fitted mean exceeds 200 keeps its own term, because
there sceptre's 1e-100 floor on a cell's likelihood can bind. A test-only,
dense, sequential replica of `run_reduced_em_algo_cpp`
(`tests/validation/r_mixture.py`) sits between the two.

### One Poisson GLM per gRNA

The mixture reuses `glm/irls.py`, one gRNA per call, so a fit never depends on
which other gRNAs share its batch. One setting differs from the discovery
engine: the floor on a fitted mean. `glm.fit` bounds it at the machine
epsilon, the discovery engine at 1e-10, and for a gRNA whose cells are
separated by a covariate the two floors give different fits. The fixture's
`glm_extreme` row is such a gRNA: R's fitted means reach 2.2e-16, `glm.fit`
stops unconverged, the EM fails and the backup rule assigns the 15 cells. With
the 1e-10 floor the port's EM converged instead and assigned nothing, so the
mixture passes `mu_floor` = machine epsilon to `fit_poisson_glm_batch`, whose
default stays 1e-10.

### The default assignment design

`mixture_design_matrix` builds the design sceptre's `assign_grnas` uses by
default: an intercept, then `log(x)` -- or `log(x + 1)` when any cell has zero
-- of `response_n_nonzero`, `response_n_umis`, `grna_n_nonzero` and
`grna_n_umis`, then any extra covariates (15 or more distinct values: left
out; categorical: treatment dummies). `n_nonzero` counts entries above 0.5, as
`compute_cell_covariates` does. sceptre's `import_data` computes two further
gRNA columns, the top gRNA and its share of the cell, and deletes both before
any formula sees them, so neither enters. `response_p_mito` is always left out.
`design_from_covariates` applies the same rule to a covariate frame exported
from R.

### Thresholding and maximum

`assign_grnas_thresholding` and `assign_grnas_maximum` port sceptre's two
simpler methods. Neither fits anything, so agreeing with sceptre comes down to
which side of each cut a value falls on, where a tie goes, and what happens to
a cell with no gRNA UMIs. The rules below come from sceptre 0.10.3's R source
and, where the decision is made in its C++ (`threshold_count_matrix`,
`compute_cell_covariates_cpp`), from probing that code.

**Thresholding is `>=`.** A gRNA is assigned to every cell in which its UMI
count is at least `threshold` (default 5), as sceptre's
`threshold_count_matrix` does. A `threshold` below 1 is refused, as sceptre
refuses it. A cell can be assigned no gRNA, or several.

**Maximum gives every cell exactly one gRNA**: the one with the most UMIs in
it, the first in row order on a tie. The top gRNA and its share of the cell's
gRNA UMIs are the two columns sceptre's `import_data` computes with
`compute_cell_covariates`, `grna_feature_w_max_expression` and
`grna_frac_umis_max_feature`, the ones the default assignment design leaves
out (above).

**A cell with no gRNA UMIs is assigned the first gRNA**, and its share is 0/0,
NaN. R names the gRNA as `rownames(matrix_in)[out$max_feature + 1L]`, and
`max_feature` is 0 for such a cell. The port keeps both, for parity with
sceptre: what removes the cell is the UMI rule below, not its assignment.

**Two rules flag a cell, and their union is `cells_w_zero_or_twoplus_grnas`**,
the cells sceptre's low-MOI QC removes:

- the top gRNA holds at most `umi_fraction_threshold` (default 0.8) of the
  cell's gRNA UMIs: `max_grna_frac_umis <= umi_fraction_threshold`;
- the cell has fewer than `min_grna_n_umis_threshold` (default 5) gRNA UMIs:
  `grna_n_umis < min_grna_n_umis_threshold`.

R selects the first set with `which()`, which drops the NaN, so the share rule
never flags an empty cell; the UMI rule does, since 0 is below any positive
threshold. At `min_grna_n_umis_threshold = 0`, which sceptre allows, neither
rule flags it, and the empty cell keeps the first gRNA, in sceptre as here. The
port returns the flagged cells and does not remove them.

**What is refused, and the default.** sceptre refuses the maximum method for a
high-MOI screen, a `umi_fraction_threshold` outside (0, 1) and a negative
`min_grna_n_umis_threshold`, and so does the port. `assign_grnas_maximum`
takes no `moi`, so the MOI check is in `assign_grnas`, which refuses
`method="maximum"` with `moi="high"`. `method="default"` is sceptre's default,
the maximum method for `moi="low"` and the mixture for `moi="high"`, so it
needs `moi`.

**After a thresholding or mixture assignment** sceptre's low-MOI rule is the
one in `process_initial_assignment_list`: flag the cells assigned no gRNA, or
two or more. `cells_w_zero_or_twoplus_grnas` ports it, and it applies
unchanged to a fishash assignment. It departs from sceptre in one case, on
purpose. When no cell is assigned anything, sceptre computes the zero-gRNA
cells as `seq(1, n_cells)[-sort(unique(unlist(list)))]`, the index is empty,
and negative indexing by an empty vector selects nothing, so no cell is
flagged. The port flags every cell, since none carries a gRNA.

### Validated against R, and what that covers

The fishash and mixture fixtures (`scripts/dump_fishash_ground_truth.R`,
`scripts/dump_mixture_ground_truth.R`) record R's internals pass by pass, and
the tests feed R's own intermediate values back in so an error in one step
cannot hide behind another:

- fishash: log p-values to 1e-12 relative, the cut, every pass's B or
  n_signif and calls, the per-cell types and strings, the imputation's factors
  to 1e-10, on simulated and hand-built cases (empty guides and cells, a guide
  in every cell, rows whose unmasked counts vanish, ties, no signal, a supplied
  background, half-integer margins, stored zeros, degenerate shapes); and both
  places where R errors.
- the mixture: the design to 1e-15, the GLM to the existing 1e-6, the EM fed
  R's fitted means to 1e-9 for posteriors and 1e-12 for log-likelihoods, and
  the end-to-end assignments exactly.

Both fixtures keep every compared quantity away from a decision boundary (log
p at least 1e-6 from the cut; posteriors at least 1e-3 from 0.8), so any
disagreement in a call is a defect rather than a last-bit difference.

The thresholding and maximum methods are checked against sceptre 0.10.3's
public `assign_grnas` instead: `scripts/dump_assignment_rules_ground_truth.R`
runs it on synthetic counts with ties, empty cells and values on each cut, and
`test_assignment_rules_vs_r.py` compares the ports with its output, recorded in
`tests/validation/assignment_rules_ground_truth.json.gz`, value for value.

### What the assignment ports do not establish

Agreement with R says the ports compute what fishash and sceptre compute, not
that any of them assigns gRNAs well. fishash and the mixture are evaluated on
simulated screens by the runners in `scripts/fishash_eval/`; the results
belong with the manuscript, not here. No port reads anything but a count
matrix: the `.h5mu` exports this repository builds hold 0/1 assignments, not
gRNA UMI counts.
