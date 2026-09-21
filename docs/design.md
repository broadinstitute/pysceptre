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
    statistic is now a sparse matmul, and permutations reach it through a
    prefix scan. Both release the GIL. The old figure survived in a
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

R places treated cells by a without-replacement Fisher-Yates shuffle. The
port draws **with** replacement and deduplicates with `np.unique`. This is an
approximation, and an intentional one: the number of treated cells for a
target is far smaller than the number of resamples, so collisions are rare.

The alternative -- a dense `(n_cells, B)` matrix, the straightforward
translation -- is what made the first port unusably slow at real scale. The
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

### Five things that look like simplifications and are not

Each of these turns the estimator into one that nothing has scored.

**`num_trt_cells` is the sum over a target's gRNAs, not the union of its
perturbed cells.** `compute_power_posthoc` groups `cells_per_grna` by target
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
and `compute_power_posthoc` raises with that explanation rather than returning
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

The estimator's inputs are not properties of the *pair*: expression mean and
dispersion belong to the gene, cell counts to the element. That is measured
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

### Building the estimator's inputs

`analytical_power/inputs.py` derives the three things `compute_power_posthoc`
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
`compute_power_posthoc` originally refused a repeated `grna_id` for that
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
