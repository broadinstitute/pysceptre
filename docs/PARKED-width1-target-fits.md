# Parked: fitting gRNA targets one at a time

**Status:** works, measured, bit-identical, and deliberately not merged.
Parked 2026-09-19; branch deleted 2026-09-21 once the reasoning was folded
in here. The implementation is recoverable from the branch's commit if it
is ever wanted, but the section below on Linux is the reason it should not
be.

This note exists so the next person -- possibly us -- does not have to
rediscover either the reasoning or the measurements.

## What the change does

`fit_all_targets` fits a chunk's gRNA targets as one batched logistic IRLS,
`k` indicator rows against a shared design matrix. This branch fits them
**one at a time** and maps those fits across worker threads, pinned by
`_TARGET_BATCH_WIDTH = 1`, mirroring `_GENE_BATCH_WIDTH = 1` on the gene
side.

It is only affordable because the design matrix's outer products became
run-scoped first (`glm.irls.x_outer_flat`, in PR #35). Before that, every
width-1 fit would have rebuilt its own 550 MB array.

## Why we tried it

**Invariance, not speed.** Batching makes a target's fitted probabilities
depend on which other targets share its chunk, and the batched solve is not
bit-for-bit across batch widths. Measured on 40 rows at 40,000 cells:

| split | vs one 40-row fit |
|---|---|
| width 20 | identical |
| width 10 | identical |
| width 5 | 8.5e-16 |
| width 1 | 2.6e-15 |

Stability above width 10 is BLAS blocking behaviour, not a promise. So
`chunk_memory_gb` was in principle a numerical knob, and had we parallelised
by *splitting* the batch across workers, `n_jobs` would have become one too.
The five-budget sweep found `chunk_memory_gb` result-neutral in practice,
but that was luck rather than guarantee.

Width 1 removes the dependence instead of bounding it: a target's fit is a
function of that target and the covariates alone.

The secondary hope was speed. Amdahl on CRT's scaling implied 46% of it ran
serially, and the serial stretch was this fit.

## What we measured

Day0, `n_jobs=8`, `chunk_memory_gb=1`, one process per run, both orderings
(these timings have been order-sensitive before):

| | wall | peak RSS | mean cores |
|---|---|---|---|
| batched (width 14) | 307.9 s / 308.8 s | 5.87 / 5.86 GB | 3.90 / 3.90 |
| width-1 | 300.6 s / 300.8 s | 5.46 / 5.51 GB | 5.24 / 5.23 |

- **wall -2.5%**, **peak -6.5%**, **CPU +31%** (1,203 -> 1,576 core-seconds)
- results **bit-identical**: 0 of 34,886 pairs moved on p-value, fold change
  or observed statistic, and stage counts are unchanged
- invariance confirmed directly: identical p-values across `n_jobs` 4 vs 8
  and across budgets 1 vs 2 GB

The feared ~1e-15 shift did not materialise at day0's dimensions. The
40-row synthetic test above did not transfer.

## Why we parked it

**The CPU is spent on work the batched version did not have to do.** Width-1
fits lose the batching amortisation, so +31% CPU buys -2.5% wall. That is
poor value where cores are billed or shared, which is the likely deployment.

**The fit is no longer the bottleneck it was written to fix.** After
prefetching the next chunk's targets (PR #35), most of the serial fit is
hidden behind the previous chunk's gene jobs. `n_jobs=4` lands within 3% of
`n_jobs=8` on this branch, so the remaining limit is elsewhere -- most
likely memory bandwidth in the per-gene-job working set, which is ~177 MB
per worker (`mu`, `w`, `a`, a `wZ` temporary, `D`, and `stacked`, which
copies `a`, `w` and `D.T` again).

**Renting fewer cores beats reclaiming them.** If CRT is flat from 4 to 8
workers, the answer is a smaller machine, not higher occupancy -- and then
width-1's CPU overhead is a cost with nothing to show for it.

## It does not help small machines either

Parking it, I speculated it might pay off where cores are scarce -- higher
occupancy letting a 4-core box reach what batched needs 8 for. Measured
across the range, it does the opposite:

| `n_jobs` | batched | width-1 | width-1 CPU |
|---|---|---|---|
| 2 | 403.6 s, 2.55 cores | **418.1 s**, 2.96 cores | +20% |
| 4 | 324.1 s, 3.43 cores | 311.5 s, 4.49 cores | +26% |
| 8 | 309.8 s, 3.90 cores | 302.9 s, 5.22 cores | +31% |

It is **slower at two workers**, and 4% and 2% faster at four and eight, for
20-31% more CPU throughout. There is no machine size at which it is the
cheaper choice, so the case for it rests entirely on invariance.

## Linux settled it: splitting the batch is the wrong mechanism

Measured later on x86_64 Linux, which the branch never saw. Width-1 across
eight threads matches a *single batched call* -- 0.85 s against 0.88 s --
while using three times the CPU, and its speedup caps at 1.59x however many
threads it is given. The work is bandwidth-bound and the per-iteration
Python overhead holds the GIL, so dividing a batch cannot help.

What did work was threads at the **pool** level rather than inside the fit:
the CRT builds a worker pool per chunk, 217 of them, and switching that pool
to threads took the CRT from 679.7 s to 531.2 s. That is now on `main` as
`gene_job_backend`.

So this branch is not merely uneconomic, it aimed at the wrong thing. The
fit does not parallelize by being split; the cost was never inside it.

## What would bring it back

- **A deployment where wall time is the only currency.** -2.5% and -6.5%
  memory for CPU nobody is counting is a fine trade.
- **A dataset where the fit dominates again** -- far more targets per gene,
  or far fewer cells, so the per-gene-job bandwidth ceiling lifts.
- **A way to keep the amortisation while parallelising.** The +31% is
  mostly redundant work, not work spread differently. Sub-batches wide
  enough to amortise but numerous enough to fill the workers would need
  chunks much wider than 14, and the sweep shows chunks past ~114 get slower
  (16 GB budget: 369.4 s against 8 GB's 352.9 s). Genuinely open.
- **Evidence that batch composition is perturbing a real result.** The
  invariance argument becomes decisive rather than precautionary, and the
  CPU stops mattering.

## How to revive it

`git cherry-pick` the single commit, or set `_TARGET_BATCH_WIDTH` in
`pipeline/discovery.py`. Any value >= the largest chunk restores the batched
behaviour without reverting the mapped-fit plumbing, which is useful for
A/B work.

## Related

- PR #35 -- prefetching, the run-scoped `X_outer_flat`, the
  `chunk_memory_gb` sweep
- `paper/status.md` -- the sweep table, and why peak RSS needs `n_jobs`
  stated beside it
