# Roadmap

Agreed improvement plan. Tiers are priority order, not dependency order.
Evidence for each measured claim is in the item; unmeasured items say so.

Two standing constraints apply to everything below:

- **Never densify.** Keep large matrices sparse or stream them. This project
  has been bitten twice: a 38,606-gene x 131k-cell densify needed ~38 GiB and
  was OOM-killed, and `no_approximation` materializing all CRT draws needs
  5.2 GB per target (see T1.1).
- **Modern on-disk formats.** Parquet for tabular data, zarr for chunked
  nd-arrays. The raw `.bin` + sidecar `.txt` layout the moi5 scripts write is
  legacy (see T3.4).

## Tier 1 — correctness and safety

### T1.1 `no_approximation` can exhaust memory

Reachable since #2. Measured at moi5 scale (33,066 pairs, one-sided, alpha=0.1):

| | |
|---|---|
| `B_total` per target | 1,653,799 draws |
| Memory per target | 5.2 GB |
| x default `target_chunk_size=200` | 1.05 TB |
| At `target_chunk_size=1` | 5.2 GB |

Transient cost is worse: RSS grew 674 MB while materializing 158 MB of draws.
R does not hit this because it processes one target at a time.

- **Guard (short term):** refuse or warn when
  `B_total * chunk_size * median_n_trt` exceeds a memory budget, and
  auto-shrink `target_chunk_size` on this path.
- **Stream (proper fix):** an empirical p-value needs only a running count of
  `#{null >= obs}`, so draws never need to be materialized at once. Slice the
  draws into batches and update counters for every gene paired with the
  target. Bounded memory, no statistical change. Restructures the per-target
  loop, so it is its own PR.

**Reprioritised — this is worth more than originally scoped.** It was filed as
a `no_approximation` fix, but profiling the moi5 run shows draws dominate the
target stage on the *default* `skew_normal` path too: 16.5 MB of the 20.7 MB
per target, so a 193-target chunk holds 3.2 GB of draws out of a 3.99 GB
working set. Streaming would cut the dominant memory term for **every**
analysis, not just the pathological one. It is currently the single
highest-value memory change available.

### T1.4 `chunk_memory_gb` under-predicts actual RSS — PARTLY ADDRESSED

Measured on moi5: a 4.0 GB budget produced a 9.58 GB peak RSS -- roughly 2.4x.
The guard is behaving correctly (it capped the chunk at 193 targets); the
budget simply counts *logical array bytes* and ignores two things:

- construction transients -- `crt_index_sampler_fast` was measured growing RSS
  by 674 MB while materializing 158 MB of logical draws, about 4x;
- allocator retention -- freed arenas are not promptly returned to the OS, so
  a high-water mark accumulates across stages.

**Partly addressed.** The knob was renamed from `max_memory_gb` to
`chunk_memory_gb`, because it governs chunk sizing rather than process memory
and the old name invited exactly this confusion; its default dropped from 4.0
to 1.0, which measured both faster and leaner; and the h5ad input removed the
largest ungoverned term. Peak on moi5 went 9.77 GB -> 3.78 GB.

What remains: the per-column factor still under-predicts at small chunk sizes
because roughly 1 GB of the gene stage is a floor from per-gene
precomputation churn that no chunk size reduces, and freed arenas are not
returned to the OS so the peak is cumulative across phases rather than the
largest phase.

A user asking for 4 GB should not get 9.6 GB. Two options: document it
explicitly as a logical-bytes budget rather than an RSS guarantee, or apply a
calibrated safety factor. **Measure the budget-to-RSS relationship across
several budget values before choosing a factor** -- do not guess a constant.

Accounting for the moi5 peak, for reference:

| component | |
|---|---|
| target stage working set | 3.99 GB (193 x 20.7 MB) |
| gene stage working set | 1.02 GB |
| retained response CSR | 0.27 GB |
| parquet load transient | ~0.36 GB plus COO/CSR copies |
| unaccounted (transients + allocator) | ~4 GB |

### T1.2 R ground truth for `no_approximation`

The only path with no value-for-value R comparison; the committed fixture
holds `skew_normal` draws only. Extend `scripts/dump_r_ground_truth.R`.
Regenerating `ground_truth.json` means recording the sceptre version in the
commit, per the gotcha in `CLAUDE.md`.

### T1.3 Surface degenerate fits

`estimate_theta` computes warning flags that `fit_all_genes` discards
(`_method` is unused), and `compute_D_matrix` does `1/sqrt(eigvals)` with no
guard against a near-singular `Zt_wZ` from collinear covariates. Silent `inf`
in a scientific pipeline is the bad outcome.

Not an issue: the test statistic itself. `lower_left - lower_right` is a
variance and stays non-negative for the real `D`; an earlier NaN report came
from a benchmark using a random `D` that violated the invariant.

## Tier 2 — performance

**Benchmarking is paused** while pysceptre keeps changing -- see
`paper/status.md`. The work below is still worth doing; measuring it on the
container is what waits, so that one rebuild covers several changes.


Ranked by measured win over effort.

| Item | Win | Confidence |
|---|---|---|
| ~~T2.1 `reduceat` for the D row sums~~ | **superseded**: the gather it reduced is gone entirely (see T2.5) | done |
| ~~T2.2 Wire up flatten-once~~ | **done**: the draw matrix is built once per target | done |
| ~~T2.5 Sparse-matmul null statistic~~ | **42.7 -> 11.2 ms** on the hot call, **1.72x end to end** | measured |
| T2.3 Parallelize `fit_all_targets` | the remaining serial stage, now a larger share | unmeasured |
| T2.4 Sparse-aware `fit_all_genes` | removes the `(n_cells, n_genes)` densify | required by the no-densify constraint |

**T2.1/T2.2/T2.5 are done.** The per-pair statistic no longer gathers
`D[:, flat_idxs]` at all: the draws are a `(B, n_cells)` 0/1 CSR matrix and
`draws @ [a, w, D.T]` produces every segment sum in one matmul, with no
158 MB temporary. That subsumes both the `reduceat` work and the
flatten-once work, since the matrix is built once per target.

It also removes the T2.1 caveat rather than guarding it: `np.add.reduceat`
returned the element at the offset rather than `0` for a zero-length segment,
and needed correcting. An empty CSR row sums to zero on its own.

Measured against the previous implementation on 2,400 pairs: `fold_change`,
`z_orig` and `stage` bit-identical, and 2,353 of 2,400 p-values bit-identical.
The 47 that moved moved by at most **8.3e-17**, against a resampling
resolution of `1/(B1+1) = 2e-3` — so no empirical quantile flipped, and
Spearman rho is exactly 1.0. A sparse matmul accumulates in a different order
than a gather, which is where the last bits come from.

**T2.3 note:** the 48-minute target stage dominates end to end, so this is the
larger prize, but it is unmeasured and interacts with the deliberate
`threadpool_limits(1)` decision in `glm/irls.py`. Needs a design note first.

**T2.4** was originally ranked low value because the README tells callers to
pre-filter genes. The no-densify constraint promotes it: `fit_all_genes`
builds a dense `(n_cells, n_genes)` `Y` regardless of what the caller passed.

### T2.5 Retained gene precomputation — DONE

Resolved by storing what upstream stores. `perform_response_precomputation`
in sceptre returns only `fitted_coefs` and `theta`; `mu`, `w`, `a` and `D` are
rebuilt inside its per-pair loop. pysceptre now keeps the same two things.

| | retained per gene | 244 genes | genome-wide |
|---|---|---|---|
| before | 10.5 MB | 2.56 GB | 405 GB |
| after | 80 B | 0.02 MB | 3.1 MB |

Measured (131k cells, p=6) settling how to hold the pieces:

| | cost |
|---|---|
| recompute all four arrays | 1.59 ms |
| memory-map `D` alone from local disk | 1.04 ms |

Reading back just the largest array costs about as much as recomputing
everything from `p+1` numbers, so **parquet, zarr and memmap are all the wrong
answer here** -- there is nothing to spill. Parquet would be doubly wrong: it
is columnar and compressed, so each access decompresses a 6.3 MB array, and it
is built for analytical scans of tabular data rather than random access to
numeric arrays in a hot loop. (Parquet/zarr remain right for the moi5 *export*
files -- T3.4.)

pysceptre recomputes once per gene per target chunk rather than once per pair,
by iterating gene-outer inside each chunk. That is ~3,660 rebuilds at moi5
scale against sceptre's 33,066:

| | rebuilds | cost |
|---|---|---|
| sceptre, per pair | 33,066 | ~0.9 min |
| pysceptre, per gene per chunk | 3,660 | ~0.1 min |

Output is bit-for-bit identical, row order included.

### Note on the performance figures

`README.md`'s performance table does not reproduce. Running
the synthetic benchmark the table cites -- 586,309 cells / 292 genes /
3,026 targets / 34,177 pairs -- took **11.0 minutes** (662 s) against the
table's **~1 hour**. (That script has since been removed: its constants were
copied from the real day0_grna20 dataset, which we now hold, so simulating it
was pointless.)

| | this run | README table |
|---|---|---|
| total | **11.0 min** | ~1 hour |
| peak RSS | 13.88 GB | not recorded |
| swaps | 0 | not recorded |
| stage 1 / stage 2 pairs | 33,491 / 686 | not recorded |

It was **not** R sceptre: the table breaks down by pysceptre's own function
names, and this is pysceptre's own benchmark script reproducing the same
shape, with `fit_all_targets` dominant in both.

Of the ~49-minute gap, about 5.6 minutes is attributable to code -- #6 made
the per-pair statistic 3.4x faster, and the table allots that stage 8 minutes.
The rest is almost certainly the machine, and specifically its memory:

- The benchmark needs **~14 GB peak**. On a 16 GB VM that is borderline and on
  anything smaller it swaps, which is slow enough to explain a 5x gap on its
  own. This run recorded `0 swaps` only because the machine has 39 GB.
- That run also predates #12/#13, when `fit_all_targets` was unbounded and
  measured 13.1 GB by itself over 586k cells. A constrained VM would have
  swapped hardest in exactly the stage the table singles out as slowest
  (48 of ~60 minutes), while the much smaller gene fit stayed fast. Bounding
  it has addressed the worst of that.

The table is left as it stands rather than overwritten: it records someone's
real measurements, and replacing them with numbers from different hardware
would trade one unattributed table for another. What it needs is the hardware
it was measured on, or a re-run somewhere documented. This run was a 14-core
Apple silicon laptop with 39 GB.

## Tier 3 — release

- **T3.1 PyPI publish.** Blocked on account/token and on whether `0.1.0` is the
  version to make public. Prefer a `release.yml` using trusted publishing on
  tag, so no long-lived token is handled.
- **T3.2** `py.typed` marker.
- **T3.3** CHANGELOG.
- **T3.4 Parquet/zarr for the moi5 export.** The scripts currently write raw
  float64 `.bin` blobs with `.txt` sidecars for labels, which carries no dtype,
  shape, or column metadata. Zarr for the matrices, parquet for the tabular
  files.
- Docs site deferred: `README.md` is ~240 lines and still readable.

## Tier 5 — exploratory (low priority)

### T5.1 GPU support for the heavy stages — TABLED

Tabled deliberately, not forgotten: profiling the CPU path removed most of the
case for it.

The item was written when `docs/ROADMAP.md` and `README.md` both put
`fit_all_targets` at ~48 minutes, i.e. 80% of a one-hour run -- a big enough
prize to justify a GPU port. Measured at the benchmark's own scale (586,309
cells, p=6), that stage is about **5.5 minutes**, so the ceiling on *any*
parallelization of it is minutes rather than tens of minutes.

CPU-side measurements that closed the case:

| stage | measured (586k cells) | what the old table claimed |
|---|---|---|
| `fit_all_genes` | 0.5 min (292 genes) | 5.4 min |
| `fit_all_targets` | 5.5 min (2,875 targets) | 48 min |
| per-pair statistic | 3.1 min (34,886 pairs, stage-1) | 8 min |

Thread-level parallelism was also measured and is weak: 2.9x at 8 workers,
because the numba counting sort and numpy's `Generator` both hold the GIL.
Real parallelism would need processes plus `shared_memory` for the response
and covariate matrices -- plausibly ~3x end to end on 14 cores, and still the
first thing to try before any accelerator.

If this is ever revisited, the constraints recorded earlier still hold:
generate draws on-device (they are the large object), VRAM makes T1.1
streaming a prerequisite, `threadpool_limits(1)` is CPU-BLAS-specific and does
not carry over, and the CPU fallback must keep working via the same optional
extra pattern numba uses.

## Tier 4 — scope extensions

Reprioritised: the preprint now drives this tier. The calibration check is
committed; the rest waits on dataset and scope decisions tracked in
`paper/status.md`.

Each changes what the package is. Listed, not sized — these need a scientific
call on whether they are wanted.

| Item | Ports | Why it matters |
|---|---|---|
| **Calibration check** | `run_calibration_check` | **COMMITTED — required for the preprint.** Establishes that p-values are calibrated rather than merely concordant with R. Reuses the existing engine unchanged; only negative-control target construction and reporting are new. See `paper/status.md`. |
| Positive-control pairs | PC pair handling | R sizes `B3` off `max(discovery, positive_control)`; we collapse that for lack of a PC set. |
| `singleton` gRNA integration | `grna_integration_strategy` | Currently union-only. |
| `permutations` mechanism | permutation resampling | Would make `B3=24999` meaningful. |
| Covariate-matrix builder | `model.matrix()` equivalent | Explicitly out of scope today. |
