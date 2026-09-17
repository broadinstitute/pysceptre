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
  `#{null >= obs}`, so draws never need to be materialized at once. Slice B3
  into batches and update counters for every gene paired with the target.
  Bounded memory, no statistical change. Restructures the per-target loop, so
  it is its own PR.

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

Ranked by measured win over effort.

| Item | Win | Confidence |
|---|---|---|
| T2.1 `reduceat` for the D row sums | 61.7 -> 20.8 ms, ~2x the per-pair stage (~8 -> ~4 min at moi5) | measured, bit-identical |
| T2.2 Wire up flatten-once | ~1.2 min at moi5 (3%) | measured |
| T2.3 Parallelize `fit_all_targets` | targets the 48-min stage | unmeasured |
| T2.4 Sparse-aware `fit_all_genes` | removes the `(n_cells, n_genes)` densify | required by the no-densify constraint |

**T2.1 caveat:** `np.add.reduceat` returns the element at the offset rather
than `0` for a zero-length segment (verified: gives `1.0` where `0` is
correct). An empty CRT draw has probability ~`exp(-n_trt)` — negligible at
real scale, not impossible for tiny targets. Needs a guard and a test.

**T2.3 note:** the 48-minute target stage dominates end to end, so this is the
larger prize, but it is unmeasured and interacts with the deliberate
`threadpool_limits(1)` decision in `glm/irls.py`. Needs a design note first.

**T2.4** was originally ranked low value because the README tells callers to
pre-filter genes. The no-densify constraint promotes it: `fit_all_genes`
builds a dense `(n_cells, n_genes)` `Y` regardless of what the caller passed.

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

### T5.1 GPU support for the heavy stages

Low priority: none of it is measured, and the CPU path has not yet had its
cheap wins taken (T2.1-T2.4). Recorded now so the design constraints are not
forgotten.

The three hot stages are all GPU-shaped:

| Stage | Work | GPU fit |
|---|---|---|
| `fit_all_targets` (~48 min, dominant) | batched logistic IRLS + CRT draws | good: dense batched matmuls, per-cell binomial draws |
| `fit_all_genes` (~5 min) | batched Poisson IRLS | good: same shape |
| per-pair statistic (~8 min) | gathers (`a[flat_idxs]`, `D[:, flat_idxs]`) then segment sums | good: gather + scatter-add are native GPU primitives |

Design constraints, to save rediscovering them:

- **Generate draws on-device.** The CRT draws are the large object (5.2 GB per
  target for `no_approximation`, ~17 MB for `skew_normal`). Host-to-device
  transfer would likely dominate, so `crt_index_sampler_fast` should draw
  directly on the GPU rather than shipping CPU draws across.
- **VRAM is much smaller than host RAM** (typically 16-80 GB), so the
  no-densify constraint and the T1.1 streaming work matter *more* here, not
  less. Do T1.1 first.
- **`threadpool_limits(1)` in `glm/irls.py` is a CPU-BLAS decision** and does
  not carry over; the thin-matmul pathology it works around is specific to
  multi-threaded OpenBLAS.
- **Keep the CPU path working.** Mirror the existing optional-dependency
  pattern: `crt/sampler.py` already try/excepts numba and falls back to pure
  numpy. A `gpu` extra in `pyproject.toml` alongside `fast` fits naturally.
- **Library choice is open.** CuPy is the most drop-in (numpy-compatible API,
  so `_segment_sums` and the IRLS loop could be largely shared);
  `numba.cuda` would avoid adding a second accelerator dependency given numba
  is already an extra. Prototype one stage before committing to either.

First concrete step, if picked up: port `fit_all_targets` alone behind the
extra and measure against the 48-minute CPU baseline. That stage dominates,
so it decides whether the rest is worth doing.

## Tier 4 — scope extensions

Each changes what the package is. Listed, not sized — these need a scientific
call on whether they are wanted.

| Item | Ports | Why it matters |
|---|---|---|
| Calibration check | `run_calibration_check` | Establishes that p-values are calibrated before discoveries are trusted. Strongest candidate. |
| Positive-control pairs | PC pair handling | R sizes `B3` off `max(discovery, positive_control)`; we collapse that for lack of a PC set. |
| `singleton` gRNA integration | `grna_integration_strategy` | Currently union-only. |
| `permutations` mechanism | permutation resampling | Would make `B3=24999` meaningful. |
| Covariate-matrix builder | `model.matrix()` equivalent | Explicitly out of scope today. |
