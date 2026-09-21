# pysceptre

A standalone Python port of the statistical engine behind
[`sceptre`](https://timothy-barry.github.io/sceptre-book/)'s discovery
analysis for single-cell CRISPR screens -- specifically the **complement
control group + CRT (conditional randomization test) resampling** path used
for high-MOI data.

It covers three of sceptre's analysis steps on that path -- the **discovery
analysis**, the **calibration check** and the **power check** -- and is *not*
a general
reimplementation. Targeting one validated path is what lets it batch the
linear-algebra work sceptre does per-gene/per-target in R/C++ loops into
vectorized numpy calls, cutting real-dataset runtimes from hours to tens of
minutes. See [Scope and limitations](#scope-and-limitations) for exactly what
is and isn't covered.

## Why this exists

<!-- --8<-- [start:why] -->

`sceptre`'s own discovery-analysis implementation is correct but processes
one (gene, gRNA-target) pair, and one resampling draw, largely with R and
per-cell C++ loops. At real dataset scale (hundreds of thousands of cells,
thousands of gRNA targets, tens of thousands of pairs) that adds up. Every
GLM fit in this pipeline shares one design matrix across many response
columns (all genes share the covariate matrix; all gRNA targets share it
too), which is exactly the shape numpy's batched matrix operations are
built for -- so the rewrite fits one batched IRLS call per resampling stage
instead of one per gene or target.

<!-- --8<-- [end:why] -->

## Installation

<!-- --8<-- [start:installation] -->

```bash
pip install -e ".[dev,fast]"
```

- `dev` installs `pytest` (needed for the test suite) plus `ruff` and
  `pre-commit` for linting and formatting.
- `fast` installs [`numba`](https://numba.readthedocs.io/), which JIT-compiles
  the CRT sampler's cell-grouping step (a counting sort). Without it,
  `pysceptre` falls back to a slower pure-numpy `argsort`-based version
  automatically -- everything still works, just slower at real dataset scale.
- `io` installs `mudata`, `anndata`, `pyarrow` and `matplotlib`. These are
  needed only by the dataset-export and benchmarking scripts under
  `scripts/`, **not** by the package: `run_discovery_analysis` takes in-memory
  arrays, so calling it does not require them.

Requires Python >= 3.10. Core dependencies: `numpy`, `scipy`, `pandas`,
`threadpoolctl`.

### Datasets

The analysis functions take plain arrays, so any route that produces them
works. The `scripts/` directory also carries one out of R end to end:
`export_sceptre_dataset.R` extracts a post-QC `sceptre_object`, and
`make_h5mu.py` converts that to a **MuData `.h5mu`** with two assays --
`rna` (cells x genes counts) and `grna` (cells x assignment units, whose
`var` marks each unit as a per-target union or an individual non-targeting
gRNA). `scripts/sceptre_io.load_export` reads one back, and with
`backed=True` serves genes from disk on demand rather than holding the whole
matrix.

<!-- --8<-- [end:installation] -->

## Quick start

<!-- --8<-- [start:quickstart] -->

```python
from pysceptre.pipeline.api import run_discovery_analysis

result = run_discovery_analysis(
    response_matrix=response_matrix,      # (n_genes, n_cells) dense array or scipy.sparse
    gene_ids=gene_ids,                    # row labels for response_matrix, in order
    covariate_matrix=covariate_matrix,    # (n_cells, p) numeric design matrix
    grna_target_cells=grna_target_cells,  # dict[target_id -> 0-based treated-cell indices]
    pairs=pairs,                          # DataFrame['response_id', 'grna_target']
    side="left",
    seed=0,
)
# result: DataFrame with one row per pair --
#   response_id, grna_target, p_value, fold_change, se_fold_change,
#   pct_change_es, pct_change_es_ci_low, pct_change_es_ci_high, z_orig, stage
```

[`examples/scanpy_interop.py`](https://github.com/broadinstitute/pysceptre/blob/main/examples/scanpy_interop.py) runs a whole screen
inside an ordinary `scanpy` workflow -- QC and covariates out of `obs`, gRNA
assignments out of the gRNA modality's `var`, discovery and calibration, then
results back onto the same object and on to PCA and clustering, with no R at
any point. Run it with `uv run --extra examples python
examples/scanpy_interop.py`.

See [TUTORIAL.md](https://github.com/broadinstitute/pysceptre/blob/main/TUTORIAL.md) for a complete, runnable walkthrough
(including how to build each input from scratch) and for guidance on
picking `target_chunk_size` for your dataset.

<!-- --8<-- [end:quickstart] -->

## API reference

### `pysceptre.pipeline.api.run_discovery_analysis`

The one function most users need.

```python
run_discovery_analysis(
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    grna_target_cells: dict[str, np.ndarray],
    pairs: pd.DataFrame,
    *,
    side: str = "both",
    resampling_approximation: str = "skew_normal",
    seed: int | None = None,
    target_chunk_size: int = 200,
    chunk_memory_gb: float = 1.0,
) -> pd.DataFrame
```

| Parameter | Type | Description |
|---|---|---|
| `response_matrix` | `(n_genes, n_cells)` dense `ndarray`, `scipy.sparse` matrix, or a backed reader | Gene expression counts. Rows must correspond 1:1 with `gene_ids`, in order. Passing a genome-wide matrix is fine: only genes appearing in `pairs` are fitted, so untested rows cost storage but not compute. (Earlier versions fitted every row and this table told you to pre-filter; that is no longer necessary.) |
| `gene_ids` | `list[str]` | Row labels for `response_matrix`, in the same order as its rows. Matched against `pairs['response_id']`. |
| `covariate_matrix` | `(n_cells, p)` `ndarray` | Already formula-expanded numeric design matrix (intercept column, `log(umis)`, batch dummies, etc. -- whatever R's `model.matrix()` would have produced). `pysceptre` does not parse an R-style formula DSL; build this matrix yourself, or extract it directly from an existing `sceptre_object`'s `@covariate_matrix` slot. |
| `grna_target_cells` | `dict[str, np.ndarray]` | Maps each gRNA target to the **0-based** indices (into `covariate_matrix`'s cell axis) of cells treated with that target. This is the "union" grna-integration-strategy convention: one entry per target, not per individual gRNA. |
| `pairs` | `pd.DataFrame` with columns `response_id`, `grna_target` | The QC-passed (gene, target) pairs to test. `pysceptre` does not run `assign_grnas()`/`run_qc()` itself -- feed it pairs that have already passed QC upstream. |
| `side` | `"left"` \| `"both"` \| `"right"` | Test sidedness, matching sceptre's own convention. Use `"left"` for expected-repression screens (e.g. CRISPRi enhancer knockdown), `"both"` for a two-sided test. |
| `resampling_mechanism` | `"crt"` \| `"permutations"` | Matches sceptre's own option for high-MOI data. The CRT (default) draws each target's synthetic treated set from that target's own fitted probabilities; permutations draw one set of random subsets, sized by the largest target, and reuse it for every target. **The choice is a real trade, and yours to make** -- see [Reproducibility](#reproducibility-and-incremental-analysis), because permutations cannot offer the invariance the CRT does. R pairs permutations with `B3 = 24999` against the CRT's `0`, so sampling is cheaper -- and the per-target logistic fit is skipped entirely, since only the CRT draws from it -- but the escalation batch is five times larger. |
| `resampling_approximation` | `"skew_normal"` \| `"no_approximation"` | `"skew_normal"` (default, matching sceptre): pairs whose initial empirical p-value (`B1=499` draws) is `<= 0.02` get a skew-normal tail fit from a further `B2=4999` draws, giving p-values far smaller than `1/(B1+1)` could resolve. `"no_approximation"` fits no curve and instead draws a third, larger empirical batch, sized by R's own rule: `B3 = ceil(mult * n_pairs / multiple_testing_alpha)`, `mult = 10` two-sided and `5` one-sided. That grows linearly in the number of pairs and is much slower -- see [Scope and limitations](#scope-and-limitations). Any other value raises `ValueError`. |
| `seed` | `int \| None` | Seeds the `numpy.random.Generator` used for all CRT draws in the run. Note this does **not** reproduce sceptre's own R/C++ RNG stream bit-for-bit (different algorithm and seeding scheme) -- see [Scope and limitations](#scope-and-limitations). |
| `target_chunk_size` | `int` | How many gRNA targets to fit and CRT-draw at once. An **upper bound, not a mandate** -- it is reduced automatically to respect `chunk_memory_gb`, so no value here can exhaust memory. Default `200`. |
| `n_jobs` | `int` | Workers for the per-pair tests, which are ~80% of the runtime. `1` (default) runs serially; a negative value uses every core. **Results do not depend on it** -- only the genes inside an already-drawn target chunk are distributed, so the resampling draws are made in the same order at any worker count, and output is bit-identical. Processes on Linux, threads elsewhere (`fork` after macOS's Accelerate BLAS can deadlock), so the ceiling is lower off Linux. Memory grows by about one gene's working arrays per worker, not by `chunk_memory_gb` per worker. |
| `chunk_memory_gb` | `float` | Budget for the arrays a *chunk* holds, which sizes how many genes or targets are processed together. **Not** a cap on the process's memory -- the input, retained state and allocator overhead sit outside it. **You should not normally need to change this.** The default is both the fastest and the leanest setting measured: a larger budget produces chunks past the point where batching still pays, costing memory for no throughput (4 GB gave 8.42 GB peak against 3.78 GB at 1 GB, for the same runtime). Default `1.0`. |

**Returns** a `pd.DataFrame`, one row per input pair, with columns:

| Column | Meaning |
|---|---|
| `response_id`, `grna_target` | Echoed from `pairs`. |
| `p_value` | The test p-value (see `stage`). |
| `fold_change` | Estimated fold change of the treated group vs. complement control (deterministic given the data -- no resampling randomness). |
| `pct_change_es` | Effect size as a percent change from baseline, `(fold_change - 1) * 100`. Named `_es` rather than `pct_change` because `DataFrame.pct_change` is a pandas *method*: `result.pct_change` would silently return the method instead of the column, and comparisons against it fail with a confusing `TypeError` rather than a `KeyError`. |
| `pct_change_es_ci_low`, `pct_change_es_ci_high` | Two-sided 95% Wald interval on `pct_change_es`, from `se_fold_change`. Deterministic, unlike the resampled p-value, so it is a useful independent read on pairs sitting near a significance threshold. Note it is a normal approximation, not sceptre's test. |
| `log2(fold_change)` | Not returned -- a pure transform of a column already present. Compute it if you need it. |
| `se_fold_change` | Standard error of `fold_change`, on the same ratio scale, so `fold_change ± 1.96 × se_fold_change` is a Wald interval around 1 (no effect). Deterministic, unlike the resampled p-value. |
| `z_orig` | The observed test statistic (before resampling). |
| `stage` | `1` = reported from the initial `B1=499`-draw empirical p-value (not significant enough to escalate). `2` = escalated to a skew-normal tail fit on `B2=4999` further draws. `3` = an empirical p-value from a further batch, reached either because `resampling_approximation="no_approximation"` (so no curve is fit, and the `B3` draws are used) or because a skew-normal fit was attempted and *rejected* (see `check_sn_tail`/`check_for_outliers` in `test_statistic/skew_normal.py`), in which case the already-drawn `B2=4999` statistics are used. |

### `pysceptre.pipeline.api.run_calibration_check`

Runs the discovery test over **synthetic negative-control targets**, built by
regrouping individual non-targeting (NTC) gRNAs. No target is real, so a
correctly calibrated method returns p-values uniform on (0, 1) -- and that
uniformity, not agreement with any other implementation, is what the check
measures.

```python
from pysceptre import run_calibration_check

calib = run_calibration_check(
    response_matrix=response_matrix,
    gene_ids=gene_ids,
    covariate_matrix=covariate_matrix,
    ntc_grna_cells=ntc_grna_cells,   # dict[NTC gRNA id -> 0-based cell indices]
    n_calibration_pairs=len(pairs),
    calibration_group_size=15,
    n_nonzero_trt_thresh=7,
    n_nonzero_cntrl_thresh=7,
    side="left",
    seed=0,
)
```

Returns the same columns as `run_discovery_analysis`, minus any `pass_qc`
column -- see below for why there isn't one.

| Argument | Notes |
|---|---|
| `ntc_grna_cells` | `dict[NTC gRNA id -> 0-based cell indices]`. Keyed by **individual gRNA**, not by target. A target-keyed mapping collapses every NTC into one entry, and in sceptre's own object omits them entirely (`"non-targeting"` is not a key in `grna_group_idxs`), leaving nothing to regroup. |
| `n_calibration_pairs` | How many pairs to test. R defaults this to the number of discovery pairs that passed QC. |
| `calibration_group_size` | NTC gRNAs per synthetic target. R's default is the median gRNAs per real target, capped at the NTC count; that median is not derivable from these arguments, so it is required here. |
| `n_nonzero_trt_thresh`, `n_nonzero_cntrl_thresh` | Pairwise QC thresholds, sceptre's defaults being `7`. Prefer your object's own values. |
| `pass_qc_rate` | R's `p_hat`, the fraction of discovery pairs clearing QC, which sizes how many synthetic groups get built. Only matters when the group count is above its floor of 100 -- but there it is decisive. |
| `negative_control_pairs` | Test exactly these pairs instead of constructing any, with `grna_target` entries being `&`-joined NTC gRNA ids. This is how you compare against an R result pair-by-pair. |

**QC works differently here, deliberately.** A discovery result reports QC
failures in-band (`pass_qc = False`, NaN p-value). A calibration check
*constructs* its pairs and only ever samples combinations that already clear
the thresholds, so every returned row passes and there is no `pass_qc`
column. Pairwise nonzero-count filtering is therefore inseparable from
building the pairs; cell-level and gRNA-level QC remain out of scope.

**R's own pair selection is not reproducible.** Nothing in sceptre's
calibration path calls `set.seed` and `sceptre_object` has no seed slot, so
re-running R gives a different pair set. Comparing pair-by-pair against an R
result means passing R's pairs back in via `negative_control_pairs`.

### `pysceptre.pipeline.api.run_power_check`

Runs the discovery test over **positive controls** -- pairs where an effect
is expected, usually a gRNA against the gene's own TSS. Where the calibration
check asks whether the pipeline invents effects it should not, this asks
whether it recovers effects it should. The two are read together.

```python
from pysceptre import run_power_check

power = run_power_check(
    response_matrix=response_matrix,
    gene_ids=gene_ids,
    covariate_matrix=covariate_matrix,
    grna_target_cells=grna_target_cells,
    positive_control_pairs=positive_control_pairs,  # response_id, grna_target
    side="left",
    seed=0,
)
```

| Argument | Notes |
|---|---|
| `positive_control_pairs` | The pairs to test. **Supply these**: which target perturbs which gene is a claim only the experiment can make. Omitted, sceptre's name-matching rule is used -- a target that is itself a gene id pairs with that gene -- which works when targets are named after genes and finds *nothing* when they are named after genomic intervals. On a real screen of the latter kind it matched 0 of 3,071 targets, so that case raises rather than quietly returning an empty result. |
| `n_nonzero_trt_thresh`, `n_nonzero_cntrl_thresh` | Pairwise QC thresholds. |

**QC is reported, not filtered** -- the opposite of the calibration check.
The result has one row per supplied pair, with `pass_qc`, `n_nonzero_trt` and
`n_nonzero_cntrl`, and NaN results where a pair did not meet the thresholds.
Dropping those would overstate power by hiding exactly the controls the
screen had too few cells to test.

**No multiple-testing correction is applied**, matching R, which returns no
`significant` column here. These are a diagnostic rather than discoveries.

### Lower-level building blocks

`run_discovery_analysis` is a thin wrapper around
`pysceptre.pipeline.discovery.run_discovery_ntcells_complement`, which
exposes a couple of additional knobs not surfaced at the top level (fixed
`B1=499, B2=4999, B3=0`, `fit_parametric_curve: bool`, `side_code: int`
instead of `side: str`). Most users won't need to go lower than
`run_discovery_analysis`, but each pipeline stage is also independently
importable and unit-tested, for anyone extending or debugging the pipeline:

| Module | What it does |
|---|---|
| `pysceptre.glm.irls` | Batched IRLS for Poisson (log link, gene fits) and binomial (logit link, gRNA-target fits) GLMs -- `fit_poisson_glm_batch`, `fit_binomial_glm_batch`. Ports R's `stats::glm.fit` algorithm and convergence criteria exactly. |
| `pysceptre.glm.nb_theta` | Negative-binomial dispersion (`theta`) estimation given a fitted mean -- `estimate_theta`. Exact port of sceptre's `estimate_theta`. |
| `pysceptre.precompute.pieces` | Per-gene precomputation pieces (the `D` matrix and friends) needed by the test statistic -- `compute_precomputation_pieces`. |
| `pysceptre.crt.sampler` | The CRT resampling draw itself -- `crt_index_sampler_fast` (sparse, real-scale-feasible; numba-accelerated if available) and `crt_index_sampler_naive` (dense reference implementation, used only for cross-checking in tests). |
| `pysceptre.test_statistic.score_stat` | The O(n_treated)-per-resample score-type test statistic -- `compute_observed_full_statistic`, `compute_null_full_statistics`. |
| `pysceptre.test_statistic.empirical_p` | Empirical p-value from a null distribution -- `compute_empirical_p_value`. |
| `pysceptre.test_statistic.skew_normal` | Skew-normal tail-fit escalation -- `fit_and_evaluate_skew_normal`. Exact port of `fit_skew_normal_funct`/`check_sn_tail`/`check_for_outliers`/`fit_and_evaluate_skew_normal`. |
| `pysceptre.test_statistic.fold_change` | Fold-change estimation -- `estimate_log_fold_change`. |
| `pysceptre.test_statistic.resampling` | Ties the above into the `B1 -> B2 -> B3` staged escalation for one pair -- `run_low_level_test_full`. |
| `pysceptre.pipeline.discovery` | Orchestration: `fit_all_genes`, `fit_all_targets`, `run_discovery_ntcells_complement`. |

## Reproducibility and incremental analysis

<!-- --8<-- [start:reproducibility] -->

A pair's result depends on `(seed, response_id, grna_target)` and the data,
and on nothing else about the run. Each target draws its CRT resamples from
its own stream keyed on the target's *name*, so results do not depend on the
order targets are processed, the chunk size, the worker count, or **which
other pairs are in the analysis**.

That makes an incremental workflow safe: run a subset, check it, then run the
full set and reuse what you already have. The pairs in common come back
identical rather than merely similar.

**This applies to the CRT, and cannot apply to permutations.** With
`resampling_mechanism="permutations"` the draws are made once and shared by
every target, sized by the largest target present, so adding a target bigger
than the current largest changes every result in the run. That is inherent to
sharing one draw set -- it is what makes permutations cheap -- not a defect
that could be fixed. Permutation runs remain fully deterministic for a fixed
pair list, and independent of chunk size and worker count; they are simply
not invariant to changing the analysis. Use the CRT if you intend to extend
an analysis and reuse earlier results.

Two limits on the CRT path, both worth knowing exactly:

| change between runs | shared pairs |
|---|---|
| more or fewer targets or pairs; different order; different `target_chunk_size` or `n_jobs` | **bitwise identical** |
| a different set of *genes* | identical to ~1e-16 |

The second is batched linear algebra, not randomness: gene GLMs are fitted in
batches, and changing the batch width changes the order of floating-point
additions. Measured at one unit in the last place (1.11e-16 on `fold_change`,
3 of 72 pairs) on OpenBLAS, and exactly zero on Apple Accelerate. It cannot
be removed without fitting genes one at a time, which is the batching this
package exists to do.

**Adjusted p-values are a different matter and must be recomputed.** A
Benjamini-Hochberg adjustment depends on every p-value in the set, so adding
pairs changes the adjusted value, and possibly the call, for pairs already
tested. That is multiple testing behaving correctly. `run_discovery_analysis`
returns raw p-values and applies no correction, so cache those and run the
adjustment over the union each time.

<!-- --8<-- [end:reproducibility] -->

## Scope and limitations

<!-- --8<-- [start:scope] -->

- **Complement control group only, high-MOI/CRT resampling only.** This is
  the one analysis path this package targets; other sceptre modes
  (permutations, non-complement control groups, low-MOI) are out of scope.
- **No `assign_grnas()` / `run_qc()`**, with one carve-out: the calibration
  check applies the *pairwise* nonzero-count thresholds, because it builds its
  own pairs and cannot select them otherwise. Cell-level and gRNA-level QC
  are still out of scope. Feed `run_discovery_analysis` pairs
  that have already passed QC (e.g. from a real `sceptre_object`'s
  `@discovery_pairs_with_info`, filtered to `pass_qc == TRUE`). This package
  is the statistical engine only.
- **No formula DSL.** `covariate_matrix` must already be a plain numeric
  design matrix; there's no `model.matrix()`-equivalent formula parser here.
- **RNG is not bit-for-bit reproducible against R.** sceptre seeds
  `boost::mt19937` with a fixed literal seed on every resampling call;
  `pysceptre` uses `numpy.random.Generator` with a different algorithm and
  seeding scheme. Validated against R by matching *distributions* (the
  resulting null-statistic and p-value distributions), not exact draws --
  see [Validation](https://github.com/broadinstitute/pysceptre/blob/main/README.md#validation).
- **`B1`/`B2`/`B3` are not exposed at the top-level API**, but they are no
  longer fixed: `run_discovery_analysis` derives them from
  `resampling_approximation` exactly as R does -- `B1=499` always, then
  `(B2, B3) = (4999, 0)` for `skew_normal` and `(0, ceil(mult * n_pairs /
  multiple_testing_alpha))` for `no_approximation`. `B3=0` on the
  `skew_normal` path is parity with R, which only uses `B3=24999` for the
  `permutations` mechanism this package doesn't implement.
  `run_discovery_ntcells_complement` and `run_low_level_test_full` accept all
  three directly if you need to override them.
- **`no_approximation` is expensive, and can be coarser at small scale.** Its
  `B3` grows linearly in pair count: a 33,066-pair one-sided run needs
  1,653,300 draws *per target* (5.2 GB), so the chunk collapses to a
  single target and the run warns that it may still exhaust memory.
  Conversely, at 4 pairs one-sided `B3 = 200`, *below* `B1 = 499`. Both are
  R's behavior, reproduced rather than corrected.
- **gRNA integration strategy: "union" only.** `grna_target_cells` is keyed
  by target, not by individual gRNA -- matches sceptre's `"union"` strategy;
  `"singleton"` is not supported.

<!-- --8<-- [end:scope] -->

## Performance

<!-- --8<-- [start:performance] -->

Benchmarked on the `day0_grna20` single-cell CRISPR screen: 567,690 cells
after QC, 237 genes appearing in pairs, 3,071 gRNA targets, 34,886 QC-passing
pairs, two-sided, CRT resampling. The headline figures are below, each
reported with the hardware it was measured on; the full tables and the
comparison against R sceptre are in the manuscript repository,
`broadinstitute/pysceptre-paper`, which stays private until the preprint is
posted.

Reproduce with `scripts/benchmark_vs_r.R` (R side, which also exports the
exact inputs) and `scripts/benchmark_pysceptre.py` (pysceptre side), or inside
the pinned container in `docker/`.

### Sizing the machine

**The two mechanisms want different machines, so `n_jobs` is worth setting
deliberately rather than to the core count.** Measured on day0, one process
per configuration, Apple M4 Max; `cores` is `(user + sys) / wall`, the mean
number actually busy.

| `resampling_mechanism` | `n_jobs` | wall | peak RSS | cores |
|---|---|---|---|---|
| `"crt"` (default) | 1 | 555.6 s | 3.63 GB | 1.09 |
| | 4 | 165.3 s | 6.20 GB | 5.00 |
| | 8 | **138.2 s** | 6.81 GB | 6.74 |
| `"permutations"` | 1 | 125.3 s | 3.47 GB | 1.06 |
| | 8 | **30.8 s** | 5.85 GB | 6.53 |

**Both mechanisms now use a large machine, so give them one.** The CRT
reaches 6.74 of 8 cores and gains 1.20x from four workers to eight;
permutations reach 6.53 and 30.8 s. Neither is memory-constrained at these
sizes -- the largest peak here is 6.8 GB, inside what a 4-vCPU cloud
instance ships with, and cloud machine types bundle memory with cores
anyway, so asking for fewer cores buys less memory rather than a cheaper
machine at the same memory.

*This guidance changed.* The CRT used to cap near 3.90 cores with four
workers within 4.4% of eight, because its per-chunk logistic fit ran on one
thread and nothing else could proceed past it. Chunks are now prepared
several deep, so several fits run concurrently, and the ceiling moved. If
you tuned `n_jobs` down for the CRT on the old advice, undo it.

**Memory rises with workers and with pipeline depth, not with
`chunk_memory_gb`.** Each worker holds one gene's working arrays and the
pipeline holds `_PREFETCH_DEPTH` chunks of draws, so peak tracks those two;
the chunk budget is a weaker lever than it looks. Raising it from 1 GB to
8 GB buys 13% on the CRT for 2.3x the memory, and past that it gets
*slower* -- 16 GB ran 369 s against 8 GB's 353 s. Leave it alone unless you
have measured otherwise on your own data. Results are identical at every
setting either way.

<!-- --8<-- [end:performance] -->

## Validation

Two independent validation layers, both against a real, installed `sceptre`
R package (pinned upstream commit), not just internal self-consistency:

1. **Synthetic ground truth** (`tests/validation/`): small hand-built
   matrices run through both sceptre's internal (`:::`) R functions
   directly and the corresponding pysceptre function, compared value-for-
   value. Covers GLM fits, NB dispersion estimation, precomputation pieces,
   the test statistic, empirical p-values, skew-normal fitting, and the CRT
   sampler's *distribution* (not exact draws -- see
   [Scope and limitations](#scope-and-limitations)). Run via
   `scripts/dump_r_ground_truth.R` + `pytest tests/validation/`.

2. **Real dataset: discovery analysis** (day0_grna20: 567,690 cells /
   292 genes / 3,026 targets / 34,886 QC-passed pairs). The same screen was
   run through `pysceptre.run_discovery_analysis` and R's
   `sceptre::run_discovery_analysis()`, with R exporting the exact object it
   analysed so the two cannot disagree about their inputs:

   | Metric | Value |
   |---|---|
   | Fold-change agreement (Pearson r) | 1.000000 (max difference 2.6e-12) |
   | p-value agreement (Spearman rho) | 0.9865 |
   | p-value agreement (Pearson r on -log10 p) | 0.9940 |

   Significance calls at BH 0.1, with R as the reference:

   | | R: significant | R: not |
   |---|---|---|
   | **pysceptre: significant** | 251 | 10 |
   | **pysceptre: not significant** | 3 | 34,622 |

   Sensitivity 0.9882, specificity 0.9997, 13 discordant pairs of 34,886.
   Reported as a 2x2 table rather than a single index, because a Jaccard
   coefficient collapses it and cannot distinguish calling extra hits from
   missing them, which are different failures.

   Fold change carries the weight here: no resampling enters it, so agreement
   to 2.6e-12 establishes that both implementations fit the same models to
   the same cells. The p-value correlation is then bounded below by Monte
   Carlo noise -- pysceptre agrees with R about as closely as it agrees with
   itself across seeds. Every one of the 13 discordant pairs has an interval
   excluding no effect and a p-value within a factor of four of the
   threshold, so the disagreement is about which side of a cutoff a resampled
   p-value fell on, never about whether an effect was detected.

3. **Calibration check** (day0_grna20: 567,690 cells / 292 genes / 2,031
   non-targeting gRNAs / 34,886 negative-control pairs), with R's own pairs
   injected so both sides test identical ones:

   | Metric | Value |
   |---|---|
   | Pairs merged | 34,886 / 34,886 |
   | Fold-change agreement (Pearson r) | 1.0 (max abs difference 4.5e-11) |
   | p-value agreement (Spearman rho) | 0.9863 |
   | KS vs `U(0,1)` | 0.02592 (pysceptre) vs 0.02594 (R) |
   | False discoveries, BH at 0.1 | 7 (pysceptre) vs 6 (R) |

   Before any p-value, a deterministic checkpoint: rebuilding R's synthetic
   groups from their names and recomputing the pairwise counts reproduces R
   **exactly** -- `n_nonzero_trt` and `n_nonzero_cntrl` both 34,886/34,886,
   zero discrepancies over ~70,000 integer comparisons.

   The negative-control p-values are **not perfectly uniform** -- KS 0.0259
   against a 5% critical value of 0.0073, with 6.1% below 0.05 where 5% is
   expected. R deviates by the same amount to four decimal places, so this is
   sceptre's behaviour on this data and not an artefact of the port; pysceptre
   reproduces the reference's mild anti-conservatism rather than adding any.

4. **Power check** (day0_grna20, 266 positive-control pairs), against R's
   stored `power_result`:

   | Metric | Value |
   |---|---|
   | rows | 266 / 266 |
   | `pass_qc` agreement | 266 / 266 |
   | `n_nonzero_trt`, `n_nonzero_cntrl` | exact, 266 / 266 each |
   | Fold-change agreement (Pearson r) | 1.000000 (max difference 1.3e-12) |
   | p-value agreement (Spearman rho), 246 testable pairs | 0.9892 |
   | median effect | -35.2% |

   The 20 pairs that fail pairwise QC are reported with NaN results by both
   implementations, and both agree on exactly which 20. A median effect of
   -35.2% is the check working: positive controls target a gene's own TSS,
   so they should repress it.

## License and attribution

`pysceptre` is licensed under the GNU General Public License v3.0 only
(SPDX: `GPL-3.0-only`) -- full text in [LICENSE](LICENSE). That is inherited,
not chosen: this package ports algorithms and formulas from
[`sceptre`](https://github.com/Katsevich-Lab/sceptre), which is `GPL-3` --
version 3 exactly, not "or later". Each module's docstring names what it
ports and from which upstream file.

The original SCEPTRE is by Timothy Barry and Eugene Katsevich. The
statistical method is theirs, not ours -- if you use this in published work,
cite `sceptre`: Barry et al. (2021), *SCEPTRE improves calibration and
sensitivity in single-cell CRISPR screen analysis*, Genome Biology 22(1),
[doi:10.1186/s13059-021-02545-2](https://doi.org/10.1186/s13059-021-02545-2).
