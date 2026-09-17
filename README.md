# pysceptre

A standalone Python port of the statistical engine behind
[`sceptre`](https://timothy-barry.github.io/sceptre-book/)'s discovery
analysis for single-cell CRISPR screens -- specifically the **complement
control group + CRT (conditional randomization test) resampling** path used
for high-MOI data.

This is *not* a general reimplementation of `sceptre`. It targets one
specific, validated analysis path so that it can batch the linear-algebra
work sceptre does per-gene/per-target in R/C++ loops into vectorized numpy
calls, cutting real-dataset runtimes from hours to tens of minutes. See
[Scope and limitations](#scope-and-limitations) for exactly what is and
isn't covered.

## Why this exists

`sceptre`'s own discovery-analysis implementation is correct but processes
one (gene, gRNA-target) pair, and one resampling draw, largely with R and
per-cell C++ loops. At real dataset scale (hundreds of thousands of cells,
thousands of gRNA targets, tens of thousands of pairs) that adds up. Every
GLM fit in this pipeline shares one design matrix across many response
columns (all genes share the covariate matrix; all gRNA targets share it
too), which is exactly the shape numpy's batched matrix operations are
built for -- so the rewrite fits one batched IRLS call per resampling stage
instead of one per gene or target.

## Installation

```bash
pip install -e ".[dev,fast]"
```

- `dev` installs `pytest` (needed for the test suite) plus `ruff` and
  `pre-commit` for linting and formatting.
- `fast` installs [`numba`](https://numba.readthedocs.io/), which JIT-compiles
  the CRT sampler's cell-grouping step (a counting sort). Without it,
  `pysceptre` falls back to a slower pure-numpy `argsort`-based version
  automatically -- everything still works, just slower at real dataset scale.

Requires Python >= 3.10. Core dependencies: `numpy`, `scipy`, `pandas`,
`threadpoolctl`.

## Quick start

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
#   response_id, grna_target, p_value, fold_change, log_2_fold_change, z_orig, stage
```

See [TUTORIAL.md](TUTORIAL.md) for a complete, runnable walkthrough
(including how to build each input from scratch) and for guidance on
picking `target_chunk_size` for your dataset's memory budget.

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
) -> pd.DataFrame
```

| Parameter | Type | Description |
|---|---|---|
| `response_matrix` | `(n_genes, n_cells)` dense `ndarray` or `scipy.sparse` matrix | Gene expression counts. Rows must correspond 1:1 with `gene_ids`, in order -- pre-filter this to only the genes that actually appear in `pairs` (do not pass a whole genome-wide matrix if only a few hundred genes are actually tested; see [TUTORIAL.md](TUTORIAL.md)). |
| `gene_ids` | `list[str]` | Row labels for `response_matrix`, in the same order as its rows. Matched against `pairs['response_id']`. |
| `covariate_matrix` | `(n_cells, p)` `ndarray` | Already formula-expanded numeric design matrix (intercept column, `log(umis)`, batch dummies, etc. -- whatever R's `model.matrix()` would have produced). `pysceptre` does not parse an R-style formula DSL; build this matrix yourself, or extract it directly from an existing `sceptre_object`'s `@covariate_matrix` slot. |
| `grna_target_cells` | `dict[str, np.ndarray]` | Maps each gRNA target to the **0-based** indices (into `covariate_matrix`'s cell axis) of cells treated with that target. This is the "union" grna-integration-strategy convention: one entry per target, not per individual gRNA. |
| `pairs` | `pd.DataFrame` with columns `response_id`, `grna_target` | The QC-passed (gene, target) pairs to test. `pysceptre` does not run `assign_grnas()`/`run_qc()` itself -- feed it pairs that have already passed QC upstream. |
| `side` | `"left"` \| `"both"` \| `"right"` | Test sidedness, matching sceptre's own convention. Use `"left"` for expected-repression screens (e.g. CRISPRi enhancer knockdown), `"both"` for a two-sided test. |
| `resampling_approximation` | `"skew_normal"` \| `"no_approximation"` | `"skew_normal"` (default, matching sceptre): pairs whose initial empirical p-value (`B1=499` draws) is `<= 0.02` get a skew-normal tail fit from a further `B2=4999` draws, giving p-values far smaller than `1/(B1+1)` could resolve. `"no_approximation"` fits no curve and instead draws a third, larger empirical batch, sized by R's own rule: `B3 = ceil(mult * n_pairs / multiple_testing_alpha)`, `mult = 10` two-sided and `5` one-sided. That grows linearly in the number of pairs and is much slower -- see [Scope and limitations](#scope-and-limitations). Any other value raises `ValueError`. |
| `seed` | `int \| None` | Seeds the `numpy.random.Generator` used for all CRT draws in the run. Note this does **not** reproduce sceptre's own R/C++ RNG stream bit-for-bit (different algorithm and seeding scheme) -- see [Scope and limitations](#scope-and-limitations). |
| `target_chunk_size` | `int` | How many gRNA targets' logistic fits + CRT draws to batch and hold in memory at once. Each target's CRT draw needs `O(B * n_treated_cells)` memory; holding *every* target's draws in memory at once does not scale to real target counts (measured: OOM-killed the process at ~3,000 targets). Lower this if you hit memory pressure; raise it for a modest speed gain if you have memory to spare. Default `200`. |

**Returns** a `pd.DataFrame`, one row per input pair, with columns:

| Column | Meaning |
|---|---|
| `response_id`, `grna_target` | Echoed from `pairs`. |
| `p_value` | The test p-value (see `stage`). |
| `fold_change` | Estimated fold change of the treated group vs. complement control (deterministic given the data -- no resampling randomness). |
| `log_2_fold_change` | `log2(fold_change)`. |
| `z_orig` | The observed test statistic (before resampling). |
| `stage` | `1` = reported from the initial `B1=499`-draw empirical p-value (not significant enough to escalate). `2` = escalated to a skew-normal tail fit on `B2=4999` further draws. `3` = an empirical p-value from a further batch, reached either because `resampling_approximation="no_approximation"` (so no curve is fit, and the `B3` draws are used) or because a skew-normal fit was attempted and *rejected* (see `check_sn_tail`/`check_for_outliers` in `test_statistic/skew_normal.py`), in which case the already-drawn `B2=4999` statistics are used. |

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

## Scope and limitations

- **Complement control group only, high-MOI/CRT resampling only.** This is
  the one analysis path this package targets; other sceptre modes
  (permutations, non-complement control groups, low-MOI) are out of scope.
- **No `assign_grnas()` / `run_qc()`.** Feed `run_discovery_analysis` pairs
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
  see [Validation](#validation).
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
  1,653,300 draws *per target*, so `target_chunk_size` must be very small.
  Conversely, at 4 pairs one-sided `B3 = 200`, *below* `B1 = 499`. Both are
  R's behavior, reproduced rather than corrected.
- **gRNA integration strategy: "union" only.** `grna_target_cells` is keyed
  by target, not by individual gRNA -- matches sceptre's `"union"` strategy;
  `"singleton"` is not supported.

## Performance

Real-dataset numbers (586k-cell, 292-gene, 3,026-target synthetic benchmark
shaped like the moi5 dataset this was validated against -- see
`scripts/benchmark_pairs.py`):

| Stage | Time |
|---|---|
| Gene GLM fits (`fit_all_genes`) | ~5.4 min |
| gRNA-target GLM fits + CRT draws (`fit_all_targets`, chunked) | ~48 min |
| Per-pair test statistic + escalation (~35k pairs) | ~8 min |
| **Total** | **~1 hour** |

For comparison, on the real moi5 dataset (244 genes, 131k cells, 2,875
targets, 33,066 pairs) an end-to-end run took **28 minutes** (~51 ms/pair).

Getting here from an initial direct port (which had a few genuine
performance bugs -- an accidentally-dense CRT sampler, unbatched per-target
GLM fits, and an OpenBLAS multi-threading pathology for many small matmuls,
among others) is itself informative for anyone optimizing similar code; see
git history / `scripts/benchmark_pairs.py` for the profiling trail.

If you don't have `numba` installed, the CRT sampler falls back to a slower
pure-numpy path -- install the `fast` extra (`pip install -e ".[fast]"`) to
get the full performance above.

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

2. **Real dataset** (moi5 single-cell CRISPR screen, 244 genes / 131k cells
   / 2,875 targets / 33,066 real QC-passed pairs): ran the actual data
   through both `pysceptre.run_discovery_analysis` and R's real
   `sceptre::run_discovery_analysis()` (same parameters: `side="left"`,
   `grna_integration_strategy="union"`, `resampling_mechanism="crt"`).
   Results:

   | Metric | Value |
   |---|---|
   | Fold-change agreement (Pearson r) | 1.0000 (exact -- deterministic) |
   | p-value agreement (Spearman rho) | 0.9964 |
   | Strong-hit calls (p < 1e-4) | 156 (pysceptre) vs. 157 (R), 152 in common, Jaccard 0.944 |

   For context, R's own CRT-vs-permutations agreement on this same dataset
   is Spearman rho = 0.996 -- pysceptre matches R about as well as R
   matches itself across resampling mechanisms. The largest individual
   p-value disagreements were all pairs both methods agree are strong hits
   (exact fold-change match in every case); the disagreement is in how
   extreme an already-astronomically-small p-value is, which is expected
   skew-normal-tail Monte Carlo noise -- confirmed by R disagreeing with
   *itself* by a similar margin on the identical pair when compared across
   its own two resampling mechanisms.

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
