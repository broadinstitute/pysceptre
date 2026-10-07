# pysceptre

A standalone Python port of the statistical engine behind
[`sceptre`](https://timothy-barry.github.io/sceptre-book/)'s discovery
analysis for single-cell CRISPR screens, for **high- and low-MOI** data: the
**complement** and **NT-cells** control groups, with **CRT** (conditional
randomization test) or **permutation** resampling.

It covers three of sceptre's analysis steps -- the **discovery analysis**, the
**calibration check** and the **power check** -- plus sceptre's **gRNA
assignment** (its mixture, thresholding and maximum methods), and is *not* a
general reimplementation: cell-level QC and R's formula DSL stay in R.
Targeting the statistical engine is what lets it batch the linear-algebra work
sceptre does per-gene/per-target in R/C++ loops into vectorized numpy calls,
cutting real-dataset runtimes from hours to tens of minutes. See
[Scope and limitations](#scope-and-limitations) for exactly what is and isn't
covered.

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
    side="both",                          # the default, and the side validated against R
    seed=0,
)
# result: DataFrame with one row per pair --
#   response_id, grna_target, p_value, fold_change, se_fold_change,
#   pct_change_es, pct_change_es_ci_low, pct_change_es_ci_high, z_orig, stage
```

A low-MOI screen takes one more input, each non-targeting gRNA's cells, and
`moi="low"` selects sceptre's low-MOI defaults: the cells carrying a
non-targeting gRNA as the control group, and permutation resampling.

```python
result = run_discovery_analysis(
    response_matrix, gene_ids, covariate_matrix, grna_target_cells, pairs,
    moi="low",
    ntc_grna_cells=ntc_grna_cells,        # dict[NT gRNA id -> 0-based cell indices]
    seed=0,
)
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
    moi: str = "high",
    control_group: str | None = None,
    ntc_grna_cells: dict[str, np.ndarray] | None = None,
    resampling_mechanism: str | None = None,
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
| `pairs` | `pd.DataFrame` with columns `response_id`, `grna_target` | The QC-passed (gene, target) pairs to test. `pysceptre` does not run `run_qc()` -- feed it pairs that have already passed QC upstream. Cell sets can come from `assign_grnas`, with any of its methods, through `pysceptre.assignment.cells_by_target`, or from R. |
| `side` | `"left"` \| `"both"` \| `"right"` | Test sidedness, matching sceptre's own convention. Use `"left"` for expected-repression screens (e.g. CRISPRi enhancer knockdown), `"both"` for a two-sided test. |
| `moi` | `"high"` \| `"low"` | sceptre's `import_data(moi=)`. It sets the defaults of `control_group` and `resampling_mechanism` as R's `set_analysis_parameters` does -- the complement and the CRT in high MOI, the NT cells and permutations in low MOI -- and it is the only way to reach the NT cells. See [Low MOI](#low-moi). Default `"high"`. |
| `control_group` | `None` \| `"complement"` \| `"nt_cells"` | Which cells a target's cells are compared with. `None` takes the MOI's default. `"complement"` is every other cell. `"nt_cells"`, low MOI only, is the cells carrying a non-targeting gRNA; the gene's GLM is then refit for every pair, on that target's cells and the NT cells, as R does. |
| `ntc_grna_cells` | `dict[str, np.ndarray]` | Each individual non-targeting gRNA's **0-based** cells. Required with `control_group="nt_cells"` and refused otherwise. The NT cells are their union, in this dict's order, which decides which cells each resample picks: keep it fixed between runs. No cell may sit under two NT gRNAs or in a tested target as well: sceptre's low-MOI QC removes every cell with more than one gRNA, and the analysis relies on it. |
| `resampling_mechanism` | `None` \| `"crt"` \| `"permutations"` | Matches sceptre's own option. `None` takes the MOI's default, `"crt"` in high MOI and `"permutations"` in low MOI. The CRT draws each target's synthetic treated set from that target's own fitted probabilities; permutations draw one set of random subsets, sized by the targets present, and reuse it for every target. **The choice is a real trade, and yours to make** -- see [Reproducibility](#reproducibility-and-incremental-analysis), because permutations cannot offer the invariance the CRT does. R pairs permutations with `B3 = 24999` against the CRT's `0`, so sampling is cheaper -- and the per-target logistic fit is skipped entirely, since only the CRT draws from it -- but the escalation batch is five times larger. |
| `resampling_approximation` | `"skew_normal"` \| `"no_approximation"` | `"skew_normal"` (default, matching sceptre): pairs whose initial empirical p-value (`B1=499` draws) is `<= 0.02` get a skew-normal tail fit from a further `B2=4999` draws, giving p-values far smaller than `1/(B1+1)` could resolve. `"no_approximation"` fits no curve and instead draws a third, larger empirical batch, sized by R's own rule: `B3 = ceil(mult * n_pairs / multiple_testing_alpha)`, `mult = 10` two-sided and `5` one-sided. That grows linearly in the number of pairs and is much slower -- see [Scope and limitations](#scope-and-limitations). Any other value raises `ValueError`. |
| `seed` | `int \| None` | Seeds the `numpy.random.Generator` used for all CRT draws in the run. Note this does **not** reproduce sceptre's own R/C++ RNG stream bit-for-bit (different algorithm and seeding scheme) -- see [Scope and limitations](#scope-and-limitations). |
| `target_chunk_size` | `int` | How many gRNA targets to fit and CRT-draw at once. An **upper bound, not a mandate** -- it is reduced automatically to respect `chunk_memory_gb`, so no value here can exhaust memory. Default `200`. |
| `n_jobs` | `int` | Workers for the per-pair tests, which are ~80% of the runtime. `1` (default) runs serially; a negative value uses every core. **Results do not depend on it** -- only the genes inside an already-drawn target chunk are distributed, so the resampling draws are made in the same order at any worker count, and output is bit-identical. Processes on Linux, threads elsewhere (`fork` after macOS's Accelerate BLAS can deadlock), so the ceiling is lower off Linux; against the NT cells, where every pair is a GLM fit, threads are capped at four because more contend for the GIL. Memory grows by about one gene's working arrays per worker, not by `chunk_memory_gb` per worker. |
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
    side="both",
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
| `moi`, `control_group`, `resampling_mechanism` | As for `run_discovery_analysis`, with the same defaults. With `control_group="nt_cells"` the whole check runs on the NT cells alone, as in R: a synthetic target is tested against the rest of them, so at least two NT gRNAs are needed. |

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
    side="both",
    seed=0,
)
```

| Argument | Notes |
|---|---|
| `positive_control_pairs` | The pairs to test. **Supply these**: which target perturbs which gene is a claim only the experiment can make. Omitted, sceptre's name-matching rule is used -- a target that is itself a gene id pairs with that gene -- which works when targets are named after genes and finds *nothing* when they are named after genomic intervals. On a real screen of the latter kind it matched 0 of 3,071 targets, so that case raises rather than quietly returning an empty result. |
| `n_nonzero_trt_thresh`, `n_nonzero_cntrl_thresh` | Pairwise QC thresholds. Against the NT cells the control count is the gene's nonzero NT cells, as in R. |
| `moi`, `control_group`, `ntc_grna_cells`, `resampling_mechanism` | As for `run_discovery_analysis`, with the same defaults. |

**QC is reported, not filtered** -- the opposite of the calibration check.
The result has one row per supplied pair, with `pass_qc`, `n_nonzero_trt` and
`n_nonzero_cntrl`, and NaN results where a pair did not meet the thresholds.
Dropping those would overstate power by hiding exactly the controls the
screen had too few cells to test.

**No multiple-testing correction is applied**, matching R, which returns no
`significant` column here. These are a diagnostic rather than discoveries.

### `pysceptre.compute_power`

A different question from the three above, and the only entry point here that
does not come from sceptre. `run_power_check` runs the real test on pairs
where an effect is expected; this estimates, in closed form and with no
resampling, what the screen *could* have detected: if this element really did
reduce this gene by X%, how likely was this screen to call it?

It is a port of
[PerturbPlan](https://github.com/Katsevich-Lab/perturbplan)'s
`compute_power_posthoc()` (MIT -- see
[THIRD_PARTY_LICENSES](https://github.com/broadinstitute/pysceptre/blob/main/THIRD_PARTY_LICENSES)),
validated against that package's own output to a relative 1e-9.

```python
from pysceptre import compute_power

power = compute_power(
    discovery_pairs,             # grna_target, response_id
    cells_per_grna,              # grna_id, grna_target, num_cells
    baseline_expression_stats,   # response_id, expression_mean, expression_size
    fold_change_mean=0.85,       # a MULTIPLIER: a 15% knockdown is 0.85
    fold_change_sd=0.13,
    cutoff=alpha / 2,            # this screen's own nominal threshold
    num_total_cells=n_cells,
    side="left",
)
```

| Argument | Notes |
|---|---|
| `cells_per_grna` | One row per **individual** gRNA. Per-gRNA granularity is required, not a convenience: the across-gRNA variance term needs the sum of *squared* per-gRNA counts, which no target-level total can supply. Note that the treated count is the **sum** over a target's gRNAs, not the size of the union of perturbed cells -- the two differ in high MOI, and the sum is what was validated. |
| `baseline_expression_stats` | `expression_size` is the NB size, i.e. theta = `1 / dispersion`, not the dispersion. The validated `expression_mean` is the size-factor-normalised mean. |
| `cutoff`, `fold_change_mean`, `fold_change_sd` | All three are **required, with no defaults**, and `cutoff` must be the analysed screen's own nominal threshold rather than a borrowed one. |
| `n_nonzero_trt_thresh`, `n_nonzero_cntrl_thresh` | Default to `0`, not to sceptre's `7`, which makes the QC factor 1 -- correct for pairs that already passed QC. Raise them to score pairs that were never tested. |

Because none of its inputs is a property of the *pair* -- expression and
dispersion belong to the gene, cell counts to the element -- it also answers
the question for pairs the screen never tested.

Read the accuracy limits in
[Design decisions](https://broadinstitute.github.io/pysceptre/design/#analytical-per-pair-power)
before using it on a single pair: the estimate is good enough to plan a screen
and to triage its negatives, not to close a question about one element-gene
pair.

<!-- --8<-- [start:tutorial-power] -->

#### Reporting power alongside a discovery result

`run_discovery_analysis` returns no power column, and `compute_power` is not
wired into it. That is deliberate: the estimator needs inputs the discovery
path has no business knowing about, and a wrapper is where a wrong cell set or
the per-target union instead of the per-gRNA sum would creep in unseen. The
join is five steps, and each one is a place to check you meant it.

```python
from pysceptre import compute_power, run_discovery_analysis
from pysceptre.analytical_power import (
    baseline_expression_stats_from_fits,
    bh_nominal_cutoff,
    cells_per_grna_from_assignments,
)
from pysceptre.pipeline.discovery import fit_all_genes

ALPHA = 0.1

# 1. the analysis
result = run_discovery_analysis(
    response_matrix, gene_ids, covariate_matrix, grna_target_cells, pairs,
    side="both", multiple_testing_alpha=ALPHA, seed=0,   # "both" is the default
)

# 2. the threshold this run actually applied, then halved. side="both" gives a
#    two-sided p-value, and compute_power's default side="left" wants the
#    one-sided threshold that corresponds to it. Do NOT halve if you ran the
#    discovery one-sided: those p-values are already one-sided.
cutoff = bh_nominal_cutoff(result["p_value"], alpha=ALPHA) / 2

# 3. the estimator's inputs, both on sceptre's own scale
fits = fit_all_genes(response_matrix, gene_ids, covariate_matrix)
baseline = baseline_expression_stats_from_fits(covariate_matrix, fits)
cells_per_grna = cells_per_grna_from_assignments(
    grna_target_data_frame, targeting_grna_cells,   # both from the export
)

# 4. power for the same pairs, at a 15% knockdown
power = compute_power(
    pairs, cells_per_grna, baseline,
    fold_change_mean=0.85, fold_change_sd=0.13,
    cutoff=cutoff, num_total_cells=covariate_matrix.shape[0],
)

# 5. one table
report = result.merge(
    power[["response_id", "grna_target", "power"]],
    on=["response_id", "grna_target"], how="left",
)
```

Four things worth knowing before you run it.

**Step 3 refits every gene, and the discovery run already did.** It discards
its fits rather than returning them, so this repeats the most expensive part
of the analysis. If that matters, call `fit_all_genes` once yourself and reuse
the result; the fit depends only on the counts and the covariate matrix, not
on the pairs.

**Step 2 raises if nothing is significant**, rather than returning a threshold
that would hand every pair zero power. On a screen with no discoveries, pass
`cutoff` yourself from a plain alpha and say so in whatever you report.

**Step 2 needs a real pair list to mean anything.** The threshold is the
largest p-value BH calls significant, so on a handful of pairs it is simply
whatever the smallest p-value happens to be: five pairs containing one strong
hit gave `1.2e-215`, and every power estimate came back 0. That is the rule R
uses, reproduced, not a defect -- but it only behaves like a threshold at
screen scale, where day0's 34,886 pairs give `7.26e-4`. On a toy example pass
`cutoff` explicitly.

**The halving in step 2 belongs to the two-sided run, not to
`compute_power`.** `side="both"` is sceptre's default and pysceptre's, and it
is the only side the discovery path is validated against R on. Its p-values
are two-sided, so the one-sided threshold `compute_power(side="left")` wants
is half of it. Run the discovery one-sided and its p-values are already
one-sided: use the threshold **unhalved**. Halving twice is the
easiest way to get a confidently wrong power estimate here, and it is
invisible in the output.

**The pairs that failed QC are the interesting ones.** They carry a NaN
p-value because they were never tested, and they are exactly where a power
estimate says something a p-value cannot. Pass them to `compute_power` too,
with `n_nonzero_trt_thresh` and `n_nonzero_cntrl_thresh` set to the
thresholds the QC used, so the estimate includes the probability the pair
would have failed QC at all.

<!-- --8<-- [end:tutorial-power] -->

### `pysceptre.run_specificity_check`

The question sceptre's calibration and power checks leave open: of the links a
screen discovered, how many are more than background? A test between an
element and a gene on another chromosome cannot be direct regulation, so how
often those tests are called at the cis cutoff is how often a test is called
without it. Like `compute_power`, it does not come from sceptre, and it has no
R counterpart to be validated against.

```python
from pysceptre import run_specificity_check
from pysceptre.analytical_power import bh_nominal_cutoff

result = run_specificity_check(
    cis_result,              # discovery on same-chromosome element-gene pairs
    trans_result,            # the same test, elements x cis genes on other chromosomes
    element_positions,       # grna_target, chrom, centre (the screen's own coordinates)
    gene_positions,          # response_id, chrom, tss
    cutoff=bh_nominal_cutoff(p_values, alpha=0.1),
    control_targets=tss_controls,
    seed=0,
)
result.by_distance                # cis rate, matched background, links above background
result.by_distance_without_broad  # the same without the broad-effect elements
result.broad_effect               # elements with far more background calls than the rest
result.far_links                  # each far link: can it run through another gene?
```

| Argument | Notes |
|---|---|
| `element_positions` | From the screen's own coordinate table, **never** parsed from target names: names can be on a different genome build from the annotation. |
| `trans_result` | Must come from the same test as `cis_result`. Pairs outside the cis elements and cis genes are ignored, as are genes on the element's chromosome and on chrY. |
| `control_targets` | Targets that are not elements, such as TSS positive controls. Left out of the counts and still used as TSS knockdowns. |
| `seed` | An int, or a `Generator` to continue drawing from. |

The background is matched gene for gene rather than pooled, because
well-expressed genes are called more often, and its intervals are a bootstrap
over elements. What the result means and what it has been checked against is in
[Design decisions](https://broadinstitute.github.io/pysceptre/design/#specificity-check).

### `pysceptre.assign_grnas`

One entry point for every assignment method, like sceptre's `assign_grnas()`.
Left at `method="default"` it takes sceptre's default for the screen's MOI,
the maximum method in low MOI and the mixture in high MOI, so it needs `moi`.
`method` can also name any of `pysceptre.assignment.ASSIGNMENT_METHODS`:
`"mixture"`, `"thresholding"`, `"maximum"` and, not from sceptre, `"fishash"`.
Each method's options pass through under its own function's keyword names.

```python
from pysceptre import assign_grnas

res = assign_grnas(grna_counts, grna_ids, moi="low")                       # sceptre's default: "maximum"
res = assign_grnas(grna_counts, grna_ids, moi="high", covariate_matrix=X)  # sceptre's default: "mixture"; X as below
res = assign_grnas(grna_counts, grna_ids, method="thresholding", threshold=3)
```

| Argument | Notes |
|---|---|
| `method` | `"default"` (the default) or one of `ASSIGNMENT_METHODS`. |
| `moi` | `"low"` or `"high"`. Required by `"default"`. `"maximum"` is refused for a high-MOI screen, as in sceptre. |
| `covariate_matrix` | The mixture's design; `mixture_design_matrix` builds sceptre's default (below). Required by `"mixture"` and refused by every other method. |
| `**hyperparameters` | The chosen function's keyword options, such as `threshold` for `"thresholding"`. One it does not take raises `TypeError`. |
| returns | That method's result: a `MixtureResult`, `ThresholdingResult`, `MaximumResult` or `FishashResult`. |

### `pysceptre.assign_grnas_mixture`

sceptre's mixture gRNA assignment, `assign_grnas(method = "mixture")`, from
raw gRNA UMI counts. For each gRNA with at least ten cells holding a count,
a Poisson GLM of its counts on the cell covariates is followed by sceptre's
two-component EM; cells whose posterior reaches 0.8 are assigned. A gRNA with
fewer cells, or whose EM does not converge, is assigned where its count
reaches 5. Validated against sceptre 0.10.3's own output, value for value.

```python
from pysceptre import assign_grnas_mixture
from pysceptre.assignment import cells_by_grna, cells_by_target, mixture_design_matrix

X, names = mixture_design_matrix(
    grna_counts,                          # (n_grnas, n_cells) raw gRNA UMI counts
    response_n_nonzero=genes_detected,    # per cell, from the gene expression
    response_n_umis=gene_umis,
)
res = assign_grnas_mixture(grna_counts, grna_ids, X)
grna_target_cells, ntc_grna_cells = cells_by_target(
    cells_by_grna(res.assigned, grna_ids), grna_target_data_frame
)
```

| Argument | Notes |
|---|---|
| `covariate_matrix` | The design, intercept included. `mixture_design_matrix` builds sceptre's default from the per-cell counts; `design_from_covariates` builds it from a covariate frame exported from R. |
| `grna_matrix` | Integer UMI counts, gRNAs as rows. Non-integer values are refused. |
| returns | A `MixtureResult`: `assigned` (CSR, gRNAs x cells), the posteriors, and a `fits` table recording each gRNA's path. |

### `pysceptre.assign_grnas_thresholding`

sceptre's thresholding assignment, `assign_grnas(method = "thresholding")`: a
gRNA is assigned to every cell in which its UMI count is at least `threshold`,
so a cell can be assigned none, one or several. Validated against sceptre
0.10.3's own output, value for value.

```python
from pysceptre import assign_grnas_thresholding

res = assign_grnas_thresholding(grna_counts, grna_ids)   # threshold=5
res.assigned          # (n_grnas, n_cells) boolean CSR
```

| Argument | Notes |
|---|---|
| `threshold` | The count a gRNA needs in a cell, reached rather than exceeded (`>=`). At least 1, as sceptre requires. Default 5. |
| returns | A `ThresholdingResult`: `assigned`, the `threshold` applied and the `grna_ids`. |

### `pysceptre.assign_grnas_maximum`

sceptre's maximum assignment, `assign_grnas(method = "maximum")`, which sceptre
allows in low MOI only: every cell is assigned the gRNA with the most UMIs in
it, the first in row order on a tie. A cell is flagged when that gRNA holds at
most `umi_fraction_threshold` of the cell's gRNA UMIs, or when the cell has
fewer than `min_grna_n_umis_threshold` gRNA UMIs; sceptre's low-MOI QC removes
the flagged cells. Validated against sceptre 0.10.3's own output, value for
value.

```python
from pysceptre import assign_grnas_maximum

res = assign_grnas_maximum(grna_counts, grna_ids)   # umi_fraction_threshold=0.8, min_grna_n_umis_threshold=5
res.assigned                        # (n_grnas, n_cells) boolean CSR, one gRNA per cell
res.cells_w_zero_or_twoplus_grnas   # the flagged cells, 0-based and ascending
```

| Argument | Notes |
|---|---|
| `umi_fraction_threshold` | In (0, 1), as sceptre requires. Default 0.8. |
| `min_grna_n_umis_threshold` | At least 0, as sceptre requires. Default 5. |
| returns | A `MaximumResult`: `assigned`, `cells_w_zero_or_twoplus_grnas`, and per cell the top gRNA's row (`max_grna`), its share of the cell's gRNA UMIs (`max_grna_frac_umis`) and the cell's gRNA UMIs (`grna_n_umis`). |

A cell with no gRNA UMIs is still assigned one, the first gRNA, with a NaN
share, because sceptre does the same; the UMI rule is what flags it, at any
`min_grna_n_umis_threshold` above 0.

### `pysceptre.assign_grnas_fishash`

Not sceptre's: a port of
[fishash](https://github.com/jackkamm/fishash) 0.99.5 (MIT -- see
[THIRD_PARTY_LICENSES](https://github.com/broadinstitute/pysceptre/blob/main/THIRD_PARTY_LICENSES)).
Each nonzero count gets a one-sided Fisher exact test of whether its gRNA and
cell co-occur more often than their totals predict; calls must pass an FDR cut
(Guo and Sarkar's block procedure by default) and a count floor, and up to
`refit` further passes take the off-cell margins from an estimate of the noise
alone. The counts are never transformed: the test conditions on each cell's
and each gRNA's totals. Validated against fishash's own output, entry for
entry.

```python
from pysceptre import assign_grnas_fishash

res = assign_grnas_fishash(grna_counts, grna_ids)   # refit=10, padj_cutoff=0.05, "GS"
res.assigned          # (n_grnas, n_cells) boolean CSR
res.log_pval          # one-sided Fisher log p-value at every nonzero count
res.demux_type        # per cell: "singlet", "doublet" or "unknown"
```

| Argument | Notes |
|---|---|
| `grna_matrix` | Integer UMI counts, gRNAs as rows. R's fishash accepts any numbers; this port refuses non-integer ones, so a normalized matrix cannot reach the count test. |
| `refit` | Maximum number of passes after the first (default 10), stopping early once the calls stop changing. `0` runs the plain Fisher test. |
| `padj_method` | `"GS"` (default), `"BH"` or `"BY"`. |

**In low MOI, sceptre's QC then removes cells, and pysceptre leaves that to
you.** After a mixture, thresholding or fishash assignment the cells to remove
are those assigned no gRNA or two or more, which
`pysceptre.assignment.cells_w_zero_or_twoplus_grnas(res.assigned)` lists. A
maximum assignment returns its own list, by the rules above, as
`res.cells_w_zero_or_twoplus_grnas`.

Read
[Design decisions](https://broadinstitute.github.io/pysceptre/design/#grna-assignment)
for what each method normalizes for, the edge cases of the thresholding and
maximum ports, and what the validation against R does and does not establish.

### Lower-level building blocks

`run_discovery_analysis` is a thin wrapper around
`pysceptre.pipeline.discovery.run_discovery_ntcells_complement` (the
complement) and `run_discovery_nt_cells` (the NT cells), which expose a
couple of additional knobs not surfaced at the top level (fixed
`B1=499, B2=4999, B3=0`, `fit_parametric_curve: bool`, `side_code: int`
instead of `side: str`, and shared permutation draws to use instead of
drawing them). Most users won't need to go lower than
`run_discovery_analysis`, but each pipeline stage is also independently
importable and unit-tested, for anyone extending or debugging the pipeline:

| Module | What it does |
|---|---|
| `pysceptre.glm.irls` | Batched IRLS for Poisson (log link, gene fits) and binomial (logit link, gRNA-target fits) GLMs -- `fit_poisson_glm_batch`, `fit_binomial_glm_batch`. Ports R's `stats::glm.fit` algorithm and convergence criteria exactly. |
| `pysceptre.glm.nb_theta` | Negative-binomial dispersion (`theta`) estimation given a fitted mean -- `estimate_theta`. Exact port of sceptre's `estimate_theta`. |
| `pysceptre.precompute.pieces` | Per-gene precomputation pieces (the `D` matrix and friends) needed by the test statistic -- `compute_precomputation_pieces`. |
| `pysceptre.crt.sampler` | The CRT resampling draw itself -- `crt_index_sampler`, which picks per target between `crt_index_sampler_fast` (sparse, real-scale-feasible, numba-accelerated if available, approximate) and `crt_index_sampler_exact` (sparse and exact, for targets above 0.2% of the cells they are tested on), and `crt_index_sampler_naive` (dense reference implementation, used only for cross-checking in tests). |
| `pysceptre.crt.permutations` | Permutation draws shared by every target -- `permutation_draws` (the complement) and `nested_permutation_draws` (the NT cells; the law of R's `hybrid_fisher_iwor_sampler`). |
| `pysceptre.test_statistic.score_stat` | The O(n_treated)-per-resample score-type test statistic -- `compute_observed_full_statistic`, `compute_null_full_statistics`. |
| `pysceptre.test_statistic.empirical_p` | Empirical p-value from a null distribution -- `compute_empirical_p_value`. |
| `pysceptre.test_statistic.skew_normal` | Skew-normal tail-fit escalation -- `fit_and_evaluate_skew_normal`. Exact port of `fit_skew_normal_funct`/`check_sn_tail`/`check_for_outliers`/`fit_and_evaluate_skew_normal`. |
| `pysceptre.test_statistic.fold_change` | Fold-change estimation -- `estimate_log_fold_change`. |
| `pysceptre.test_statistic.resampling` | Ties the above into the `B1 -> B2 -> B3` staged escalation for one pair -- `run_low_level_test_full`. |
| `pysceptre.pipeline.discovery` | Orchestration: `fit_all_genes`, `fit_all_targets`, `run_discovery_ntcells_complement`, `run_discovery_nt_cells`. |

## Low MOI

<!-- --8<-- [start:lowmoi] -->

In a low-MOI screen each cell carries at most one gRNA, so the cells carrying a
non-targeting (NT) gRNA are untreated and can be the control group on their
own. sceptre calls that the NT-cells control group and makes it the low-MOI
default, together with permutation resampling. `moi="low"` selects both, as
R's `set_analysis_parameters` does, and either can be overridden:

| `moi` | default `control_group` | default `resampling_mechanism` |
|---|---|---|
| `"high"` (default) | `"complement"` | `"crt"` |
| `"low"` | `"nt_cells"` | `"permutations"` |

`control_group="nt_cells"` is refused in high MOI, where R would quietly use
the complement instead. Low MOI
with `control_group="complement"` runs exactly the high-MOI engine.

**What the NT cells need.** Pass `ntc_grna_cells`, each individual NT gRNA's
cells. The analysis assumes what sceptre's low-MOI QC guarantees: every cell
carries at most one gRNA, because `run_qc` removes cells with zero or two or
more. pysceptre does not run that QC, so it refuses what it can see breaking
the assumption -- a cell under two NT gRNAs, or a tested target sharing a cell
with the NT cells -- and those cells have to be removed upstream. If pysceptre
made the assignment, `pysceptre.assignment.cells_w_zero_or_twoplus_grnas`, or a
maximum assignment's field of that name, lists the cells sceptre's QC would
remove. A cell in two targets is accepted: one guide can belong to two
overlapping elements.

**Each pair gets its own fit.** Against the NT cells a pair is tested on its
target's cells together with the NT cells, so the gene's GLM is refit for every
pair, as in R, rather than once per gene. The fits are independent and spread
across workers with `n_jobs`. A covariate constant on one target's cells and on
every NT cell makes that pair's GLM unfittable even when the full design is
full rank; R stops there, and pysceptre refuses it before fitting anything,
naming the target.

**The calibration check runs on the NT cells alone**, as in R: a synthetic
target is tested against the rest of the NT cells, so at least two NT gRNAs are
needed. **The power check** counts a pair's control cells as the gene's
nonzero NT cells.

**The covariates are yours to build.** sceptre's default formula leaves out the
gRNA-count covariates (`grna_n_nonzero`, `grna_n_umis`) in low MOI and keeps
them in high MOI. There is no formula DSL here, so build `covariate_matrix` the
same way to match an R run.

The reasoning, and what it was checked against, is in
[Design decisions](https://broadinstitute.github.io/pysceptre/design/#low-moi-and-the-nt-cells-control-group).

<!-- --8<-- [end:lowmoi] -->

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

**Against the NT cells the same split holds.** Each pair is fit on its own
target's cells and the NT cells, so under the CRT a pair's result depends on
its gene, its target, the NT cells and the seed, and not on the rest of the
analysis. Permutation draws there are sized by the number of NT cells and by
the smallest and largest target, so adding a target outside that range changes
every result.

One exception, by construction: under `resampling_approximation="no_approximation"`
the third batch is sized by the number of pairs, so changing the pair list
re-draws every resample, on both mechanisms.

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

- **High and low MOI, both control groups.** The complement control group in
  either MOI and the NT-cells control group in low MOI, for all three analyses.
  A low-MOI analysis needs each cell to carry at most one gRNA, which sceptre's
  QC enforces and pysceptre can only partly check -- see
  [Low MOI](https://github.com/broadinstitute/pysceptre/blob/main/README.md#low-moi).
- **Both resampling mechanisms are validated against R, in different ways.**
  Permutation p-values are checked value for value on R's own draws, which a
  test-only replica of R's two permutation samplers replays, under both
  control groups. R's CRT draws are not replicated, so the CRT is checked
  exactly on everything deterministic and on its p-values in distribution.
  Run normally, pysceptre draws with numpy rather than R's generator, so a
  result from either mechanism agrees with R in distribution, not draw for
  draw -- see
  [Validation](https://github.com/broadinstitute/pysceptre/blob/main/README.md#validation),
  items 5 and 6.
- **`assign_grnas()`: all three of sceptre's methods. No `run_qc()`**, with
  one carve-out: the calibration and power checks apply the *pairwise*
  nonzero-count thresholds, because they build or receive their own pairs and
  cannot select them otherwise. The mixture, thresholding and maximum
  assignment methods are ported; cell-level and gRNA-level QC are still out of
  scope, including low MOI's removal of cells with zero or two or more gRNAs.
  pysceptre reports which cells those are (`cells_w_zero_or_twoplus_grnas`,
  and a maximum assignment's field of that name) but never removes them.
  Feed `run_discovery_analysis` pairs that have already passed QC (e.g. from a
  real `sceptre_object`'s `@discovery_pairs_with_info`, filtered to
  `pass_qc == TRUE`).
- **`assign_grnas_fishash` is not sceptre's.** It ports fishash 0.99.5, the
  version after the preprint's 0.3.0; the two differ only where 0.3.0 would
  divide 0 by 0. All four assignment methods take raw integer counts, and the
  `.h5mu` exports this repository builds carry 0/1 assignments rather than gRNA
  UMI counts, so they cannot feed them.
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
  `permutations` mechanism -- which this package does implement, and sizes the
  same way.
  `run_discovery_ntcells_complement`, `run_discovery_nt_cells` and
  `run_low_level_test_full` accept all three directly if you need to override
  them.
- **`no_approximation` is expensive, and can be coarser at small scale.** Its
  `B3` grows linearly in pair count: a 33,066-pair one-sided run needs
  1,653,300 draws *per target* (5.2 GB), so the chunk collapses to a
  single target and the run warns that it may still exhaust memory.
  Conversely, at 4 pairs one-sided `B3 = 200`, *below* `B1 = 499`. Both are
  R's behavior, reproduced rather than corrected.
- **All three gRNA integration strategies.** `"union"` (the default),
  `"singleton"` and `"bonferroni"`, matching sceptre's own option. Under the
  latter two, `grna_target_cells` is keyed by guide rather than by target and
  `grna_target_data_frame` supplies the design; the pair expansion is
  validated against sceptre's own on a real screen, 36,450 pairs to 515,972.
  Nothing statistical differs between them -- only which cells count as
  treated, and what happens to the results afterwards.
- **`compute_power` is not a sceptre path, and its limits are its
  own.** It estimates in closed form what a screen *could* have detected,
  which is a different question from the three analyses above, and it is a
  port of [PerturbPlan](https://github.com/Katsevich-Lab/perturbplan) (MIT)
  rather than of sceptre. It is validated against that package's own output,
  not against sceptre's. Within it: complement control group only, an explicit
  `cutoff` only, per-gRNA cell counts required rather than the per-target
  union, and no minimum-detectable-effect-size path. It is accurate enough to
  plan a screen and triage its negatives, and **not** accurate enough to
  settle a question about one element-gene pair -- see
  [Design decisions](https://broadinstitute.github.io/pysceptre/design/#analytical-per-pair-power).
- **Two ways to build its baseline statistics, and they are not
  interchangeable.** `baseline_expression_stats_from_fits` takes the mean and
  theta from the same negative-binomial fit the discovery test uses, so the
  estimate and the test it predicts are on one expression scale. **Prefer it.**
  `baseline_expression_stats` instead computes a size-factor-normalised mean,
  which is what the published comparison used; measured on one screen that
  mean sits about **16 % below** the scale sceptre's own model works on, which
  makes the estimate conservative rather than wrong-shaped. Use it only to
  reproduce those numbers.
- **A per-pair alternative that carries the covariates.**
  `matched_expression_stats` gives each (target, gene) pair the mean that
  matches the information sceptre's per-cell fit gives that target's cells,
  which corrects batch-driven genes and the larger libraries of perturbed
  cells. It improved agreement with simulation on two DC-TAP screens but has
  not been checked on day0, so it is an option beside
  `baseline_expression_stats_from_fits`, not a replacement -- see
  [Design decisions](https://broadinstitute.github.io/pysceptre/design/#covariates-through-the-information-matched-mean).
- **`run_specificity_check` is also not sceptre's, and has no R reference.**
  It counts how many discovered links exceed a background measured on tests
  across chromosomes. Its reference is the analysis it was ported from, which it
  reproduces on three screens; nothing external has checked it. The
  fingerprint part of that analysis is not included -- see
  [Design decisions](https://broadinstitute.github.io/pysceptre/design/#specificity-check).

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
exact inputs) and `scripts/benchmark_pysceptre.py` (pysceptre side). The
pinned R-plus-pysceptre image those numbers were measured in lives with the
comparison itself, in the private development archive, rather than here:
`docker/` in this repository holds only the pysceptre runtime image, which
carries no R.

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

Every layer below compares against a real, installed `sceptre` R package
(pinned upstream commit), not just internal self-consistency:

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

5. **Low MOI, value for value** (`tests/validation/test_low_moi_vs_r.py`,
   against sceptre 0.10.3 on its own simulated `lowmoi_example_data`, 100
   genes by 1,000 cells). Permutation p-values are compared on R's own
   draws, replayed by `tests/validation/r_samplers.py`, a test-only replica
   of R's two permutation samplers that is itself checked draw for draw
   against R. Every per-pair fit on a target's cells and the NT cells, every
   statistic, fold change and stage, and every p-value matches R, for
   discovery, the power check and the calibration check, under both control
   groups. CRT draws are not replicated, so there the comparison is of
   everything deterministic, and the p-values distributionally.

6. **sceptredata's two real screens**, both control groups and both
   mechanisms: Papalexi et al. 2021 (low MOI, 299 genes, 20,729 cells, 26
   targets and 9 non-targeting gRNAs) and Gasperini et al. 2019 (high MOI, 526
   genes, 45,919 cells), each analysed by sceptre's documented pipeline in R
   and by pysceptre on the exported object, with pysceptre's own draws:

   | screen | control group, mechanism | fold change, max difference | p-value Spearman | BH 0.1, both / pysceptre only / R only |
   |---|---|---|---|---|
   | Papalexi | NT cells, permutations | 1.8e-12 | 0.990 | 449 / 13 / 4 |
   | Papalexi | NT cells, CRT | 1.8e-12 | 0.989 | 451 / 5 / 11 |
   | Papalexi | complement, permutations | 2.3e-12 | 0.991 | 631 / 14 / 21 |
   | Papalexi | complement, CRT | 2.3e-12 | 0.991 | 629 / 15 / 13 |
   | Gasperini | complement, CRT | 1.8e-12 | 0.996 | 11 / 2 / 0 |
   | Gasperini | complement, permutations | 1.8e-12 | 0.996 | 12 / 0 / 0 |

   The fold change is the log2 fold change, compared against R's; the power
   check's nonzero counts and QC calls match R exactly in all six. The
   calibration checks, fed R's own pairs, land within 0.008 of R's KS
   statistic against uniformity in all six, with false discoveries of 0
   against R's 0 for the NT cells, 343 and 356 against R's 346 and 345 for
   the complement on Papalexi, and 1 against 0 on Gasperini. The Papalexi
   complement figure is sceptre's own miscalibration, reproduced rather than
   introduced, and it is the case for the NT cells being low MOI's default.

   **On R's own permutation draws** the same three permutation runs agree in
   stage for every pair, and in p-value to 3.4e-8 relative or better for
   every pair above 1e-10 (in discovery, 89 to 96% of them bit for bit), and
   they make exactly R's discovery and calibration calls. Below about 1e-19 the two can differ by orders
   of magnitude: R's skew-normal tail loses precision to cancellation there,
   far past any threshold.

   The test statistic agrees with R to 2.4e-10 relative for 99% of pairs in
   every analysis and to 9.9e-9 for all but one. That one, in NT-cells
   discovery, differs by 1e-3: a gene with no overdispersion on its cells,
   whose dispersion estimate R and pysceptre resolve differently; see
   [Design decisions](https://broadinstitute.github.io/pysceptre/design/#a-dispersion-estimate-can-stop-on-a-rounding-accident).

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
