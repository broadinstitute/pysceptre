# Recipes

Patterns that are not a single function call. Each one has been run before
being written down, and each names the way it is most easily got wrong.

The three public analyses are in the [User guide](guide.md); this page is for
the joins and the designs around them.

## Per-gRNA tests instead of per-target

Pass `grna_integration_strategy="singleton"` with the design table, and
`grna_target_cells` keyed by guide:

```python
result = run_discovery_analysis(
    response_matrix, gene_ids, covariate_matrix,
    targeting_grna_cells,          # keyed by guide, both from the export
    pairs,                         # still (response_id, grna_target)
    grna_integration_strategy="singleton",
    grna_target_data_frame=design,
    side="left", seed=0,
)
# -> response_id, grna_id, grna_target, p_value, pct_change_es, ...
```

Each pair fans out to one row per guide of its target, and the result carries
both the guide tested and the target it belongs to, sorted by p-value with
missing values last. That is R's shape and R's ordering. `"bonferroni"` takes
the same per-guide tests and collapses them back to one row per target: the
smallest p-value among the guides that passed QC, multiplied by how many
passed and capped at 1, carrying that guide's other numbers.

**Nothing statistical changes between the strategies.** The same CRT, the same
score statistic, the same escalation. Only the treated cell set differs, which
is exactly how sceptre works: it tests whatever it calls a `grna_group`, and
the strategy only decides what a group is.

Three things worth knowing.

**It is roughly a guides-per-target multiple of the work.** On one real screen
36,450 pairs expand to 515,972, so a singleton run is about 14 times a union
run, not a flag flip. A guide shared by two overlapping targets is tested once
and reported under both, since the gene and the treated cells are identical.

**The multiple-testing burden is per guide.** A threshold derived from a
union run is the wrong one; derive it from this run's own p-values.

**QC is per guide too.** A guide's cell count is far below its target's, so
pairs that passed QC at target resolution can fail at guide resolution. The
discovery path is fed pairs already judged testable and does not recheck, so
if that distinction matters, recompute the pairwise counts at guide resolution
before you trust a null.

## A design matrix with interactions

`covariate_matrix` is a plain numeric design matrix and the engine imposes
nothing on its columns, so anything you can write as columns works. There is
no formula DSL, which means no `model.matrix()` to fight and also no
`model.matrix()` to catch your mistakes.

```python
timepoint = ...   # (n_cells,)
batch = ...       # (n_cells,) 0/1

covariate_matrix = np.column_stack([
    np.ones(n_cells),          # intercept
    np.log(response_n_umis),   # sceptre's own covariates, if you want them
    timepoint,
    batch,
    timepoint * batch,         # the interaction
])
```

**Hold out a reference level.** The one real hazard is rank deficiency, and
hand-built designs are where it happens: a factor's full dummy set beside an
intercept, a covariate repeated on two scales, or a column that is constant
within another's levels. Any of those makes the weighted least squares
singular. `run_discovery_analysis` refuses such a matrix before fitting
anything and names the redundant columns, so the failure is a sentence rather
than a `LinAlgError` from inside the solver -- but it is still your design to
fix.

Two smaller notes. Every fit scales with the number of columns, so a wide
design costs real time on a large screen. And the columns you pass are the
columns the *null model* conditions on, which is what makes a covariate a
control rather than a variable of interest: adding `timepoint` asks whether
the perturbation has an effect *beyond* what time explains.

## Power for pairs the screen could not test

The pairs that fail pairwise QC carry a NaN p-value because no test was ever
run on them. They are the clearest case for an analytical estimate, since a
p-value has nothing to say and the question -- could this screen ever have
seen this -- is exactly what the closed form answers.

```python
power = compute_power(
    untested_pairs,                # the pass_qc == False rows
    cells_per_grna, baseline_expression_stats,
    fold_change_mean=0.85, fold_change_sd=0.13,
    cutoff=cutoff, num_total_cells=n_cells,
    n_nonzero_trt_thresh=7,        # the thresholds the QC actually used
    n_nonzero_cntrl_thresh=7,
)
```

**Set the thresholds here, unlike everywhere else.** They default to 0, which
is right for pairs that already passed QC: their power should not be
discounted for a risk that did not materialise. For a pair that was never
tested the discount is the point, and the returned `qc_failure_prob` column
says how much of the answer it accounts for. A pair with high `power` and high
`qc_failure_prob` is one the screen could have detected had it captured the
cells.

--8<-- "README.md:recipe-power"
