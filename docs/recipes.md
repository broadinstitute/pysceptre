# Recipes

Patterns that are not a single function call. Each one has been run before
being written down, and each names the way it is most easily got wrong.

The three public analyses are in the [User guide](guide.md); this page is for
the joins and the designs around them.

## Per-gRNA tests instead of per-target

[Scope](scope.md) says gRNA integration is `"union"` only, and that is true of
sceptre's *named* strategy. But the engine never learns what the keys of
`grna_target_cells` mean: it builds one indicator per key, fits the binomial
model and draws the CRT from it. So key that dict by gRNA and you get one test
per guide.

```python
# targeting_grna_cells comes straight from an export: grna_id -> cell indices
pairs = pd.DataFrame(
    [(gene, guide) for gene in genes_of_interest for guide in targeting_grna_cells],
    columns=["response_id", "grna_target"],
)

result = run_discovery_analysis(
    response_matrix, gene_ids, covariate_matrix,
    targeting_grna_cells,          # keyed by guide, not by target
    pairs,
    side="left", seed=0,
)
```

On a screen where one guide of four carries a real knockdown, that guide comes
back significant and its three siblings do not.

**Three things this is not.** It is not sceptre's `"singleton"` strategy
reproduced: there is no per-target aggregation afterwards, and no attempt to
match how sceptre reports singleton results. The multiple-testing burden is
now per guide rather than per target, so a threshold derived from a
per-target run is the wrong one -- derive it from this run's own p-values.
And a guide's cell count is much smaller than its target's, so pairs that
passed QC at target resolution can fail it at guide resolution; check that
before reading a null result as biology.

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
