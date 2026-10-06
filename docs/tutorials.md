# Tutorials

One worked example end to end, then five shorter ones for the tasks that are
not a single function call. Every code block here has been run before being
written down, and each one names the way it is most easily got wrong.

The three sceptre analyses themselves are in the [User guide](guide.md).

---

## A discovery analysis end to end

Synthetic inputs, small enough to run in seconds.

--8<-- "TUTORIAL.md:body"

---

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
    seed=0,                        # side defaults to "both", as in sceptre
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

One thing you may be warned about. A design table that lists the same guide
twice against the same target has its rows kept, because that is what R does,
and you will see a warning saying so. It is inert under `"union"`, where a
target's cells are a set union over its guides, but under `"singleton"` the
guide appears twice in the result and under `"bonferroni"` it is counted twice
in the correction, inflating the corrected p-value. Pass
`drop_duplicate_design_rows=True` to collapse them, at the cost of no longer
matching R's row count.

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

--8<-- "README.md:tutorial-power"

## A low-MOI screen

Synthetic again: 3,000 cells, each carrying exactly one of fourteen gRNAs, as
sceptre's low-MOI QC leaves a screen. Five targets have two guides each, four
guides are non-targeting, and one target knocks its gene down.

```python
import numpy as np
import pandas as pd
from scipy import sparse

from pysceptre import run_calibration_check, run_discovery_analysis, run_power_check

rng = np.random.default_rng(0)
n_cells, n_genes = 3000, 8
covariate_matrix = np.column_stack([np.ones(n_cells), rng.normal(size=n_cells)])

guides = [f"t{k}_g{j}" for k in range(5) for j in range(2)] + [f"nt{j}" for j in range(4)]
guide_of_cell = rng.integers(0, len(guides), size=n_cells)     # one gRNA per cell
cells = {g: np.flatnonzero(guide_of_cell == i) for i, g in enumerate(guides)}
grna_target_cells = {
    f"t{k}": np.sort(np.concatenate([cells[f"t{k}_g0"], cells[f"t{k}_g1"]])) for k in range(5)
}
ntc_grna_cells = {f"nt{j}": cells[f"nt{j}"] for j in range(4)}   # one entry per NT guide

mu = np.exp(1.0 + 0.3 * covariate_matrix[:, 1])
counts = rng.negative_binomial(5, 5 / (5 + mu), size=(n_genes, n_cells))
counts[0, grna_target_cells["t0"]] //= 3                         # t0 knocks down g0
response_matrix = sparse.csr_matrix(counts.astype(float))
gene_ids = [f"g{i}" for i in range(n_genes)]
pairs = pd.DataFrame(
    [(g, t) for g in gene_ids for t in grna_target_cells], columns=["response_id", "grna_target"]
)

result = run_discovery_analysis(
    response_matrix, gene_ids, covariate_matrix, grna_target_cells, pairs,
    moi="low", ntc_grna_cells=ntc_grna_cells, seed=0,
)
result.sort_values("p_value").head(3)[["response_id", "grna_target", "p_value", "pct_change_es"]]
#  response_id grna_target       p_value  pct_change_es
#           g0          t0  1.383583e-92     -70.540403
#           g6          t4  9.139733e-03       7.983770
#           g4          t2  1.334429e-02       7.187633
```

`moi="low"` is what changed: every target is tested against the cells carrying
a non-targeting guide, by permutation, which are sceptre's low-MOI defaults.
The calibration and power checks take the same switch:

```python
calibration = run_calibration_check(
    response_matrix, gene_ids, covariate_matrix, ntc_grna_cells,
    n_calibration_pairs=len(pairs), calibration_group_size=2, moi="low", seed=0,
)
power = run_power_check(
    response_matrix, gene_ids, covariate_matrix, grna_target_cells,
    positive_control_pairs=pd.DataFrame({"response_id": ["g0"], "grna_target": ["t0"]}),
    moi="low", ntc_grna_cells=ntc_grna_cells, seed=0,
)
# calibration: 40 pairs, 2.5% with p < 0.05
# power: g0 / t0, n_nonzero_trt 213, n_nonzero_cntrl 753 (nonzero NT cells), p 1.38e-92
```

Three things worth knowing.

**One gRNA per cell is a precondition, and it is checked where it can be.**
sceptre's QC removes cells with zero or two or more gRNAs before a low-MOI
analysis; pysceptre does not run that QC. A cell listed under two NT guides, or
under a tested target and an NT guide, is refused with the offending target
named:

```text
ValueError: 1 tested target(s) share cells with the NT cells, e.g. 't1' (2 cells).
With control_group='nt_cells' a cell must carry at most one gRNA, ...
```

**The cost is per pair.** Against the NT cells a gene's GLM is refit for every
pair, as in R, because each pair has its own cells. The fits parallelize across
genes with `n_jobs` like everything else.

**Both defaults can be overridden.** `resampling_mechanism="crt"` keeps the
NT cells and resamples from a logistic fit instead, and results then stop
depending on which other targets are in the run. `control_group="complement"`
runs the high-MOI engine on a low-MOI screen.

---

Two longer tutorials are planned and not yet written: a real screen from
`.h5mu` through to a result frame, and the scanpy/MuData interop path. Until
they exist,
[`examples/scanpy_interop.py`](https://github.com/broadinstitute/pysceptre/blob/main/examples/scanpy_interop.py)
in the repository shows the second as runnable code.
