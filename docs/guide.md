# User guide

--8<-- "README.md:quickstart"

## The other two sceptre analyses

The calibration check and the power check take the same inputs as the
discovery analysis and return the same shape of frame. They differ in where
their pairs come from, and that difference shows up in the result:

| | pairs | QC failures |
|---|---|---|
| `run_discovery_analysis` | you supply them | reported in-band: `pass_qc = False`, `p_value = NaN` |
| `run_calibration_check` | constructed from non-targeting gRNAs | cannot occur -- pairs are chosen to pass, so there is no `pass_qc` column |
| `run_power_check` | you supply positive controls | reported in-band, like discovery |

The asymmetry is deliberate. A positive control is a claim about one specific
pair, so silently dropping the ones with too few cells would overstate power;
a calibration pair is interchangeable, so one that would fail is simply never
drawn. See [Design decisions](design.md#pairwise-qc-appears-in-two-different-shapes).

Signatures and parameters for all three are in the
[API reference](api.md).

## Low MOI

--8<-- "README.md:lowmoi"

A worked example is in [Tutorials](tutorials.md#a-low-moi-screen).

## Choosing what counts as one treated unit

`run_discovery_analysis` takes `grna_integration_strategy`, matching sceptre's
own option:

| | one test per | result identifies |
|---|---|---|
| `"union"` (default) | gRNA target, on the union of its guides' cells | `grna_target` |
| `"singleton"` | individual guide | `grna_id` and `grna_target` |
| `"bonferroni"` | guide, then aggregated back to the target | `grna_target` |

**Nothing statistical differs between them.** The same CRT, the same score
statistic, the same escalation; only which cells count as treated, and in
`bonferroni`'s case what happens to the results afterwards. The two per-guide
strategies want `grna_target_cells` keyed by guide and the design table in
`grna_target_data_frame`, and they cost roughly a guides-per-target multiple
of the work. [Tutorials](tutorials.md#per-grna-tests-instead-of-per-target)
has a worked example.

## A fourth entry point, which is not sceptre's

`compute_power` estimates in closed form what a screen *could* have detected,
per pair and without simulation, so it speaks to the pairs a p-value cannot:
the ones that failed QC and were never tested. It is a port of
[PerturbPlan](https://github.com/Katsevich-Lab/perturbplan) rather than of
sceptre and is validated against that package instead, so read its limits in
[Design decisions](design.md#analytical-per-pair-power) before using it on a
single pair. `run_discovery_analysis` does not call it, and
[Tutorials](tutorials.md) shows the join. `matched_expression_stats` is an
optional per-pair input that carries the covariates into the estimate; see
[Design decisions](design.md#covariates-through-the-information-matched-mean).

## A fifth, also not sceptre's

`run_specificity_check` asks how many of the links a screen discovered are
more than background: how often a test is called without regulation, measured
on tests between elements and genes on other chromosomes, matched gene for
gene and bootstrapped over elements. It needs a discovery result on the cis
pairs, one from the same test on those elements against genes elsewhere, and
positions for both. It has no R counterpart; read
[Design decisions](design.md#specificity-check) for what it has been checked
against.

## Assigning gRNAs to cells

The analyses take each target's cells as given. `assign_grnas` computes them
from raw gRNA UMI counts, like sceptre's function of the same name, with one of
four methods: sceptre's mixture, thresholding and maximum methods, each
validated against sceptre's own, or fishash's one-sided Fisher test, which is
not sceptre's and is validated against fishash's R. Left at its default it
takes sceptre's choice, the maximum method in low MOI and the mixture in high
MOI. Each method is also a function of its own: `assign_grnas_mixture`,
`assign_grnas_thresholding`, `assign_grnas_maximum` and `assign_grnas_fishash`.
`pysceptre.assignment.cells_by_target` turns any result into the
`grna_target_cells` and `ntc_grna_cells` the analyses take.

In low MOI, sceptre's QC then removes cells, and that removal is left to you:
a maximum assignment returns the cells it flags, and after any other method
`pysceptre.assignment.cells_w_zero_or_twoplus_grnas` lists the cells assigned
no gRNA or two or more. Read [Design decisions](design.md#grna-assignment) for
what each method normalizes for, and
[Thresholding and maximum](design.md#thresholding-and-maximum) for those two
methods' edge cases.

## If a design matrix is refused

`covariate_matrix` is checked before any fitting and a rank-deficient design
is refused by name, because there is no formula DSL here to catch an aliased
contrast for you. The usual causes are a factor's full dummy set alongside an
intercept, a covariate included twice on different scales, or a column that is
constant within another's levels. Hold out a reference level, or drop the
column the error names.

--8<-- "README.md:reproducibility"

--8<-- "README.md:performance"
