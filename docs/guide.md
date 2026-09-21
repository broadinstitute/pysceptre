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
[Tutorials](tutorials.md) shows the join.

## If a design matrix is refused

`covariate_matrix` is checked before any fitting and a rank-deficient design
is refused by name, because there is no formula DSL here to catch an aliased
contrast for you. The usual causes are a factor's full dummy set alongside an
intercept, a covariate included twice on different scales, or a column that is
constant within another's levels. Hold out a reference level, or drop the
column the error names.

--8<-- "README.md:reproducibility"

--8<-- "README.md:performance"
