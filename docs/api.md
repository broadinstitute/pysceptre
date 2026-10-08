# API reference

Generated from the source. Eleven functions make up the public API, and all
eleven are importable from the top level (`from pysceptre import
run_discovery_analysis`) as well as from their defining module.

!!! note "Reading these signatures"
    `pysceptre.pipeline.api` is the documented path and the top-level
    re-export is a convenience; both are supported and both will keep
    working.

## Public API

::: pysceptre.pipeline.api.run_discovery_analysis

::: pysceptre.pipeline.api.run_calibration_check

::: pysceptre.pipeline.api.run_power_check

## Analytical power

A different question from the three above, and the one function here does not
run a test at all. `run_power_check` is sceptre's positive-control
diagnostic -- it runs the real test on pairs where an effect is expected.
`compute_power` estimates in closed form what the screen *could* have
detected, per pair, including for pairs the screen never tested.

!!! warning "Three arguments have no defaults on purpose"
    `cutoff`, `fold_change_mean` and `fold_change_sd` are all required, and
    `cutoff` must be the analysed screen's own nominal threshold. Borrowing
    another design's was measured at more than double the error with a
    systematically optimistic bias. See
    [Design decisions](design.md#analytical-per-pair-power).

::: pysceptre.analytical_power.estimate.compute_power

::: pysceptre.analytical_power.estimate.target_cell_counts

### Building its inputs

Three things `compute_power` needs that no other part of pysceptre
produces. They take arrays and frames rather than a loaded export, because
nothing under `src/` imports the `io` extra; the file-reading glue lives in
`scripts/`.

::: pysceptre.analytical_power.inputs.cells_per_grna_from_assignments

::: pysceptre.analytical_power.inputs.baseline_expression_stats_from_fits

::: pysceptre.analytical_power.inputs.matched_expression_stats

::: pysceptre.analytical_power.inputs.baseline_expression_stats

::: pysceptre.analytical_power.inputs.poscounts_size_factors

::: pysceptre.analytical_power.inputs.bh_nominal_cutoff

## Specificity check

How many discovered links are more than background, measured on tests between
elements and genes on other chromosomes. Not from sceptre; see
[Design decisions](design.md#specificity-check).

::: pysceptre.specificity.check.run_specificity_check

::: pysceptre.specificity.check.SpecificityResult

### Its pieces

::: pysceptre.specificity.background.cis_links

::: pysceptre.specificity.background.background_pairs

::: pysceptre.specificity.background.above_background_by_distance

::: pysceptre.specificity.background.broad_effect_elements

::: pysceptre.specificity.background.nearest_tss

::: pysceptre.specificity.lookup.tss_targets

::: pysceptre.specificity.lookup.gene_lookup

::: pysceptre.specificity.lookup.detour_check

## gRNA assignment

Which cells carry which gRNA, from raw gRNA UMI counts. `assign_grnas` is the
one entry point, like sceptre's, and runs any of the four methods in
`pysceptre.assignment.ASSIGNMENT_METHODS`. `assign_grnas_mixture`,
`assign_grnas_thresholding` and `assign_grnas_maximum` are sceptre's three
methods; `assign_grnas_fishash` is not sceptre's (a port of fishash). See
[Design decisions](design.md#grna-assignment).

::: pysceptre.assignment.api.assign_grnas

::: pysceptre.assignment.mixture.assign_grnas_mixture

::: pysceptre.assignment.mixture.MixtureResult

::: pysceptre.assignment.thresholding.assign_grnas_thresholding

::: pysceptre.assignment.thresholding.ThresholdingResult

::: pysceptre.assignment.maximum.assign_grnas_maximum

::: pysceptre.assignment.maximum.MaximumResult

::: pysceptre.assignment.fishash.assign_grnas_fishash

::: pysceptre.assignment.fishash.FishashResult

### Its pieces

::: pysceptre.assignment.design.mixture_design_matrix

::: pysceptre.assignment.design.design_from_covariates

::: pysceptre.assignment.design.cell_count_covariates

::: pysceptre.assignment.fishash.impute_masked_counts

::: pysceptre.assignment.hypergeom.log_phyper

::: pysceptre.assignment.cells.cells_by_grna

::: pysceptre.assignment.cells.cells_by_target

::: pysceptre.assignment.cells.cells_w_zero_or_twoplus_grnas

## The dose test

Not from sceptre. Each cell carries a target with a weight set by its gRNA UMI
count instead of being called; the three analyses take the weights through
`grna_target_weights` (`ntc_grna_weights` for the calibration check). See
[Design decisions](design.md#the-dose-test).

::: pysceptre.assignment.dose.dose_weights

::: pysceptre.assignment.dose.DoseWeights

::: pysceptre.assignment.dose.dose_ramp

::: pysceptre.assignment.dose.estimate_dose_floor

::: pysceptre.assignment.dose.DoseFloor

## Lower-level building blocks

Most users will not need to go below `run_discovery_analysis`. Each pipeline
stage is independently importable and unit-tested, for anyone extending or
debugging the pipeline.

### The GLM layer

::: pysceptre.glm.irls.fit_poisson_glm_batch

::: pysceptre.glm.irls.fit_binomial_glm_batch

::: pysceptre.glm.nb_theta.estimate_theta

### Precomputation and resampling

::: pysceptre.precompute.pieces.compute_precomputation_pieces

::: pysceptre.crt.sampler.crt_index_sampler

::: pysceptre.crt.sampler.crt_index_sampler_fast

::: pysceptre.crt.sampler.crt_index_sampler_exact

::: pysceptre.crt.permutations.permutation_draws

::: pysceptre.crt.permutations.nested_permutation_draws

### The test statistic

::: pysceptre.test_statistic.score_stat.compute_observed_full_statistic

::: pysceptre.test_statistic.score_stat.compute_null_full_statistics

::: pysceptre.test_statistic.empirical_p.compute_empirical_p_value

::: pysceptre.test_statistic.skew_normal.fit_and_evaluate_skew_normal

::: pysceptre.test_statistic.fold_change.estimate_log_fold_change

::: pysceptre.test_statistic.resampling.run_low_level_test_full

### Orchestration

::: pysceptre.pipeline.discovery.run_discovery_ntcells_complement

::: pysceptre.pipeline.discovery.run_discovery_nt_cells

::: pysceptre.pipeline.api.resolve_analysis_settings

::: pysceptre.pipeline.api.nt_cell_pool

::: pysceptre.pipeline.discovery.parallel_backend

::: pysceptre.pipeline.discovery.gene_job_backend

### Analytical power primitives

The pieces `compute_power` is assembled from, each a direct port of
the correspondingly named function in PerturbPlan (MIT -- see
`THIRD_PARTY_LICENSES`).

::: pysceptre.analytical_power.closed_form.test_stat_distribution

::: pysceptre.analytical_power.closed_form.qc_failure_prob

::: pysceptre.analytical_power.closed_form.rejection_prob

::: pysceptre.analytical_power.closed_form.zero_prob

::: pysceptre.analytical_power.closed_form.var_nb
