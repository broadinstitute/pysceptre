# API reference

Generated from the source. Four functions make up the public API, and all
four are importable from the top level (`from pysceptre import
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
`compute_power_posthoc` estimates in closed form what the screen *could* have
detected, per pair, including for pairs the screen never tested.

!!! warning "Three arguments have no defaults on purpose"
    `cutoff`, `fold_change_mean` and `fold_change_sd` are all required, and
    `cutoff` must be the analysed screen's own nominal threshold. Borrowing
    another design's was measured at more than double the error with a
    systematically optimistic bias. See
    [Design decisions](design.md#analytical-per-pair-power).

::: pysceptre.analytical_power.posthoc.compute_power_posthoc

::: pysceptre.analytical_power.posthoc.target_cell_counts

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

::: pysceptre.crt.sampler.crt_index_sampler_fast

### The test statistic

::: pysceptre.test_statistic.score_stat.compute_observed_full_statistic

::: pysceptre.test_statistic.score_stat.compute_null_full_statistics

::: pysceptre.test_statistic.empirical_p.compute_empirical_p_value

::: pysceptre.test_statistic.skew_normal.fit_and_evaluate_skew_normal

::: pysceptre.test_statistic.fold_change.estimate_log_fold_change

::: pysceptre.test_statistic.resampling.run_low_level_test_full

### Orchestration

::: pysceptre.pipeline.discovery.run_discovery_ntcells_complement

::: pysceptre.pipeline.discovery.parallel_backend

::: pysceptre.pipeline.discovery.gene_job_backend

### Analytical power primitives

The pieces `compute_power_posthoc` is assembled from, each a direct port of
the correspondingly named function in PerturbPlan (MIT -- see
`THIRD_PARTY_LICENSES`).

::: pysceptre.analytical_power.closed_form.test_stat_distribution

::: pysceptre.analytical_power.closed_form.qc_failure_prob

::: pysceptre.analytical_power.closed_form.rejection_prob

::: pysceptre.analytical_power.closed_form.zero_prob

::: pysceptre.analytical_power.closed_form.var_nb
