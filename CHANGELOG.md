# Changelog

Notable changes per release. Dates are the release date.

## 0.2.0

### Added

- **`compute_power`**, a closed-form per-pair power estimate: if this element
  really did reduce this gene by X%, would this screen have detected it? No
  simulation and no fitted parameters, so it answers that for pairs the screen
  never tested as well as for the ones it did. A port of
  [PerturbPlan](https://github.com/Katsevich-Lab/perturbplan)'s
  `compute_power_posthoc()` (MIT, commit `43232419fe26`; see
  `THIRD_PARTY_LICENSES`), validated against that package's own output to a
  relative 1e-9 across 18 cached cases. Three arguments deliberately have no
  default, and `num_trt_cells` is the sum over a target's gRNAs rather than
  the union; `docs/design.md` says why for each.
- **`analytical_power.inputs`**, the three helpers that build its inputs:
  per-gRNA cell counts, per-gene baseline statistics, and the screen's own
  nominal cutoff. `baseline_expression_stats_from_fits` is the recommended
  route, taking the mean and theta off the same negative-binomial fit the
  discovery test uses, so the estimate and the test it predicts share one
  expression scale by construction.
- **`singleton` and `bonferroni` gRNA integration strategies**, via
  `grna_integration_strategy` on `run_discovery_analysis`. Nothing statistical
  differs between the strategies; only which cells count as treated, and what
  happens to the results afterwards. The pair expansion is validated against
  sceptre's own `update_dfs_based_on_grouping_strategy`, including on a real
  screen where 36,450 pairs expand to 515,972 and the gRNA-to-target map is
  many-to-many.
- **A rank check on `covariate_matrix`.** A rank-deficient design previously
  failed as a bare `LinAlgError` from inside the batched solve; it is now
  refused before any fitting, naming the redundant columns. Transposed
  matrices, row-count mismatches and non-finite entries are caught there too.
- **Four tutorials** in the documentation, beside the existing worked example:
  per-gRNA tests, design matrices with interactions, power for pairs that
  failed QC, and power reported alongside a discovery result.
- **Per-gRNA targeting units and `--all-cells` in the export**, so an export
  can answer questions the discovery pair list does not anticipate. An export
  now carries every gene and every cell by default.

### Changed

- **An export carries everything by default.** `--all-genes` and `--all-cells`
  are now no-ops; `--pair-genes-only` and `--qc-cells-only` narrow it. The
  choice is not recoverable afterwards, because poscounts size factors are a
  per-cell median against a geometric mean over every cell and every gene.
- **The container is pysceptre-only and distroless**, 776 MB to 585 MB, and no
  longer carries R: the R-plus-pysceptre comparison image belongs with the
  comparison. A `.dockerignore` keeps a 4.3 GB build context out of the
  daemon.
- **Scope documentation corrected in two places it was wrong.** Permutation
  resampling is implemented, though exercised rather than validated against R;
  and `analytical_power` is a fourth path that does not come from sceptre.

### Fixed

- `compute_power` rejected any repeated `grna_id` in `cells_per_grna`, which
  is stricter than R and wrong for overlapping candidate elements: a guide
  inside two of them belongs to both targets and contributes its cells to
  each. Uniqueness now applies to the `(grna_id, grna_target)` pair.

### Validated

- **The power estimate is scored against per-pair simulation**, not only
  against the code it ports. On day0's 34,886 pairs with 100 simulations
  each, at the 0.8 bar: MCC 0.941 at a 15 % knockdown and between 0.911 and
  0.961 from 10 % to 50 %, with no degradation as the signal grows. This is
  new evidence rather than a re-run, because day0 was analysed under the CRT
  where the published comparison used sceptre's permutation test.
- **The two expression conventions are scored against each other**, and the
  documented default wins at every effect size: MCC 0.941 against 0.910 at a
  15 % knockdown, widening to 0.925 against 0.856 at 50 %. That is
  corroboration rather than adjudication: which mean is correct is settled by
  PerturbPlan's own definition of the input, `avg_library_size *
  relative_expression`, an expected raw count per cell. See
  `docs/design.md`.

### Notes

- `0.2.0rc1` was tagged during development and is superseded by this release.
- The version skips `0.1.1` deliberately; this is a feature release.

## 0.1.0

First public release. The statistical engine behind sceptre's discovery
analysis for the complement control group and CRT resampling, plus the
calibration check and the power check, validated against R.
