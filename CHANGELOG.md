# Changelog

Notable changes per release. Dates are the release date.

## 0.5.0

### Added

- **The dose test**, not from sceptre: `run_discovery_analysis` and
  `run_power_check` take `grna_target_weights`, and `run_calibration_check`
  takes `ntc_grna_weights`, per-cell weights that replace the 0/1 treatment in
  sceptre's score statistic, so a cell carries a target with a weight set by
  its gRNA UMI count rather than being called. Resampled cells take weights
  drawn within their propensity quartile. Without weights, or with weights all
  1, every result is sceptre's exactly. Complement control group, CRT and the
  union strategy only.
- **`dose_weights`** builds the dose test's cells and weights from raw gRNA
  UMI counts: a cell's largest count over a target's gRNAs, weighted linearly
  in its log between `floor` and `ceiling` (`dose_ramp`). The floor is
  estimated from the counts by default (`floor="auto"`,
  `estimate_dose_floor`): the largest count whose entries are still mostly
  spread over gRNAs like single-UMI noise.
- **gRNA UMI counts in the dataset export.** `scripts/export_sceptre_dataset.R`
  writes every gRNA's counts and `make_h5mu.py` stores them as an optional
  third assay, `grna_counts`; `load_h5mu` returns them as
  `SceptreExport.grna_counts`, with their ids in `grna_count_ids`.

## 0.4.0

### Added

- **`assign_grnas_mixture`**, sceptre's mixture gRNA assignment
  (`assign_grnas(method = "mixture")`) from raw gRNA UMI counts: a Poisson
  GLM per gRNA, then sceptre's two-component EM from its five fixed starts,
  with the backup rule for gRNAs with fewer than ten cells or an EM that does
  not converge. `mixture_design_matrix` builds sceptre's default design and
  `design_from_covariates` builds it from a covariate frame exported from R.
  Validated against sceptre 0.10.3 value for value, internals included.
- **`assign_grnas_fishash`**, not from sceptre: a port of fishash 0.99.5
  (jackkamm/fishash, commit `5eabd3c`; MIT -- see `THIRD_PARTY_LICENSES`), a
  one-sided Fisher test per (gRNA, cell) with Guo and Sarkar's block FDR and
  fishash's refit against Simpson's paradox. Its p-values come from a port of
  R's `phyper` (`assignment/hypergeom.py`, from R's nmath, GPL-2-or-later),
  with a numba kernel and a numpy fallback. Validated against fishash's own
  output pass by pass.
- **`assign_grnas_thresholding`**, sceptre's thresholding assignment
  (`assign_grnas(method = "thresholding")`): a gRNA is assigned to every cell
  where its UMI count reaches `threshold` (default 5; `>=`, as in sceptre's
  C++). Validated against sceptre 0.10.3's public `assign_grnas` value for
  value.
- **`assign_grnas_maximum`**, sceptre's maximum assignment
  (`assign_grnas(method = "maximum")`, low MOI only): each cell is assigned its
  gRNA with the most UMIs, the first on a tie, and the cells sceptre's low-MOI
  QC removes come back as `cells_w_zero_or_twoplus_grnas`: those whose top
  gRNA holds at most `umi_fraction_threshold` (default 0.8) of their gRNA UMIs,
  or with fewer than `min_grna_n_umis_threshold` (default 5) gRNA UMIs. A cell
  with no gRNA UMIs is assigned the first gRNA, as in sceptre, and the UMI rule
  flags it at any threshold above 0. Validated against sceptre 0.10.3's public
  `assign_grnas` value for value.
- **`assign_grnas`**, one entry point like sceptre's: `method="default"` is
  sceptre's rule, `"maximum"` for `moi="low"` and `"mixture"` for
  `moi="high"`, and `method` can name any of `ASSIGNMENT_METHODS`
  (`"mixture"`, `"thresholding"`, `"maximum"`, `"fishash"`), each method's
  options passed through by keyword. It and the two ports above are importable
  from the top level, as the mixture and fishash ports are.
- **`pysceptre.assignment.cells_by_grna` and `cells_by_target`**, which turn an
  assignment into the `grna_target_cells` and `ntc_grna_cells` the analyses
  take.
- **`pysceptre.assignment.cells_w_zero_or_twoplus_grnas`**, sceptre's low-MOI
  rule after a thresholding or mixture assignment
  (`process_initial_assignment_list`): the cells assigned no gRNA, or two or
  more. It departs from sceptre in one case: when no cell is assigned
  anything, sceptre flags no cell as having zero gRNAs, and this flags every
  cell.

### Changed

- **`fit_poisson_glm_batch` takes `mu_floor`**, the lower bound on a fitted
  mean. The default stays 1e-10; the mixture assignment passes R's
  machine-epsilon floor.
- **The covariate validator moved to `glm/design.py`** as
  `validate_design_matrix`, unchanged; `pipeline/api.py` imports it.

## 0.3.0

### Added

- **Low MOI.** `moi="low"` on `run_discovery_analysis`, `run_power_check` and
  `run_calibration_check` selects sceptre's low-MOI defaults: the cells
  carrying a non-targeting gRNA as the control group, and permutation
  resampling. `control_group` (`"complement"` or `"nt_cells"`) and
  `resampling_mechanism` override either, as R's `set_analysis_parameters`
  does; `None` means the MOI's default, so a high-MOI call keeps the
  defaults it had.
- **The NT-cells control group**, for all three analyses. Each pair is tested
  on its target's cells together with the NT cells, so the gene's GLM is refit
  for every pair, as in R's `discovery_ntcells_perm_test` and
  `discovery_ntcells_crt`. The calibration check runs on the NT cells alone,
  and the power check counts a pair's control cells as the gene's nonzero NT
  cells. Discovery and power take the NT gRNAs as `ntc_grna_cells`.
- **`resampling_mechanism` on the calibration and power checks**, which
  previously always used the CRT. Under permutations the calibration check
  sizes its shared draws by R's rule, the `calibration_group_size` largest NT
  gRNAs together, and the power check by every target supplied.
- **`nested_permutation_draws`**, the shared permutation draws for the NT
  cells: the law of R's `hybrid_fisher_iwor_sampler`, built backwards from a
  uniform ordering, with the equivalence proved in `docs/design.md` and tested
  against a forward port of R.
- **`crt_index_sampler_exact`**, the CRT sampler the NT cells need. The fast
  sampler's with-replacement placement is accurate only while inclusion
  probabilities are tiny; against the NT cells a target is a large share of
  its combined cells, and it made the test conservative.
- **Low-MOI checks R relies on its QC for.** A cell under two NT gRNAs, or a
  tested target sharing a cell with the NT cells, is refused, as is a
  covariate that cannot be estimated on a target's cells together with the NT
  cells, which R stops on. A target whose cells the covariates separate from
  the NT cells, such as a batch with no NT cells, is reported NaN with a
  warning, since there is nothing left to test.
- **`control_group="nt_cells"` is refused in high MOI.** R quietly replaces
  any high-MOI control group with the complement; refusing means an explicit
  setting is never ignored.
- **Low-MOI exports.** The export carries `low_moi`, `control_group`,
  `resampling_mechanism` and R's NT-cell order, and `load_export(...)
  .analysis_kwargs()` configures a matching run.

### Changed

- **The CRT sampler is chosen per target.** A target above 0.2% of the cells
  it is tested on now draws from `crt_index_sampler_exact`, which has R's law
  exactly; below that the fast sampler is kept. The fast sampler repeats about
  half a target's share of its listings, which makes the test conservative:
  1.6% repeats on sceptredata's Gasperini screen and 2.3% on Papalexi's against
  the complement. **CRT p-values therefore move for every target above 0.2% of
  the cells**, which includes every target of both sceptredata screens. day0's
  targets, and its calibration check's synthetic ones, are all below it, so
  day0's results are bit-identical to 0.2.0.
- **`resampling_mechanism` defaults to `None`**, meaning the MOI's default:
  `"crt"` in high MOI, as before, and `"permutations"` in low MOI.
- **Faster per-pair fits.** BLAS threads are limited through one cached
  `ThreadpoolController` instead of a library scan on every fit, and a
  permutation stage's draw matrix is built from the shared array in one step.
  Results are bit-identical; on sceptredata's Papalexi screen against the NT
  cells, on an Apple M4 Max, a run went from 32.0 s to 24.9 s serially and from
  49.9 s to 21.7 s at `n_jobs=8`, where the scans had made eight threads slower
  than one.
- **The permutation prefix scan is now chosen by cost, not taken always.**
  `PermutationPrefixSums` computes one gene's segment sums for every target
  at once, which is cheaper than a sparse matmul per target only when enough
  targets read it: the scan costs about eight times the matmul per element,
  so a gene tested against a single target paid 6 to 7x for the same numbers.
  `prefix_scan_pays` now gates it on `sum(n_trt) >= 8 * m` and `_gene_job`
  consults it before building the scan at all. The two routes are
  bit-identical, so no result moves; single-target permutation calls get
  about 2x. Reported as
  [issue #2](https://github.com/broadinstitute/pysceptre/issues/2), and
  `docs/design.md` has the measurements.

### Fixed

- **The export of an NT-cells `sceptre_object` wrote every NT gRNA's cells
  wrong.** R stores them there as positions within its NT cells, and they were
  written as cell indices; the bounds check could not notice. They are now
  decoded and checked against the gRNA assignment.
- **A spurious warning on small runs**: "reducing target_chunk_size from 200
  to N" fired whenever a run had fewer targets than `target_chunk_size`, though
  nothing was reduced.
- **A pair with no treated cells was reported as a strong discovery.** A
  target or guide left without cells, possible under `singleton` and
  `bonferroni` after QC, gave a NaN statistic, which exceeds no null value and
  read as p = 2 / (B + 1); `bonferroni` then counted it. Such pairs, and any
  pair whose statistic is not finite, now come back NaN and are left out of the
  Bonferroni factor.
- **The skew-normal fit could abort a whole run** with a `math domain error`
  on a degenerate null distribution, where sceptre's C++ produces NaN and falls
  back to the empirical p-value. It now does the same.
- The calibration check counted non-targeting gRNAs with no cells among the
  possible groups; R drops them after QC, and so does pysceptre now.
- The calibration check gave up when R's group-count rule asked for at least
  every possible group, where R uses them all; it now does too.
- Pairs naming a target missing from `grna_target_cells` were silently
  dropped, and a gene missing from `gene_ids` failed mid-run with a bare
  `KeyError`. Both are refused up front, naming them.
- A categorical `grna_target` column made unused categories count as tested
  targets.
- Two analyses started at once in one process could corrupt each other through
  shared worker state; they now run one after the other, under one BLAS-thread
  limit for the whole call.

### Validated

- **Low MOI, value for value against sceptre 0.10.3**, on its simulated
  example data: on R's own permutation draws, replayed by a test-only replica
  of R's samplers that is checked draw for draw against R, every per-pair fit,
  statistic, fold change, stage and p-value matches R, for discovery and the
  power and calibration checks, under both control groups.
- **Permutations are validated against R for the first time.** The same
  replay on sceptredata's two real screens, Papalexi (low MOI) and Gasperini
  (high MOI): every pair reaches R's stage, and every p-value above 1e-10
  agrees to 3.4e-8 relative or better.
- **Both MOIs on real data with pysceptre's own draws**, six runs across both
  sceptredata screens, both control groups and both mechanisms: fold changes
  within 2.3e-12 of R, p-value Spearman 0.989 to 0.996, power-check counts
  exact, and calibration matching R's uniformity and false discoveries. The
  table is in README's Validation section.
- **day0 is unchanged.** Discovery and calibration on day0's real targets are
  bit-identical to 0.2.0.

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
