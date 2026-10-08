# pysceptre

Standalone Python port of the statistical engine behind
[`sceptre`](https://github.com/Katsevich-Lab/sceptre)'s discovery analysis for
single-cell CRISPR screens.

**Scope: sceptre's three analyses and its gRNA assignment, plus four
things that are not sceptre's (an estimator, a check, fishash's gRNA
assignment and the dose test), not a general sceptre reimplementation.** The three analyses are
discovery analysis, the calibration check and the power check, for high- and
low-MOI screens: the complement and NT-cells control groups, with CRT
(conditional randomization test) or permutation resampling. The assignment is
all three of sceptre's `assign_grnas()` methods (mixture, thresholding,
maximum). Cell-level `run_qc()` and R's formula DSL are deliberately out of
scope -- see "Scope and limitations" in `README.md` before adding either.

**Two things that statement used to get wrong, and a reader should not have to
discover by grepping.**

`resampling_mechanism="permutations"` **exists**, on all three entry points,
and since 0.3.0 it is validated differently from the CRT rather than less.
`tests/validation/test_permutations.py` is internal consistency only, but
`test_low_moi_vs_r.py` replays R's own permutation draws
(`tests/validation/r_samplers.py`) and gets R's p-values back value for value,
for both control groups. pysceptre's own permutation draws are numpy's, so a
permutation result agrees with R in distribution, not draw for draw -- the same
footing as the CRT.

`analytical_power/` is a **fourth** thing and it does not come from sceptre.
It is a port of PerturbPlan's closed-form power estimate for a screen that
has already been run (MIT, see `THIRD_PARTY_LICENSES`), it answers what a
screen *could* have detected rather than what it did, and its ground truth is
PerturbPlan's own R rather than sceptre's. Its scope limits are its own:
complement control group only, explicit cutoff only, and no
minimum-detectable-effect-size path. `docs/design.md` has them.

`specificity/` is a **fifth**, and it has no external ground truth at all.
`run_specificity_check` measures how many discovered links exceed a background
taken from tests across chromosomes. It is ported from WattEG-paper's
`analysis/direct_indirect.py`, and `test_specificity_days.py` reproduces that
notebook on three screens; nothing outside it validates the method. The
notebook's fingerprint check is deliberately left out. `docs/design.md`,
"Specificity check", has the rest.

`assignment/` is a **sixth**, and most of it is sceptre's.
`assign_grnas_mixture`, `assign_grnas_thresholding` and `assign_grnas_maximum`
port sceptre 0.10.3's three `assign_grnas()` methods, `assign_grnas` its one
entry point and default (maximum in low MOI, mixture in high), and
`cells_w_zero_or_twoplus_grnas` its low-MOI rule after a thresholding or
mixture assignment. The mixture is validated against sceptre value for value,
internals included (`test_mixture_vs_r.py`); thresholding and maximum against
its public `assign_grnas` (`test_assignment_rules_vs_r.py`, fixture from
`scripts/dump_assignment_rules_ground_truth.R`). Two behaviours look like bugs
and are not: a maximum assignment gives a cell with no gRNA UMIs the first
gRNA, as sceptre does, and `cells_w_zero_or_twoplus_grnas` departs from sceptre
in one case on purpose; `docs/design.md`, "Thresholding and maximum", has both.
`assign_grnas_fishash` ports the R package fishash 0.99.5 (MIT;
`THIRD_PARTY_LICENSES`), a one-sided Fisher test per (gRNA, cell), and its
ground truth is fishash's own R (`test_fishash_vs_r.py`). Its p-values come
from `assignment/hypergeom.py`, a port of R's `phyper` and the nmath functions
under it (GPL-2-or-later): scipy's `hypergeom` rejects the non-integer margins
the refit passes produce, which R rounds half to even. All four methods take
raw integer counts only and refuse anything else. The mixture's Poisson fits
pass `mu_floor` = machine epsilon to `glm/irls.py` (R's `glm.fit` floor); the
discovery engine keeps the default 1e-10 -- don't unify them,
`docs/design.md`, "One Poisson GLM per gRNA", says why. The evaluation that
motivated this (fishash against the mixture, Gaussian
mixtures and the lab's CMO procedure, on the fishash preprint's simulations)
lives in `scripts/fishash_eval/`; its results belong to the manuscript.

The **dose test** is a **seventh**, and it is not sceptre's either. With
per-cell weights (`grna_target_weights`; `ntc_grna_weights` for the
calibration check) the three analyses replace the 0/1 treatment in sceptre's
score statistic by a weight, so a cell carries a target with a weight set by
its gRNA UMI count instead of being called. `assignment/dose.py`
(`dose_weights`, `dose_ramp`) builds the weights from counts, with the floor
estimated from the counts by default (`estimate_dose_floor`). It is a route
next to sceptre's test, never a replacement: without weights, or with weights
all 1, every result must stay sceptre's exactly, and `test_dose_test.py`
checks that. It has no external ground truth; `docs/design.md`, "The dose
test", has the statistic, the resampled weights and what was measured.

**One carve-out from "no `run_qc()`".** The calibration and power checks
*construct or receive* their own pairs, so both must decide which are testable
at all. Pairwise nonzero-count filtering therefore lives in
`pipeline/pairwise_qc.py`, shared by both. Cell-level and gRNA-level QC stay
out of scope.

The two use it in **opposite** ways, and that is deliberate. Calibration
samples its pairs, so it avoids ones that would fail and every returned row
passes -- there is no `pass_qc` column. A positive control is a specific
claim about a specific pair, so the power check *reports* failures with a NaN
result; dropping them would overstate power by hiding the controls the screen
had too few cells to test. Discovery behaves like the power check.

`README.md` is the user-facing reference (full API table, validation numbers,
performance figures). This file is the contributor-facing complement: don't
duplicate the README here, point at it.

## Layout
src-layout -- the importable package lives under `src/`, so it is only on
`sys.path` once installed (editable is fine).

- `src/pysceptre/`
  - `glm/`              -- batched IRLS (`irls.py`), NB dispersion (`nb_theta.py`)
                           and the design-matrix rank check (`design.py`).
  - `precompute/`       -- per-gene precomputation pieces reused across draws.
  - `crt/`              -- the CRT samplers (`sampler.py`) and the permutation
                           draws every target shares (`permutations.py`).
  - `test_statistic/`   -- score statistic, empirical p, skew-normal escalation,
                           fold change, and the per-pair `B1 -> B2 -> B3` staging.
  - `pipeline/`         -- `discovery.py` (orchestration, one path per
                           control group), `api.py` (the public
                           entry points) and `grouping.py` (the gRNA
                           integration strategies, which are pair bookkeeping
                           only: nothing statistical differs between them).
  - `analytical_power/` -- the closed-form per-pair power estimate, ported
                           from PerturbPlan (MIT, `THIRD_PARTY_LICENSES`). Not
                           from sceptre, and not part of the discovery path.
  - `specificity/`      -- the specificity check: links above a background
                           measured across chromosomes. Tables in, tables out;
                           no count matrix.
  - `assignment/`       -- gRNA-to-cell assignment: sceptre's three methods
                           (the mixture in `mixture.py` and `design.py`,
                           `thresholding.py`, `maximum.py`) and, not from
                           sceptre, fishash (`fishash.py`, with R's `phyper`
                           in `hypergeom.py`), all behind `assign_grnas` in
                           `api.py`; `cells.py` turns an assignment into cell
                           sets, and `dose.py` turns counts into the dose
                           test's cells and weights. gRNAs x cells, sparse
                           throughout.
- `tests/validation/`   -- the whole suite, in one place. **Most** files
                           compare against R ground truth rather than only
                           internal consistency, but not all:
                           `test_permutations.py` is internal-consistency
                           only (`test_low_moi_vs_r.py` is where permutations
                           meet R), and the analytical power tests compare against
                           PerturbPlan's R, not sceptre's, and the
                           specificity tests against the notebook they were
                           ported from. Slow on a fresh
                           venv while numba JIT-compiles, fast once its cache
                           is warm.
- `docker/`             -- one Dockerfile, the minimal pysceptre runtime
                           image. Two-stage, distroless, non-root, and it
                           carries **no R**: the R-plus-pysceptre comparison
                           image belongs with the comparison, in the private
                           development archive.
- `scripts/`            -- dataset export (`export_sceptre_dataset.R` +
                           `make_h5mu.py`), the R-comparison benchmark setup, and
                           the validation runners. **Not shipped in the wheel**.

`run_discovery_analysis` is also re-exported at the top level
(`from pysceptre import run_discovery_analysis`). `README.md` documents the
fully-qualified `pysceptre.pipeline.api` path; both work, keep both working.

## Commands
```bash
uv venv --python 3.12 .venv           # numba is not verified above 3.13
uv sync --extra dev --extra fast --extra io   # reproducible, uses uv.lock
uv lock --check                       # is the committed lockfile current?

pytest                                # full suite; no R needed (see Gotchas)
pytest tests/validation/test_glm_fits.py -q

ruff check . && ruff format .         # same config the pre-commit hook uses
pre-commit install                    # lint + format on commit
pre-commit run --all-files

uv build                              # sdist + wheel into dist/
uvx twine check dist/*                # metadata check before any upload

uv pip install -e ".[docs]"           # mkdocs-material + mkdocstrings
mkdocs serve                          # docs site at localhost:8000
mkdocs build --strict                 # what CI runs; a broken link fails it

Rscript scripts/dump_r_ground_truth.R tests/validation/ground_truth.json 4
```

Python 3.10+ (`requires-python`). Verified passing on 3.10, 3.11, 3.12, 3.13.

## Conventions
- **Indentation: 4 spaces**, enforced by `ruff format`. Lint and format config
  live in `pyproject.toml` under `[tool.ruff]`, so the CLI and the pre-commit
  hook cannot disagree. `E501` is off -- the formatter owns line length, and
  leaving it on would flag the long porting-rationale prose in the docstrings.
- **`[tool.ruff.format]` excludes `*.md`.** Recent ruff formats Python inside
  Markdown code blocks, which rewrites the hand-aligned snippets in `README.md`
  and `TUTORIAL.md`. Don't drop the exclude.
- **Commit messages: UPPERCASE verb prefix** -- `ADD`, `FIX`, `UPDATE`,
  `REWRITE`, `RELEASE`, `REMOVE`.
- **Docs accuracy is a hard rule**: every concrete detail (defaults, versions,
  flags, measured timings) must be confirmable from source. If you can't verify
  it, omit it.
- **Two words not to use: "arm" and "harness".** Anywhere: code, comments,
  docstrings, docs, commit messages. Say **group** for the treated or
  complement side of a comparison (`control group` is sceptre's own parameter
  name, so it is the precise term as well as the plain one), and name the
  thing itself, the dumper or the fixture or the suite or the runner, instead
  of calling it a harness.
- **No em dashes or en dashes.** Write `--` where you want the pause, which
  is what `README.md` and every source docstring already do, and a plain
  hyphen in a compound like `cis-trans`. A numeric range gets `to`, not a
  dash. This is enforced by reading, not by a hook, so check a Markdown file
  before committing it: `grep -n -e ' -- ' -e '-' <file>` must print nothing.
- **Never commit real screen data.** `.gitignore` covers
  `test_data/`, `*.rds`, `*.csv`. Only synthetic, fixed-seed
  fixtures belong in the repo.
- **Plots must be colorblind-safe.** Categorical/discrete series use the
  Okabe-Ito palette; continuous scales use `cividis`. Okabe-Ito in order:

  | | hex | | hex |
  |---|---|---|---|
  | black          | `#000000` | yellow         | `#F0E442` |
  | orange         | `#E69F00` | blue           | `#0072B2` |
  | sky blue       | `#56B4E9` | vermillion     | `#D55E00` |
  | bluish green   | `#009E73` | reddish purple | `#CC79A7` |

  `cividis` ships with matplotlib (`cmap="cividis"`), so no extra
  dependency. Don't use `viridis` for gradients here, and never `jet`/
  `rainbow`. Nothing under `src/` plots; the scripts that do keep a local
  `OKABE_ITO` dict (`scripts/plot_calibration_check.py` is the model).

- **Docstrings describe the function. Comments stay short. Rationale lives
  in the docs.** A docstring says what something does, what it takes and
  returns, and what a caller must honour. Design decisions, what R does, why
  this differs and what was measured belong in the documentation (ROADMAP
  T3.5). The narrative lives in a separate repository,
  `pysceptre-paper` (private): this one is strictly the tool.

  Source should read as code, not as a memoir. Where a choice looks
  arbitrary or invites "simplification", leave a **one-line pointer** to the
  relevant section rather than the argument itself -- enough to say a reason
  exists and where it is. `threadpool_limits` looks pointless until
  something tells you where to read why it is not.

  **The decisive reason is that a measurement is not a property of the
  code.** Run the same unchanged function on another backend, machine or
  dataset and its numbers are wrong, so keeping them in source forces a
  commit that touches the function without changing it. `git blame` and
  `git log -p` on that function then answer "when did this last change?"
  with a list of prose edits. `parallel_backend` is the worked example: its
  behaviour was untouched for months while its docstring asserted "1.85x
  with threads, 3.54x with processes", which a later measurement on the same
  code reversed for one mechanism.

  It also matters ahead of the API reference, which renders docstrings: a
  measured figure there becomes a published claim, machine- and
  dataset-specific, and stale the moment anything moves.

  **`docs/design.md` is where the reasoning goes**, and its headings are the
  anchors a pointer in the source names. Most docstrings and comments do not
  follow this yet; migrating them is ROADMAP T3.6, a module at a time. The
  backend selection in `pipeline/discovery.py` is the worked example of the
  finished shape.

- **The docs site includes `README.md` and `TUTORIAL.md` by marker, not by
  copy.** `docs/` pages pull sections through `pymdownx.snippets` using the
  `<!-- --8<-- [start:name] -->` comments in those two files, so the README
  stays canonical and cannot drift from the site. Don't delete a marker, and
  don't answer a docs gap by pasting README prose into `docs/`. Links inside
  a marked section must be absolute URLs -- a relative one resolves
  differently on GitHub and in the rendered site, and `mkdocs build
  --strict` will fail the build.
- **GPL-3.0-only is inherited, not chosen** -- upstream `sceptre` is `GPL-3`,
  which in R packaging means version 3 exactly. Don't "modernize" it to
  `-or-later`: that grants more than was received. The SPDX expression is
  PEP 639, hence `setuptools>=77` in `[build-system]` and deliberately no
  `License ::` classifier (PEP 639 forbids both together).

- **`threadpool_limits(limits=1, user_api="blas")` in `glm/irls.py` is
  deliberate, not a leftover.** The IRLS matmuls are "thin" (p is a handful of
  covariates) and called in a tight loop, so multi-threaded OpenBLAS is a net
  loss -- measured 68.7s -> 17.7s for a 150-column batch over 100k cells by
  forcing one thread. Scoped to this module so it doesn't clobber BLAS
  threading process-wide. See the module docstring.

  **It is a silent no-op on macOS.** numpy there is built against Apple
  Accelerate, which `threadpoolctl` cannot introspect -- `threadpool_info()`
  returns `[]` and a matmul takes the same time inside and outside the
  context manager (measured ratio 0.98). So local benchmarks on a
  Mac do not exercise this path, while CI (ubuntu-latest, OpenBLAS) does.
  Any measurement of this optimization must name the BLAS it was taken on.

- **numba is optional and both code paths must keep working.**
  `crt/sampler.py` try/excepts the import and falls back to a pure-numpy
  `argsort` grouping. The jitted path is a counting sort -- the grouping step
  was the dominant cost of the module at real scale.

- **`crt_index_sampler_fast` places cells with replacement and does not
  deduplicate**, an intentional approximation of R's without-replacement
  Fisher-Yates placement: a cell landing twice on one resample is counted
  twice, and about half a target's share of the cells repeats. Don't "fix" it
  into a dense `(n_cells, B)` matrix, which is what made the first port
  unusably slow.

- **The CRT sampler is chosen per target, by `crt_index_sampler`**: exact
  (`crt_index_sampler_exact`, geometric gaps, R's law) when the target is more
  than 0.2% of the cells it is tested on, fast below. Below the line day0 is
  bit-identical to 0.2.0; above it the fast sampler measurably biases the test
  conservative (5.6% of null p-values below 0.1 against 10.0%,
  `scripts/measure_nt_cells_crt_sampler.py`; docs/design.md, "The CRT sampler
  draws with replacement"). The rule reads counts, never fitted probabilities,
  so chunking cannot flip it. Retiring the threshold by making the fast path
  exact (repair duplicates after placement) is a ROADMAP item that needs day0
  re-validated.

- **The NT-cells path refits the gene's GLM for every pair.** That is R's
  method (`discovery_ntcells_perm_test`, `discovery_ntcells_crt`): each pair
  is fit on `c(trt_idxs, all_nt_idxs)`, so nothing is shared across a gene's
  targets. It is not a missed batching opportunity, and pairs are fit one at a
  time for the same reason as `_GENE_BATCH_WIDTH = 1`.

- **Cell order on the NT-cells path is R's, on purpose.** The combined cells
  are the target's cells ascending, then the NT pool in `ntc_grna_cells` dict
  order (R's `all_nt_idxs` when the dict is in R's gRNA order). Order moves no
  fitted value, but a resample is a list of positions, and matching R's order
  is what lets R's own draws be injected and compared value for value. Don't
  sort the pool.

- **`nested_permutation_draws` is R's `hybrid_fisher_iwor_sampler` built
  backwards**: a uniform ordering, then peel one position per step. Same joint
  law over the whole chain of prefixes, proved in docs/design.md and tested
  against a forward port of R in `test_nt_cells_sampler.py`. It looks nothing
  like the C++; that is not a reason to "port it properly".

- **The low-MOI QC invariant is checked, partially, on purpose.** R's engine
  never checks that each cell carries at most one gRNA, because its QC
  guarantees it. pysceptre refuses a cell under two NT gRNAs and a tested
  target sharing a cell with the NT cells. A cell in two *targets* is accepted:
  one guide in two overlapping elements does that legitimately.

- **`tests/validation/r_samplers.py` is a test-only exact replica of R's two
  permutation samplers** (boost `mt19937(4)`, `u = raw / 2**32`), checked
  against R's own draws in `lowmoi_ground_truth.json.gz`. It is what makes
  permutation p-values comparable to R value for value. It stays out of
  `src/`: the package does not reproduce R's RNG, by design.

- **RNG is not bit-for-bit reproducible against R.** sceptre seeds
  `boost::mt19937`; this uses `numpy.random.Generator`. Validation matches
  *distributions*, not draws. Don't chase exact agreement.

- **The dose test's resampled weights are drawn per stage, not up front.**
  `_StratifiedWeights` in `pipeline/discovery.py` draws a stage's weights
  only when a pair reaches it, from a generator seeded by the stage bounds
  and one number taken from the target's stream *after* its index draws. Two
  things depend on that: the index draws stay sceptre's, and a result does
  not depend on which pairs escalated first or on `n_jobs`. Drawing all
  `B1 + B2 + B3` weights eagerly was the dominant extra cost of the dose path.
  The test refuses the NT-cells control group, permutations and the per-guide
  strategies; lifting one is a design change, not a missing branch.

- **`B1`/`B2`/`B3` are derived in `api.py::_resampling_budget`, porting R's
  own sizing** (`s4_analysis_functs_1.R`: set in `run_discovery_analysis`,
  then B3 recomputed in `run_qc_pt_2`). `B1=499` always; `skew_normal` gives
  `(4999, 0)`; `no_approximation` gives `(0, ceil(mult * n_pairs / alpha))`.
  `B3=0` on the `skew_normal` path is **parity with R**, not a stub -- R only
  uses `B3=24999` for `permutations`, which this package sizes the same way.
  Don't "fix" the `skew_normal` zero.

- **`stage == 3` is reachable on the default path.** It is entered whenever
  the skew-normal fit is *rejected* (`sn_fit_used=False`), regardless of `B3`,
  and the p-value then comes from the already-drawn `B2=4999` statistics. Only
  `stage == 2` implies a skew-normal fit was actually used.

- **`tests/validation/ground_truth.json` is a committed cache.**
  `tests/validation/conftest.py` regenerates it via `Rscript` *only if the file
  is missing* -- so the suite runs in CI with no R installed, but if you change
  `scripts/dump_r_ground_truth.R` you must delete the JSON or the tests keep
  silently validating against the stale fixture. Neither the dumper nor the
  JSON records which `sceptre` version produced the fixture -- if that matters
  for a change you're making, regenerate it and note the version in the commit.

- **The assignment fixtures guard themselves against going stale.**
  `tests/validation/fishash_ground_truth.json.gz`,
  `mixture_ground_truth.json.gz` and `assignment_rules_ground_truth.json.gz`
  (from `scripts/dump_fishash_ground_truth.R`,
  `scripts/dump_mixture_ground_truth.R` and
  `scripts/dump_assignment_rules_ground_truth.R`) follow the only-if-missing
  rule too, but each records the md5 of the dumper that made it, and a test
  fails when the dumper on disk differs: change a dumper, delete its `.gz`,
  rerun the tests. All three record package versions and install SHAs and
  contain no timestamps, so regenerating gives the same bytes. The fishash and
  mixture dumpers pick the first seed in a fixed list that keeps every compared
  value away from a decision boundary, so any flipped call in those tests is a
  defect. The thresholding and maximum dumper does the opposite on purpose: its
  values sit exactly on the cuts, which is safe because every comparison there
  is exact arithmetic on integers.

- **There are more ground-truth fixtures now, and all have the same trap.**
  `tests/validation/perturbplan_ground_truth.json` caches PerturbPlan's own
  output for the analytical power port, regenerated by
  `scripts/dump_perturbplan_ground_truth.R` under the same
  only-if-missing rule as `ground_truth.json`. Change the dumper and you must
  delete the JSON. Unlike the sceptre fixture this one *does* record the
  version and commit SHA that produced it; the dumper's header explains why
  that SHA is not the one the published comparison used, and why it makes no
  difference.

  `tests/validation/lowmoi_ground_truth.json.gz` is the low-MOI one, from
  `scripts/dump_lowmoi_ground_truth.R`, same rule, and it records the sceptre
  version and install SHA. It is built from **sceptre's own** simulated
  `lowmoi_example_data`. The sceptredata package ships *real* data under the
  same name; the dumper reads with `package = "sceptre"` for that reason, and
  must keep doing so.

- **Real screen data is never committed, so anything that needs it takes a
  path from the environment and fails immediately when it is missing.**
  `tests/validation/test_day0_regression.py` reads `PYSCEPTRE_DAY0_EXPORT` and
  skips when it is unset; `test_sceptredata_realdata.py` reads
  `PYSCEPTRE_SCEPTREDATA_DIR`, the output of
  `scripts/run_sceptredata_examples.R` (sceptredata's Papalexi and Gasperini
  subsets, real data under MIT). Don't reintroduce absolute paths, and don't give a
  dataset-specific default: a wrong-but-plausible default is worse than a
  missing one when the output is a benchmark or a validation number.

- **The `realdata` pytest marker is named for the kind of test, not a
  dataset.** Those tests are deselected by default (`addopts` in
  `pyproject.toml`); run them with `pytest -m realdata`. Swapping which
  screen they run on should not mean renaming the marker -- it did once.

- **The dataset is MuData (`.h5mu`) with two assays, and the `grna` assay's
  `var` is load-bearing.** sceptre keeps gRNA assignments at exactly two
  resolutions: `grna_group_idxs`, one entry per *target* (the union of that
  target's gRNAs), and `indiv_nt_grna_idxs`, one entry per individual
  *non-targeting* gRNA. Targeting gRNAs are never kept individually -- they are
  only ever used as a union -- and NTCs are, because the calibration check
  regroups them into synthetic targets. **`"non-targeting"` is not a key in the
  target-keyed table** (2,974 keys against 2,975 distinct targets on one real
  screen), so
  a target-keyed export silently drops every NTC and makes the calibration
  check impossible. Both kinds are stored as rows of one annotated `var` with a
  `unit_kind` of `target` or `ntc_grna`. Don't collapse them back. Raw gRNA
  UMI counts, which the dose test needs, are an optional third assay,
  `grna_counts`, indexed by position with the gRNA ids in `var["grna_id"]`:
  MuData needs `var` names unique across assays, and the `grna` assay already
  uses those ids.

- **Calibration QC is a filter on construction, not a reported column.**
  A discovery result reports failures in-band (`pass_qc = False`, NaN p-value:
  34,256 rows of which 1,121 fail on a real screen). A calibration result has
  no
  `pass_qc` column at all, because every row passed by construction (33,135
  rows, zero failures). Don't "add the missing column".

- **R's calibration pair selection is not reproducible, and this is not a bug
  to chase.** Nothing in sceptre's calibration path calls `set.seed` and
  `sceptre_object` has no seed slot. Two R runs of the same object here gave
  the *identical* 100 groups but shared only ~20 of each group's 331 genes.
  Pair-by-pair comparison against R therefore requires feeding R's own
  `grna_target` column back through
  `calibration.negative_control_pairs_from_names`; the constructor itself is
  validated distributionally, which is what a calibration check measures.

- **`sample_combinations_v2` is C++ and its rule was recovered by probing it,
  not by reading it.** An installed sceptre exposes only `.Call(...)`, but the
  function is callable, so the group count was derived empirically (sceptre
  0.10.3) as `max(100, ceil(5 * n_calibration_pairs / (n_genes * p_hat)))`,
  verified on eight cases in `tests/validation/test_calibration_pairs.py`.
  R *enumerates* every combination instead when `choose(n_ntc, size) <= 100`.
  If those tests start failing, sceptre changed the rule -- re-derive it the
  same way rather than patching the expectations.

- **`uv.lock` is committed but CI installs unlocked.** The lockfile exists so
  `uv sync` reproduces a known-good dev environment; the test matrix still
  runs `uv pip install -e ".[dev,fast]"` and resolves fresh against the
  pyproject ranges, so a breaking upstream release surfaces in CI instead of
  being masked by pins. A separate `lock` job runs `uv lock --check` to stop
  the committed file drifting from `pyproject.toml` -- if you change
  dependencies, run `uv lock` and commit the result or that job fails. The
  lockfile does not constrain anyone installing pysceptre from PyPI; it is not
  in the wheel.

- **The `io` extra is dev-only and nothing under `src/` imports it.**
  `mudata` (the two-assay .h5mu) and `pyarrow` (result parquet) are used by
  `scripts/`, not
  by the package: `run_discovery_analysis` takes in-memory arrays and needs
  neither. Keep them out of `dependencies` -- a user who already has their
  data in memory should not be made to install zarr, which `anndata` pulls in.

- **Implicit namespace packages were the previous state.** `__init__.py` files
  now exist in every package dir; without them `setuptools.find_packages`
  returns `[]` and only the pyproject `packages.find` default of
  `namespaces = true` was making the build work. Keep them.
