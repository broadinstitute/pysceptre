# pysceptre

Standalone Python port of the statistical engine behind
[`sceptre`](https://github.com/Katsevich-Lab/sceptre)'s discovery analysis for
single-cell CRISPR screens.

**Scope: two validated analysis paths, not a general sceptre reimplementation.**
Discovery analysis and the calibration check, both on the complement control
group + CRT (conditional randomization test) resampling, high-MOI path.
Permutations, non-complement control groups, low-MOI, `assign_grnas()`,
`run_qc()`, the power check, and R's formula DSL are all deliberately out of
scope — see "Scope and limitations" in `README.md` before adding any of them.

**One carve-out from "no `run_qc()`".** The calibration check *constructs* its
pairs, and sceptre only ever samples combinations that already clear
`n_nonzero_trt_thresh` / `n_nonzero_cntrl_thresh`, so pairwise nonzero-count
filtering is inseparable from building the pairs and lives in
`pipeline/calibration.py`. Cell-level and gRNA-level QC stay out of scope.

`README.md` is the user-facing reference (full API table, validation numbers,
performance figures). This file is the contributor-facing complement: don't
duplicate the README here, point at it.

## Layout
src-layout — the importable package lives under `src/`, so it is only on
`sys.path` once installed (editable is fine).

- `src/pysceptre/`
  - `glm/`            — batched IRLS (`irls.py`) and NB dispersion (`nb_theta.py`).
  - `precompute/`     — per-gene precomputation pieces reused across draws.
  - `crt/`            — the CRT resampling draw (`sampler.py`).
  - `test_statistic/` — score statistic, empirical p, skew-normal escalation,
                        fold change, and the per-pair `B1 -> B2 -> B3` staging.
  - `pipeline/`       — `discovery.py` (orchestration) and `api.py` (the one
                        public entry point, `run_discovery_analysis`).
- `tests/validation/` — the whole suite (21 tests; ~20s on a fresh venv while
                        numba JIT-compiles, ~2s once its cache is warm). Every
                        test compares against R ground truth, not just internal
                        consistency.
- `scripts/`          — dataset export (`export_sceptre_dataset.R` +
                        `make_h5mu.py`), the R-comparison benchmark setup, and the
                        validation runners. **Not shipped in the wheel**.

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

Rscript scripts/dump_r_ground_truth.R tests/validation/ground_truth.json 4
```

Python 3.10+ (`requires-python`). Verified passing on 3.10, 3.11, 3.12, 3.13.

## Conventions
- **Indentation: 4 spaces**, enforced by `ruff format`. Lint and format config
  live in `pyproject.toml` under `[tool.ruff]`, so the CLI and the pre-commit
  hook cannot disagree. `E501` is off — the formatter owns line length, and
  leaving it on would flag the long porting-rationale prose in the docstrings.
- **`[tool.ruff.format]` excludes `*.md`.** Recent ruff formats Python inside
  Markdown code blocks, which rewrites the hand-aligned snippets in `README.md`
  and `TUTORIAL.md`. Don't drop the exclude.
- **Commit messages: UPPERCASE verb prefix** — `ADD`, `FIX`, `UPDATE`,
  `REWRITE`, `RELEASE`, `REMOVE`.
- **Docs accuracy is a hard rule**: every concrete detail (defaults, versions,
  flags, measured timings) must be confirmable from source. If you can't verify
  it, omit it.
- **Never commit real screen data.** `.gitignore` covers
  `tests/validation/moi5_real/`, `*.rds`, `*.csv`. Only synthetic, fixed-seed
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
  `rainbow`. Nothing in the repo plots yet — this applies to whatever
  does first.

- Docstrings here carry the *why* of each port decision (what R does, why the
  Python differs, what was measured). Preserve that when editing — it is the
  main defense against someone "simplifying" a deliberate choice.

## Gotchas (non-obvious)

- **GPL-3.0-only is inherited, not chosen** — upstream `sceptre` is `GPL-3`,
  which in R packaging means version 3 exactly. Don't "modernize" it to
  `-or-later`: that grants more than was received. The SPDX expression is
  PEP 639, hence `setuptools>=77` in `[build-system]` and deliberately no
  `License ::` classifier (PEP 639 forbids both together).

- **`threadpool_limits(limits=1, user_api="blas")` in `glm/irls.py` is
  deliberate, not a leftover.** The IRLS matmuls are "thin" (p is a handful of
  covariates) and called in a tight loop, so multi-threaded OpenBLAS is a net
  loss — measured 68.7s -> 17.7s for a 150-column batch over 100k cells by
  forcing one thread. Scoped to this module so it doesn't clobber BLAS
  threading process-wide. See the module docstring.

  **It is a silent no-op on macOS.** numpy there is built against Apple
  Accelerate, which `threadpoolctl` cannot introspect —
  `threadpool_info()` returns `[]` and a matmul takes the same time inside and
  outside the context manager (measured ratio 0.98). So local benchmarks on a
  Mac do not exercise this path, while CI (ubuntu-latest, OpenBLAS) does.
  Any measurement of this optimization must name the BLAS it was taken on.

- **numba is optional and both code paths must keep working.**
  `crt/sampler.py` try/excepts the import and falls back to a pure-numpy
  `argsort` grouping. The jitted path is a counting sort — the grouping step
  was the dominant cost of the module at real scale.

- **The CRT sampler's with-replacement draw + `np.unique` dedupe is an
  intentional approximation** of R's without-replacement Fisher-Yates
  placement, justified in the `crt/sampler.py` docstring (M_j << B, so
  collisions are rare). It is not a bug; don't "fix" it into a dense
  `(n_cells, B)` matrix, which is what made the first port unusably slow.

- **RNG is not bit-for-bit reproducible against R.** sceptre seeds
  `boost::mt19937`; this uses `numpy.random.Generator`. Validation matches
  *distributions*, not draws. Don't chase exact agreement.

- **`B1`/`B2`/`B3` are derived in `api.py::_resampling_budget`, porting R's
  own sizing** (`s4_analysis_functs_1.R`: set in `run_discovery_analysis`,
  then B3 recomputed in `run_qc_pt_2`). `B1=499` always; `skew_normal` gives
  `(4999, 0)`; `no_approximation` gives `(0, ceil(mult * n_pairs / alpha))`.
  `B3=0` on the `skew_normal` path is **parity with R**, not a stub — R only
  uses `B3=24999` for `permutations`, which is out of scope. Don't "fix" it.

- **`stage == 3` is reachable on the default path.** It is entered whenever
  the skew-normal fit is *rejected* (`sn_fit_used=False`), regardless of `B3`,
  and the p-value then comes from the already-drawn `B2=4999` statistics. Only
  `stage == 2` implies a skew-normal fit was actually used.

- **`tests/validation/ground_truth.json` is a committed cache.**
  `tests/validation/conftest.py` regenerates it via `Rscript` *only if the file
  is missing* — so the suite runs in CI with no R installed, but if you change
  `scripts/dump_r_ground_truth.R` you must delete the JSON or the tests keep
  silently validating against the stale fixture. Neither the dumper nor the
  JSON records which `sceptre` version produced the fixture — if that matters
  for a change you're making, regenerate it and note the version in the commit.

- **The moi5 scripts take their paths from the environment**, because the data
  they touch is real screen data that is never committed:

  | var | used by | what |
  |---|---|---|
  | `PYSCEPTRE_MOI5_DIR` | all three | the moi5 export directory |
  | `WTC11_BASE`         | the two R scripts | root of the wtc-11 data tree |
  | `SCEPTRE_IO_R`       | the two R scripts | `sceptre_io.R` from the element-gene-power-analysis repo |

  `run_moi5_validation.py` also takes `--data-dir`, which wins over the env
  var. Each script fails immediately with a message naming the missing
  variable, rather than part-way through. These were hardcoded
  `/mnt/disks/sw-dev-disk/...` cloud-VM paths; don't reintroduce absolute
  paths here.

- **The dataset is MuData (`.h5mu`) with two assays, and the `grna` assay's
  `var` is load-bearing.** sceptre keeps gRNA assignments at exactly two
  resolutions: `grna_group_idxs`, one entry per *target* (the union of that
  target's gRNAs), and `indiv_nt_grna_idxs`, one entry per individual
  *non-targeting* gRNA. Targeting gRNAs are never kept individually — they are
  only ever used as a union — and NTCs are, because the calibration check
  regroups them into synthetic targets. **`"non-targeting"` is not a key in the
  target-keyed table** (2,974 keys against 2,975 distinct targets on moi5), so
  a target-keyed export silently drops every NTC and makes the calibration
  check impossible. Both kinds are stored as rows of one annotated `var` with a
  `unit_kind` of `target` or `ntc_grna`. Don't collapse them back.

- **Calibration QC is a filter on construction, not a reported column.**
  A discovery result reports failures in-band (`pass_qc = False`, NaN p-value:
  34,256 rows of which 1,121 fail on moi5). A calibration result has no
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
  If those tests start failing, sceptre changed the rule — re-derive it the
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
