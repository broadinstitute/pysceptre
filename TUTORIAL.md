# Tutorial

This walks through building every input `run_discovery_analysis` needs from
scratch, running it, and reading the output. It uses small synthetic data
(no real dataset required) so you can copy-paste and run this directly.

## 1. Install

```bash
pip install -e ".[dev,fast]"
```

## 2. Build the inputs

`pysceptre` needs five things: a response (gene expression) matrix, gene
IDs, a covariate (design) matrix, a mapping from gRNA target to treated
cells, and the list of (gene, target) pairs to test.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)

n_cells = 5_000
n_genes = 8
n_targets = 15

# --- covariate_matrix: (n_cells, p) numeric design matrix ---
# In a real analysis this comes from whatever produced your sceptre_object's
# @covariate_matrix (e.g. R's model.matrix() on log(n_umis), batch, etc.) --
# pysceptre does not parse a formula DSL, so build/export this matrix
# yourself. Here: intercept + two continuous covariates.
covariate_matrix = np.column_stack([
    np.ones(n_cells),
    rng.normal(size=n_cells),   # e.g. log(response_n_umis)
    rng.normal(size=n_cells),   # e.g. log(grna_n_umis + 1)
])

# --- response_matrix: (n_genes, n_cells), one row per gene ---
# Only include genes that actually appear in `pairs` below -- don't pass a
# whole genome-wide matrix if just a few genes are tested (see README).
gene_ids = [f"gene_{i}" for i in range(n_genes)]
beta = rng.normal(loc=[1.0, 0.2, -0.1], scale=0.1, size=(n_genes, 3))
theta_true = rng.uniform(5, 50, size=n_genes)  # NB dispersion
mu_true = np.exp(covariate_matrix @ beta.T)     # (n_cells, n_genes)
response_matrix = rng.negative_binomial(
    n=theta_true[None, :], p=theta_true[None, :] / (theta_true[None, :] + mu_true)
).T.astype(float)  # (n_genes, n_cells)

# --- grna_target_cells: dict[target_id -> 0-based treated-cell indices] ---
# "union" convention: one entry per gRNA *target*, not per individual gRNA.
grna_target_cells = {
    f"target_{t}": rng.choice(n_cells, size=rng.integers(50, 400), replace=False)
    for t in range(n_targets)
}

# --- pairs: DataFrame['response_id', 'grna_target'] ---
# The QC-passed pairs to test. In a real analysis this comes from an
# upstream assign_grnas()/run_qc() step -- here, every gene x every target.
pairs = pd.DataFrame([
    {"response_id": g, "grna_target": t}
    for g in gene_ids for t in grna_target_cells
])
```

## 3. Run discovery analysis

```python
from pysceptre.pipeline.api import run_discovery_analysis

result = run_discovery_analysis(
    response_matrix=response_matrix,
    gene_ids=gene_ids,
    covariate_matrix=covariate_matrix,
    grna_target_cells=grna_target_cells,
    pairs=pairs,
    side="both",       # "left" for expected-repression screens (e.g. CRISPRi)
    seed=0,
)
print(result.head())
```

Output columns: `response_id`, `grna_target`, `p_value`, `fold_change`,
`log_2_fold_change`, `z_orig`, `stage` (see [README.md](README.md#api-reference)
for what each means, in particular `stage`, which tells you whether a pair's
p-value came from the initial empirical draws or a skew-normal tail fit).

## 4. Inject a real effect (sanity check)

A good way to build confidence in a new analysis is to plant a known signal
and confirm it comes back significant. Make one target's treated cells have
systematically lower expression of one gene (simulating, e.g., CRISPRi
knockdown of an enhancer):

```python
strong_target = "target_0"
strong_gene = "gene_0"
trt = grna_target_cells[strong_target]
response_matrix[gene_ids.index(strong_gene), trt] = rng.negative_binomial(
    n=theta_true[0], p=theta_true[0] / (theta_true[0] + mu_true[trt, 0] * 0.3)
)  # ~70% knockdown in treated cells

result = run_discovery_analysis(
    response_matrix=response_matrix, gene_ids=gene_ids,
    covariate_matrix=covariate_matrix, grna_target_cells=grna_target_cells,
    pairs=pairs, side="left", seed=0,
)
hit = result[(result.response_id == strong_gene) & (result.grna_target == strong_target)]
print(hit)  # expect a very small p_value, log_2_fold_change well below 0, stage == 2
```

## 5. Tuning for your dataset's scale

- **`target_chunk_size`** (default `200`) controls how many gRNA targets'
  logistic fits + CRT draws are held in memory simultaneously. If you're
  running out of memory at real dataset scale (thousands of targets, 500k+
  cells), lower this. If you have memory to spare, raising it gives a modest
  speed gain from better batching. See `scripts/benchmark_pairs.py` for how
  to measure this on synthetic data shaped like your real dataset before
  committing to a full run.
- **Install the `fast` extra** (`pip install -e ".[fast]"`) to get
  `numba`-accelerated CRT sampling. This is close to a strict improvement
  with no downside at real dataset scale; without it, `pysceptre` still
  works correctly, just slower.
- **Filter `response_matrix` and `gene_ids` to only genes that appear in
  `pairs`** before calling `run_discovery_analysis` -- fitting every gene in
  a genome-wide matrix when only a few hundred are actually tested wastes
  most of the work (this was the single biggest real-world memory/time
  mistake made in early testing against a real dataset: a naive densify of
  a 38,606-gene x 131k-cell matrix needed ~38 GiB and OOM-killed the
  process, versus ~256 MiB after subsetting to the 244 genes actually
  tested).

## 6. Benchmarking against your own dataset's shape

`scripts/benchmark_pairs.py` builds a synthetic dataset with the same
dimensions as a real target dataset (cell count, gene count, target count,
pair count) and times each pipeline stage. Edit the `N_CELLS`, `N_GENES`,
`N_TARGETS`, `N_PAIRS` constants at the top to match your real data's shape,
then run:

```bash
python scripts/benchmark_pairs.py
```

This is the recommended way to get a realistic time estimate *before*
committing to a full real-data run, and to catch memory issues (like the one
above) on synthetic data first.
