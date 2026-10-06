"""Why the NT-cells CRT uses an exact sampler: the figures in docs/design.md.

Prints how often `crt_index_sampler_fast` lists a cell twice in one resample,
first at fixed inclusion probabilities and then at the dimensions of four
screens (a target's cells over the cells it is tested on), and then the null
p-values of a synthetic low-MOI screen with no effects, tested against the NT
cells by permutations, by the CRT with the exact sampler, and by the CRT with
the fast sampler swapped back in.

Usage: python scripts/measure_nt_cells_crt_sampler.py [seed]
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
from scipy import sparse

import pysceptre.pipeline.discovery as discovery
from pysceptre import run_discovery_analysis
from pysceptre.crt.sampler import crt_index_sampler_fast


def repeated_fraction(p: float, n_cells: int = 2000, B: int = 5498) -> float:
    """Share of the fast sampler's listings that repeat a cell already in that resample."""
    draws = crt_index_sampler_fast(np.full(n_cells, p), B, np.random.default_rng(0))
    listed = sum(d.size for d in draws)
    return sum(d.size - np.unique(d).size for d in draws) / listed


# (label, cells tested on, treated cells): day0's scale, then sceptredata's two
# screens, Gasperini's against every cell and Papalexi's against every cell or
# against the NT cells.
REGIMES = [
    ("day0, complement", 567_690, 400),
    ("Gasperini, complement", 44_308, 1_200),
    ("Papalexi, complement", 15_645, 600),
    ("Papalexi, NT cells", 2_400, 600),
]


def regime_repeated_fraction(n_cells: int, n_trt: int, B: int = 5498) -> float:
    """The fast sampler's repeated share for probabilities spread around `n_trt / n_cells`."""
    rng = np.random.default_rng(0)
    p = np.clip(rng.gamma(4, n_trt / n_cells / 4, size=n_cells), 1e-6, 0.95)
    draws = crt_index_sampler_fast(p, B, np.random.default_rng(2))[:300]
    return sum(d.size - np.unique(d).size for d in draws) / sum(d.size for d in draws)


def null_screen(seed: int = 1):
    """6,000 cells each carrying one of 30 gRNAs (12 targets of two, 6 NT), no effects."""
    rng = np.random.default_rng(seed)
    n_cells, n_genes, n_targets, n_ntc = 6000, 40, 12, 6
    X = np.column_stack(
        [np.ones(n_cells), rng.normal(size=n_cells), (rng.random(n_cells) < 0.5).astype(float)]
    )
    grnas = [f"t{k}_g{j}" for k in range(n_targets) for j in range(2)] + [
        f"nt{j}" for j in range(n_ntc)
    ]
    assign = rng.integers(0, len(grnas), size=n_cells)
    cells_of = {g: np.flatnonzero(assign == i) for i, g in enumerate(grnas)}
    targets = {
        f"t{k}": np.sort(np.concatenate([cells_of[f"t{k}_g0"], cells_of[f"t{k}_g1"]]))
        for k in range(n_targets)
    }
    ntc = {f"nt{j}": cells_of[f"nt{j}"] for j in range(n_ntc)}
    mu = np.exp(0.8 + 0.25 * X[:, 1] + 0.3 * X[:, 2] + rng.normal(scale=0.3, size=(n_genes, 1)))
    Y = rng.negative_binomial(4, 4 / (4 + mu)).astype(float)
    genes = [f"g{i}" for i in range(n_genes)]
    pairs = pd.DataFrame(
        [(g, t) for g in genes for t in targets], columns=["response_id", "grna_target"]
    )
    return sparse.csr_matrix(Y), genes, X, targets, ntc, pairs


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    warnings.simplefilter("ignore")
    for p in (0.0012, 0.1, 0.25, 0.4):
        print(f"fast sampler, p = {p}: {repeated_fraction(p):.4f} of listings repeated")
    for label, n_cells, n_trt in REGIMES:
        share = n_trt / n_cells
        rep = regime_repeated_fraction(n_cells, n_trt)
        print(f"{label:22s} target share {share:.4%}: {rep:.4%} of listings repeated")

    resp, genes, X, targets, ntc, pairs = null_screen()
    common = dict(moi="low", ntc_grna_cells=ntc, seed=seed)
    sizes = [len(c) for c in targets.values()]
    n_nt = sum(len(c) for c in ntc.values())
    print(
        f"\n{len(pairs)} null pairs, targets of {min(sizes)} to {max(sizes)} cells, {n_nt} NT cells"
    )
    perm = run_discovery_analysis(resp, genes, X, targets, pairs, **common)
    exact = run_discovery_analysis(
        resp, genes, X, targets, pairs, resampling_mechanism="crt", **common
    )
    chooser = discovery.crt_index_sampler
    discovery.crt_index_sampler = lambda p, B, rng, n_trt: crt_index_sampler_fast(p, B, rng)
    try:
        fast = run_discovery_analysis(
            resp, genes, X, targets, pairs, resampling_mechanism="crt", **common
        )
    finally:
        discovery.crt_index_sampler = chooser
    for name, r in [
        ("permutations", perm),
        ("CRT, exact sampler", exact),
        ("CRT, fast sampler", fast),
    ]:
        p = r["p_value"].to_numpy()
        print(f"{name:20s} mean null p = {p.mean():.3f}, P(p < 0.1) = {np.mean(p < 0.1):.3f}")


if __name__ == "__main__":
    main()
