"""A result must depend on its own inputs, and nothing else.

Re-running an analysis with one more pair in the list should not move the
pairs that were already there. That sounds obvious and is easy to get wrong:
with a single shared generator consumed target by target, inserting one
target shifts every later target's draws, and every later p-value with them.

Each target now seeds its own stream from its *name*
(`discovery.target_seed_sequence`), so its draws depend on `(seed,
target_id)` and on nothing about the rest of the run.

**There are two guarantees here, not one, and they differ in strength.**

*Changing the targets or pairs* is bitwise exact. A target's draws come from
its own name-keyed stream, and the binomial fit is batched per chunk but
measured identical on both BLAS libraries tested.

*Changing the gene set* is exact only to floating point. The gene GLM is
batched across whichever genes share a chunk, and floating-point addition is
not associative, so a different batch width can change a fit in its last
bits. Measured on OpenBLAS: removing one gene from a ten-gene run moved
`fold_change` on 3 of 72 shared pairs, by at most **1.11e-16** -- one unit in
the last place. On Accelerate the same comparison was exactly zero. This
cannot be fixed while the fits are batched, and batching them is the reason
this package is fast, so it is documented rather than chased.

The practical consequence for reusing results across runs: **adding pairs or
targets is safe to the bit; adding genes is safe to about 1e-16.** No
scientific conclusion turns on the difference, but "identical" is the wrong
word for the second case and "identical to floating-point precision" is the
right one.

**What is *not* reusable, and should not be.** A Benjamini-Hochberg adjusted
p-value depends on every p-value in the set, so adding pairs changes the
adjusted value -- and possibly the significance call -- for pairs that were
already there. That is multiple testing working correctly, not a
reproducibility failure. It happens to be outside this package's output
entirely: `run_discovery_analysis` returns raw p-values and applies no
correction, so the column this guarantee covers is exactly the one worth
caching, and the column that must be recomputed over the union is one the
caller owns anyway.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pysceptre.pipeline.api import run_discovery_analysis
from pysceptre.pipeline.discovery import target_seed_sequence

N_GENES, N_CELLS, N_TARGETS = 10, 2000, 8


@pytest.fixture(scope="module")
def world():
    """A few more genes and targets than any single run uses."""
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(N_CELLS), rng.normal(size=(N_CELLS, 2))])
    theta, mu = 5.0, 20.0
    counts = rng.negative_binomial(theta, theta / (theta + mu), size=(N_GENES + 2, N_CELLS)).astype(
        float
    )
    genes = [f"g{i}" for i in range(N_GENES + 2)]
    targets = {
        f"t{j}": np.sort(rng.choice(N_CELLS, 200, replace=False)) for j in range(N_TARGETS + 2)
    }
    return counts, genes, X, targets


def _run(world, gene_sel, target_sel, **kwargs):
    counts, genes, X, targets = world
    pairs = pd.DataFrame(
        [(g, t) for g in gene_sel for t in target_sel],
        columns=["response_id", "grna_target"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_discovery_analysis(
            response_matrix=counts,
            gene_ids=genes,
            covariate_matrix=X,
            grna_target_cells={t: targets[t] for t in target_sel},
            pairs=pairs,
            seed=0,
            chunk_memory_gb=8.0,
            **kwargs,
        )
    return result.set_index(["response_id", "grna_target"])


def _assert_shared_pairs_identical(base, other, what):
    """Bitwise. Used where the guarantee really is exact."""
    common = base.index.intersection(other.index)
    assert len(common) > 0
    for col in ("p_value", "fold_change", "z_orig", "stage"):
        np.testing.assert_array_equal(
            base.loc[common, col].to_numpy(),
            other.loc[common, col].to_numpy(),
            err_msg=f"{col} moved after {what}",
        )


def _assert_shared_pairs_agree(base, other, what, rtol=1e-12):
    """To floating point, for the batched-gene-fit case.

    `stage` is still compared exactly: it is a discrete escalation decision,
    and a one-ULP perturbation flipping it would be a real finding worth
    failing on rather than a rounding detail to absorb.
    """
    common = base.index.intersection(other.index)
    assert len(common) > 0
    for col in ("p_value", "fold_change", "z_orig"):
        np.testing.assert_allclose(
            base.loc[common, col].to_numpy(),
            other.loc[common, col].to_numpy(),
            rtol=rtol,
            err_msg=f"{col} moved materially after {what}",
        )
    np.testing.assert_array_equal(
        base.loc[common, "stage"].to_numpy(),
        other.loc[common, "stage"].to_numpy(),
        err_msg=f"stage flipped after {what}",
    )


BASE_GENES = [f"g{i}" for i in range(N_GENES)]
BASE_TARGETS = [f"t{j}" for j in range(N_TARGETS)]


@pytest.fixture(scope="module")
def baseline(world):
    return _run(world, BASE_GENES, BASE_TARGETS)


@pytest.mark.parametrize(
    "what,targets",
    [
        ("adding a target", [*BASE_TARGETS, "t8"]),
        ("adding two targets", [*BASE_TARGETS, "t8", "t9"]),
        ("removing a target", BASE_TARGETS[:-1]),
        ("reversing target order", list(reversed(BASE_TARGETS))),
    ],
)
def test_changing_the_targets_leaves_other_pairs_bitwise_identical(world, baseline, what, targets):
    """The guarantee that matters for adding pairs to an existing analysis."""
    _assert_shared_pairs_identical(baseline, _run(world, BASE_GENES, targets), what)


@pytest.mark.parametrize(
    "what,genes",
    [
        ("adding a gene", [*BASE_GENES, "g10"]),
        ("removing a gene", BASE_GENES[:-1]),
    ],
)
def test_changing_the_genes_leaves_other_pairs_agreeing_to_floating_point(
    world, baseline, what, genes
):
    """Weaker on purpose: the gene GLM is batched, so the batch width moves.

    Measured residue on OpenBLAS is one ULP (1.11e-16 on `fold_change`, 3 of
    72 pairs); on Accelerate it is exactly zero. `rtol=1e-12` is far above the
    former and far below anything that could matter, so this fails if the
    residue ever grows by four orders of magnitude.
    """
    _assert_shared_pairs_agree(baseline, _run(world, genes, BASE_TARGETS), what)


@pytest.mark.parametrize("chunk", [1, 3, 5])
def test_target_chunk_size_does_not_move_results(world, baseline, chunk):
    _assert_shared_pairs_identical(
        baseline,
        _run(world, BASE_GENES, BASE_TARGETS, target_chunk_size=chunk),
        f"target_chunk_size={chunk}",
    )


@pytest.mark.parametrize("n_jobs", [2, 4])
def test_worker_count_does_not_move_results(world, baseline, n_jobs):
    _assert_shared_pairs_identical(
        baseline, _run(world, BASE_GENES, BASE_TARGETS, n_jobs=n_jobs), f"n_jobs={n_jobs}"
    )


def test_seeding_is_keyed_on_the_name_not_the_position():
    a = target_seed_sequence(0, "target_alpha")
    b = target_seed_sequence(0, "target_beta")
    assert a.spawn_key != b.spawn_key
    # Same name, same seed, same stream -- whatever else is in the run.
    assert target_seed_sequence(0, "target_alpha").spawn_key == a.spawn_key
    assert np.array_equal(
        np.random.default_rng(a).integers(0, 1_000_000, 20),
        np.random.default_rng(target_seed_sequence(0, "target_alpha")).integers(0, 1_000_000, 20),
    )


def test_seeding_still_responds_to_the_seed():
    a = target_seed_sequence(0, "t")
    b = target_seed_sequence(1, "t")
    assert a.entropy != b.entropy
    assert not np.array_equal(
        np.random.default_rng(a).integers(0, 1_000_000, 20),
        np.random.default_rng(b).integers(0, 1_000_000, 20),
    )


def test_seeding_does_not_use_pythons_salted_string_hash():
    """`hash("x")` varies per process unless PYTHONHASHSEED is fixed.

    If the key came from it, the whole guarantee would evaporate between
    runs -- silently, and only outside a single session. Pinning a literal
    catches a change to the hashing that would reintroduce that.
    """
    key = target_seed_sequence(0, "target_alpha").spawn_key[0]
    assert key == 11_214_916_989_929_217_540, "the per-target key derivation changed"
