"""The trans background reproduces the reference run's test, and its screens are exact.

`run_trans_analysis` must give every pair it tests in full the result
`run_discovery_analysis` gives it, bit for bit: same draws, same target fits
(each target fit with its reference chunk-mates), same gene fits. Its two
screens may skip work, but never in a way any table of `run_specificity_check`
can see. See docs/design.md, "The trans background".
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

pytest.importorskip("numba")

from pysceptre import run_specificity_check  # noqa: E402
from pysceptre.crt import sampler as crt_sampler  # noqa: E402
from pysceptre.pipeline.api import run_discovery_analysis, run_trans_analysis  # noqa: E402
from pysceptre.pipeline.discovery import fit_all_genes  # noqa: E402
from pysceptre.pipeline.gene_cache import fit_gene_cache  # noqa: E402
from pysceptre.pipeline.trans import (  # noqa: E402
    iter_trans_analysis,
    trans_fit_chunks,
    trans_results_frame,
)

N_CELLS, N_GENES, N_TARGETS, SEED, CHUNK = 40_000, 24, 18, 7, 4
RESULT_COLUMNS = ["p_value", "fold_change", "se_fold_change", "z_orig", "stage"]


@pytest.fixture(scope="module")
def screen():
    """A small high-MOI screen: depth and batch covariates, planted down and up effects.

    Targets have 50 to 120 cells, so both CRT samplers are used (the exact one
    above 0.2% of the cells, 80 here). Genes and targets sit on three
    chromosomes so the end-to-end check has cis and trans pairs.
    """
    rng = np.random.default_rng(20261014)
    depth = rng.normal(0.0, 0.3, N_CELLS)
    batch = rng.integers(0, 3, N_CELLS)
    X = np.column_stack(
        [
            np.ones(N_CELLS),
            depth,
            (batch == 1).astype(float),
            (batch == 2).astype(float),
            rng.normal(size=N_CELLS),
        ]
    )
    targets = {
        f"t{j:02d}": np.sort(rng.choice(N_CELLS, int(rng.integers(50, 121)), replace=False))
        for j in range(N_TARGETS)
    }
    base = rng.uniform(0.5, 6.0, N_GENES)
    theta = rng.uniform(2.0, 20.0, N_GENES)
    effect = np.ones((N_GENES, N_CELLS))
    for g, t, factor in [
        (0, 0, 0.3),
        (1, 1, 0.4),
        (2, 2, 0.5),
        (3, 5, 1.8),
        (4, 7, 0.45),
        (6, 9, 2.2),
    ]:
        effect[g, targets[f"t{t:02d}"]] *= factor
    mu = base[:, None] * np.exp(0.8 * depth + 0.2 * (batch == 1) - 0.3 * (batch == 2))[None, :]
    mu = mu * effect
    counts = rng.negative_binomial(theta[:, None], theta[:, None] / (theta[:, None] + mu)).astype(
        float
    )
    genes = [f"g{i:02d}" for i in range(N_GENES)]
    gene_chrom = np.array(["chr1", "chr2", "chr3"])[np.arange(N_GENES) % 3]
    target_chrom = np.array(["chr1", "chr2", "chr3"])[np.arange(N_TARGETS) % 3]
    pairs = pd.DataFrame(
        [(g, t) for g in genes for t in targets], columns=["response_id", "grna_target"]
    )
    same = (
        gene_chrom[pairs["response_id"].str[1:].astype(int)]
        == target_chrom[pairs["grna_target"].str[1:].astype(int)]
    )
    return dict(
        counts=counts,
        genes=genes,
        X=X,
        targets=targets,
        pairs=pairs,
        cis=pairs.loc[same].reset_index(drop=True),
        trans=pairs.loc[~same].reset_index(drop=True),
        gene_chrom=gene_chrom,
        target_chrom=target_chrom,
    )


def _discovery(s, pairs, side="both"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_discovery_analysis(
            s["counts"],
            s["genes"],
            s["X"],
            s["targets"],
            pairs,
            side=side,
            seed=SEED,
            target_chunk_size=CHUNK,
        )


def _trans(s, pairs, reference, mode=None, side="both", **kwargs):
    if mode is not None:
        pairs = pairs.assign(mode=mode)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_trans_analysis(
            s["counts"],
            s["genes"],
            s["X"],
            s["targets"],
            pairs,
            reference_pairs=reference,
            seed=SEED,
            side=side,
            target_chunk_size=CHUNK,
            **kwargs,
        )


def _assert_same_results(a: pd.DataFrame, b: pd.DataFrame):
    """Exact equality of the result columns, NaN equal to NaN, keyed by pair."""
    key = ["response_id", "grna_target"]
    a = a.set_index(key).sort_index()
    b = b.set_index(key).sort_index()
    assert a.index.equals(b.index)
    for c in RESULT_COLUMNS:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        assert np.array_equal(x, y, equal_nan=True), c


def test_full_mode_matches_discovery_on_the_reference_pairs(screen):
    got = _trans(screen, screen["pairs"], screen["pairs"], mode="full")
    _assert_same_results(got, _discovery(screen, screen["pairs"]))
    assert (got["screen"] == "none").all()


def test_full_mode_matches_discovery_on_pairs_outside_the_reference(screen):
    # Both pair sets cover every target, so the reference layout and the one a
    # discovery run on the trans pairs builds are the same chunks.
    n = len(screen["X"])
    assert trans_fit_chunks(
        screen["targets"], screen["cis"], n, target_chunk_size=CHUNK, chunk_memory_gb=1.0
    ) == trans_fit_chunks(
        screen["targets"], screen["trans"], n, target_chunk_size=CHUNK, chunk_memory_gb=1.0
    )
    got = _trans(screen, screen["trans"], screen["cis"], mode="full")
    _assert_same_results(got, _discovery(screen, screen["trans"]))


@pytest.mark.parametrize("side", ["both", "left", "right"])
def test_screens_are_exact(screen, side):
    full = _trans(screen, screen["pairs"], screen["pairs"], mode="full", side=side)
    scr = _trans(screen, screen["pairs"], screen["pairs"], side=side)
    for c in ("fold_change", "se_fold_change", "z_orig"):
        assert np.array_equal(full[c], scr[c], equal_nan=True), c
    none = scr["screen"] == "none"
    for c in RESULT_COLUMNS:
        assert np.array_equal(full.loc[none, c], scr.loc[none, c], equal_nan=True), c
    # Every pair that escalated with a negative effect was tested in full.
    escalated_down = (full["stage"] >= 2) & (full["fold_change"] < 1)
    assert none[escalated_down].all()
    fc = scr["screen"] == "fold_change"
    assert (full.loc[fc, "fold_change"] >= 1).all()
    assert scr.loc[fc, "p_value"].isna().all()
    early = scr["screen"] == "early_stop"
    assert (full.loc[early, "stage"] == 1).all()
    assert (full.loc[early, "p_value"] >= scr.loc[early, "p_value_floor"]).all()
    assert (scr.loc[early, "p_value_floor"] >= 0.05).all()
    assert scr.loc[early, "p_value"].isna().all()
    assert early.any() and fc.any()
    # Right-sided, a negative effect has a large p-value and never escalates.
    assert escalated_down.any() or side == "right"
    assert (scr.loc[early, "n_null_evaluated"] < 499).all()


def test_audit_rows_carry_both_results(screen):
    mode = np.where(np.arange(len(screen["pairs"])) % 5 == 0, "audit", "screened")
    got = _trans(screen, screen["pairs"], screen["pairs"], mode=mode)
    full = _trans(screen, screen["pairs"], screen["pairs"], mode="full")
    scr = _trans(screen, screen["pairs"], screen["pairs"])
    audit = mode == "audit"
    for c in RESULT_COLUMNS:
        assert np.array_equal(got[c], scr[c], equal_nan=True), c
        assert np.array_equal(got.loc[audit, f"audit_{c}"], full.loc[audit, c], equal_nan=True), c
        assert got.loc[~audit, f"audit_{c}"].isna().all()


def test_shards_of_reference_chunks_match_one_run(screen):
    n = len(screen["X"])
    chunks = trans_fit_chunks(
        screen["targets"], screen["cis"], n, target_chunk_size=CHUNK, chunk_memory_gb=1.0
    )
    assert len(chunks) == 5
    whole = _trans(screen, screen["trans"], screen["cis"])
    parts = []
    for ids in ([3, 0], [4, 1, 2]):
        mine = {t for i in ids for t in chunks[i]}
        sub = screen["trans"].loc[screen["trans"]["grna_target"].isin(mine)].reset_index(drop=True)
        parts.append(_trans(screen, sub, screen["cis"], fit_chunk_ids=ids))
    _assert_same_results(pd.concat(parts, ignore_index=True), whole)


@pytest.mark.parametrize(("n_jobs", "backend"), [(3, "fork"), (3, "thread")])
def test_workers_and_backend_change_nothing(screen, n_jobs, backend):
    one = _trans(screen, screen["trans"], screen["cis"])
    many = _trans(screen, screen["trans"], screen["cis"], n_jobs=n_jobs, backend=backend)
    _assert_same_results(many, one)
    assert many["screen"].equals(one["screen"])


def test_reused_gene_fits_change_nothing(screen):
    fits = fit_all_genes(
        screen["counts"], screen["genes"], screen["X"], batch_width=1, chunk_memory_gb=1.0
    )
    fresh = _trans(screen, screen["trans"], screen["cis"])
    reused = _trans(screen, screen["trans"], screen["cis"], gene_fits=fits)
    _assert_same_results(reused, fresh)
    cache = fit_gene_cache(screen["counts"], screen["genes"], screen["X"])
    cached = _trans(screen, screen["trans"], screen["cis"], gene_fits=cache)
    _assert_same_results(cached, fresh)


def _chunk_pairs(s, ids):
    """The trans pairs whose targets lie in the reference chunks `ids`."""
    chunks = trans_fit_chunks(
        s["targets"], s["cis"], len(s["X"]), target_chunk_size=CHUNK, chunk_memory_gb=1.0
    )
    mine = {t for i in ids for t in chunks[i]}
    return s["trans"].loc[s["trans"]["grna_target"].isin(mine)].reset_index(drop=True)


def test_a_gene_fit_cache_is_checked_against_the_data(screen):
    s = screen
    cache = fit_gene_cache(s["counts"], s["genes"], s["X"])
    pairs = _chunk_pairs(s, [0])
    pairs = pairs.loc[pairs["response_id"].isin(s["genes"][:6])].reset_index(drop=True)
    used, unused = pairs["response_id"].iloc[0], s["genes"][-1]

    changed = s["counts"].copy()
    changed[s["genes"].index(used), 0] += 1
    with pytest.raises(ValueError, match=rf"counts of 1 cached gene\(s\) differ.*'{used}'"):
        _trans({**s, "counts": changed}, pairs, s["cis"], gene_fits=cache)
    # Only the genes the call tests are checked.
    changed = s["counts"].copy()
    changed[s["genes"].index(unused), 0] += 1
    alone = _trans({**s, "counts": changed}, pairs, s["cis"], gene_fits=cache)
    _assert_same_results(alone, _trans(s, pairs, s["cis"]))

    nudged = s["X"].copy()
    nudged[17, 1] = np.nextafter(nudged[17, 1], np.inf)
    with pytest.raises(ValueError, match="covariate_matrix differs"):
        _trans({**s, "X": nudged}, pairs, s["cis"], gene_fits=cache)
    assert s["X"].flags.c_contiguous
    with pytest.raises(ValueError, match="memory layout 'F'"):
        _trans({**s, "X": np.asfortranarray(s["X"])}, pairs, s["cis"], gene_fits=cache)
    fewer = fit_gene_cache(s["counts"][:, :-1], s["genes"], s["X"][:-1], genes=[used])
    with pytest.raises(ValueError, match="covariate_matrix has shape"):
        _trans(s, pairs, s["cis"], gene_fits=fewer)


def _blas_threads() -> list[int]:
    return sorted({d["num_threads"] for d in threadpool_info() if d["user_api"] == "blas"})


@pytest.mark.parametrize("first_done", ["started_last", "started_first"])
def test_interleaved_generators_keep_their_own_inputs(screen, first_done):
    """Two generators advanced in turn in one thread give each its own results, bit for bit.

    One BLAS thread holds until the last of them finishes, whichever that is,
    and the limit found before them comes back after. They run under an outer
    limit of two BLAS threads, so that a limit left behind shows.
    """
    s = screen
    a_pairs, b_pairs = _chunk_pairs(s, [0]), _chunk_pairs(s, [1, 2])

    def generator(pairs, ids):
        return iter_trans_analysis(
            s["counts"],
            s["genes"],
            s["X"],
            s["targets"],
            pairs,
            reference_pairs=s["cis"],
            seed=SEED,
            target_chunk_size=CHUNK,
            fit_chunk_ids=ids,
        )

    with threadpool_limits(limits=2, user_api="blas"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = _blas_threads()
        alone_a = trans_results_frame(a_pairs, generator(a_pairs, [0]))
        alone_b = trans_results_frame(b_pairs, generator(b_pairs, [1, 2]))
        assert _blas_threads() == before
        ga, gb = generator(a_pairs, [0]), generator(b_pairs, [1, 2])
        got_a = [next(ga)]
        got_b = [next(gb)]
        # `ga` resumes while `gb` is suspended with its inputs in place.
        got_a.append(next(ga))
        held = _blas_threads()
        if first_done == "started_last":
            got_b.extend(gb)
            between = _blas_threads()
            got_a.extend(ga)
        else:
            got_a.extend(ga)
            between = _blas_threads()
            got_b.extend(gb)
        after = _blas_threads()
    one = [1] if before else []
    assert held == one and between == one
    assert after == before
    assert len(got_a) > 2 and len(got_b) > 1
    pd.testing.assert_frame_equal(trans_results_frame(a_pairs, got_a), alone_a, check_exact=True)
    pd.testing.assert_frame_equal(trans_results_frame(b_pairs, got_b), alone_b, check_exact=True)


def test_refusals(screen, monkeypatch):
    s = screen
    with pytest.raises(ValueError, match="seed"):
        run_trans_analysis(
            s["counts"],
            s["genes"],
            s["X"],
            s["targets"],
            s["trans"],
            reference_pairs=s["cis"],
            seed=None,
        )
    with pytest.raises(ValueError, match="skew_normal"):
        _trans(s, s["trans"], s["cis"], resampling_approximation="no_approximation")
    with pytest.raises(ValueError, match="expected"):
        _trans(s, s["trans"], s["cis"], expected_fit_chunks=(6, 1746))
    with pytest.raises(ValueError, match="fit_chunk_ids"):
        _trans(s, s["trans"], s["cis"], fit_chunk_ids=[0])
    with pytest.raises(ValueError, match="mode"):
        _trans(s, s["trans"], s["cis"], mode="sometimes")
    outside = pd.DataFrame({"response_id": ["g00"], "grna_target": ["t99"]})
    with pytest.raises((ValueError, KeyError)):
        _trans(s, outside, s["cis"])
    monkeypatch.setattr(crt_sampler, "_HAVE_NUMBA", False)
    with pytest.raises(RuntimeError, match="numba"):
        _trans(s, s["trans"], s["cis"])


def test_specificity_check_sees_no_difference(screen):
    """Every table of the check is the same from a screened and a full background."""
    s = screen
    centre = {t: 1_000_000.0 * (1 + int(t[1:]) // 3) for t in s["targets"]}
    tss = {g: 1_000_000.0 * (1 + int(g[1:]) // 3) + 2_000.0 * int(g[1:]) for g in s["genes"]}
    element_positions = pd.DataFrame(
        {
            "grna_target": list(s["targets"]),
            "chrom": s["target_chrom"],
            "centre": [centre[t] for t in s["targets"]],
        }
    )
    gene_positions = pd.DataFrame(
        {"response_id": s["genes"], "chrom": s["gene_chrom"], "tss": [tss[g] for g in s["genes"]]}
    )
    cis = _discovery(s, s["cis"])
    cis["log_2_fold_change"] = np.log2(cis["fold_change"])
    results = {}
    for name, mode in (("full", "full"), ("screened", None)):
        trans = _trans(s, s["trans"], s["cis"], mode=mode)
        trans["log_2_fold_change"] = np.log2(trans["fold_change"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results[name] = run_specificity_check(
                cis,
                trans,
                element_positions,
                gene_positions,
                cutoff=0.05,
                distance_bins=(0.0, 1e4, 1e5, np.inf),
                n_bootstrap=200,
                seed=1,
            )
    full, scr = results["full"], results["screened"]
    assert full.background_rate == scr.background_rate
    assert full.background["background"].sum() > 0
    for name in (
        "by_distance",
        "by_distance_without_broad",
        "broad_effect",
        "per_gene",
        "links",
        "tss_targets",
        "gene_lookup",
        "far_links",
    ):
        pd.testing.assert_frame_equal(getattr(full, name), getattr(scr, name), check_exact=True)
    cols = [c for c in full.background.columns if c in ("grna_target", "response_id", "background")]
    pd.testing.assert_frame_equal(full.background[cols], scr.background[cols], check_exact=True)


def test_structured_full_mode_matches_structured_discovery(screen):
    s = screen
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = run_discovery_analysis(
            s["counts"],
            s["genes"],
            s["X"],
            s["targets"],
            s["pairs"],
            seed=SEED,
            target_chunk_size=CHUNK,
            glm_solver="structured",
        )
    got = _trans(s, s["pairs"], s["pairs"], mode="full", glm_solver="structured")
    _assert_same_results(got, expected)


def test_gene_fits_of_another_solver_are_refused(screen):
    from pysceptre.pipeline.gene_cache import fit_gene_cache

    s = screen
    outer = fit_gene_cache(s["counts"], s["genes"], s["X"], genes=s["genes"][:4])
    with pytest.raises(ValueError, match="glm_solver='outer', not 'structured'"):
        _trans(s, s["trans"], s["cis"], gene_fits=outer, glm_solver="structured")


def test_a_target_the_covariates_explain_is_nan_in_both(screen):
    # A covariate column equal to t03's indicator; the reference run reports t03 NaN, so must trans.
    cells = screen["targets"]["t03"]
    s = {**screen, "X": np.column_stack([screen["X"], np.isin(np.arange(N_CELLS), cells)])}
    pairs = screen["pairs"][screen["pairs"]["response_id"].isin(screen["genes"][:3])]
    pairs = pairs.reset_index(drop=True)
    reference = _discovery(s, pairs)
    spanned = reference["grna_target"] == "t03"
    assert reference.loc[spanned, RESULT_COLUMNS].isna().all().all()
    assert reference.loc[~spanned, "p_value"].notna().all()
    _assert_same_results(_trans(s, pairs, pairs, mode="full"), reference)
    for mode in ("screened", "audit"):
        got = _trans(s, pairs, pairs, mode=mode)
        rows = got["grna_target"] == "t03"
        assert got.loc[rows, ["p_value", "z_orig"]].isna().all().all()
        assert (got.loc[rows, "screen"] == "none").all()
