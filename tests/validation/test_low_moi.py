"""Low MOI: the NT-cells control group, R's MOI defaults, and their refusals.

These are internal-consistency tests on synthetic data where every cell
carries exactly one gRNA, as sceptre's low-MOI QC leaves it. Agreement with R
itself is tested in `test_low_moi_vs_r.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from pysceptre import run_calibration_check, run_discovery_analysis, run_power_check
from pysceptre.crt.permutations import draws_for_target, nested_permutation_draws
from pysceptre.glm.irls import fit_poisson_glm_batch
from pysceptre.glm.nb_theta import estimate_theta
from pysceptre.pipeline.api import nt_cell_pool, resolve_analysis_settings
from pysceptre.pipeline.discovery import (
    combined_cells,
    resolve_entropy,
    run_discovery_nt_cells,
    run_discovery_ntcells_complement,
)
from pysceptre.pipeline.pairwise_qc import nonzero_counts
from pysceptre.precompute.pieces import compute_precomputation_pieces
from pysceptre.test_statistic.fold_change import estimate_log_fold_change
from pysceptre.test_statistic.score_stat import (
    FirstStagePermutationDraws,
    PermutationSliceDraws,
    draws_to_matrix,
    prefix_matrix,
)


def _screen(
    seed: int = 1, n_cells: int = 2400, n_genes: int = 6, n_targets: int = 5, n_ntc: int = 4
):
    """One gRNA per cell: two per target plus `n_ntc` NT gRNAs, and a knockdown of g0 by t0."""
    rng = np.random.default_rng(seed)
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
    Y[0, targets["t0"]] = rng.binomial(Y[0, targets["t0"]].astype(int), 0.3)
    genes = [f"g{i}" for i in range(n_genes)]
    pairs = pd.DataFrame(
        [(g, t) for g in genes for t in targets], columns=["response_id", "grna_target"]
    )
    return sparse.csr_matrix(Y), genes, X, targets, ntc, pairs


@pytest.fixture(scope="module")
def screen():
    return _screen()


# --- R's defaults and refusals -----------------------------------------------------------


def test_moi_sets_rs_defaults():
    assert resolve_analysis_settings("high", None, None) == ("complement", "crt")
    assert resolve_analysis_settings("low", None, None) == ("nt_cells", "permutations")
    assert resolve_analysis_settings("low", "complement", "crt") == ("complement", "crt")
    assert resolve_analysis_settings("high", "complement", "permutations") == (
        "complement",
        "permutations",
    )


@pytest.mark.parametrize(
    "args",
    [("high", "nt_cells", None), ("medium", None, None), ("low", "ntc", None), ("low", None, "x")],
)
def test_invalid_settings_are_refused(args):
    with pytest.raises(ValueError):
        resolve_analysis_settings(*args)


def test_nt_cells_are_refused_in_high_moi(screen):
    resp, genes, X, targets, ntc, pairs = screen
    with pytest.raises(ValueError, match="needs moi='low'"):
        run_discovery_analysis(
            resp, genes, X, targets, pairs, control_group="nt_cells", ntc_grna_cells=ntc
        )


def test_nt_cells_need_the_nt_grnas(screen):
    resp, genes, X, targets, ntc, pairs = screen
    with pytest.raises(ValueError, match="needs ntc_grna_cells"):
        run_discovery_analysis(resp, genes, X, targets, pairs, moi="low")


def test_nt_grnas_are_refused_where_they_would_be_ignored(screen):
    resp, genes, X, targets, ntc, pairs = screen
    with pytest.raises(ValueError, match="only used with control_group='nt_cells'"):
        run_discovery_analysis(
            resp,
            genes,
            X,
            targets,
            pairs,
            moi="low",
            control_group="complement",
            ntc_grna_cells=ntc,
        )


def test_a_cell_under_two_nt_grnas_is_refused(screen):
    resp, genes, X, targets, ntc, pairs = screen
    shared = dict(ntc, nt1=np.concatenate([ntc["nt1"], ntc["nt0"][:2]]))
    with pytest.raises(ValueError, match="more than one non-targeting gRNA"):
        run_discovery_analysis(resp, genes, X, targets, pairs, moi="low", ntc_grna_cells=shared)


def test_a_target_sharing_cells_with_the_nt_cells_is_refused(screen):
    resp, genes, X, targets, ntc, pairs = screen
    shared = dict(ntc, nt0=np.concatenate([ntc["nt0"], targets["t2"][:3]]))
    with pytest.raises(ValueError, match="share cells with the NT cells, e.g. 't2'"):
        run_discovery_analysis(resp, genes, X, targets, pairs, moi="low", ntc_grna_cells=shared)


def test_a_covariate_constant_on_a_targets_cells_and_the_nt_cells_is_refused(screen):
    """R stops on an NA coefficient here; a column that is zero on t1's cells and
    on every NT cell is exactly that, though the full design is full rank."""
    resp, genes, X, targets, ntc, pairs = screen
    pool = nt_cell_pool(ntc, X.shape[0])
    extra = np.ones(X.shape[0])
    extra[targets["t1"]] = 0.0
    extra[pool] = 0.0
    X2 = np.column_stack([X, extra])
    with pytest.raises(ValueError, match="cannot be estimated on the cells of 1 target"):
        run_discovery_analysis(resp, genes, X2, targets, pairs, moi="low", ntc_grna_cells=ntc)


# --- the low-MOI complement path is the high-MOI engine ------------------------------------


@pytest.mark.parametrize("mechanism", ["crt", "permutations"])
def test_low_moi_complement_is_the_high_moi_engine(screen, mechanism):
    resp, genes, X, targets, ntc, pairs = screen
    low = run_discovery_analysis(
        resp,
        genes,
        X,
        targets,
        pairs,
        moi="low",
        control_group="complement",
        resampling_mechanism=mechanism,
        seed=3,
    )
    high = run_discovery_analysis(
        resp, genes, X, targets, pairs, resampling_mechanism=mechanism, seed=3
    )
    pd.testing.assert_frame_equal(low, high, check_exact=True)


# --- the NT-cells path ------------------------------------------------------------------


@pytest.mark.parametrize("mechanism", ["crt", "permutations"])
def test_nt_cells_results_do_not_depend_on_workers_or_chunking(screen, mechanism):
    resp, genes, X, targets, ntc, pairs = screen
    common = dict(moi="low", ntc_grna_cells=ntc, resampling_mechanism=mechanism, seed=7)
    a = run_discovery_analysis(resp, genes, X, targets, pairs, **common)
    b = run_discovery_analysis(
        resp, genes, X, targets, pairs, **common, n_jobs=3, target_chunk_size=2
    )
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_nt_cells_crt_results_do_not_depend_on_the_other_pairs(screen):
    """Each pair's fit uses its own cells and each target draws from its own
    stream, so a subset analysis reproduces the full one exactly."""
    resp, genes, X, targets, ntc, pairs = screen
    common = dict(moi="low", ntc_grna_cells=ntc, resampling_mechanism="crt", seed=7)
    full = run_discovery_analysis(resp, genes, X, targets, pairs, **common)
    subset = pairs[pairs["grna_target"].isin(["t3", "t1"]) & pairs["response_id"].ne("g2")]
    part = run_discovery_analysis(
        resp, genes, X, {t: targets[t] for t in ("t1", "t3")}, subset, **common
    )
    merged = part.merge(full, on=["response_id", "grna_target"], suffixes=("", "_full"))
    assert len(merged) == len(subset)
    for col in ("p_value", "fold_change", "z_orig"):
        np.testing.assert_array_equal(merged[col], merged[f"{col}_full"])


def test_each_pair_is_fit_on_its_target_and_the_nt_cells(screen):
    """The fold change and statistic come from a fit on R's `c(trt_idxs, all_nt_idxs)`."""
    resp, genes, X, targets, ntc, pairs = screen
    result = run_discovery_analysis(
        resp, genes, X, targets, pairs, moi="low", ntc_grna_cells=ntc, seed=1
    )
    pool = nt_cell_pool(ntc, X.shape[0])
    Y = resp.toarray()
    for gene, target in [("g0", "t0"), ("g3", "t4")]:
        cells = combined_cells(targets[target], pool)
        y, Xc = Y[int(gene[1:])][cells], X[cells]
        fit = fit_poisson_glm_batch(Xc, y)
        theta = min(max(estimate_theta(y=y, mu=fit.fitted_values, dfr=y.size - 3)[0], 0.01), 1000)
        pieces = compute_precomputation_pieces(y, Xc, fit.coefs, theta)
        trt = np.arange(targets[target].size)
        fc, _ = estimate_log_fold_change(y, pieces.mu, trt)
        z = pieces.a[trt].sum() / np.sqrt(
            pieces.w[trt].sum() - np.sum(pieces.D[:, trt].sum(axis=1) ** 2)
        )
        row = result.query("response_id == @gene and grna_target == @target").iloc[0]
        assert row["fold_change"] == pytest.approx(fc, rel=1e-12)
        assert row["z_orig"] == pytest.approx(z, rel=1e-10)


def test_the_knockdown_is_found_against_the_nt_cells(screen):
    resp, genes, X, targets, ntc, pairs = screen
    for mechanism in ("crt", "permutations"):
        r = run_discovery_analysis(
            resp,
            genes,
            X,
            targets,
            pairs,
            moi="low",
            ntc_grna_cells=ntc,
            resampling_mechanism=mechanism,
            seed=2,
        )
        hit = r.query("response_id == 'g0' and grna_target == 't0'").iloc[0]
        assert hit["p_value"] < 1e-6
        assert hit["fold_change"] < 0.5


def test_injected_permutations_replace_the_samplers_draws(screen):
    """The hook the R comparison relies on. Injecting exactly what the engine
    would have drawn reproduces its result, so injection changes nothing else."""
    resp, genes, X, targets, ntc, pairs = screen
    pool = nt_cell_pool(ntc, X.shape[0])
    sizes = [len(c) for c in targets.values()]
    B = 499 + 4999 + 24999
    common = dict(
        B1=499, B2=4999, B3=24999, resampling_mechanism="permutations", seed=0, nt_cells=pool
    )
    own = run_discovery_nt_cells(resp, genes, X, targets, pairs=pairs, **common)
    perms = nested_permutation_draws(
        pool.size, min(sizes), max(sizes), B, np.random.default_rng(resolve_entropy(0))
    )
    injected = run_discovery_nt_cells(
        resp, genes, X, targets, pairs=pairs, permutations=perms, **common
    )
    pd.testing.assert_frame_equal(own, injected, check_exact=True)
    with pytest.raises(ValueError, match="shape"):
        run_discovery_nt_cells(
            resp, genes, X, targets, pairs=pairs, permutations=perms[:10], **common
        )
    with pytest.raises(ValueError, match="wide"):
        run_discovery_nt_cells(
            resp, genes, X, targets, pairs=pairs, permutations=perms[:, :5], **common
        )


def test_the_first_stage_cache_changes_nothing():
    rng = np.random.default_rng(4)
    perms = np.stack([rng.permutation(52)[:20] for _ in range(30)])
    plain = PermutationSliceDraws(perms, 12, 52)
    cached = FirstStagePermutationDraws(perms, 12, 52)
    for lo, hi in [(0, 10), (0, 10), (10, 30)]:
        assert (plain.slice(lo, hi) != cached.slice(lo, hi)).nnz == 0
    assert cached.slice(0, 10) is cached.slice(0, 10)


# --- pairwise QC against the NT cells -----------------------------------------------------


def test_the_control_count_is_the_genes_nonzero_nt_cells(screen):
    resp, genes, X, targets, ntc, pairs = screen
    pool = nt_cell_pool(ntc, X.shape[0])
    groups = [targets["t0"], targets["t1"]]
    trt, cntrl = nonzero_counts(resp, groups, X.shape[0], control_cells=pool)
    trt_c, _ = nonzero_counts(resp, groups, X.shape[0])
    Y = resp.toarray() > 0
    np.testing.assert_array_equal(trt, trt_c)
    expected = Y[:, pool].sum(axis=1)
    np.testing.assert_array_equal(cntrl, np.repeat(expected[:, None], 2, axis=1))


def test_power_check_counts_the_nt_cells_and_reports_failures(screen):
    resp, genes, X, targets, ntc, pairs = screen
    # "tiny" is three of t4's cells: too few to pass the default trt threshold of 7.
    with_tiny = dict(targets, tiny=targets["t4"][:3])
    pc = pd.DataFrame({"response_id": ["g0", "g1"], "grna_target": ["t0", "tiny"]})
    pool = nt_cell_pool(ntc, X.shape[0])
    Y = resp.toarray() > 0
    with pytest.raises(ValueError, match="no positive control pair passes"):
        run_power_check(
            resp,
            genes,
            X,
            with_tiny,
            positive_control_pairs=pc,
            moi="low",
            ntc_grna_cells=ntc,
            n_nonzero_cntrl_thresh=10**9,
            seed=1,
        )
    out = run_power_check(
        resp, genes, X, with_tiny, positive_control_pairs=pc, moi="low", ntc_grna_cells=ntc, seed=1
    )
    assert out["n_nonzero_cntrl"].tolist() == [int(Y[0, pool].sum()), int(Y[1, pool].sum())]
    assert out["pass_qc"].tolist() == [True, False]
    assert np.isnan(out.loc[1, "p_value"]) and out.loc[0, "p_value"] < 1e-6


# --- the calibration check on the NT cells alone ------------------------------------------


def test_nt_cells_calibration_is_the_complement_check_on_the_nt_cells(screen):
    """R restricts the whole universe to the NT cells; doing that by hand and
    running the complement check gives the same result, bit for bit."""
    resp, genes, X, targets, ntc, pairs = screen
    common = dict(n_calibration_pairs=12, calibration_group_size=2, seed=5)
    for mechanism in ("crt", "permutations"):
        nt = run_calibration_check(
            resp, genes, X, ntc, moi="low", resampling_mechanism=mechanism, **common
        )
        pool = nt_cell_pool(ntc, X.shape[0])
        pos = np.full(X.shape[0], -1)
        pos[pool] = np.arange(pool.size)
        by_hand = run_calibration_check(
            resp[:, pool],
            genes,
            X[pool],
            {g: pos[c] for g, c in ntc.items()},
            moi="low",
            control_group="complement",
            resampling_mechanism=mechanism,
            **common,
        )
        pd.testing.assert_frame_equal(nt, by_hand, check_exact=True)


def test_nt_cells_calibration_needs_two_nt_grnas(screen):
    resp, genes, X, targets, ntc, pairs = screen
    with pytest.raises(ValueError, match="at least two non-targeting gRNAs"):
        run_calibration_check(
            resp,
            genes,
            X,
            {"nt0": ntc["nt0"]},
            n_calibration_pairs=5,
            calibration_group_size=1,
            moi="low",
        )


def test_calibration_permutations_do_not_depend_on_which_groups_were_drawn(screen):
    """R sizes the shared draws by the largest NT gRNAs, not by the groups that
    happened to be sampled, so adding groups leaves the others untouched."""
    resp, genes, X, targets, ntc, pairs = screen
    small = pd.DataFrame({"response_id": ["g1", "g2"], "grna_target": ["nt0&nt1", "nt0&nt1"]})
    large = pd.concat(
        [small, pd.DataFrame({"response_id": ["g1"], "grna_target": ["nt2&nt3"]})],
        ignore_index=True,
    )
    common = dict(
        n_calibration_pairs=3,
        calibration_group_size=2,
        moi="low",
        control_group="complement",
        resampling_mechanism="permutations",
        seed=9,
    )
    a = run_calibration_check(resp, genes, X, ntc, negative_control_pairs=small, **common)
    b = run_calibration_check(resp, genes, X, ntc, negative_control_pairs=large, **common)
    pd.testing.assert_frame_equal(a, b.head(2), check_exact=True)


def test_nt_cells_calibration_reads_a_backed_matrix(screen):
    """A backed reader serves rows; the NT-cells check restricts their columns on read."""
    resp, genes, X, targets, ntc, pairs = screen

    class Backed:
        def __init__(self, m):
            self._m = sparse.csr_matrix(m)
            self.shape = self._m.shape

        def rows(self, start, stop):
            return self._m[start:stop]

        def __getitem__(self, i):
            return self._m[int(i)]

    common = dict(n_calibration_pairs=8, calibration_group_size=2, moi="low", seed=2)
    a = run_calibration_check(resp, genes, X, ntc, **common)
    b = run_calibration_check(Backed(resp), genes, X, ntc, **common)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_the_complement_engine_honours_an_explicit_permutation_width(screen):
    resp, genes, X, targets, ntc, pairs = screen
    widest = max(len(c) for c in targets.values())
    with pytest.raises(ValueError, match="narrower than the largest target"):
        run_discovery_ntcells_complement(
            resp,
            genes,
            X,
            targets,
            pairs.head(3),
            resampling_mechanism="permutations",
            permutation_width=widest - 1,
        )
    with pytest.raises(ValueError, match="apply only to permutations"):
        run_discovery_ntcells_complement(
            resp, genes, X, targets, pairs.head(3), permutation_width=widest
        )


def test_the_prefix_matrix_is_the_row_list_matrix_exactly():
    """`prefix_matrix` replaced building a list of row prefixes; same entries, same order."""
    rng = np.random.default_rng(5)
    perms = np.stack([rng.permutation(90)[:40] for _ in range(60)])
    for lo, hi, k in [(0, 60, 40), (7, 33, 13), (0, 1, 1), (59, 60, 25)]:
        new = prefix_matrix(perms, lo, hi, k, 90)
        old = draws_to_matrix(draws_for_target(perms[lo:hi], k), 90)
        np.testing.assert_array_equal(new.indptr, old.indptr)
        np.testing.assert_array_equal(new.indices, old.indices)
        np.testing.assert_array_equal(new.data, old.data)


def test_nt_cells_calibration_under_the_crt_is_calibrated():
    """The calibration check against the NT cells runs on a universe where a
    synthetic target is a third of the cells. The fast sampler's repeats made it
    conservative there: 4% of null p-values below 0.1 instead of 10%."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from measure_nt_cells_crt_sampler import null_screen

    resp, genes, X, targets, ntc, pairs = null_screen()
    calibration = run_calibration_check(
        resp,
        genes,
        X,
        ntc,
        n_calibration_pairs=480,
        calibration_group_size=2,
        moi="low",
        resampling_mechanism="crt",
        seed=1,
    )
    p = calibration["p_value"].to_numpy()
    assert 0.07 < np.mean(p < 0.1) < 0.13
    assert p.mean() < 0.53


@pytest.mark.parametrize(("moi", "mechanism"), [("high", "crt"), ("low", "permutations")])
def test_a_guide_with_no_cells_is_untested_not_significant(screen, moi, mechanism):
    """Under singleton a target's guide can be left without cells after QC. It used
    to get p = 2 / (B + 1), the floor, because a NaN statistic exceeds nothing;
    bonferroni then counted it and reported the target as a discovery."""
    resp, genes, X, targets, ntc, pairs = screen
    guides = {
        "t0_a": targets["t0"][::2],
        "t0_b": targets["t0"][1::2],
        "t0_empty": np.array([], int),
    }
    design = pd.DataFrame({"grna_id": list(guides), "grna_target": ["t0"] * 3})
    common = dict(
        grna_target_data_frame=design,
        moi=moi,
        resampling_mechanism=mechanism,
        ntc_grna_cells=ntc if moi == "low" else None,
        seed=4,
    )
    pc = pd.DataFrame({"response_id": ["g1", "g2"], "grna_target": ["t0", "t0"]})
    single = run_discovery_analysis(
        resp, genes, X, guides, pc, grna_integration_strategy="singleton", **common
    )
    empty = single[single["grna_id"] == "t0_empty"]
    assert empty["p_value"].isna().all() and len(empty) == 2
    tested = single[single["grna_id"] != "t0_empty"]
    assert tested["p_value"].notna().all()
    bonf = run_discovery_analysis(
        resp, genes, X, guides, pc, grna_integration_strategy="bonferroni", **common
    )
    for gene in ("g1", "g2"):
        expected = min(2 * tested.loc[tested["response_id"] == gene, "p_value"].min(), 1.0)
        got = bonf.loc[bonf["response_id"] == gene, "p_value"].item()
        assert got == pytest.approx(expected, rel=1e-12)


# --- inputs the reviewers found mishandled ---------------------------------------------


@pytest.mark.parametrize("mechanism", ["crt", "permutations"])
def test_a_batch_without_nt_cells_makes_its_targets_untested(screen, mechanism):
    """A covariate that is zero on every NT cell and one on a target's cells
    predicts treatment exactly there: the statistic is zero over zero. Those
    pairs used to come back p = 1e-250, or abort the run under the CRT."""
    resp, genes, X, targets, ntc, pairs = screen
    batch = np.zeros(X.shape[0])
    batch[targets["t3"]] = 1.0
    for t in ("t0", "t1", "t2", "t4"):  # mixed, so these stay testable
        batch[targets[t][::2]] = 1.0
    with pytest.warns(UserWarning, match="1 target"):
        r = run_discovery_analysis(
            resp,
            genes,
            np.column_stack([X, batch]),
            targets,
            pairs,
            moi="low",
            ntc_grna_cells=ntc,
            resampling_mechanism=mechanism,
            seed=1,
        )
    assert r.loc[r["grna_target"] == "t3", "p_value"].isna().all()
    assert r.loc[r["grna_target"] != "t3", "p_value"].notna().all()


def test_a_misaligned_response_matrix_is_refused_by_the_nt_cells_calibration(screen):
    resp, genes, X, targets, ntc, pairs = screen
    extra = sparse.hstack([sparse.csr_matrix((resp.shape[0], 5)), resp]).tocsr()
    with pytest.raises(ValueError, match="rows but the response matrix has"):
        run_calibration_check(
            extra, genes, X, ntc, n_calibration_pairs=5, calibration_group_size=2, moi="low"
        )


def test_an_unused_category_is_not_a_tested_target(screen):
    resp, genes, X, targets, ntc, pairs = screen
    cat = pairs[pairs["grna_target"].isin(["t0", "t1"])].copy()
    cat["grna_target"] = pd.Categorical(cat["grna_target"], categories=["t0", "t1", "t2", "t3"])
    shared = dict(targets, t3=np.concatenate([targets["t3"], ntc["nt0"][:2]]))
    plain = cat.astype({"grna_target": str})
    a = run_discovery_analysis(resp, genes, X, shared, cat, moi="low", ntc_grna_cells=ntc, seed=1)
    b = run_discovery_analysis(resp, genes, X, shared, plain, moi="low", ntc_grna_cells=ntc, seed=1)
    pd.testing.assert_frame_equal(
        a.astype({"grna_target": str}), b, check_exact=True, check_categorical=False
    )


@pytest.mark.parametrize("moi", ["high", "low"])
def test_pairs_naming_unknown_targets_or_genes_are_refused(screen, moi):
    resp, genes, X, targets, ntc, pairs = screen
    common = dict(moi=moi, ntc_grna_cells=ntc if moi == "low" else None)
    bad_target = pd.DataFrame({"response_id": ["g0"], "grna_target": ["nope"]})
    with pytest.raises(ValueError, match="not in grna_target_cells"):
        run_discovery_analysis(resp, genes, X, targets, bad_target, **common)
    bad_gene = pd.DataFrame({"response_id": ["gX"], "grna_target": ["t0"]})
    with pytest.raises(ValueError, match="not in gene_ids"):
        run_discovery_analysis(resp, genes, X, targets, bad_gene, **common)


def test_no_pairs_gives_the_documented_columns(screen):
    resp, genes, X, targets, ntc, pairs = screen
    for moi in ("high", "low"):
        r = run_discovery_analysis(
            resp,
            genes,
            X,
            targets,
            pairs.head(0),
            moi=moi,
            ntc_grna_cells=ntc if moi == "low" else None,
        )
        assert r.empty and "p_value" in r.columns and "stage" in r.columns


def test_supplied_nt_calibration_groups_are_checked(screen):
    resp, genes, X, targets, ntc, pairs = screen
    common = dict(n_calibration_pairs=1, calibration_group_size=2, moi="low", seed=1)
    every = "&".join(ntc)
    with pytest.raises(ValueError, match="hold every NT cell"):
        run_calibration_check(
            resp,
            genes,
            X,
            ntc,
            negative_control_pairs=pd.DataFrame({"response_id": ["g1"], "grna_target": [every]}),
            **common,
        )
    # An NT gRNA with no cells may still be named; it contributes nothing.
    with_empty = dict(ntc, ntE=np.array([], dtype=int))
    named = pd.DataFrame({"response_id": ["g1"], "grna_target": ["nt0&ntE"]})
    out = run_calibration_check(resp, genes, X, with_empty, negative_control_pairs=named, **common)
    alone = run_calibration_check(
        resp,
        genes,
        X,
        ntc,
        negative_control_pairs=pd.DataFrame({"response_id": ["g1"], "grna_target": ["nt0"]}),
        **common,
    )
    np.testing.assert_allclose(out["z_orig"], alone["z_orig"], rtol=1e-12)


def test_a_backed_reader_without_toarray_works_on_every_path(screen):
    resp, genes, X, targets, ntc, pairs = screen

    class Rows:
        def __init__(self, m):
            self._m = sparse.csr_matrix(m)
            self.shape = self._m.shape

        def rows(self, start, stop):
            return self._m[start:stop]

        def __getitem__(self, i):
            return self._m[int(i)]

    common = dict(moi="low", ntc_grna_cells=ntc, seed=2)
    a = run_discovery_analysis(resp, genes, X, targets, pairs.head(10), **common)
    b = run_discovery_analysis(Rows(resp), genes, X, targets, pairs.head(10), **common)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
