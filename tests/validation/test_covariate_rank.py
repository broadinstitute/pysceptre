"""The design-matrix check at the API boundary.

`covariate_matrix` is built by the caller -- there is no formula DSL to catch
an aliased contrast -- so a rank-deficient design is a likely mistake. Left
unchecked it surfaces as a bare `numpy.linalg.LinAlgError: Singular matrix`
from inside the batched weighted least squares, with no mention of
covariates. These pin the error instead.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pysceptre import run_discovery_analysis
from pysceptre.pipeline.api import _validate_covariate_matrix

N_CELLS = 600


@pytest.fixture
def screen():
    rng = np.random.default_rng(2)
    counts = rng.poisson(10.0, size=(2, N_CELLS)).astype(float)
    gene_ids = ["g0", "g1"]
    targets = {"t0": rng.choice(N_CELLS, size=80, replace=False)}
    pairs = pd.DataFrame({"response_id": gene_ids, "grna_target": ["t0", "t0"]})
    return counts, gene_ids, targets, pairs, rng


def _run(screen, X):
    counts, gene_ids, targets, pairs, _ = screen
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_discovery_analysis(counts, gene_ids, X, targets, pairs, seed=0)


def test_an_interaction_design_is_accepted(screen):
    """The case this check must not break: a hand-built interaction term."""
    _, _, _, _, rng = screen
    time = rng.normal(size=N_CELLS)
    batch = rng.integers(0, 2, size=N_CELLS).astype(float)
    X = np.column_stack([np.ones(N_CELLS), time, batch, time * batch])
    result = _run(screen, X)
    assert np.isfinite(result["p_value"]).all()


@pytest.mark.parametrize(
    "label",
    ["scaled_duplicate", "aliased_dummies", "exact_duplicate", "constant_within_levels"],
)
def test_rank_deficient_designs_are_refused_by_name(screen, label):
    """Each is a different way to build the same mistake, and all four are realistic."""
    _, _, _, _, rng = screen
    time = rng.normal(size=N_CELLS)
    batch = rng.integers(0, 2, size=N_CELLS).astype(float)
    ones = np.ones(N_CELLS)
    X = {
        # A covariate included twice on different scales, e.g. raw and doubled.
        "scaled_duplicate": np.column_stack([ones, time, 2.0 * time]),
        # A factor's full dummy set alongside an intercept: the classic aliased contrast.
        "aliased_dummies": np.column_stack([ones, batch, 1.0 - batch]),
        "exact_duplicate": np.column_stack([ones, time, time]),
        # A column that is a multiple of the intercept, i.e. constant.
        "constant_within_levels": np.column_stack([ones, time, np.full(N_CELLS, 3.0)]),
    }[label]
    with pytest.raises(ValueError, match="rank deficient"):
        _run(screen, X)


def test_the_error_names_the_later_column_of_a_collinear_pair(screen):
    """Left to right, so the intercept survives and the dependent column is the one named.

    R's `lm` drops aliased terms in the same direction. Naming column 0 would
    be technically true and useless, since it is almost always the intercept.
    """
    _, _, _, _, rng = screen
    time = rng.normal(size=N_CELLS)
    X = np.column_stack([np.ones(N_CELLS), time, 2.0 * time])
    with pytest.raises(ValueError) as excinfo:
        _run(screen, X)
    message = str(excinfo.value)
    assert "[2]" in message, message
    assert "rank 2 of 3" in message, message


def test_a_transposed_matrix_says_so_rather_than_complaining_about_rank(screen):
    """(p, n_cells) instead of (n_cells, p) is the common shape slip."""
    with pytest.raises(ValueError, match="rows but the response matrix has"):
        _run(screen, np.ones((3, N_CELLS)))


def test_a_row_count_mismatch_is_caught_before_any_fitting(screen):
    with pytest.raises(ValueError, match="rows but the response matrix has"):
        _run(screen, np.ones((N_CELLS - 1, 2)))


def test_non_finite_entries_name_their_column(screen):
    _, _, _, _, rng = screen
    time = rng.normal(size=N_CELLS)
    time[7] = np.nan
    with pytest.raises(ValueError, match=r"non-finite values in column\(s\) \[1\]"):
        _run(screen, np.column_stack([np.ones(N_CELLS), time]))


@pytest.mark.parametrize(
    ("X", "match"),
    [
        (np.ones(10), "must be 2-D"),
        (np.ones((0, 3)), "empty"),
        (np.ones((10, 0)), "empty"),
        (np.zeros((10, 2)), "all zeros"),
    ],
)
def test_degenerate_shapes_are_refused(X, match):
    """Checked directly, since some of these cannot reach the public entry point."""
    with pytest.raises(ValueError, match=match):
        _validate_covariate_matrix(X)


def test_a_realistically_scaled_full_rank_design_is_a_no_op():
    """It must not reject anything usable. sceptre's own covariates are all logs.

    day0's real design is 567,690 x 11 with a singular-value ratio of 0.00144,
    four orders clear of the tolerance, so the guard is nowhere near biting on
    a genuine screen.
    """
    rng = np.random.default_rng(5)
    X = np.column_stack(
        [
            np.ones(400),
            rng.normal(size=400),
            rng.normal(size=400) * 1e3,
            rng.normal(size=400) * 1e-2,
        ]
    )
    assert _validate_covariate_matrix(X) is None


def test_the_rank_it_computes_is_numpys():
    """The property worth pinning, over designs on both sides of the line.

    The check used to take rank from the Gram matrix's eigenvalues, which are
    the *squares* of the singular values and so square the condition number.
    That rejected full-rank designs. Agreement with `matrix_rank` is the
    guarantee that replaced it.
    """
    rng = np.random.default_rng(9)
    for _ in range(40):
        n = int(rng.integers(50, 200))
        p = int(rng.integers(2, 8))
        X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
        # Half the time, alias a column onto an earlier one.
        if rng.random() < 0.5 and p > 2:
            j, k = rng.choice(p, size=2, replace=False)
            X[:, j] = X[:, k] * float(rng.choice([1.0, -2.5, 1e4]))
        expected_full = np.linalg.matrix_rank(X) == p
        try:
            _validate_covariate_matrix(X)
            got_full = True
        except ValueError:
            got_full = False
        assert got_full == expected_full, (n, p, np.linalg.matrix_rank(X))


def test_an_extreme_scale_spread_is_deficient_and_that_is_not_a_bug():
    """Pinned because it looks like a false positive and is not.

    Columns spanning fourteen orders of magnitude have a singular-value ratio
    near 1e-14, which is below the relative tolerance, so `matrix_rank` calls
    them deficient too. Loosening the tolerance to admit them would admit
    genuinely singular designs, and the GLM fit would then fail where this
    check used to speak.
    """
    rng = np.random.default_rng(5)
    X = np.column_stack([np.ones(400), rng.normal(size=400) * 1e-7, rng.normal(size=400) * 1e7])
    assert np.linalg.matrix_rank(X) == 2, "numpy agrees this is deficient"
    with pytest.raises(ValueError, match="rank deficient"):
        _validate_covariate_matrix(X)
