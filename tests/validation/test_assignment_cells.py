"""The helpers that turn an assignment matrix into the discovery engine's cell sets.

Plain unit tests on hand-built matrices: no R reference is involved.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from pysceptre.assignment import cells_by_grna, cells_by_target

ASSIGNED = np.array(
    [
        [1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0],
    ],
    dtype=bool,
)
IDS = ["g1", "g2", "g3", "nt1", "nt2"]


@pytest.mark.parametrize("as_sparse", [True, False])
def test_cells_by_grna(as_sparse):
    m = sparse.csr_matrix(ASSIGNED) if as_sparse else ASSIGNED
    out = cells_by_grna(m, IDS)
    assert list(out) == IDS
    expected = {"g1": [0, 2], "g2": [], "g3": [1, 2, 4], "nt1": [0, 1], "nt2": [3]}
    for g, cells in expected.items():
        assert out[g].dtype == np.int64
        np.testing.assert_array_equal(out[g], cells)


def test_cells_by_grna_ignores_stored_false():
    m = sparse.csr_matrix(ASSIGNED)
    m.data[0] = False
    out = cells_by_grna(m, IDS)
    np.testing.assert_array_equal(out["g1"], [2])


def test_cells_by_grna_checks_ids():
    with pytest.raises(ValueError, match="entries"):
        cells_by_grna(ASSIGNED, IDS[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        cells_by_grna(ASSIGNED, ["g1", "g1", "g3", "nt1", "nt2"])


def test_cells_by_target():
    grna_cells = cells_by_grna(ASSIGNED, IDS)
    design = pd.DataFrame(
        {
            "grna_id": ["g3", "g1", "nt1", "g2", "nt2", "g_absent"],
            "grna_target": ["A", "A", "non-targeting", "B", "non-targeting", "C"],
        }
    )
    targets, ntc = cells_by_target(grna_cells, design)
    assert list(targets) == ["A", "B", "C"]
    np.testing.assert_array_equal(targets["A"], [0, 1, 2, 4])
    np.testing.assert_array_equal(targets["B"], [])
    np.testing.assert_array_equal(targets["C"], [])
    assert list(ntc) == ["nt1", "nt2"]
    np.testing.assert_array_equal(ntc["nt1"], [0, 1])
    np.testing.assert_array_equal(ntc["nt2"], [3])


def test_cells_by_target_needs_its_columns():
    with pytest.raises(KeyError, match="grna_target"):
        cells_by_target({}, pd.DataFrame({"grna_id": ["g1"]}))
