"""The export's gRNA-unit contract, checked without R.

`scripts/` is how a sceptre object becomes a dataset, and the one part of it
that is easy to break silently is the `grna` assay's `var`: three kinds of unit
share one matrix, and every reader picks the kind it wants out of it. A reader
that excluded kinds rather than selecting them, or a writer that collapsed the
many-to-many gRNA -> target map, would produce a file that loads cleanly and is
wrong.

The R half of the export needs a real sceptre object and is exercised by
running it; this half needs neither, so it runs in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mudata", reason="the io extra is not installed")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from sceptre_io import SceptreExport, load_export, write_h5mu  # noqa: E402

N_CELLS = 40
N_GENES = 3


def _export(
    *,
    targeting: bool = True,
    design: pd.DataFrame | None = None,
    in_use: np.ndarray | None = None,
) -> SceptreExport:
    """A tiny export with all three unit kinds.

    `guide_shared` deliberately belongs to both targets, which is the case a
    real screen produces whenever two candidate elements overlap -- 1,673 of
    43,736 guides on day0.
    """
    rng = np.random.default_rng(0)
    from scipy import sparse

    counts = sparse.csr_matrix(rng.poisson(2.0, size=(N_GENES, N_CELLS)).astype(float))
    target_cells = {
        "elem_A": np.array([0, 1, 2, 3, 4, 5]),
        "elem_B": np.array([4, 5, 6, 7]),
    }
    targeting_cells = {
        "guide_A1": np.array([0, 1, 2]),
        "guide_A2": np.array([3]),
        "guide_shared": np.array([4, 5]),
        "guide_B1": np.array([6, 7]),
    }
    if design is None:
        design = pd.DataFrame(
            {
                "grna_id": ["guide_A1", "guide_A2", "guide_shared", "guide_shared", "guide_B1"],
                "grna_target": ["elem_A", "elem_A", "elem_A", "elem_B", "elem_B"],
            }
        )
    pairs = pd.DataFrame({"response_id": ["gene0", "gene1"], "grna_target": ["elem_A", "elem_B"]})
    return SceptreExport(
        response_matrix=counts,
        gene_ids=[f"gene{i}" for i in range(N_GENES)],
        covariate_matrix=np.column_stack([np.ones(N_CELLS), rng.normal(size=N_CELLS)]),
        grna_target_cells=target_cells,
        pairs=pairs,
        metadata={
            "n_cells": N_CELLS,
            "n_genes": N_GENES,
            "n_covariates": 2,
            "covariate_names": ["(Intercept)", "x"],
            "n_targets": len(target_cells),
            "n_pairs": len(pairs),
            "n_nonzero": counts.nnz,
            "side_code": 0,
            "run_permutations": False,
            "B1": 499,
            "B2": 4999,
            "B3": 0,
            "sceptre_version": "0.99.0",
        },
        ntc_grna_cells={"ntc_1": np.array([10, 11]), "ntc_2": np.array([12])},
        targeting_grna_cells=targeting_cells if targeting else None,
        in_use=np.ones(N_CELLS, dtype=bool) if in_use is None else in_use,
        grna_target_data_frame=design if targeting else None,
        discovery_pairs_with_info=pd.DataFrame(
            {
                "response_id": ["gene0", "gene1", "gene2"],
                "grna_target": ["elem_A", "elem_B", "elem_A"],
                "n_nonzero_trt": [9, 8, 2],
                "n_nonzero_cntrl": [30, 31, 30],
                "pass_qc": [True, True, False],
            }
        ),
    )


@pytest.fixture
def written(tmp_path):
    return load_export(write_h5mu(_export(), tmp_path / "dataset.h5mu"))


def test_each_kind_is_selected_not_inferred_by_exclusion(written):
    """The three kinds stay separate, and adding one does not enlarge the others.

    `grna_target_cells` is what an analysis runs on. If it were built by
    excluding the NTCs rather than selecting the targets, the individual
    targeting guides would land in it and every target's cell set would be
    wrong.
    """
    assert set(written.grna_target_cells) == {"elem_A", "elem_B"}
    assert set(written.ntc_grna_cells) == {"ntc_1", "ntc_2"}
    assert set(written.targeting_grna_cells) == {
        "guide_A1",
        "guide_A2",
        "guide_shared",
        "guide_B1",
    }


def test_cells_survive_the_round_trip(written):
    original = _export()
    for attr in ("grna_target_cells", "ntc_grna_cells", "targeting_grna_cells"):
        before, after = getattr(original, attr), getattr(written, attr)
        assert set(before) == set(after)
        for unit in before:
            assert np.array_equal(np.sort(before[unit]), np.sort(after[unit]))


def test_a_targets_cells_are_the_union_of_its_guides_cells(written):
    """The property a power simulation depends on.

    It gives each guide its own effect size and then tests the target, so a
    guide missing from a target -- which is what collapsing the many-to-many
    map does -- would simulate a perturbation the test never sees.
    """
    design = written.grna_target_data_frame
    for target, cells in written.grna_target_cells.items():
        guides = design.loc[design["grna_target"] == target, "grna_id"]
        union = np.unique(np.concatenate([written.targeting_grna_cells[g] for g in guides]))
        assert np.array_equal(union, np.sort(cells)), target


def test_a_multi_target_guide_is_labelled_rather_than_assigned_one_target(tmp_path):
    """`var` has one row per unit and so cannot hold two targets for one guide.

    Picking one silently would look right and read wrong, so the writer says
    `<multiple>` and leaves the real answer to `grna_target_data_frame`.
    """
    import mudata

    path = write_h5mu(_export(), tmp_path / "dataset.h5mu")
    var = mudata.read_h5mu(path)["grna"].var
    assert var.loc["guide_shared", "grna_target"] == "<multiple>"
    assert var.loc["guide_A1", "grna_target"] == "elem_A"
    assert var.loc["guide_A1", "unit_kind"] == "targeting_grna"
    # and the frame that does hold both
    design = load_export(path).grna_target_data_frame
    assert set(design.loc[design["grna_id"] == "guide_shared", "grna_target"]) == {
        "elem_A",
        "elem_B",
    }


def test_an_export_without_targeting_guides_still_loads(tmp_path):
    """Datasets written before this existed have to keep working: the fields
    are absent, not empty, and a reader is expected to check."""
    back = load_export(write_h5mu(_export(targeting=False), tmp_path / "dataset.h5mu"))
    assert back.targeting_grna_cells is None
    assert back.grna_target_data_frame is None
    assert set(back.grna_target_cells) == {"elem_A", "elem_B"}
    assert set(back.ntc_grna_cells) == {"ntc_1", "ntc_2"}


def test_discovery_pairs_with_info_keeps_its_qc_columns(written):
    """The QC columns are the reason this frame is carried at all -- `pairs`
    already holds the ids of the passing ones."""
    dpwi = written.discovery_pairs_with_info
    assert {"n_nonzero_trt", "n_nonzero_cntrl", "pass_qc"} <= set(dpwi.columns)
    assert len(dpwi) == 3
    # A logical must come back logical; int8 would make boolean indexing fail.
    assert dpwi["pass_qc"].astype(bool).tolist() == [True, True, False]
    assert len(written.pairs) == int(dpwi["pass_qc"].astype(bool).sum())


def _masked_export() -> SceptreExport:
    """An `--all-cells` export in miniature: cell 2 failed QC.

    Cell 2 carries `guide_A1` and belongs to `elem_A`, so subsetting has to
    move both the matrix and the units, and drop that one membership from each.
    """
    mask = np.ones(N_CELLS, dtype=bool)
    mask[2] = False
    export = _export(in_use=mask)
    export.metadata["n_cells_in_use"] = int(mask.sum())
    export.metadata["all_cells"] = True
    return export


def test_the_default_read_subsets_to_the_qc_passing_cells(tmp_path):
    """The contract that lets an analysis ignore how the file was written."""
    path = write_h5mu(_masked_export(), tmp_path / "dataset.h5mu")
    sub = load_export(path)

    assert sub.response_matrix.shape == (N_GENES, N_CELLS - 1)
    assert sub.covariate_matrix.shape == (N_CELLS - 1, 2)
    assert sub.metadata["n_cells"] == N_CELLS - 1
    assert sub.in_use.all()
    # Cell 2 is gone and everything above it has shifted down by one, in the
    # matrix and in every unit kind alike.
    assert np.array_equal(sub.grna_target_cells["elem_A"], [0, 1, 2, 3, 4])
    assert np.array_equal(sub.targeting_grna_cells["guide_A1"], [0, 1])
    assert np.array_equal(sub.ntc_grna_cells["ntc_1"], [9, 10])
    full = _export()
    assert np.array_equal(
        sub.response_matrix.toarray(),
        full.response_matrix.toarray()[:, [c for c in range(N_CELLS) if c != 2]],
    )


def test_all_cells_returns_every_cell(tmp_path):
    """The simulation path: expression statistics need the cells QC removed."""
    path = write_h5mu(_masked_export(), tmp_path / "dataset.h5mu")
    full = load_export(path, all_cells=True)

    assert full.response_matrix.shape == (N_GENES, N_CELLS)
    assert full.in_use.sum() == N_CELLS - 1
    assert not full.in_use[2]
    # Units are in the file's space here, so the cell that failed QC is still
    # addressable and still carries its guide.
    assert np.array_equal(full.targeting_grna_cells["guide_A1"], [0, 1, 2])


def test_a_file_without_the_mask_is_unaffected_by_either_path(tmp_path):
    """Every dataset written before `--all-cells` existed: the two reads agree
    because there is nothing to subset."""
    path = write_h5mu(_export(), tmp_path / "dataset.h5mu")
    default, explicit = load_export(path), load_export(path, all_cells=True)
    assert default.response_matrix.shape == explicit.response_matrix.shape == (N_GENES, N_CELLS)
    for unit, cells in default.grna_target_cells.items():
        assert np.array_equal(cells, explicit.grna_target_cells[unit])


def test_a_backed_read_of_an_all_cells_file_is_refused_not_mis_served(tmp_path):
    """`BackedResponseMatrix` serves columns straight out of the file, so it
    cannot honour the subset. Refusing beats handing back the wrong cells."""
    path = write_h5mu(_masked_export(), tmp_path / "dataset.h5mu")
    with pytest.raises(ValueError, match="all_cells=True"):
        load_export(path, backed=True)
