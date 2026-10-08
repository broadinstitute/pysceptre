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

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mudata", reason="the io extra is not installed")

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

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


# --- low MOI --------------------------------------------------------------------------------
#
# An nt_cells object stores its NT gRNAs as positions within an NT-cell pool, `all_nt_idxs`.
# The exporter decodes them to cells and carries the pool in R's order, which is not sorted.

NT_POOL = [30, 31, 12, 13, 14]


def _lowmoi_export(*, in_use: np.ndarray | None = None) -> SceptreExport:
    """A low-MOI nt_cells export: one gRNA per cell, so the NT units are disjoint and share
    no cell with a target. The pool is the NT units' cells concatenated in unit order, which
    is how R builds it, and deliberately unsorted."""
    export = _export(in_use=in_use)
    export.ntc_grna_cells = {"ntc_1": np.array([30, 31]), "ntc_2": np.array([12, 13, 14])}
    export.all_nt_idxs = np.array(NT_POOL)
    export.metadata.update(
        {
            "low_moi": True,
            "control_group": "nt_cells",
            "resampling_mechanism": "permutations",
            "control_group_complement": False,
            "run_permutations": True,
            "side": "both",
            "resampling_approximation": "skew_normal",
            "multiple_testing_alpha": 0.1,
            "B3": 24999,
        }
    )
    return export


def test_low_moi_settings_survive_the_round_trip(tmp_path):
    back = load_export(write_h5mu(_lowmoi_export(), tmp_path / "dataset.h5mu"))
    assert back.low_moi is True
    assert (back.moi, back.control_group, back.resampling_mechanism) == (
        "low",
        "nt_cells",
        "permutations",
    )
    assert back.metadata["side"] == back.side == "both"


def test_the_nt_pool_keeps_rs_order(tmp_path):
    """Sorting the pool would move every cell R's permutation draws index by position."""
    back = load_export(write_h5mu(_lowmoi_export(), tmp_path / "dataset.h5mu"))
    assert np.array_equal(back.all_nt_idxs, NT_POOL)
    assert list(back.ntc_grna_cells) == ["ntc_1", "ntc_2"]
    assert np.array_equal(np.concatenate(list(back.ntc_grna_cells.values())), back.all_nt_idxs)


def test_analysis_kwargs_carry_the_nt_cells_for_an_nt_cells_export(tmp_path):
    back = load_export(write_h5mu(_lowmoi_export(), tmp_path / "dataset.h5mu"))
    kwargs = back.analysis_kwargs()
    assert {k: v for k, v in kwargs.items() if k != "ntc_grna_cells"} == {
        "moi": "low",
        "control_group": "nt_cells",
        "resampling_mechanism": "permutations",
        "side": "both",
        "resampling_approximation": "skew_normal",
        "multiple_testing_alpha": 0.1,
    }
    assert kwargs["ntc_grna_cells"] is back.ntc_grna_cells


def test_analysis_kwargs_for_a_complement_export_leave_the_nt_cells_out(written):
    assert written.analysis_kwargs() == {
        "moi": "high",
        "control_group": "complement",
        "resampling_mechanism": "crt",
        "side": "both",
    }
    assert written.all_nt_idxs is None


@pytest.mark.parametrize(
    ("r_logicals", "expected"),
    [
        ({}, ("high", "complement", "crt")),
        (
            {"control_group_complement": True, "run_permutations": True},
            ("high", "complement", "permutations"),
        ),
    ],
)
def test_an_export_without_the_settings_keys_falls_back(tmp_path, r_logicals, expected):
    """Old exports carry R's two logicals, or nothing: high MOI, complement and CRT."""
    export = _export()
    export.metadata.pop("run_permutations")
    export.metadata.update(r_logicals)
    back = load_export(write_h5mu(export, tmp_path / "dataset.h5mu"))
    assert (back.moi, back.control_group, back.resampling_mechanism) == expected


def test_an_nt_cells_export_without_its_pool_is_refused(tmp_path):
    """What the exporter wrote for an nt_cells object before it decoded the NT positions:
    `control_group_complement` false, no settings keys, no pool. Its NT units are positions,
    so reading them as cells would be silently wrong."""
    export = _export()
    export.metadata["control_group_complement"] = False
    path = write_h5mu(export, tmp_path / "dataset.h5mu")
    with pytest.raises(ValueError, match="Re-export"):
        load_export(path)


@pytest.mark.parametrize(
    "pool",
    [
        [30, 31, 12, 13],  # a cell missing
        [30, 31, 12, 13, 14, 15],  # a cell no NT unit carries
        [30, 31, 12, 13, 13],  # a cell twice
        [30, 31, 12, 13, 400],  # out of range
    ],
)
def test_a_pool_the_nt_units_do_not_partition_is_refused(tmp_path, pool):
    export = _lowmoi_export()
    export.all_nt_idxs = np.array(pool)
    with pytest.raises(ValueError, match="all_nt_idxs"):
        load_export(write_h5mu(export, tmp_path / "dataset.h5mu"))


def test_subsetting_moves_the_nt_pool_without_reordering_it(tmp_path):
    """Cell 2 fails QC, so every NT cell above it shifts down by one, in place."""
    mask = np.ones(N_CELLS, dtype=bool)
    mask[2] = False
    export = _lowmoi_export(in_use=mask)
    export.metadata["all_cells"] = True
    export.metadata["n_cells_in_use"] = int(mask.sum())
    path = write_h5mu(export, tmp_path / "dataset.h5mu")

    assert np.array_equal(load_export(path, all_cells=True).all_nt_idxs, NT_POOL)
    sub = load_export(path)
    assert np.array_equal(sub.all_nt_idxs, [29, 30, 11, 12, 13])
    assert np.array_equal(np.concatenate(list(sub.ntc_grna_cells.values())), sub.all_nt_idxs)


def _write_intermediate(export: SceptreExport, directory: Path) -> Path:
    """The columnar layout `export_sceptre_dataset.R` writes, from a SceptreExport."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(json.dumps(export.metadata))
    coo = export.response_matrix.tocoo()
    pd.DataFrame({"gene_index": coo.row, "cell_index": coo.col, "value": coo.data}).to_parquet(
        directory / "response_matrix.parquet"
    )
    pd.DataFrame(
        {"gene_index": np.arange(len(export.gene_ids)), "response_id": export.gene_ids}
    ).to_parquet(directory / "gene_ids.parquet")
    pd.DataFrame(export.covariate_matrix, columns=export.metadata["covariate_names"]).to_parquet(
        directory / "covariate_matrix.parquet"
    )
    pd.DataFrame({"in_use": export.in_use}).to_parquet(directory / "cell_annotation.parquet")
    units = [(u, "target", u, c) for u, c in export.grna_target_cells.items()]
    units += [(u, "ntc_grna", "non-targeting", c) for u, c in (export.ntc_grna_cells or {}).items()]
    pd.DataFrame(
        {
            "unit_id": [u for u, _, _, c in units for _ in c],
            "cell_index": np.concatenate([c for *_, c in units]),
        }
    ).to_parquet(directory / "grna_assignments.parquet")
    pd.DataFrame(
        {
            "unit_id": [u for u, *_ in units],
            "grna_target": [t for _, _, t, _ in units],
            "unit_kind": [k for _, k, _, _ in units],
        }
    ).to_parquet(directory / "grna_annotation.parquet")
    export.pairs.to_parquet(directory / "pairs.parquet")
    if export.all_nt_idxs is not None:
        pd.DataFrame({"cell_index": export.all_nt_idxs}).to_parquet(
            directory / "all_nt_idxs.parquet"
        )
    return directory


def test_make_h5mu_carries_the_nt_pool_and_removes_its_parquet(tmp_path):
    """The conversion an R export goes through, end to end on the Python side."""
    pytest.importorskip("pyarrow", reason="the io extra is not installed")
    directory = _write_intermediate(_lowmoi_export(), tmp_path / "export")
    subprocess.run(
        [sys.executable, str(SCRIPTS / "make_h5mu.py"), str(directory)],
        check=True,
        capture_output=True,
    )
    assert not (directory / "all_nt_idxs.parquet").exists()
    back = load_export(directory)
    assert np.array_equal(back.all_nt_idxs, NT_POOL)
    assert (back.moi, back.control_group, back.resampling_mechanism) == (
        "low",
        "nt_cells",
        "permutations",
    )


def test_an_old_nt_cells_intermediate_is_refused(tmp_path):
    pytest.importorskip("pyarrow", reason="the io extra is not installed")
    export = _export(targeting=False)
    export.metadata["control_group_complement"] = False
    directory = _write_intermediate(export, tmp_path / "export")
    with pytest.raises(ValueError, match="Re-export"):
        load_export(directory)


def _with_counts(export: SceptreExport) -> SceptreExport:
    from dataclasses import replace

    from scipy import sparse

    counts = np.zeros((3, N_CELLS), dtype=np.int64)
    counts[0, [0, 2, 9]] = [5, 120, 1]
    counts[1, [3, 4]] = [2, 40]
    counts[2, [2, 12]] = [7, 3]
    return replace(
        export,
        grna_counts=sparse.csr_matrix(counts),
        grna_count_ids=["guide_A1", "guide_A2", "ntc_1"],
    )


def test_grna_counts_survive_the_round_trip(tmp_path):
    exp = _with_counts(_export())
    got = load_export(write_h5mu(exp, tmp_path / "dataset.h5mu"))
    assert got.grna_count_ids == exp.grna_count_ids
    np.testing.assert_array_equal(got.grna_counts.toarray(), exp.grna_counts.toarray())
    assert "3 gRNAs" in got.describe()


def test_grna_counts_are_subset_to_the_qc_passing_cells(tmp_path):
    exp = _with_counts(_masked_export())
    sub = load_export(write_h5mu(exp, tmp_path / "dataset.h5mu"))
    keep = [c for c in range(N_CELLS) if c != 2]
    np.testing.assert_array_equal(sub.grna_counts.toarray(), exp.grna_counts.toarray()[:, keep])


def test_an_export_without_grna_counts_still_loads(written):
    assert written.grna_counts is None and written.grna_count_ids is None
