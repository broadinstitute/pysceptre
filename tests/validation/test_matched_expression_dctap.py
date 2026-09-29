"""`matched_expression_stats` against the script it was ported from, on a real screen.

The script is WattEG-paper's `analysis/perturbplan/info_matched_mean.py`,
and its cached table holds the per-element power (`pp_elem`) and information
ratio (`info_ratio_pert`) for every pair of one DC-TAP screen and pair set.
This rebuilds both from the same inputs and asks for the same numbers. The
simulation those columns are scored against is not re-scored here; see
`docs/design.md`, "Covariates through the information-matched mean".

    PYSCEPTRE_DCTAP_PREPARED=.../prepared \\
    PYSCEPTRE_DCTAP_PREFIX=.../compared_dctap/dctap_k562 \\
    PYSCEPTRE_DCTAP_INFOMEAN_TSV=.../dctap_k562_cis_infomean_fsd0.tsv \\
        pytest -m realdata tests/validation/test_matched_expression_dctap.py

`PREPARED` holds `sim_input.h5` and `discovery_threshold.txt`; `PREFIX` is
the path stem of `_cells_per_grna.tsv` and `_total_cells.txt`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysceptre.analytical_power import compute_power, matched_expression_stats

pytestmark = pytest.mark.realdata

KEYS = ["grna_target", "response_id"]
ENV = ("PYSCEPTRE_DCTAP_PREPARED", "PYSCEPTRE_DCTAP_PREFIX", "PYSCEPTRE_DCTAP_INFOMEAN_TSV")


@dataclass
class _Fit:
    fitted_coefs: np.ndarray
    theta: float


def _decode(values) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else v for v in values]


@pytest.fixture(scope="module")
def rebuilt() -> tuple[pd.DataFrame, pd.DataFrame]:
    unset = [v for v in ENV if not os.environ.get(v)]
    if unset:
        pytest.skip(f"unset: {', '.join(unset)}")
    h5py = pytest.importorskip("h5py")
    prepared = Path(os.environ["PYSCEPTRE_DCTAP_PREPARED"])
    prefix = os.environ["PYSCEPTRE_DCTAP_PREFIX"]

    with h5py.File(prepared / "sim_input.h5") as f:
        genes = _decode(f["genes"][:])
        coefs = f["fitted_coefs"][:]
        Z = f["covariate_matrix"][:]
        phi = f["row_data/dispersion"][:]
        target_ids = _decode(f["perts/target_ids"][:])
        cp = f["perts/cre_perts"]
        indptr, indices = cp["indptr"][:], cp["indices"][:]
    fits = {g: _Fit(coefs[k], 1.0 / phi[k]) for k, g in enumerate(genes)}
    union = {t: indices[indptr[k] : indptr[k + 1]] for k, t in enumerate(target_ids)}

    cells_per_grna = pd.read_csv(f"{prefix}_cells_per_grna.tsv", sep="\t")
    total = int(Path(f"{prefix}_total_cells.txt").read_text())
    cutoff = float((prepared / "discovery_threshold.txt").read_text())
    expected = pd.read_csv(os.environ["PYSCEPTRE_DCTAP_INFOMEAN_TSV"], sep="\t")

    stats = matched_expression_stats(expected[KEYS], Z, fits, union, cells_per_grna, total)
    power = compute_power(
        expected[KEYS],
        cells_per_grna,
        stats,
        fold_change_mean=0.85,
        fold_change_sd=0.0,
        cutoff=cutoff,
        num_total_cells=total,
        side="both",
    )
    return expected, stats.assign(power=power["power"].to_numpy())


def test_reproduces_the_information_ratio(rebuilt):
    expected, got = rebuilt
    np.testing.assert_allclose(got["info_ratio_pert"], expected["info_ratio_pert"], rtol=1e-9)


def test_reproduces_the_per_element_power(rebuilt):
    expected, got = rebuilt
    np.testing.assert_allclose(got["power"], expected["pp_elem"], rtol=1e-9, atol=1e-12)
