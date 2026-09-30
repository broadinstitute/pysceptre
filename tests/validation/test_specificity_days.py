"""`run_specificity_check` against the notebook it was ported from, on three real screens.

The notebook is WattEG-paper's `analysis/direct_indirect.py` (day0, day2, day4
of an endothelial differentiation). There is no R for this check, so the
notebook's results are the reference: the far-bin estimates and intervals,
the background rates, the broad-effect counts and the detour statuses it
reports, reproduced here from the same inputs. The draws follow the
notebook's order (each day with all elements, then without the broad-effect
ones, days in order from one seed), so the intervals are its intervals.

    PYSCEPTRE_WATTEG_PAPER=.../WattEG-paper \\
    PYSCEPTRE_GENCODE_GTF=.../gencode.v44.genes_only.gtf \\
        pytest -m realdata tests/validation/test_specificity_days.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysceptre import run_specificity_check
from pysceptre.analytical_power import bh_nominal_cutoff

pytestmark = pytest.mark.realdata

DAYS = ("day0", "day2", "day4")
SEED = 20260929
ALPHA = 0.1

# (above, lo, hi) in the far bin, all elements and without broad-effect ones; background rate;
# broad-effect elements; detour statuses of the far links.
EXPECTED = {
    "day0": dict(
        far=(0.20, -0.30, 0.44),
        far_without=(0.18, -0.33, 0.43),
        rate=0.000844,
        broad=4,
        detour={"can't check": 25, "no detour": 1},
    ),
    "day2": dict(
        far=(0.45, 0.28, 0.57),
        far_without=(0.50, 0.28, 0.63),
        rate=0.001422,
        broad=40,
        detour={"can't check": 41, "detour": 12, "no detour": 9, "detour (nominal)": 5},
    ),
    "day4": dict(
        far=(0.55, 0.37, 0.66),
        far_without=(0.66, 0.53, 0.74),
        rate=0.000872,
        broad=16,
        detour={"can't check": 47, "no detour": 5},
    ),
}


def _gtf_genes(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 8)
            if f[2] != "gene" or not f[0].startswith("chr"):
                continue
            name = f[8].split('gene_name "', 1)[1].split('"', 1)[0]
            rows.append((name, f[0], int(f[3]) if f[6] == "+" else int(f[4])))
    return pd.DataFrame(rows, columns=["response_id", "chrom", "tss"])


@pytest.fixture(scope="module")
def results() -> dict:
    root, gtf = os.environ.get("PYSCEPTRE_WATTEG_PAPER"), os.environ.get("PYSCEPTRE_GENCODE_GTF")
    if not (root and gtf):
        pytest.skip("set PYSCEPTRE_WATTEG_PAPER and PYSCEPTRE_GENCODE_GTF")
    pytest.importorskip("mudata")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from sceptre_io import load_h5mu  # noqa: PLC0415

    genes = _gtf_genes(Path(gtf))
    rng = np.random.default_rng(SEED)
    out = {}
    for day in DAYS:
        x = load_h5mu(
            Path(root) / f"sceptre_objects/{day}/export/dataset_permutations.h5mu",
            backed=True,
            all_cells=True,
        )
        cis = x.discovery_result
        trans = pd.read_csv(Path(root) / f"discovery/{day}_trans_discovery.tsv", sep="\t")
        cutoff = bh_nominal_cutoff(
            cis.loc[cis["pass_qc"].astype(bool), "p_value"].to_numpy(), ALPHA
        )
        # These screens name each target by its hg38 coordinates, which is where the notebook
        # took positions from. That is a property of these screens, not a rule: DC-TAP's names
        # are hg19, and a caller passes positions from the screen's own coordinate table.
        targets = pd.Series(pd.concat([cis["grna_target"], trans["grna_target"]]).unique())
        c = targets.str.extract(r"^(chr[^:]+):(\d+)-(\d+)")
        named = c[0].notna()
        positions = pd.DataFrame(
            {
                "grna_target": targets[named],
                "chrom": c.loc[named, 0],
                "centre": (c.loc[named, 1].astype(int) + c.loc[named, 2].astype(int)) / 2,
            }
        )
        g = x.grna_target_data_frame
        out[day] = run_specificity_check(
            cis,
            trans,
            positions,
            genes,
            cutoff,
            control_targets=set(g.loc[g["grna_id"].str.contains("TSS"), "grna_target"]),
            measured_genes=list(x.gene_ids),
            seed=rng,
        )
    return out


@pytest.mark.parametrize("day", DAYS)
def test_far_links_above_background(results, day):
    exp = EXPECTED[day]
    for table, want in (
        (results[day].by_distance, exp["far"]),
        (results[day].by_distance_without_broad, exp["far_without"]),
    ):
        far = table.iloc[-1]
        assert far["distance_bin"] == "> 100 kb"
        got = (far["above"], far["above_lo"], far["above_hi"])
        assert got == pytest.approx(want, abs=0.005)


@pytest.mark.parametrize("day", DAYS)
def test_near_links_are_above_background(results, day):
    """The plan's other headline: within 50 kb, at least 95% of links are above background."""
    near = results[day].by_distance.iloc[:3]
    assert (near["above"] >= 0.95).all()


@pytest.mark.parametrize("day", DAYS)
def test_background_rate_and_broad_effect_elements(results, day):
    exp = EXPECTED[day]
    assert results[day].background_rate == pytest.approx(exp["rate"], abs=5e-7)
    assert int(results[day].broad_effect["broad_effect"].sum()) == exp["broad"]


@pytest.mark.parametrize("day", DAYS)
def test_detour_statuses(results, day):
    assert results[day].far_links["status"].value_counts().to_dict() == EXPECTED[day]["detour"]
