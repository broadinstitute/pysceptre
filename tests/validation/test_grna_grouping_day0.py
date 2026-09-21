"""The singleton expansion against sceptre's own, at real scale.

`test_grna_grouping.py` pins the arithmetic on four small cases from R. This
runs the same comparison on a real screen, where the gRNA-to-target map is
genuinely many-to-many -- on day0, 1,673 of 43,736 guides sit inside two or
three overlapping candidate elements -- and where 36,450 pairs expand to
515,972. A transcription that silently deduplicated by guide would pass the
small cases and fail here.

Needs a post-QC sceptre object, which is real screen data and never
committed:

    PYSCEPTRE_DAY0_SCEPTRE_OBJECT=path/to/sceptre_object.rds pytest -m realdata
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pysceptre.pipeline.grouping import singleton_pairs

pytestmark = pytest.mark.realdata

_DUMP_R = r"""
suppressPackageStartupMessages(library(sceptre))
args <- commandArgs(trailingOnly = TRUE)
object_path <- args[1]; out <- args[2]
so <- readRDS(object_path)
if (nrow(so@discovery_pairs) == 0) stop("this object carries no discovery pairs")
# Their function, with the strategy flipped. Nothing else is touched.
so@grna_integration_strategy <- "singleton"
expanded <- sceptre:::update_dfs_based_on_grouping_strategy(so)
cat(sprintf("%d pairs -> %d, over %d design rows\n", nrow(so@discovery_pairs),
            nrow(expanded@discovery_pairs), nrow(so@grna_target_data_frame)))
writeLines(jsonlite::toJSON(list(
  pairs_in = so@discovery_pairs[, c("response_id", "grna_target")],
  gtdf = so@grna_target_data_frame[, c("grna_id", "grna_target")],
  out_response = expanded@discovery_pairs$response_id,
  out_grna_id = expanded@discovery_pairs$grna_group,
  out_target = expanded@discovery_pairs$grna_target
), auto_unbox = TRUE), out)
"""


@pytest.fixture(scope="module")
def r_expansion() -> dict:
    obj = os.environ.get("PYSCEPTRE_DAY0_SCEPTRE_OBJECT")
    if not obj or not Path(obj).exists():
        pytest.skip("set PYSCEPTRE_DAY0_SCEPTRE_OBJECT to a post-QC sceptre object")
    try:
        ok = subprocess.run(
            ["Rscript", "-e", "library(sceptre); library(jsonlite)"],
            capture_output=True,
            timeout=180,
        ).returncode
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("Rscript not available")
    if ok != 0:
        pytest.skip("R lacks sceptre or jsonlite")
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "dump.R"
        script.write_text(_DUMP_R)
        payload = Path(tmp) / "payload.json"
        run = subprocess.run(
            ["Rscript", str(script), obj, str(payload)], capture_output=True, text=True
        )
        if run.returncode != 0:
            pytest.skip(f"could not expand in R: {run.stderr.strip()[-300:]}")
        with open(payload) as f:
            return json.load(f)


def test_the_expansion_matches_r_row_for_row(r_expansion):
    """Exactly, and in the same order, over half a million rows."""
    d = r_expansion
    got = singleton_pairs(pd.DataFrame(d["pairs_in"]), pd.DataFrame(d["gtdf"]))
    expected = pd.DataFrame(
        {
            "response_id": d["out_response"],
            "grna_id": d["out_grna_id"],
            "grna_target": d["out_target"],
        }
    )
    assert len(got) == len(expected), f"{len(got)} rows against R's {len(expected)}"
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)


def test_the_real_design_is_many_to_many(r_expansion):
    """Otherwise the test above proves less than it appears to."""
    gtdf = pd.DataFrame(r_expansion["gtdf"])
    shared = gtdf["grna_id"].duplicated().sum()
    assert shared > 0, "this screen's guides each sit in one target; the join is untested"
    print(f"\n{shared} of {gtdf['grna_id'].nunique()} guides sit in more than one target")


def test_every_expanded_guide_belongs_to_its_row_s_target(r_expansion):
    """A guide must never be carried under a target it was not designed against."""
    d = r_expansion
    got = singleton_pairs(pd.DataFrame(d["pairs_in"]), pd.DataFrame(d["gtdf"]))
    design = set(
        zip(
            pd.DataFrame(d["gtdf"])["grna_id"],
            pd.DataFrame(d["gtdf"])["grna_target"],
            strict=True,
        )
    )
    pairs_seen = set(zip(got["grna_id"], got["grna_target"], strict=True))
    assert pairs_seen <= design
