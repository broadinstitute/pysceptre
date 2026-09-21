import json
import subprocess
from pathlib import Path

import pytest

VALIDATION_DIR = Path(__file__).parent
GROUND_TRUTH_PATH = VALIDATION_DIR / "ground_truth.json"
DUMP_SCRIPT = VALIDATION_DIR.parent.parent / "scripts" / "dump_r_ground_truth.R"


def _r_sceptre_available() -> bool:
    try:
        result = subprocess.run(
            ["Rscript", "-e", "library(sceptre)"],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="session")
def ground_truth():
    if not GROUND_TRUTH_PATH.exists():
        if not _r_sceptre_available():
            pytest.skip("R/sceptre not available and no cached ground_truth.json present")
        subprocess.run(["Rscript", str(DUMP_SCRIPT), str(GROUND_TRUTH_PATH), "4"], check=True)
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)


PERTURBPLAN_GROUND_TRUTH_PATH = VALIDATION_DIR / "perturbplan_ground_truth.json"
PERTURBPLAN_DUMP_SCRIPT = (
    VALIDATION_DIR.parent.parent / "scripts" / "dump_perturbplan_ground_truth.R"
)


def _r_perturbplan_available() -> bool:
    try:
        result = subprocess.run(
            ["Rscript", "-e", "library(perturbplan)"],
            capture_output=True,
            timeout=60,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="session")
def perturbplan_ground_truth():
    """PerturbPlan's own output for the analytical power port.

    Regenerated only if the JSON is missing, like `ground_truth`, so CI needs
    neither R nor perturbplan. If you change
    `scripts/dump_perturbplan_ground_truth.R`, delete the JSON or the tests
    keep validating against the stale fixture.
    """
    if not PERTURBPLAN_GROUND_TRUTH_PATH.exists():
        if not _r_perturbplan_available():
            pytest.skip(
                "R/perturbplan not available and no cached perturbplan_ground_truth.json present"
            )
        subprocess.run(
            ["Rscript", str(PERTURBPLAN_DUMP_SCRIPT), str(PERTURBPLAN_GROUND_TRUTH_PATH)],
            check=True,
        )
    with open(PERTURBPLAN_GROUND_TRUTH_PATH) as f:
        return json.load(f)
