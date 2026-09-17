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
