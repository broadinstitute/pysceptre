"""Standalone Python port of sceptre's discovery-analysis statistical engine.

Covers sceptre's discovery analysis, calibration check and power check for
high- and low-MOI single-cell CRISPR screens -- the complement and NT-cells
control groups, with CRT or permutation resampling -- and batches the
per-gene/per-target linear algebra into vectorized numpy calls. It also ports
sceptre's gRNA assignment methods (mixture, thresholding, maximum) and, not from
sceptre, fishash. See README.md for scope, limitations, and validation against
the R packages.
"""

from importlib.metadata import PackageNotFoundError, version

# Note the two senses of "power" sitting side by side. `run_power_check` is
# sceptre's positive-control diagnostic: it runs the real test on pairs where
# an effect is expected. `compute_power` is the analytical estimate:
# no test is run at all, and it answers what the screen *could* have detected.
from .analytical_power import compute_power
from .assignment import (
    assign_grnas,
    assign_grnas_fishash,
    assign_grnas_maximum,
    assign_grnas_mixture,
    assign_grnas_thresholding,
)
from .pipeline.api import (
    run_calibration_check,
    run_discovery_analysis,
    run_power_check,
)
from .specificity import run_specificity_check

try:
    __version__ = version("pysceptre")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "unknown"

__all__ = [
    "assign_grnas",
    "assign_grnas_fishash",
    "assign_grnas_maximum",
    "assign_grnas_mixture",
    "assign_grnas_thresholding",
    "compute_power",
    "run_discovery_analysis",
    "run_calibration_check",
    "run_power_check",
    "run_specificity_check",
    "__version__",
]
