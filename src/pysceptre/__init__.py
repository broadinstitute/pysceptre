"""Standalone Python port of sceptre's CRT discovery-analysis statistical engine.

Targets one validated analysis path -- the complement control group + CRT
resampling mechanism used for high-MOI single-cell CRISPR screens -- and
batches the per-gene/per-target linear algebra into vectorized numpy calls.
See README.md for scope, limitations, and validation against the R package.
"""

from importlib.metadata import PackageNotFoundError, version

# Note the two senses of "power" sitting side by side. `run_power_check` is
# sceptre's positive-control diagnostic: it runs the real test on pairs where
# an effect is expected. `compute_power` is the analytical estimate:
# no test is run at all, and it answers what the screen *could* have detected.
from .analytical_power import compute_power
from .pipeline.api import (
    run_calibration_check,
    run_discovery_analysis,
    run_power_check,
)

try:
    __version__ = version("pysceptre")
except PackageNotFoundError:  # not installed (e.g. running from a source tree)
    __version__ = "unknown"

__all__ = [
    "compute_power",
    "run_discovery_analysis",
    "run_calibration_check",
    "run_power_check",
    "__version__",
]
