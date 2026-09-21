"""Closed-form per-pair power, ported from PerturbPlan (MIT).

Named to stay distinct from `pipeline/power.py`, which is sceptre's
positive-control *power check* -- a different thing that shares the word.
"""

from .closed_form import (
    qc_failure_prob,
    rejection_prob,
    test_stat_distribution,
    var_nb,
    zero_prob,
)
from .posthoc import compute_power_posthoc, target_cell_counts

__all__ = [
    "compute_power_posthoc",
    "qc_failure_prob",
    "rejection_prob",
    "target_cell_counts",
    "test_stat_distribution",
    "var_nb",
    "zero_prob",
]
