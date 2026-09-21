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
from .estimate import compute_power, target_cell_counts
from .inputs import (
    baseline_expression_stats,
    baseline_expression_stats_from_fits,
    bh_nominal_cutoff,
    cells_per_grna_from_assignments,
    poscounts_size_factors,
)

__all__ = [
    "baseline_expression_stats",
    "baseline_expression_stats_from_fits",
    "bh_nominal_cutoff",
    "cells_per_grna_from_assignments",
    "compute_power",
    "poscounts_size_factors",
    "qc_failure_prob",
    "rejection_prob",
    "target_cell_counts",
    "test_stat_distribution",
    "var_nb",
    "zero_prob",
]
