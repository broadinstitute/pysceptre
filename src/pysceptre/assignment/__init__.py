"""gRNA-to-cell assignment: sceptre's mixture, thresholding and maximum methods, and fishash.

`mixture.py` and `design.py` port sceptre's mixture assignment, `thresholding.py` and
`maximum.py` its two simpler methods, and `api.py` its single entry point, `assign_grnas`.
`fishash.py` is not from sceptre: it ports the MIT-licensed R package fishash, and
`hypergeom.py` ports the hypergeometric tail it relies on from R's nmath (GPL-2-or-later); see
`THIRD_PARTY_LICENSES`. See `docs/design.md`, "gRNA assignment".
"""

from __future__ import annotations

from .api import ASSIGNMENT_METHODS, assign_grnas
from .cells import cells_by_grna, cells_by_target, cells_w_zero_or_twoplus_grnas
from .design import cell_count_covariates, design_from_covariates, mixture_design_matrix
from .fishash import (
    FishashIteration,
    FishashResult,
    ImputedCounts,
    assign_grnas_fishash,
    impute_masked_counts,
)
from .maximum import MaximumResult, assign_grnas_maximum
from .mixture import G_PERT_GUESSES, PI_GUESSES, MixtureResult, assign_grnas_mixture
from .thresholding import ThresholdingResult, assign_grnas_thresholding

__all__ = [
    "ASSIGNMENT_METHODS",
    "FishashIteration",
    "FishashResult",
    "G_PERT_GUESSES",
    "ImputedCounts",
    "MaximumResult",
    "MixtureResult",
    "PI_GUESSES",
    "ThresholdingResult",
    "assign_grnas",
    "assign_grnas_fishash",
    "assign_grnas_maximum",
    "assign_grnas_mixture",
    "assign_grnas_thresholding",
    "cell_count_covariates",
    "cells_by_grna",
    "cells_by_target",
    "cells_w_zero_or_twoplus_grnas",
    "design_from_covariates",
    "impute_masked_counts",
    "mixture_design_matrix",
]
