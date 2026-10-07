"""gRNA-to-cell assignment: sceptre's mixture method, and fishash.

`mixture.py` and `design.py` port sceptre's mixture assignment. `fishash.py` is not from
sceptre: it ports the MIT-licensed R package fishash, and `hypergeom.py` ports the
hypergeometric tail it relies on from R's nmath (GPL-2-or-later); see `THIRD_PARTY_LICENSES`.
See `docs/design.md`, "gRNA assignment".
"""

from __future__ import annotations

from .cells import cells_by_grna, cells_by_target
from .design import cell_count_covariates, design_from_covariates, mixture_design_matrix
from .fishash import (
    FishashIteration,
    FishashResult,
    ImputedCounts,
    assign_grnas_fishash,
    impute_masked_counts,
)
from .mixture import G_PERT_GUESSES, PI_GUESSES, MixtureResult, assign_grnas_mixture

__all__ = [
    "FishashIteration",
    "FishashResult",
    "G_PERT_GUESSES",
    "ImputedCounts",
    "MixtureResult",
    "PI_GUESSES",
    "assign_grnas_fishash",
    "assign_grnas_mixture",
    "cell_count_covariates",
    "cells_by_grna",
    "cells_by_target",
    "design_from_covariates",
    "impute_masked_counts",
    "mixture_design_matrix",
]
