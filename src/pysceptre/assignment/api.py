"""One entry point for every assignment method, like sceptre's `assign_grnas`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse

from .fishash import FishashResult, assign_grnas_fishash
from .maximum import MaximumResult, assign_grnas_maximum
from .mixture import MixtureResult, assign_grnas_mixture
from .thresholding import ThresholdingResult, assign_grnas_thresholding

__all__ = ["ASSIGNMENT_METHODS", "assign_grnas"]

ASSIGNMENT_METHODS = ("mixture", "thresholding", "maximum", "fishash")


def assign_grnas(
    grna_matrix: sparse.spmatrix | np.ndarray,
    grna_ids: Sequence[str],
    *,
    method: str = "default",
    moi: str | None = None,
    covariate_matrix: np.ndarray | None = None,
    **hyperparameters: Any,
) -> MixtureResult | ThresholdingResult | MaximumResult | FishashResult:
    """Assign gRNAs to cells with one of the package's methods.

    `method="default"` is sceptre's default: "maximum" for a low-MOI screen and "mixture" for a
    high-MOI one, so it needs `moi`. The methods are "mixture" (`assign_grnas_mixture`, which
    needs `covariate_matrix`), "thresholding" (`assign_grnas_thresholding`), "maximum"
    (`assign_grnas_maximum`, refused for a high-MOI screen as in sceptre) and "fishash"
    (`assign_grnas_fishash`, not from sceptre). Each method's options are passed through
    `hyperparameters` under that function's keyword names.

    Args:
        grna_matrix: `(n_grnas, n_cells)` raw integer gRNA UMI counts, scipy.sparse or dense.
        grna_ids: The `n_grnas` row names, distinct.
        method: "default" or one of `ASSIGNMENT_METHODS`.
        moi: "low" or "high"; required by "default".
        covariate_matrix: The mixture's `(n_cells, p)` design; `mixture_design_matrix`
            builds sceptre's default.
        **hyperparameters: Keyword options of the chosen method's function.

    Returns:
        The chosen method's result: `MixtureResult`, `ThresholdingResult`, `MaximumResult` or
        `FishashResult`.

    Raises:
        ValueError: an unknown `method` or `moi`, "default" without `moi`, "maximum" for a
            high-MOI screen, "mixture" without `covariate_matrix`, or `covariate_matrix` given
            to another method.
        TypeError: a hyperparameter the chosen method does not take.
    """
    if moi is not None and moi not in ("low", "high"):
        raise ValueError(f"moi must be 'low' or 'high', got {moi!r}")
    if method == "default":
        if moi is None:
            raise ValueError("method='default' needs moi: 'maximum' for low, 'mixture' for high")
        method = "maximum" if moi == "low" else "mixture"
    if method not in ASSIGNMENT_METHODS:
        raise ValueError(f"method must be 'default' or one of {ASSIGNMENT_METHODS}, got {method!r}")
    if method == "maximum" and moi == "high":
        raise ValueError("method 'maximum' is for low-MOI screens only, as in sceptre")
    if method == "mixture":
        if covariate_matrix is None:
            raise ValueError(
                "method 'mixture' needs covariate_matrix; mixture_design_matrix builds sceptre's default"
            )
        return assign_grnas_mixture(grna_matrix, grna_ids, covariate_matrix, **hyperparameters)
    if covariate_matrix is not None:
        raise ValueError(f"covariate_matrix is used by method 'mixture' only, not {method!r}")
    if method == "thresholding":
        return assign_grnas_thresholding(grna_matrix, grna_ids, **hyperparameters)
    if method == "maximum":
        return assign_grnas_maximum(grna_matrix, grna_ids, **hyperparameters)
    return assign_grnas_fishash(grna_matrix, grna_ids, **hyperparameters)
