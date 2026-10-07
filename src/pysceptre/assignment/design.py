"""sceptre's default covariate matrix for gRNA assignment.

Port of the parts of sceptre 0.10.3 that build the mixture method's design: the per-cell
`n_nonzero` and `n_umis` of `compute_cell_covariates`, and `auto_construct_formula_object` with
`include_grna_covariates = TRUE` followed by `model.matrix`. It builds this one design; it is
not a formula language. See `docs/design.md`, "The default assignment design".
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import sparse

from ..glm.design import validate_design_matrix

__all__ = ["cell_count_covariates", "design_from_covariates", "mixture_design_matrix"]

# sceptre drops a non-count covariate with this many distinct values or more from the formula.
_MAX_N_LEVELS = 15
_COUNT_COVARIATE = re.compile("n_umis|n_nonzero")
_RESERVED = (
    "response_n_nonzero",
    "response_n_umis",
    "response_p_mito",
    "grna_n_nonzero",
    "grna_n_umis",
)


def cell_count_covariates(matrix: sparse.spmatrix | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell `(n_nonzero, n_umis)` of a `(n_features, n_cells)` count matrix.

    As sceptre's `compute_cell_covariates`: `n_nonzero` counts the entries above 0.5 and
    `n_umis` is the column sum.
    """
    c = sparse.csc_matrix(matrix, dtype=np.float64)
    c.sum_duplicates()
    c.sort_indices()
    n_cells = c.shape[1]
    cols = np.repeat(np.arange(n_cells, dtype=np.int64), np.diff(c.indptr))
    n_nonzero = np.bincount(cols, weights=(c.data > 0.5).astype(np.float64), minlength=n_cells)
    n_umis = np.bincount(cols, weights=c.data, minlength=n_cells)
    return n_nonzero, n_umis


def _categorical_columns(name: str, values) -> tuple[list[np.ndarray], list[str]]:
    s = pd.Series(values)
    if isinstance(s.dtype, pd.CategoricalDtype):
        levels = list(s.cat.categories)
    elif s.dtype == bool:
        levels = [False, True]
    else:
        levels = sorted(s.astype(str).unique())
        s = s.astype(str)
    columns, names = [], []
    for level in levels[1:]:
        columns.append((s == level).to_numpy(dtype=np.float64))
        names.append(f"{name}{'TRUE' if level is True else level}")
    return columns, names


def mixture_design_matrix(
    grna_matrix: sparse.spmatrix | np.ndarray,
    *,
    response_n_nonzero: np.ndarray | None = None,
    response_n_umis: np.ndarray | None = None,
    extra_covariates: pd.DataFrame | None = None,
) -> tuple[np.ndarray, list[str]]:
    """The design sceptre's mixture assignment uses by default, and its column names.

    The covariates, in sceptre's order, are `response_n_nonzero` and `response_n_umis` (when
    given), `grna_n_nonzero` and `grna_n_umis` (computed from `grna_matrix`), then the columns
    of `extra_covariates`. A covariate whose name contains `n_umis` or `n_nonzero` enters as
    `log(x)`, or `log(x + 1)` when any cell has zero; any other covariate with 15 or more
    distinct values is left out; the rest enter as they are, categorical ones as treatment
    dummies with the first level as reference. An intercept comes first.

    Args:
        grna_matrix: `(n_grnas, n_cells)` gRNA UMI counts.
        response_n_nonzero: `(n_cells,)` number of genes detected per cell, or None.
        response_n_umis: `(n_cells,)` gene expression UMIs per cell, or None. Give both response
            covariates or neither; with neither, the design uses the gRNA covariates only.
        extra_covariates: Optional per-cell covariates, one row per cell. Categorical levels
            order as pandas categories, or sorted as strings.

    Returns:
        `(X, column_names)`, `X` of shape `(n_cells, p)`.

    Raises:
        ValueError: only one response covariate given, lengths disagree, an extra covariate
            uses a name sceptre reserves, or the design is not finite and full rank.
    """
    grna_n_nonzero, grna_n_umis = cell_count_covariates(grna_matrix)
    n_cells = grna_n_nonzero.size
    if (response_n_nonzero is None) != (response_n_umis is None):
        raise ValueError("give both response_n_nonzero and response_n_umis, or neither")
    frame: dict[str, np.ndarray | pd.Series] = {}
    if response_n_nonzero is not None:
        frame["response_n_nonzero"] = np.asarray(response_n_nonzero, dtype=np.float64)
        frame["response_n_umis"] = np.asarray(response_n_umis, dtype=np.float64)
    frame["grna_n_nonzero"] = grna_n_nonzero
    frame["grna_n_umis"] = grna_n_umis
    if extra_covariates is not None:
        clash = [c for c in extra_covariates.columns if c in _RESERVED]
        if clash:
            raise ValueError(f"extra_covariates uses names sceptre reserves: {clash}")
        if len(extra_covariates) != n_cells:
            raise ValueError(
                f"extra_covariates has {len(extra_covariates)} rows for {n_cells} cells"
            )
        for c in extra_covariates.columns:
            frame[str(c)] = extra_covariates[c].reset_index(drop=True)
    for name, v in frame.items():
        if len(v) != n_cells:
            raise ValueError(f"{name} has {len(v)} values for {n_cells} cells")

    return design_from_covariates(pd.DataFrame(frame))


def design_from_covariates(covariates: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """sceptre's default formula applied to a covariate frame, as `model.matrix` would.

    `covariates` is sceptre's `covariate_data_frame`, columns in its order. `response_p_mito` is
    left out; a column whose name contains `n_umis` or `n_nonzero` enters as `log(x)`, or
    `log(x + 1)` when any value is 0; any other column with 15 or more distinct values is left
    out; the rest enter as they are, categorical ones as treatment dummies (first level as
    reference). An intercept comes first.

    Returns:
        `(X, column_names)`.

    Raises:
        ValueError: the design is not finite and full rank.
    """
    n_cells = len(covariates)
    columns = [np.ones(n_cells)]
    names = ["(Intercept)"]
    for name in covariates.columns:
        if name == "response_p_mito":
            continue
        v = covariates[name].reset_index(drop=True)
        if _COUNT_COVARIATE.search(str(name)):
            v = v.to_numpy(dtype=np.float64)
            if np.any(v == 0):
                columns.append(np.log(v + 1.0))
                names.append(f"log({name} + 1)")
            else:
                columns.append(np.log(v))
                names.append(f"log({name})")
            continue
        if v.nunique(dropna=False) >= _MAX_N_LEVELS:
            continue
        if pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v):
            columns.append(v.to_numpy(dtype=np.float64))
            names.append(str(name))
        else:
            cols, cnames = _categorical_columns(str(name), v)
            columns.extend(cols)
            names.extend(cnames)
    X = np.column_stack(columns)
    validate_design_matrix(X, n_cells=n_cells)
    return X, names
