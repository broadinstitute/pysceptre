"""Top-level public entry point.

Assumes gRNA assignment and QC have already happened upstream (out of
scope -- this package is the statistical engine only, not sceptre's
`assign_grnas`/`run_qc`), and that the covariate matrix is already a plain
numeric design matrix (no `model.matrix`-equivalent formula DSL -- exact
factor-contrast parity with R across languages is its own project, and
orthogonal to the statistical engine targeted here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .discovery import _DEFAULT_TARGET_CHUNK_SIZE, run_discovery_ntcells_complement


def run_discovery_analysis(
    response_matrix,
    gene_ids: list[str],
    covariate_matrix: np.ndarray,
    grna_target_cells: dict[str, np.ndarray],
    pairs: pd.DataFrame,
    *,
    side: str = "both",
    resampling_approximation: str = "skew_normal",
    seed: int | None = None,
    target_chunk_size: int = _DEFAULT_TARGET_CHUNK_SIZE,
) -> pd.DataFrame:
    """response_matrix: (n_genes, n_cells) dense ndarray or scipy.sparse matrix.
    gene_ids: row labels for response_matrix, in order.
    covariate_matrix: (n_cells, p) already formula-expanded design matrix.
    grna_target_cells: dict[target -> 0-based treated-cell indices].
    pairs: DataFrame['response_id', 'grna_target'] -- QC-passed pairs to test.
    target_chunk_size: how many gRNA targets' logistic fits + CRT draws to
        batch/hold in memory at once (see pipeline/discovery.py) -- lower this
        if you hit memory pressure, raise it for a modest speed gain if you
        have memory to spare.

    Targets sceptre's complement-control-group + CRT discovery-analysis path
    (the only valid combination for high-MOI data -- see pipeline/discovery.py).
    """
    side_code = {"left": -1, "both": 0, "right": 1}[side]
    fit_parametric_curve = resampling_approximation == "skew_normal"

    return run_discovery_ntcells_complement(
        response_matrix=response_matrix,
        gene_ids=gene_ids,
        covariate_matrix=covariate_matrix,
        grna_target_cells=grna_target_cells,
        pairs=pairs,
        fit_parametric_curve=fit_parametric_curve,
        side_code=side_code,
        seed=seed,
        target_chunk_size=target_chunk_size,
    )
