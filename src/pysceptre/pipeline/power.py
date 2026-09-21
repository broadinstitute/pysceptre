"""Positive-control pair construction for the power check.

A power check runs the *same* discovery test over pairs where an effect is
expected -- a gRNA target and the gene it is known to perturb, typically its
own TSS. Nothing statistical is new; it is a sanity check that the pipeline
detects effects it should, and it is read alongside the calibration check,
which asks the opposite question.

**Two ways to get the pairs, and real screens use the second.** R's
`construct_positive_control_pairs` pairs each gRNA target with the gene of
the same name::

    pc_targets <- unique(grna_target[grna_target %in% response_ids])
    data.frame(grna_target = pc_targets, response_id = pc_targets)

That works when targets are named after the genes they hit. It finds nothing
on a screen whose targets are genomic intervals: on day0_grna20 the rule
matches **0 of 3,071 targets**, because they are named `chr4:55636048-...`
while the genes are `NMU`. Such screens supply the mapping explicitly, and
sceptre stores it in `positive_control_pairs`. Both routes are supported
here; the explicit one is the default because it is the one real data uses.

**QC is reported, not filtered.** This is the opposite of the calibration
check and the same as discovery. Calibration *samples* its pairs, so it can
simply avoid ones that would fail; a positive control is a specific
biological claim about a specific pair, and quietly dropping the ones that
fail QC would overstate power by hiding exactly the pairs the screen had too
few cells to test. R reports all of them: on day0, 266 pairs of which 246
pass and 20 carry a NaN p-value. So does this.

**No multiple-testing correction**, matching R, which returns no
`significant` column for a power check. These pairs are not discoveries;
they are a diagnostic, and adjusting them against each other would answer a
question nobody asked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pairwise_qc import nonzero_counts


def construct_positive_control_pairs(gene_ids: list[str], grna_targets: list[str]) -> pd.DataFrame:
    """R's name-matching rule: a target that is also a gene id pairs with it.

    Returns an empty frame when no target is named after a gene, which is the
    common case for screens that target genomic intervals -- see the module
    docstring. An empty result is not an error; it means the mapping has to
    be supplied.
    """
    genes = set(gene_ids)
    matched = sorted({t for t in grna_targets if t in genes})
    return pd.DataFrame({"response_id": matched, "grna_target": matched})


def annotate_pairwise_qc(
    pairs: pd.DataFrame,
    response_matrix,
    gene_ids: list[str],
    grna_target_cells: dict[str, np.ndarray],
    n_cells: int,
    *,
    n_nonzero_trt_thresh: int = 7,
    n_nonzero_cntrl_thresh: int = 7,
) -> pd.DataFrame:
    """Add `n_nonzero_trt`, `n_nonzero_cntrl` and `pass_qc` to `pairs`.

    The counts come from the same sparse matmul the calibration check uses,
    so the two paths cannot drift apart on what "enough cells" means. The
    control group is the complement, so its count is a subtraction rather than
    a second pass.
    """
    targets = list(dict.fromkeys(pairs["grna_target"]))
    missing = [t for t in targets if t not in grna_target_cells]
    if missing:
        raise KeyError(f"targets not present in grna_target_cells: {missing[:5]}")
    gene_index = {g: i for i, g in enumerate(gene_ids)}
    unknown = [g for g in pairs["response_id"].unique() if g not in gene_index]
    if unknown:
        raise KeyError(f"genes not present in gene_ids: {unknown[:5]}")

    trt, cntrl = nonzero_counts(response_matrix, [grna_target_cells[t] for t in targets], n_cells)
    tcol = {t: j for j, t in enumerate(targets)}
    rows = pairs["response_id"].map(gene_index).to_numpy()
    cols = pairs["grna_target"].map(tcol).to_numpy()

    out = pairs.copy().reset_index(drop=True)
    out["n_nonzero_trt"] = trt[rows, cols]
    out["n_nonzero_cntrl"] = cntrl[rows, cols]
    out["pass_qc"] = (out["n_nonzero_trt"] >= n_nonzero_trt_thresh) & (
        out["n_nonzero_cntrl"] >= n_nonzero_cntrl_thresh
    )
    return out


def merge_qc_failures(tested: pd.DataFrame, annotated: pd.DataFrame) -> pd.DataFrame:
    """Put the QC failures back, with NaN where a result would have been.

    The engine is only ever handed testable pairs, so the failures never
    reach it and have to be reinstated here. They are reinstated rather than
    dropped because a positive control that could not be tested is a fact
    about the screen -- too few cells carry that gRNA, or too few express
    that gene -- and omitting it would make the power check look better than
    the data supports.

    Row order follows `annotated`, so the output lists the positive controls
    in the order they were supplied rather than in whatever order survived
    QC.
    """
    key = ["response_id", "grna_target"]
    merged = annotated.merge(tested, on=key, how="left", suffixes=("", "_res"))
    # A failed pair has no result columns; make that explicit rather than
    # leaving whatever the merge produced.
    result_cols = [c for c in tested.columns if c not in key]
    failed = ~merged["pass_qc"].to_numpy(dtype=bool)
    for col in result_cols:
        if col in merged and merged[col].dtype.kind in "fc":
            merged.loc[failed, col] = np.nan
    return merged
