"""Pairwise QC: per (gene, target) nonzero-cell counts.

Shared by the calibration check and the power check, which is why it lives
here rather than in either. Both *construct* the pairs they test -- one from
synthetic groups of non-targeting gRNAs, one from positive controls -- so
both have to decide which of the pairs they built are testable at all. A
discovery analysis is handed its pairs and does not.

This is the narrow carve-out from the "no `run_qc()`" scope rule in
CLAUDE.md. Pairwise nonzero-count filtering is inseparable from building
pairs; cell-level and gRNA-level QC remain out of scope.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _membership_matrix(group_cell_lists: list[np.ndarray], n_cells: int) -> sparse.csc_matrix:
    """(n_cells, n_groups) 0/1 membership, as CSC for the matmul below."""
    n_groups = len(group_cell_lists)
    indptr = np.zeros(n_groups + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([c.size for c in group_cell_lists])
    indices = np.concatenate(group_cell_lists) if n_groups else np.empty(0, dtype=np.int64)
    data = np.ones(indices.size, dtype=np.int32)
    return sparse.csc_matrix((data, indices, indptr), shape=(n_cells, n_groups))


def nonzero_counts(
    response_matrix,
    group_cell_lists: list[np.ndarray],
    n_cells: int,
    gene_chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Per (gene, group): treated and control nonzero-cell counts.

    Returns `(n_nonzero_trt, n_nonzero_cntrl)`, both (n_genes, n_groups) int64.

    This is R's `compute_n_trt_cells_matrix`, done as one sparse-sparse matmul
    of the binarized response matrix against the membership matrix rather than
    a loop over pairs. The control arm is the *complement*, so its count is
    `total nonzero for the gene - treated nonzero` and needs no second pass --
    which is the whole reason the complement control group is cheap here.

    The response matrix is binarized, never densified: only the sparsity
    pattern matters for a nonzero count, so `data` is replaced with ones and
    the (n_genes, n_cells) structure is left alone.
    """
    G = _membership_matrix(group_cell_lists, n_cells)

    def block_counts(block):
        """Treated counts and per-gene totals for a slab of genes."""
        b = sparse.csr_matrix(block) if not sparse.isspmatrix_csr(block) else block
        binary = sparse.csr_matrix(
            (np.ones(b.nnz, dtype=np.int32), b.indices, b.indptr), shape=b.shape
        )
        return np.asarray((binary @ G).todense(), dtype=np.int64), np.diff(binary.indptr).astype(
            np.int64
        )

    if hasattr(response_matrix, "rows"):
        # Backed input: sweep contiguous gene slabs so the full sparsity
        # pattern is never resident. Only the (n_genes, n_groups) counts are
        # kept, which are small -- 38,606 genes x 100 groups of int64 is 31 MB.
        n_genes = response_matrix.shape[0]
        trts, tots = [], []
        for start in range(0, n_genes, gene_chunk):
            t_block, tot_block = block_counts(
                response_matrix.rows(start, min(start + gene_chunk, n_genes))
            )
            trts.append(t_block)
            tots.append(tot_block)
        trt = np.vstack(trts)
        n_nonzero_tot = np.concatenate(tots)
    else:
        trt, n_nonzero_tot = block_counts(response_matrix)

    cntrl = n_nonzero_tot[:, None] - trt
    return trt, cntrl
