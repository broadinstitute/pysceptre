"""Rank of a design matrix, which of its columns are redundant, and its indicator structure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def redundant_columns(X: np.ndarray) -> tuple[int, list[int]]:
    """`(rank, redundant)` for a finite `(n, p)` design matrix.

    `redundant` lists, left to right, the columns that are linear combinations
    of the ones before them -- the columns R's `glm.fit` would report as NA.
    It is empty when `X` has full column rank.

    Rank comes from the singular values of `R` in a thin QR of `X`, with the
    relative tolerance `numpy.linalg.matrix_rank` uses. Never from `X.T @ X`,
    whose eigenvalues square the condition number. See docs/design.md, "A rank
    check that does not square the condition number".
    """
    n, p = X.shape
    eps = np.finfo(float).eps
    r = np.linalg.qr(np.asarray(X, dtype=float), mode="r")

    def _rank(block: np.ndarray) -> int:
        sv = np.linalg.svd(block, compute_uv=False)
        if sv.size == 0 or sv[0] <= 0:
            return 0
        return int(np.sum(sv > sv[0] * max(n, block.shape[1]) * eps))

    rank = _rank(r)
    if rank == p:
        return rank, []
    kept: list[int] = []
    for j in range(p):
        trial = kept + [j]
        if _rank(r[:, trial]) == len(trial):
            kept.append(j)
    return rank, [j for j in range(p) if j not in kept]


def targets_in_span(
    X: np.ndarray,
    cells_of: dict[str, np.ndarray],
    weights_of: dict[str, np.ndarray] | None = None,
) -> set[str]:
    """The targets whose treated indicator (or dose weights) the columns of `X` span.

    A target is flagged when the share of its indicator outside the span,
    `1 - |Q_T' t|^2 / |t|^2` with `Q` from a thin QR of `X`, is under `sqrt(eps)`.
    `Q` is orthonormal, so the test does not depend on how `X` is scaled. Targets
    with no cells are never flagged. docs/design.md, "A target the covariates explain
    is not tested".
    """
    cells_of = {t: c for t, c in cells_of.items() if len(c)}
    if not cells_of:
        return set()
    q = np.linalg.qr(np.asarray(X, dtype=float), mode="reduced")[0]
    tol = np.sqrt(np.finfo(float).eps)
    out = set()
    for t, cells in cells_of.items():
        w = np.ones(len(cells)) if weights_of is None else np.asarray(weights_of[t], dtype=float)
        s = q[cells].T @ w
        if 1.0 - (s @ s) / (w @ w) < tol:
            out.add(t)
    return out


def validate_design_matrix(covariate_matrix: np.ndarray, n_cells: int | None = None) -> None:
    """Refuse a design matrix the GLM cannot fit, and say which columns are at fault.

    Without this a rank-deficient design fails deep inside the batched weighted
    least squares as a bare `LinAlgError`, with no mention of covariates. The
    rank test is `glm.design.redundant_columns`.

    Raises:
        ValueError: not 2-D, empty, non-finite entries, a row count that
            disagrees with the data, or rank deficiency.
    """
    X = np.asarray(covariate_matrix)
    if X.ndim != 2:
        raise ValueError(f"covariate_matrix must be 2-D (n_cells, p), got shape {X.shape}")
    n, p = X.shape
    if n == 0 or p == 0:
        raise ValueError(f"covariate_matrix is empty, shape {X.shape}")
    if n_cells is not None and n != n_cells:
        raise ValueError(
            f"covariate_matrix has {n} rows but the response matrix has {n_cells} cells"
        )
    if p > n:
        raise ValueError(
            f"covariate_matrix has shape {X.shape}, more columns ({p}) than rows ({n}), so it "
            "cannot be full rank. It is expected as (n_cells, p); this looks transposed."
        )
    X = X.astype(float, copy=False)
    if not np.all(np.isfinite(X)):
        bad = np.flatnonzero(~np.all(np.isfinite(X), axis=0)).tolist()
        raise ValueError(f"covariate_matrix has non-finite values in column(s) {bad[:10]}")

    # Redundant columns are named left to right, so the first of a collinear set is kept --
    # R's convention, and the useful one, since column 0 is usually the intercept.
    rank, redundant = redundant_columns(X)
    if rank == 0:
        raise ValueError("covariate_matrix is all zeros")
    if not redundant:
        return
    raise ValueError(
        f"covariate_matrix is rank deficient: rank {rank} of {p} columns. "
        f"Column(s) {redundant} are linear combinations of the ones before them, so the "
        "weighted least squares inside the GLM fit is singular. This is usually an aliased "
        "contrast (a factor's full dummy set alongside an intercept), a duplicated column, or "
        "a covariate that is constant within another's levels. Drop the named column(s), or "
        "build the design with one reference level held out."
    )


@dataclass(frozen=True)
class IndicatorGroup:
    """Columns of a design matrix that are 0/1 and never 1 in the same row, typically one factor's dummies.

    `columns` are their positions in the design matrix. `codes[i]` is the
    position within `columns` of the column row `i` has its 1 in, or
    `len(columns)` when it has none (the reference level).
    """

    columns: np.ndarray
    codes: np.ndarray

    @property
    def n_levels(self) -> int:
        """Columns in the group; the reference level is not one of them."""
        return int(self.columns.size)


@dataclass(frozen=True)
class StructuredDesign:
    """A design matrix as its dense columns plus groups of indicator columns.

    `dense` are the positions of the columns in no group, the intercept among
    them, and `X_dense` is `X[:, dense]`, C-contiguous. Built by
    `structured_design`; the IRLS in `glm.irls` uses it to form the weighted
    normal equations without the `(n, p^2)` array of per-row outer products.
    """

    n: int
    p: int
    dense: np.ndarray
    X_dense: np.ndarray
    groups: tuple[IndicatorGroup, ...]
    # Every group's `codes` as the rows of one `(n_groups, n)` array, and each
    # group's `n_levels`: the layout the compiled per-level sums read.
    codes_matrix: np.ndarray
    levels: np.ndarray

    def to_matrix(self) -> np.ndarray:
        """The `(n, p)` design matrix this was built from."""
        X = np.zeros((self.n, self.p))
        X[:, self.dense] = self.X_dense
        for g in self.groups:
            rows = g.codes < g.n_levels
            X[np.flatnonzero(rows), g.columns[g.codes[rows]]] = 1.0
        return X


def structured_design(X: np.ndarray) -> StructuredDesign:
    """Split `X` into dense columns and groups of mutually exclusive 0/1 columns.

    A column is an indicator when every entry is exactly 0 or 1 and it is
    neither all 0 nor all 1. Indicators are grouped greedily, left to right:
    each joins the first group none of whose columns is 1 in a row where it
    is 1, or starts a new one. A factor's dummies with a reference level held
    out form one group, and factors whose dummies share rows form separate
    groups; a column of a partially crossed factor can join another factor's
    group, which changes only how many pair tables there are. Every other
    column, the intercept among them, is dense.

    The split is a property of `X` alone and changes no fitted value in exact
    arithmetic: the normal equations are the same sums, accumulated in another
    order. See docs/design.md, "Indicator columns in the IRLS".
    """
    X = np.ascontiguousarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    n, p = X.shape
    dense: list[int] = []
    members: list[list[int]] = []
    masks: list[list[np.ndarray]] = []
    occupied: list[np.ndarray] = []
    for j in range(p):
        col = X[:, j]
        ones = col == 1.0
        n_ones = int(np.count_nonzero(ones))
        if not (0 < n_ones < n) or not bool(np.all(ones | (col == 0.0))):
            dense.append(j)
            continue
        for g, occ in enumerate(occupied):
            if not np.any(occ & ones):
                members[g].append(j)
                masks[g].append(ones)
                occ |= ones
                break
        else:
            members.append([j])
            masks.append([ones])
            occupied.append(ones.copy())
    codes_matrix = np.empty((len(members), n), dtype=np.int64)
    groups = []
    for g, (cols, group_masks) in enumerate(zip(members, masks, strict=True)):
        codes = codes_matrix[g]
        codes[:] = len(cols)
        for level, ones in enumerate(group_masks):
            codes[ones] = level
        groups.append(IndicatorGroup(columns=np.asarray(cols, dtype=np.int64), codes=codes))
    dense_idx = np.asarray(dense, dtype=np.int64)
    return StructuredDesign(
        n=n,
        p=p,
        dense=dense_idx,
        X_dense=np.ascontiguousarray(X[:, dense_idx]),
        groups=tuple(groups),
        codes_matrix=codes_matrix,
        levels=np.asarray([len(c) for c in members], dtype=np.int64),
    )
