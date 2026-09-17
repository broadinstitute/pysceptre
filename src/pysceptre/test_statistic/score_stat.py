"""Port of `compute_observed_full_statistic_v2` / `compute_null_full_statistics`
(low_level_full_test.cpp): the O(n_trt)-per-resample score-type test statistic
that lets sceptre evaluate B resampled treatment assignments without
refitting a GLM for each one.

  top         = sum_{i in trt} a[i]
  lower_left  = sum_{i in trt} w[i]
  lower_right = sum_k ( sum_{i in trt} D[k, i] )^2
  z = top / sqrt(lower_left - lower_right)

All indices here are 0-based (unlike the R/C++ source, which is 1-based).

`compute_null_full_statistics` computes this for every one of the B
(ragged-length) synthetic index sets. A direct port would loop over B in
Python, but at real scale (B up to ~5,498, called once per (gene, target)
pair, ~34,886 pairs) that loop dominates runtime -- this instead flattens
the B ragged arrays into one long array plus a per-element "which resample"
label, and computes all B sums at once via `np.bincount` (segment sums),
which does the O(B)-scale work in C rather than a Python loop. The
flattening (`flatten_synthetic_idxs`) only depends on the target's
synthetic draws, not the gene, so callers that reuse the same
`synthetic_idxs` across many genes (see pipeline/discovery.py) should
flatten once and reuse the result via `compute_null_full_statistics_flat`.
"""

from __future__ import annotations

import numpy as np


def compute_observed_full_statistic(a: np.ndarray, w: np.ndarray, D: np.ndarray, trt_idxs: np.ndarray) -> float:
    top = a[trt_idxs].sum()
    lower_left = w[trt_idxs].sum()
    lower_right = np.square(D[:, trt_idxs].sum(axis=1)).sum()
    return float(top / np.sqrt(lower_left - lower_right))


def flatten_synthetic_idxs(synthetic_idxs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns (flat_idxs, segment_ids, B): flat_idxs is the concatenation of
    all B ragged index arrays, segment_ids labels each element of flat_idxs
    with which of the B resamples it belongs to (for use with np.bincount)."""
    B = len(synthetic_idxs)
    lengths = np.fromiter((len(idxs) for idxs in synthetic_idxs), dtype=np.int64, count=B)
    if lengths.sum() == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), B
    flat_idxs = np.concatenate(synthetic_idxs)
    segment_ids = np.repeat(np.arange(B, dtype=np.int64), lengths)
    return flat_idxs, segment_ids, B


def compute_null_full_statistics_flat(a: np.ndarray, w: np.ndarray, D: np.ndarray, flat_idxs: np.ndarray, segment_ids: np.ndarray, B: int) -> np.ndarray:
    """Vectorized core: see module docstring. `(flat_idxs, segment_ids, B)` is
    the output of `flatten_synthetic_idxs`, which callers looping over many
    genes for the same target's synthetic draws should compute once and reuse."""
    if flat_idxs.size == 0:
        return np.full(B, np.nan)
    top = np.bincount(segment_ids, weights=a[flat_idxs], minlength=B)
    lower_left = np.bincount(segment_ids, weights=w[flat_idxs], minlength=B)
    # per-row (covariate-dimension) segment sums of D, then sum of squares over rows
    D_row_sums = np.stack([np.bincount(segment_ids, weights=D[k, flat_idxs], minlength=B) for k in range(D.shape[0])])
    lower_right = np.sum(D_row_sums**2, axis=0)
    return top / np.sqrt(lower_left - lower_right)


def compute_null_full_statistics(a: np.ndarray, w: np.ndarray, D: np.ndarray, synthetic_idxs: list[np.ndarray]) -> np.ndarray:
    """synthetic_idxs: a list of B (possibly ragged, i.e. differently-sized)
    0-based integer index arrays -- one synthetic treated-cell set per resample.
    This targets sceptre's `use_all_cells=True` case (the only one the
    complement-control-group CRT path uses): each resample's own set size is
    used directly, rather than a fixed external n_trt.

    Convenience wrapper around `compute_null_full_statistics_flat` for callers
    with a single synthetic_idxs list to consume once; prefer flattening once
    and calling `compute_null_full_statistics_flat` directly when the same
    synthetic_idxs will be reused across many (a, w, D) triples."""
    flat_idxs, segment_ids, B = flatten_synthetic_idxs(synthetic_idxs)
    return compute_null_full_statistics_flat(a, w, D, flat_idxs, segment_ids, B)
