"""Negative-control pair construction for the calibration check.

A calibration check runs the *same* CRT machinery as discovery, over synthetic
negative-control targets built by regrouping individual non-targeting (NTC)
gRNAs. Nothing statistical is new here; what this module adds is construction
and the pairwise QC that construction is filtered by.

**The structural difference from discovery.** Discovery pairs are given by the
caller: R tests all of them and reports QC failures in-band, with
`pass_qc = FALSE` and a NaN p-value (measured on a 567,690-cell screen: 34,256 rows of which
1,121 fail). Calibration pairs are *constructed*: R samples only combinations
that already clear `n_nonzero_trt_thresh` / `n_nonzero_cntrl_thresh`, so every
returned row passes (measured on the same screen: 33,135 rows, zero
failures, zero NaN).
QC is therefore a filter on construction here, not a reported outcome, which is
why this module has to know the thresholds at all.

That is a deliberate, narrow carve-out from the "no `run_qc()`" scope rule in
CLAUDE.md: pairwise nonzero-count filtering is intrinsic to *building* negative
control pairs and cannot be separated from it. Cell-level and gRNA-level QC
remain out of scope.

**R's group count, recovered empirically.** `sample_combinations_v2` is C++
(`.Call(_sceptre_sample_combinations_v2)`), so its rule cannot be read from an
installed sceptre. It can be *called*, though, so the rule was recovered by
probing it across the argument space (sceptre 0.10.3):

    n_groups = max(100, ceil(5 * n_calibration_pairs / (n_genes * pass_qc_rate)))

verified exactly on eight cases spanning the floor and well above it --
(33,135 pairs, 9,045 genes, 0.967) -> 100, (500,000, 9,045, 0.967) -> 286,
(33,135, 100, 0.967) -> 1,714, (33,135, 10, 0.9) -> 18,409, and the floor
boundary at (271,350, 9,045, 1.0) -> 150 against (180,000, 9,045, 1.0) -> 100.
The 100 floor is sceptre's `N_POSSIBLE_GROUPS_THRESHOLD`; the factor 5
oversamples candidate pairs so enough survive QC.

**Confirmed independently on a real dataset.** The day0 object had already been
through `run_calibration_check` in R and stores the pairs it used: 625 groups,
over 292 genes, for 34,886 pairs, with 2,031 NTC gRNAs -- every parameter
different from the probe grid, and far off the floor. The rule reproduces
625 exactly,
but only when `pass_qc_rate` is R's own `mean(discovery_pairs_with_info$pass_qc)`
= 0.9571; assuming 1.0 gives 598. So the rate matters whenever the result is
not pinned to the floor, and the dataset export records it.

**What is *not* reproducible against R, and why that is fine.** Nothing in
sceptre's calibration path calls `set.seed`, and `sceptre_object` has no seed
slot, so R's own choice of pairs varies run to run. Two R runs of the same
object here produced the identical 100 groups but shared only about 20 of each
group's 331 genes. So a pair-by-pair comparison against R is only meaningful
when R's pairs are *injected* (see `negative_control_pairs_from_names`); the
constructor below is validated distributionally instead, which is what a
calibration check measures anyway.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from .pairwise_qc import nonzero_counts

# sceptre's N_POSSIBLE_GROUPS_THRESHOLD, which doubles as the floor on the
# number of synthetic groups.
_N_GROUPS_FLOOR = 100
# Oversampling factor on candidate pairs, recovered from sample_combinations_v2.
_CANDIDATE_MULTIPLIER = 5

# R joins the member gRNA ids with "&" to name a synthetic target, e.g.
# "guide_37990&guide_38049&...". Kept identical so a pysceptre result and an R
# result can be merged on (response_id, grna_target) without translation.
GROUP_NAME_SEPARATOR = "&"


def n_synthetic_groups(
    n_calibration_pairs: int,
    n_genes: int,
    pass_qc_rate: float = 1.0,
) -> int:
    """How many synthetic negative-control targets to build.

    Ports the rule recovered from R's `sample_combinations_v2` (see the module
    docstring). `pass_qc_rate` is R's `p_hat`: the fraction of candidate pairs
    expected to clear pairwise QC, which R estimates as
    `mean(discovery_pairs_with_info$pass_qc)` and falls back to 0.1 for when
    there are no discovery pairs to estimate from.

    It defaults to 1.0 only because pysceptre is handed pairs that already
    passed QC and so cannot recompute the rate. **Pass the real rate when you
    have it** -- the dataset export records it as
    `metadata["discovery_pass_qc_rate"]`. The default is harmless when the
    group count is pinned to its floor of 100,
    where R's 0.967 and 1.0 both land on the floor of 100, and wrong on day0,
    where R's 0.9571 gives its actual 625 groups and 1.0 gives 598.
    """
    if n_calibration_pairs <= 0:
        raise ValueError(f"n_calibration_pairs must be positive, got {n_calibration_pairs}")
    if n_genes <= 0:
        raise ValueError(f"n_genes must be positive, got {n_genes}")
    if not 0.0 < pass_qc_rate <= 1.0:
        raise ValueError(f"pass_qc_rate must be in (0, 1], got {pass_qc_rate}")
    needed = int(np.ceil(_CANDIDATE_MULTIPLIER * n_calibration_pairs / (n_genes * pass_qc_rate)))
    return max(_N_GROUPS_FLOOR, needed)


def sample_ntc_groups(
    ntc_grna_ids: list[str],
    calibration_group_size: int,
    n_groups: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    """Draw `n_groups` distinct combinations of `calibration_group_size` NTC gRNAs.

    Members are returned as indices into `ntc_grna_ids`, sorted ascending, so
    the "&"-joined name is a canonical form -- two draws of the same set name
    the same target, and the name can be split back into members.

    Distinctness is enforced with a set of frozensets rather than by rejection
    on the name string, so it costs one hash per draw. When the number of
    possible combinations is small enough that distinct draws become hard to
    find, this gives up rather than spinning: `choose(n, k)` is astronomically
    large for any realistic screen (choose(1499, 15) for 1,499 NTC gRNAs in
    groups of 15), so exhausting it
    means the caller asked for something degenerate.
    """
    n_ntc = len(ntc_grna_ids)
    if calibration_group_size < 1:
        raise ValueError(f"calibration_group_size must be >= 1, got {calibration_group_size}")
    if calibration_group_size > n_ntc:
        raise ValueError(
            f"calibration_group_size ({calibration_group_size}) exceeds the number of "
            f"non-targeting gRNAs available ({n_ntc}). R caps the size at the NTC count; "
            "pass a smaller calibration_group_size."
        )

    seen: set[frozenset[int]] = set()
    groups: list[list[int]] = []
    # Generous but finite: distinct draws are near-certain in practice, and the
    # bound only matters in the degenerate small-`choose` case.
    max_attempts = 50 * n_groups + 1000
    for _ in range(max_attempts):
        if len(groups) == n_groups:
            break
        members = rng.choice(n_ntc, size=calibration_group_size, replace=False)
        key = frozenset(int(m) for m in members)
        if key in seen:
            continue
        seen.add(key)
        groups.append(sorted(int(m) for m in members))
    if len(groups) < n_groups:
        raise RuntimeError(
            f"could only draw {len(groups)} distinct groups of {calibration_group_size} "
            f"from {n_ntc} non-targeting gRNAs after {max_attempts} attempts; "
            f"{n_groups} were requested"
        )
    return groups


def build_ntc_groups(
    ntc_grna_ids: list[str],
    calibration_group_size: int,
    n_calibration_pairs: int,
    n_genes: int,
    pass_qc_rate: float = 1.0,
    rng: np.random.Generator | None = None,
) -> list[list[int]]:
    """Choose the synthetic negative-control groups, as R does: enumerate or sample.

    R branches on how many combinations exist at all::

        n_possible <- choose(n_nt_grnas, calibration_group_size)
        if (n_possible <= 100) iterate_over_combinations(...)   # every one
        else                   sample_combinations_v2(...)      # a sample

    so with few enough NTC gRNAs the group set is *exhaustive and
    deterministic*, not random, and is exactly `choose(n, k)` groups -- which
    can be fewer than the floor of 100, because you cannot draw 100 distinct
    groups out of 6. Only the sampling branch uses the count rule in
    `n_synthetic_groups`.

    Missing this branch is not academic: with 4 NTC gRNAs in groups of 2 there
    are 6 possible groups, and asking for 100 distinct ones cannot terminate.
    """
    n_ntc = len(ntc_grna_ids)
    if calibration_group_size < 1:
        raise ValueError(f"calibration_group_size must be >= 1, got {calibration_group_size}")
    if calibration_group_size > n_ntc:
        raise ValueError(
            f"calibration_group_size ({calibration_group_size}) exceeds the number of "
            f"non-targeting gRNAs available ({n_ntc}). R caps the size at the NTC count; "
            "pass a smaller calibration_group_size."
        )
    n_possible = math.comb(n_ntc, calibration_group_size)
    if n_possible <= _N_GROUPS_FLOOR:
        return [list(c) for c in itertools.combinations(range(n_ntc), calibration_group_size)]
    n_groups = n_synthetic_groups(n_calibration_pairs, n_genes, pass_qc_rate)
    return sample_ntc_groups(
        ntc_grna_ids,
        calibration_group_size,
        n_groups,
        np.random.default_rng() if rng is None else rng,
    )


def group_name(member_ids: list[str]) -> str:
    """R's `get_undercover_group_names`: member ids joined with "&"."""
    return GROUP_NAME_SEPARATOR.join(member_ids)


def group_cells(
    member_ids: list[str],
    ntc_grna_cells: dict[str, np.ndarray],
    n_cells: int,
) -> np.ndarray:
    """Treated cells of a synthetic target: the *union* of its members' cells.

    R does `unique(unlist(indiv_nt_grna_idxs[members]))`. A cell carrying two
    of the group's gRNAs is one treated cell, not two, which is why this
    deduplicates rather than concatenating.
    """
    missing = [g for g in member_ids if g not in ntc_grna_cells]
    if missing:
        raise KeyError(f"non-targeting gRNAs not present in ntc_grna_cells: {missing[:5]}")
    if not member_ids:
        return np.empty(0, dtype=np.int64)
    union = np.unique(np.concatenate([ntc_grna_cells[g] for g in member_ids]))
    if union.size and (union[-1] >= n_cells or union[0] < 0):
        raise ValueError(
            f"NTC gRNA cell indices out of range for n_cells={n_cells}: [{union[0]}, {union[-1]}]"
        )
    return union.astype(np.int64, copy=False)


def build_negative_control_pairs(
    response_matrix,
    gene_ids: list[str],
    ntc_grna_cells: dict[str, np.ndarray],
    n_cells: int,
    *,
    n_calibration_pairs: int,
    calibration_group_size: int,
    n_nonzero_trt_thresh: int = 7,
    n_nonzero_cntrl_thresh: int = 7,
    pass_qc_rate: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Build synthetic negative-control targets and the pairs to test.

    Returns `(synthetic_target_cells, pairs)`, shaped exactly like the
    `grna_target_cells` / `pairs` arguments `run_discovery_analysis` takes, so
    the caller hands them straight to the existing engine.

    Only pairs clearing both thresholds are eligible, and `n_calibration_pairs`
    are then drawn from the eligible set uniformly without replacement. If
    fewer are eligible than requested, all of them are returned and the caller
    can see the shortfall in `len(pairs)` -- raising would throw away a usable
    (if smaller) calibration check, and R likewise proceeds on what it finds,
    erroring only when *nothing* passes.

    The default thresholds are sceptre's defaults; they are exported alongside
    the dataset, so prefer passing the object's own values over relying on
    these.
    """
    rng = np.random.default_rng() if rng is None else rng
    ntc_ids = list(ntc_grna_cells)
    if not ntc_ids:
        raise ValueError(
            "no non-targeting gRNAs supplied; a calibration check needs individual "
            "NTC gRNAs to regroup (grna_target_cells collapses them into one target)"
        )
    n_genes = len(gene_ids)
    if response_matrix.shape[0] != n_genes:
        raise ValueError(
            f"response_matrix has {response_matrix.shape[0]} rows but {n_genes} gene_ids"
        )

    member_idx = build_ntc_groups(
        ntc_ids, calibration_group_size, n_calibration_pairs, n_genes, pass_qc_rate, rng
    )

    names = [group_name([ntc_ids[i] for i in members]) for members in member_idx]
    cells = [
        group_cells([ntc_ids[i] for i in members], ntc_grna_cells, n_cells)
        for members in member_idx
    ]

    trt, cntrl = nonzero_counts(response_matrix, cells, n_cells)
    eligible = (trt >= n_nonzero_trt_thresh) & (cntrl >= n_nonzero_cntrl_thresh)
    gene_rows, group_cols = np.nonzero(eligible)
    if gene_rows.size == 0:
        raise ValueError(
            "no negative control pair passes pairwise QC "
            f"(thresholds trt >= {n_nonzero_trt_thresh}, cntrl >= {n_nonzero_cntrl_thresh})"
        )

    take = min(n_calibration_pairs, gene_rows.size)
    chosen = rng.choice(gene_rows.size, size=take, replace=False)
    gene_rows, group_cols = gene_rows[chosen], group_cols[chosen]

    used = np.unique(group_cols)
    synthetic_target_cells = {names[j]: cells[j] for j in used}
    pairs = pd.DataFrame(
        {
            "response_id": [gene_ids[i] for i in gene_rows],
            "grna_target": [names[j] for j in group_cols],
        }
    )
    return synthetic_target_cells, pairs


def negative_control_pairs_from_names(
    pairs: pd.DataFrame,
    ntc_grna_cells: dict[str, np.ndarray],
    n_cells: int,
) -> dict[str, np.ndarray]:
    """Rebuild synthetic targets from "&"-joined names in an existing result.

    This is the validation path. R's calibration pair selection is unseeded and
    so differs run to run (module docstring), which makes a pair-by-pair
    comparison meaningless unless both sides test the *same* pairs. Feeding R's
    own `grna_target` column back through this reconstructs its groups exactly,
    so the engine can be compared on identical input -- the same principle as
    exporting the object R analyses rather than re-deriving it.
    """
    out: dict[str, np.ndarray] = {}
    for name in pd.unique(pairs["grna_target"].astype(str)):
        members = name.split(GROUP_NAME_SEPARATOR)
        out[name] = group_cells(members, ntc_grna_cells, n_cells)
    return out
