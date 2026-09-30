"""The specificity check: how many discovered links are more than background.

Not from sceptre, and not validated against R. Named next to sceptre's
calibration and power checks because it answers the question they leave
open. See `docs/design.md`, "Specificity check".
"""

from .background import (
    DEFAULT_DISTANCE_BINS,
    above_background_by_distance,
    background_pairs,
    broad_effect_elements,
    cis_links,
    nearest_tss,
    per_gene_background,
)
from .check import SpecificityResult, run_specificity_check
from .lookup import DETOUR_STATUSES, detour_check, gene_lookup, tss_targets

__all__ = [
    "DEFAULT_DISTANCE_BINS",
    "DETOUR_STATUSES",
    "SpecificityResult",
    "above_background_by_distance",
    "background_pairs",
    "broad_effect_elements",
    "cis_links",
    "detour_check",
    "gene_lookup",
    "nearest_tss",
    "per_gene_background",
    "run_specificity_check",
    "tss_targets",
]
