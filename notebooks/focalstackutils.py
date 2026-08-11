"""Re-export shim.

The field-flattening core logic moved to ``yeastprep.core.focus`` (see
design.md's Feature 1 plan) so it can be shared by the yeastprep UI/CLI as
well as these notebooks. Kept here so existing notebook cells (`import
focalstackutils as F`) keep working unchanged.
"""

from yeastprep.core.focus import (  # noqa: F401
    TileInfo,
    TileVarianceStack,
    compute_tile_variance_stack,
    peaks_from_variance_stack,
    find_focus_index_map,
    fit_focal_indices_to_poly2d,
    resample_focal_slice,
)
