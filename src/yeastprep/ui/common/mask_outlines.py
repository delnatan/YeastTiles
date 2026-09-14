"""RGBA outline overlay for a cellpose label image, shared by the Preview,
Segmentation and Tile Generation previews.

Uses cellpose's own `masks_to_outlines` when cellpose is installed, so the
outlines match the Cellpose GUI's exactly. Without it (a
`classification`-only install can still preview masks segmented on another
machine) it falls back to marking each labeled pixel that has a 4-neighbor
with a different label -- the same boundary pixels, give or take the
interior holes cellpose's external-contour trace skips.
"""

import numpy as np

from yeastprep.optional_deps import missing


def _label_boundaries(masks: np.ndarray) -> np.ndarray:
    padded = np.pad(masks, 1)
    center = padded[1:-1, 1:-1]
    differs = (
        (padded[:-2, 1:-1] != center)
        | (padded[2:, 1:-1] != center)
        | (padded[1:-1, :-2] != center)
        | (padded[1:-1, 2:] != center)
    )
    return differs & (masks > 0)


def outline_rgba(masks: np.ndarray, color: tuple[float, float, float, float]) -> np.ndarray:
    if missing("cellpose"):
        outlines = _label_boundaries(masks)
    else:
        from cellpose.utils import masks_to_outlines

        outlines = masks_to_outlines(masks)
    rgba = np.zeros((*masks.shape, 4), dtype=np.float32)
    rgba[outlines] = color
    return rgba
