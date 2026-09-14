"""Focal-slice preview with cellpose mask outlines overlaid -- the panel
used to judge segmentation quality, whether that's from a live in-session
recompute or a saved mask loaded straight off disk.

Built on the shared vispy-based `SingleImagePreviewPanel` (see
`ui/common/single_image_preview.py`), using `RawStackCanvas.set_overlay_mask`
for the outline instead of a second matplotlib `imshow` layer. `set_data`
keeps its original signature so `segmentation_page.py` needs no changes
beyond this import.
"""

import numpy as np

from ..common.mask_outlines import outline_rgba
from ..common.single_image_preview import SingleImagePreviewPanel

_OUTLINE_RGBA = (1.0, 0.85, 0.0, 1.0)  # yellow, matches cellpose GUI's outline color


class SegmentationPreviewPanel(SingleImagePreviewPanel):
    def set_data(self, focal_slice: np.ndarray, masks: np.ndarray):
        self.set_image(focal_slice)

        self.canvas.set_overlay_mask(outline_rgba(masks, _OUTLINE_RGBA))

        n_cells = int(masks.max())
        self.status_label.setText(f"{n_cells} cell{'s' if n_cells != 1 else ''} detected")
