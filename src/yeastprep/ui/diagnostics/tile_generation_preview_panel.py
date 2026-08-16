"""Focal-slice preview with cellpose mask outlines plus each cell's crop
window drawn on top -- the panel used to judge, before batch export,
whether the chosen tile size actually fits each cell (too small clips the
cell; too large wastes tile area / risks overlapping neighbors).

Built on the shared vispy-based `SingleImagePreviewPanel` (see
`ui/common/single_image_preview.py`), using `RawStackCanvas`'s
`set_overlay_mask`/`set_overlay_boxes` instead of a second matplotlib
`imshow` layer plus `Rectangle` patches. `set_data`/`clear` keep their
original signatures so `tile_generation_page.py` needs no changes beyond
this import.
"""

import numpy as np
from cellpose.utils import masks_to_outlines

from ..common.single_image_preview import SingleImagePreviewPanel
from yeastprep.core.tiles import cell_geometry

_OUTLINE_RGBA = (1.0, 0.85, 0.0, 1.0)  # yellow, matches cellpose GUI's outline color
_BOX_COLOR = "#4da6ff"


class TileGenerationPreviewPanel(SingleImagePreviewPanel):
    def set_data(self, focal_slice: np.ndarray, masks: np.ndarray, tile_size: int):
        self.set_image(focal_slice)

        outline_rgba = np.zeros((*masks.shape, 4), dtype=np.float32)
        outline_rgba[masks_to_outlines(masks)] = _OUTLINE_RGBA
        self.canvas.set_overlay_mask(outline_rgba)
        self.canvas.set_overlay_boxes(self._crop_boxes(masks, tile_size), color=_BOX_COLOR)

        n_cells = int(masks.max())
        self.status_label.setText(
            f"{n_cells} cell{'s' if n_cells != 1 else ''} -> {n_cells} tile(s) queued for export"
        )

    def _crop_boxes(
        self, masks: np.ndarray, tile_size: int
    ) -> list[tuple[float, float, float, float]]:
        boxes = []
        for geom in cell_geometry(masks, tile_size):
            row_src, col_src = geom.src
            row_dst, col_dst = geom.dst
            # Reconstruct the *unclipped* window origin from src/dst: the
            # window's top-left in image coordinates is src.start - dst.start.
            y0 = row_src.start - row_dst.start
            x0 = col_src.start - col_dst.start
            boxes.append((x0, y0, tile_size, tile_size))
        return boxes
