"""Focal-slice preview with cellpose mask outlines plus each cell's crop
window drawn on top -- the panel used to judge, before batch export,
whether the chosen tile size actually fits each cell (too small clips the
cell; too large wastes tile area / risks overlapping neighbors).
Structurally mirrors segmentation_preview_panel.py."""

import numpy as np
from cellpose.utils import masks_to_outlines
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from tileclass.contrast import compute_percentile_clim
from yeastprep.core.tiles import cell_geometry

_OUTLINE_RGBA = (1.0, 0.85, 0.0, 1.0)  # yellow, matches cellpose GUI's outline color
_BOX_COLOR = "#4da6ff"


class TileGenerationPreviewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._im = None
        self._overlay = None
        self._boxes = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.figure = Figure(figsize=(3, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Crop window preview")
        self.ax.axis("off")

        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def set_data(self, focal_slice: np.ndarray, masks: np.ndarray, tile_size: int):
        vmin, vmax = compute_percentile_clim(focal_slice)
        outline_rgba = np.zeros((*masks.shape, 4), dtype=np.float32)
        outline_rgba[masks_to_outlines(masks)] = _OUTLINE_RGBA

        if self._im is None:
            self._im = self.ax.imshow(focal_slice, cmap="gray", vmin=vmin, vmax=vmax)
            self._overlay = self.ax.imshow(outline_rgba)
        else:
            self._im.set_data(focal_slice)
            self._im.set_clim(vmin, vmax)
            self._overlay.set_data(outline_rgba)

        for box in self._boxes:
            box.remove()
        self._boxes = self._draw_crop_boxes(masks, tile_size)

        self.canvas.draw_idle()

        n_cells = int(masks.max())
        self.status_label.setText(
            f"{n_cells} cell{'s' if n_cells != 1 else ''} -> {n_cells} tile(s) queued for export"
        )

    def _draw_crop_boxes(self, masks: np.ndarray, tile_size: int) -> list:
        boxes = []
        for geom in cell_geometry(masks, tile_size):
            row_src, col_src = geom.src
            row_dst, col_dst = geom.dst
            # Reconstruct the *unclipped* window origin from src/dst: the
            # window's top-left in image coordinates is src.start - dst.start.
            y0 = row_src.start - row_dst.start
            x0 = col_src.start - col_dst.start
            rect = Rectangle(
                (x0, y0),
                tile_size,
                tile_size,
                linewidth=0.8,
                edgecolor=_BOX_COLOR,
                facecolor="none",
            )
            self.ax.add_patch(rect)
            boxes.append(rect)
        return boxes

    def clear(self):
        for box in self._boxes:
            box.remove()
        self._boxes = []
        self._im = None
        self._overlay = None
        self.ax.clear()
        self.ax.set_title("Crop window preview")
        self.ax.axis("off")
        self.canvas.draw_idle()
        self.status_label.setText("")
