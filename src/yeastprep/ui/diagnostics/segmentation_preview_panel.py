"""Focal-slice preview with cellpose mask outlines overlaid -- the panel
used to judge segmentation quality, whether that's from a live in-session
recompute or a saved mask loaded straight off disk."""

import numpy as np
from cellpose.utils import masks_to_outlines
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget

from tileclass.contrast import compute_percentile_clim

_OUTLINE_RGBA = (1.0, 0.85, 0.0, 1.0)  # yellow, matches cellpose GUI's outline color


class SegmentationPreviewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._im = None
        self._overlay = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.figure = Figure(figsize=(3, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Segmentation preview")
        self.ax.axis("off")

        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def set_data(self, focal_slice: np.ndarray, masks: np.ndarray):
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
        self.canvas.draw_idle()

        n_cells = int(masks.max())
        self.status_label.setText(f"{n_cells} cell{'s' if n_cells != 1 else ''} detected")
