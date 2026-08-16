"""Coarse per-tile focus-index heatmap, click-to-inspect a tile.

Matches troubleshoot_wildtype.ipynb's `plt.imshow(grids, vmin=8, vmax=18);
plt.colorbar()`, except vmin/vmax are derived from the stack's Z range
rather than hardcoded.
"""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget


class CoarseHeatmapPanel(QWidget):
    tile_clicked = Signal(int, int)  # (tile_row, tile_col)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._im = None
        self._colorbar = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.figure = Figure(figsize=(3, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Coarse focus map (click a tile)")
        self.ax.axis("off")
        layout.addWidget(self.canvas, 1)

        self.canvas.mpl_connect("button_press_event", self._on_click)

    def clear(self):
        self._data = None
        # Colorbar must be removed before the image it's attached to --
        # Colorbar.remove() restores the parent axes' subplotspec, which
        # breaks if that axes has already lost its image.
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None
        if self._im is not None:
            self._im.remove()
            self._im = None
        self.ax.set_title("Coarse focus map (click 'Do it' to view)")
        self.canvas.draw_idle()

    def set_data(self, coarse_focal_indices, n_z: int):
        self._data = coarse_focal_indices
        vmin, vmax = 0, max(n_z - 1, 1)
        if self._im is None:
            self._im = self.ax.imshow(coarse_focal_indices, vmin=vmin, vmax=vmax, cmap="viridis")
            self._colorbar = self.figure.colorbar(self._im, ax=self.ax)
        else:
            self._im.set_data(coarse_focal_indices)
            self._im.set_clim(vmin, vmax)
        self.canvas.draw_idle()

    def _on_click(self, event):
        if self._data is None or event.inaxes != self.ax or event.xdata is None:
            return
        j = int(round(event.xdata))
        i = int(round(event.ydata))
        if 0 <= i < self._data.shape[0] and 0 <= j < self._data.shape[1]:
            self.tile_clicked.emit(i, j)
