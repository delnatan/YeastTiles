"""Full-resolution fitted focus surface, click-to-inspect a tile.

Matches troubleshoot_wildtype.ipynb's `plt.imshow(focus_indices)`. Clicks
are mapped back to the containing coarse tile via TileInfo so this panel
drives the same tile-variance inspector as CoarseHeatmapPanel.
"""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget


class FocusSurfacePanel(QWidget):
    tile_clicked = Signal(int, int)  # (tile_row, tile_col)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tile_info = None
        self._im = None
        self._colorbar = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.figure = Figure(figsize=(3, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Fitted focus surface")
        self.ax.axis("off")
        layout.addWidget(self.canvas, 1)

        self.canvas.mpl_connect("button_press_event", self._on_click)

    def clear(self):
        self._tile_info = None
        # Colorbar must be removed before the image it's attached to --
        # Colorbar.remove() restores the parent axes' subplotspec, which
        # breaks if that axes has already lost its image.
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None
        if self._im is not None:
            self._im.remove()
            self._im = None
        self.ax.set_title("Fitted focus surface (click 'Do it' to view)")
        self.canvas.draw_idle()

    def set_data(self, fine_focus_indices, tile_info):
        self._tile_info = tile_info
        if self._im is None:
            self._im = self.ax.imshow(fine_focus_indices, cmap="viridis")
            self._colorbar = self.figure.colorbar(self._im, ax=self.ax)
        else:
            self._im.set_data(fine_focus_indices)
            self._im.set_clim(fine_focus_indices.min(), fine_focus_indices.max())
        self.canvas.draw_idle()

    def _on_click(self, event):
        if self._tile_info is None or event.inaxes != self.ax or event.xdata is None:
            return
        j = int(event.xdata // self._tile_info.tile_width)
        i = int(event.ydata // self._tile_info.tile_height)
        self.tile_clicked.emit(i, j)
