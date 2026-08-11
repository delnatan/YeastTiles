"""Resampled focal-plane preview -- the panel most directly useful for
judging whether `offset_um` is right."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QVBoxLayout, QWidget

from tileclass.contrast import compute_percentile_clim


class FocalPreviewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._im = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.figure = Figure(figsize=(3, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Flattened focal-plane preview")
        self.ax.axis("off")
        layout.addWidget(self.canvas, 1)

    def set_data(self, focal_slice):
        vmin, vmax = compute_percentile_clim(focal_slice)
        if self._im is None:
            self._im = self.ax.imshow(focal_slice, cmap="gray", vmin=vmin, vmax=vmax)
        else:
            self._im.set_data(focal_slice)
            self._im.set_clim(vmin, vmax)
        self.canvas.draw_idle()
