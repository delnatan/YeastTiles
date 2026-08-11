"""Brightfield Z-slice preview with its own scrubber -- re-slices the
already-in-memory img3d directly on the UI thread (no worker needed),
since this is the one interaction in the diagnostic tab that must feel
instant, matching troubleshoot_wildtype.ipynb's plt.imshow(img3d[z])."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from tileclass.contrast import compute_percentile_clim


class RawSlicePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._img3d = None
        self._im = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.figure = Figure(figsize=(3, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Raw brightfield slice")
        self.ax.axis("off")
        layout.addWidget(self.canvas, 1)

        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Z:"))
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.valueChanged.connect(self._redraw)
        z_row.addWidget(self.z_slider, 1)
        self.z_label = QLabel("-")
        self.z_label.setFixedWidth(50)
        z_row.addWidget(self.z_label)
        layout.addLayout(z_row)

    def clear(self):
        self._img3d = None
        if self._im is not None:
            self._im.remove()
            self._im = None
        self.z_slider.blockSignals(True)
        self.z_slider.setRange(0, 0)
        self.z_slider.blockSignals(False)
        self.z_label.setText("-")
        self.ax.set_title("Raw brightfield slice (not loaded for this view)")
        self.canvas.draw_idle()

    def set_volume(self, img3d):
        self._img3d = img3d
        Nz = img3d.shape[0]
        self.z_slider.blockSignals(True)
        self.z_slider.setRange(0, max(Nz - 1, 0))
        self.z_slider.setValue(Nz // 2)
        self.z_slider.blockSignals(False)
        self._redraw()

    def _redraw(self):
        if self._img3d is None:
            return
        z = self.z_slider.value()
        plane = self._img3d[z]
        vmin, vmax = compute_percentile_clim(plane)
        if self._im is None:
            self._im = self.ax.imshow(plane, cmap="gray", vmin=vmin, vmax=vmax)
        else:
            self._im.set_data(plane)
            self._im.set_clim(vmin, vmax)
        self.z_label.setText(f"{z + 1}/{self._img3d.shape[0]}")
        self.canvas.draw_idle()
