"""Raw-stack viewer tab: multi-channel, multi-Z visual check for the
currently selected raw file (design.md Feature 1, step 1).

Loads the file's lazy 5D proxy directly via pyvistra.io.load_image (the
format reader is genuinely needed -- .ims/.czi/.nd2 support isn't worth
reimplementing) but renders through the vendored RawStackCanvas rather
than a pyvistra QMainWindow, so this stays a self-contained tab instead of
popping open a separate foreign window.
"""

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pyvistra.io import load_image

from .canvas import RawStackCanvas
from .channel_controls import ChannelRow


class RawStackViewerPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._proxy = None
        self._meta = None
        self._channel_rows: list[ChannelRow] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = RawStackCanvas()
        left_layout.addWidget(self.canvas, 1)

        z_row = QHBoxLayout()
        z_row.addWidget(QLabel("Z:"))
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.valueChanged.connect(self._update_slice)
        z_row.addWidget(self.z_slider, 1)
        self.z_label = QLabel("-")
        self.z_label.setFixedWidth(60)
        z_row.addWidget(self.z_label)
        left_layout.addLayout(z_row)

        self.info_label = QLabel(" ")
        left_layout.addWidget(self.info_label)

        splitter.addWidget(left)

        right = QScrollArea()
        right.setWidgetResizable(True)
        right.setFixedWidth(260)
        self.controls_container = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_container)
        self.controls_layout.setContentsMargins(4, 4, 4, 4)

        self.auto_contrast_btn = QPushButton("Auto Contrast")
        self.auto_contrast_btn.clicked.connect(self.canvas.auto_contrast)
        self.controls_layout.addWidget(self.auto_contrast_btn)
        self.controls_layout.addStretch(1)

        right.setWidget(self.controls_container)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)

    def load_file(self, path):
        self._clear_channel_rows()
        try:
            proxy, meta = load_image(str(path))
        except Exception as exc:
            self.info_label.setText(f"Failed to load: {exc}")
            return

        self._proxy = proxy
        self._meta = meta
        _T, Z, C, _Y, _X = proxy.shape
        dz, dy, dx = meta["scale"]

        self.canvas.set_channels(C, dtype=proxy.dtype, scale=(dy, dx))

        clim_hi = 65535.0 if np.issubdtype(proxy.dtype, np.integer) else 1.0
        channels_meta = meta.get("channels") or []
        for c in range(C):
            name = (
                str(channels_meta[c].get("name"))
                if c < len(channels_meta) and channels_meta[c].get("name")
                else f"Channel {c}"
            )
            row = ChannelRow(c, name, self.canvas.display, clim_bounds=(0.0, clim_hi))
            self._channel_rows.append(row)
            self.controls_layout.insertWidget(self.controls_layout.count() - 1, row)

        self.z_slider.blockSignals(True)
        self.z_slider.setRange(0, max(Z - 1, 0))
        self.z_slider.setValue(Z // 2)
        self.z_slider.blockSignals(False)

        self.info_label.setText(
            f"{proxy.shape[1]} Z x {C} C, {proxy.shape[3]}x{proxy.shape[4]} px, "
            f"scale ({dz:.3f}, {dy:.3f}, {dx:.3f}) um"
        )

        self._update_slice()
        self.canvas.reset_camera()
        self.canvas.auto_contrast()

    def _update_slice(self):
        if self._proxy is None:
            return
        z = self.z_slider.value()
        plane = np.asarray(self._proxy[0, z, :, :, :])
        self.canvas.set_plane(plane)
        self.z_label.setText(f"{z + 1}/{self._proxy.shape[1]}")

    def _clear_channel_rows(self):
        for row in self._channel_rows:
            row.disconnect_from_display()
            row.setParent(None)
            row.deleteLater()
        self._channel_rows = []
