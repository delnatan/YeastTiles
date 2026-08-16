"""Shared single-image preview built on the vispy `RawStackCanvas` (see
`ui/rawstack/canvas.py`), used by the Segmentation and Tile Generation
stage pages -- replaces each stage's own matplotlib boilerplate with one
shared widget, using `RawStackCanvas`'s overlay API for mask outlines /
crop boxes instead of a second `imshow` layer per overlay.
"""

import numpy as np
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..rawstack.canvas import RawStackCanvas


class SingleImagePreviewPanel(QWidget):
    """One grayscale image + optional overlay(s), with a status label and
    a Reset View / Auto Contrast button row replacing
    `NavigationToolbar2QT`. Base class for the Segmentation and Tile
    Generation preview panels, which add their own `set_data()` on top of
    `set_image()`/the canvas's overlay methods."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loaded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.canvas = RawStackCanvas()
        layout.addWidget(self.canvas, 1)

        button_row = QHBoxLayout()
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.canvas.reset_camera)
        button_row.addWidget(reset_btn)
        contrast_btn = QPushButton("Auto Contrast")
        contrast_btn.clicked.connect(self.canvas.auto_contrast)
        button_row.addWidget(contrast_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def set_image(self, image: np.ndarray):
        if not self._loaded:
            self.canvas.set_channels(1, image.dtype, scale=(1.0, 1.0))
            self._loaded = True
        self.canvas.set_plane(image)
        self.canvas.reset_camera()
        self.canvas.auto_contrast()

    def clear(self):
        self.canvas.clear_plane()
        self.canvas.clear_overlay_mask()
        self.canvas.clear_overlay_boxes()
        self.status_label.setText("")
