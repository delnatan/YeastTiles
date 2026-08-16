"""Shared before/after image preview built on the vispy `RawStackCanvas`
(see `ui/rawstack/canvas.py`), used by the Denoise and Deconvolve stage
pages -- replaces each stage's own matplotlib
Figure/FigureCanvasQTAgg/NavigationToolbar2QT/contrast-scaling boilerplate
with one shared widget.
"""

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from ..rawstack.canvas import RawStackCanvas


class _SingleCanvasColumn(QWidget):
    """One labeled canvas -- half of a before/after pair. Swaps to a text
    placeholder instead of the canvas when there's no image for this half
    yet (e.g. a freshly selected file that hasn't been processed), so
    "nothing to show" never gets rendered as a duplicate of the other
    half -- that duplicate previously read as "processing already
    happened" when it hadn't."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)

        self._body = QStackedLayout()
        layout.addLayout(self._body, 1)

        self.canvas = RawStackCanvas()
        self._body.addWidget(self.canvas)

        self._empty_label = QLabel("Not yet processed")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._body.addWidget(self._empty_label)

        self._loaded = False

    def set_image(self, image: np.ndarray):
        if not self._loaded:
            self.canvas.set_channels(1, image.dtype, scale=(1.0, 1.0))
            self._loaded = True
        self.canvas.set_plane(image)
        self.canvas.reset_camera()
        self.canvas.auto_contrast()
        self._body.setCurrentWidget(self.canvas)

    def set_empty(self, message: str = "Not yet processed"):
        self._empty_label.setText(message)
        self._body.setCurrentWidget(self._empty_label)


class BeforeAfterPreviewPanel(QWidget):
    """Side-by-side before/after image preview. Public API (`set_data`/
    `clear`) matches the matplotlib-based panels it replaces
    (`DenoisePreviewPanel`/`DeconvolvePreviewPanel`), so page code needs no
    changes beyond the import line. Shows a placeholder until the first
    `set_data()` call, instead of a blank white figure."""

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)

        self._stack = QStackedLayout()
        outer.addLayout(self._stack, 1)

        self._placeholder = QLabel("No file selected")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._stack.addWidget(self._placeholder)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        panes_row = QHBoxLayout()
        self._before = _SingleCanvasColumn("Before")
        self._after = _SingleCanvasColumn("After")
        panes_row.addWidget(self._before, 1)
        panes_row.addWidget(self._after, 1)
        content_layout.addLayout(panes_row, 1)
        # Same shape/source image on both sides -- panning/zooming one
        # should move the other the same way.
        self._before.canvas.view.camera.link(self._after.canvas.view.camera)

        button_row = QHBoxLayout()
        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self._reset_view)
        button_row.addWidget(reset_btn)
        contrast_btn = QPushButton("Auto Contrast")
        contrast_btn.clicked.connect(self._auto_contrast)
        button_row.addWidget(contrast_btn)
        button_row.addStretch(1)
        content_layout.addLayout(button_row)

        self._stack.addWidget(content)
        self._stack.setCurrentWidget(self._placeholder)

    def set_data(self, before: np.ndarray, after: "np.ndarray | None", label: str = ""):
        """`after=None` means no processed result exists yet for the
        currently selected file -- show a blank placeholder on that side
        rather than the raw `before` image again, which would otherwise
        look like processing had already happened."""
        self._before.title_label.setText(f"{label} -- before".strip(" -") or "Before")
        self._after.title_label.setText(f"{label} -- after".strip(" -") or "After")
        self._before.set_image(before)
        if after is None:
            self._after.set_empty("Not yet processed")
        else:
            self._after.set_image(after)
        self._stack.setCurrentIndex(1)

    def clear(self):
        self._stack.setCurrentIndex(0)

    def _reset_view(self):
        self._before.canvas.reset_camera()
        self._after.canvas.reset_camera()

    def _auto_contrast(self):
        self._before.canvas.auto_contrast()
        self._after.canvas.auto_contrast()
