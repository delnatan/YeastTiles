"""Preview page: a read-only composite viewer for any already-produced 2D
file -- 01_reduced/, 02_denoised/, or 03_deconvolved/ -- brightfield +
target/fluorescence overlay of the combined-channel tiff (plus a saved
segmentation mask outline, if one exists for that file), with the same
per-channel colormap/contrast/histogram controls as the Raw Stack tab
(`ui/rawstack/panel.py`), reusing its `ChannelRow` (which itself embeds
pyvistra's `CompactHistogramWidget`).

Every one of these stages produces a file that already exists on disk by
the time it's selectable in the tree, so "Preview" is always one of the
actions `SelectionActionsPanel` offers for it, alongside whichever
processing tasks (Denoise/Deconvolve/Segment/...) currently apply -- see
`ui/selection_actions.py`. Raw files don't offer a Preview action here;
the rawstack viewer on Data Reduction already fills that role for that
stage.
"""

from pathlib import Path

import numpy as np
from cellpose.utils import masks_to_outlines
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core.combined_tiff import BRIGHTFIELD_CHANNEL, TARGET_CHANNEL, load_combined_channels
from yeastprep.core.segmentation import load_saved_masks, seg_npy_path

from .. import settings
from ..common.preview_source_label import PreviewSourceLabel
from ..project_tree_panel import ProjectTreePanel
from ..rawstack.canvas import RawStackCanvas
from ..rawstack.channel_controls import ChannelRow
from .page_progress import PageProgress  # noqa: F401 -- shape of progress_changed, unused here

_ROLE_BY_CHANNEL = {BRIGHTFIELD_CHANNEL: "brightfield", TARGET_CHANNEL: "target"}
_LABEL_BY_CHANNEL = {BRIGHTFIELD_CHANNEL: "Brightfield", TARGET_CHANNEL: "Target/fluorescence"}
_MASK_OUTLINE_RGBA = (1.0, 0.85, 0.0, 1.0)  # yellow, matches cellpose GUI / SegmentationPreviewPanel


class PreviewPage(QWidget):
    # Declared for main_window.py's uniform per-page wiring loop; this page
    # does no background work, so it never actually emits.
    progress_changed = Signal(object)  # PageProgress

    def __init__(self, tree_panel: ProjectTreePanel, parent=None):
        super().__init__(parent)
        self.tree_panel = tree_panel
        self._loaded = False
        self._channel_rows: list[ChannelRow] = []

        layout = QVBoxLayout(self)

        self.preview_source_label = PreviewSourceLabel()
        layout.addWidget(self.preview_source_label)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        self.canvas = RawStackCanvas()
        splitter.addWidget(self.canvas)

        right = QScrollArea()
        right.setWidgetResizable(True)
        right.setFixedWidth(280)
        self.controls_container = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_container)
        self.controls_layout.setContentsMargins(4, 4, 4, 4)

        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.canvas.reset_camera)
        self.controls_layout.addWidget(self.reset_view_btn)
        self.auto_contrast_btn = QPushButton("Auto Contrast")
        self.auto_contrast_btn.clicked.connect(self.canvas.auto_contrast)
        self.controls_layout.addWidget(self.auto_contrast_btn)

        self.controls_layout.addStretch(1)
        right.setWidget(self.controls_container)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)

        self.status_label = QLabel(
            "Select a reduced, denoised, or deconvolved file in the tree and click "
            "'Preview' to see its brightfield + target/fluorescence overlay."
        )
        layout.addWidget(self.status_label)

    def load_selection(self, stage: str, path: str, mode: str = "live"):
        try:
            brightfield, target = load_combined_channels(path)
        except Exception as exc:
            self.status_label.setText(f"Failed to load {path}: {exc}")
            return

        composite = np.stack([brightfield, target], axis=0)
        if not self._loaded:
            self.canvas.set_channels(2, composite.dtype, scale=(1.0, 1.0))
            self._build_channel_rows()
            self._loaded = True

        # Colors are re-applied (not just set once at first load) so a
        # colormap picked in the Raw Stack tab after this page's first use
        # still carries over on the next file selection -- see
        # settings.get_channel_colormap.
        for channel_idx, role in _ROLE_BY_CHANNEL.items():
            self.canvas.display.set_colormap_name(
                channel_idx, settings.get_channel_colormap(role)
            )

        self.canvas.set_plane(composite)
        self.canvas.reset_camera()
        self.canvas.auto_contrast()
        for row in self._channel_rows:
            row.set_data(composite[row.channel_idx])
        self.preview_source_label.set_path(path)

        masks = load_saved_masks(seg_npy_path(path))
        n_cells_msg = ""
        if masks is not None:
            outline_rgba = np.zeros((*masks.shape, 4), dtype=np.float32)
            outline_rgba[masks_to_outlines(masks)] = _MASK_OUTLINE_RGBA
            self.canvas.set_overlay_mask(outline_rgba)
            n_cells_msg = f" -- {int(masks.max())} cell(s) segmented"
        else:
            self.canvas.clear_overlay_mask()
        self.status_label.setText(f"{Path(path).name}{n_cells_msg}")

    def _build_channel_rows(self):
        for channel_idx in (BRIGHTFIELD_CHANNEL, TARGET_CHANNEL):
            row = ChannelRow(
                channel_idx,
                _LABEL_BY_CHANNEL[channel_idx],
                self.canvas.display,
                clim_bounds=(0.0, 65535.0),
            )
            self._channel_rows.append(row)
            self.controls_layout.insertWidget(self.controls_layout.count() - 1, row)

    def shutdown(self):
        pass
