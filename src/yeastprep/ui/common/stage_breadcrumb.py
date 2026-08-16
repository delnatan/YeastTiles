"""Compact pipeline-stage indicator ("Raw -> Reduced -> Denoise ->
Deconvolve -> Segment -> Tile"), tied to the shared `ProjectTreePanel`'s
project state. Lives once in `main_window.py`, above the sidebar/stack
split, so it stays visible as a workflow guide no matter which page is
showing -- not owned by any one page. Reads
`yeastprep.core.stages.pipeline_status()` -- the one place stage
sequencing/status logic lives -- rather than re-deriving it, and reuses
`status_icons.STATUS_COLORS` so its colors match the tree's per-file status
dots.
"""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget

from yeastprep.core import project as project_core
from yeastprep.core import stages as stages_core

from ..status_icons import STATUS_COLORS

_STATUS_TO_DOT_STATUS = {
    "empty": "unprocessed",
    "stale": "stale",
    "done": "done",
}


class PipelineBreadcrumb(QWidget):
    def __init__(self, tree_panel, parent=None):
        super().__init__(parent)
        self._tree_panel = tree_panel

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self._chips: dict[str, QLabel] = {}
        for i, spec in enumerate(stages_core.PIPELINE):
            if i > 0:
                arrow = QLabel("→")
                arrow.setStyleSheet("color: #8c8c8c;")
                layout.addWidget(arrow)
            chip = QLabel(spec.label)
            chip.setAlignment(Qt.AlignCenter)
            layout.addWidget(chip)
            self._chips[spec.key] = chip
        layout.addStretch(1)

        tree_panel.project_root_changed.connect(self.refresh)
        tree_panel.segmentation_source_changed.connect(self.refresh)
        tree_panel.refreshed.connect(self.refresh)

        self.refresh()

    def refresh(self, *_args):
        paths = self._tree_panel.project_paths()
        if paths is None:
            for spec in stages_core.PIPELINE:
                self._style_chip(spec.key, "empty", active=False, optional=spec.optional)
            return

        config = project_core.load_project_config(self._tree_panel.project_root())
        for state in stages_core.pipeline_status(paths, config):
            self._style_chip(state.key, state.status, state.active, state.optional)

    def _style_chip(self, key: str, status: str, active: bool, optional: bool):
        chip = self._chips.get(key)
        if chip is None:
            return
        color = STATUS_COLORS.get(_STATUS_TO_DOT_STATUS.get(status, "unprocessed"), "#8c8c8c")
        border = "2px solid #4da6ff" if active else "1px solid transparent"
        weight = "bold" if active else "normal"
        chip.setStyleSheet(
            f"padding: 2px 8px; border-radius: 8px; background-color: {color}33; "
            f"color: {color}; border: {border}; font-weight: {weight};"
        )
        tooltip = status + (" (optional)" if optional else "")
        chip.setToolTip(tooltip)
