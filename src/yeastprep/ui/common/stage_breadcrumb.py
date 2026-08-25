"""Compact pipeline-stage indicator ("Raw -> Reduced -> Denoise ->
Deconvolve -> Segment -> Tile"), tied to the shared `ProjectTreePanel`'s
project state. Lives once in `main_window.py`, above the sidebar/stack
split, so it stays visible as a workflow guide no matter which page is
showing -- not owned by any one page.

Reads `tree_panel.last_scan_snapshot().pipeline_states` rather than calling
`yeastprep.core.stages.pipeline_status()` itself: that call globs and
`stat()`s every stage folder (same walk `ProjectTreePanel` needs for its
own tree), and doing it a second time here, synchronously on the GUI
thread, would defeat the point of `ProjectTreePanel` having moved that walk
to a background thread (see `core/project_scan.py`). Reuses
`status_icons.STATUS_COLORS` so its colors match the tree's per-file status
dots.
"""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget

from yeastprep.core import stages as stages_core

from ..status_icons import STATUS_COLORS

_STATUS_TO_DOT_STATUS = {
    "empty": "unprocessed",
    "stale": "stale",
    "done": "done",
    "archived": "archived",
}

_STATUS_TOOLTIPS = {
    "empty": "Nothing produced yet",
    "done": "Up to date",
    "stale": "Source changed since this was last produced",
    "archived": "Source folder isn't present on this computer (likely moved "
    "to storage) -- can't verify freshness, showing as up to date",
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

        # Reset to "empty" the moment the project changes (cheap, no I/O)
        # rather than leaving the previous project's chips lingering while
        # its background scan is still in flight.
        tree_panel.project_root_changed.connect(self._reset_chips)
        tree_panel.segmentation_source_changed.connect(self.refresh)
        tree_panel.refreshed.connect(self.refresh)

        self.refresh()

    def _reset_chips(self, *_args):
        for spec in stages_core.PIPELINE:
            self._style_chip(spec.key, "empty", active=False, optional=spec.optional)

    def refresh(self, *_args):
        paths = self._tree_panel.project_paths()
        if paths is None:
            self._reset_chips()
            return

        snapshot = self._tree_panel.last_scan_snapshot()
        if snapshot is None or snapshot.root != str(paths.root):
            # This project's background scan hasn't finished yet -- leave
            # the chips as they are; `refreshed` fires again (and calls
            # back into here) once it has.
            return
        for state in snapshot.pipeline_states:
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
        tooltip = _STATUS_TOOLTIPS.get(status, status) + (" (optional)" if optional else "")
        chip.setToolTip(tooltip)
