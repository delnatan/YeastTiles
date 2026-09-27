"""The window's one navigation strip ("Raw -> Reduced -> Denoise ->
Deconvolve -> Segment -> Tile | Classifier Training, Classify Tiles"),
spanning the top of `main_window.py`. Each chip does three jobs:

- shows its stage's status (colored like the tree's per-file status dots),
  with the active Segmentation/Tile Generation source stage outlined;
- opens the page that produces that stage when clicked (`page_requested`),
  without loading any file -- loading a file is the Selection panel's job;
- underlines the page currently showing, and carries a thin progress bar
  while that page's background batch/training run is going, which keeps
  running (and reporting here) after navigating away.

"Raw" has no producing page, so its chip is status-only. Classifier
Training and Classify Tiles aren't pipeline stages (they work on their own
pool of projects), so they sit after a separator with no status color.

Reads `tree_panel.last_scan_snapshot().pipeline_states` rather than calling
`yeastprep.core.stages.pipeline_status()` itself: that call globs and
`stat()`s every stage folder, which `ProjectTreePanel` already does on a
background thread (see `core/project_scan.py`).
"""

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

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

_NEUTRAL_COLOR = "#a0a0a0"


class _Chip(QWidget):
    """A chip (clickable if it has a page) over a thin progress bar that
    keeps its space when hidden, so the strip doesn't jump around."""

    def __init__(self, label: str, clickable: bool, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        if clickable:
            self.button = QToolButton()
            self.button.setText(label)
            self.button.setCursor(Qt.PointingHandCursor)
        else:
            self.button = QLabel(label)
            self.button.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.button)

        self.progress = QProgressBar()
        self.progress.setFixedHeight(3)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            "QProgressBar { border: none; background: #3a3a3a; }"
            "QProgressBar::chunk { background: #4da6ff; }"
        )
        # Ignored: take the chip's width rather than the bar's own minimum.
        policy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        policy.setRetainSizeWhenHidden(True)
        self.progress.setSizePolicy(policy)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.color = _NEUTRAL_COLOR
        self.source_active = False
        self.current = False
        self.status_tooltip = ""

    def restyle(self):
        border = "2px solid #4da6ff" if self.source_active else "1px solid transparent"
        # rgba(), not 8-digit hex: Qt reads "#xxxxxxxx" as #AARRGGBB.
        r, g, b = (int(self.color[i : i + 2], 16) for i in (1, 3, 5))
        alpha = 0.4 if self.current else 0.13
        decoration = "underline" if self.current else "none"
        self.button.setStyleSheet(
            f"padding: 2px 8px; border-radius: 8px; background-color: rgba({r}, {g}, {b}, {alpha}); "
            f"color: {self.color}; border: {border}; text-decoration: {decoration};"
        )


class PipelineBreadcrumb(QWidget):
    page_requested = Signal(str)  # page_key

    def __init__(
        self,
        tree_panel,
        stage_pages: dict[str, str],
        extra_pages: list[tuple[str, str]],
        parent=None,
    ):
        """`stage_pages`: pipeline stage key -> key of the page that produces
        it. `extra_pages`: (label, page_key) entries shown after the
        pipeline, e.g. the classification pages."""
        super().__init__(parent)
        self._tree_panel = tree_panel
        self._stage_chips: dict[str, _Chip] = {}
        self._page_chips: dict[str, _Chip] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        for i, spec in enumerate(stages_core.PIPELINE):
            if i > 0:
                arrow = QLabel("→")
                arrow.setStyleSheet("color: #8c8c8c;")
                layout.addWidget(arrow)
            chip = self._add_chip(layout, spec.label, stage_pages.get(spec.key))
            self._stage_chips[spec.key] = chip

        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        layout.addSpacing(8)
        layout.addWidget(separator)
        layout.addSpacing(8)
        for label, page_key in extra_pages:
            chip = self._add_chip(layout, label, page_key)
            chip.status_tooltip = "Works on its own pool of projects, independent of the pipeline"
            self._update_tooltip(chip)
        layout.addStretch(1)

        # Reset to "empty" the moment the project changes (cheap, no I/O)
        # rather than leaving the previous project's chips lingering while
        # its background scan is still in flight.
        tree_panel.project_root_changed.connect(self._reset_chips)
        tree_panel.segmentation_source_changed.connect(self.refresh)
        tree_panel.refreshed.connect(self.refresh)

        self._reset_chips()
        self.refresh()

    def _add_chip(self, layout, label: str, page_key: str | None) -> _Chip:
        chip = _Chip(label, clickable=page_key is not None)
        if page_key is not None:
            chip.button.clicked.connect(
                lambda _checked=False, key=page_key: self.page_requested.emit(key)
            )
            self._page_chips[page_key] = chip
        chip.restyle()
        layout.addWidget(chip)
        return chip

    # ------------------------------------------------------------------
    # Current page / background progress

    def set_current_page(self, page_key: str | None):
        """Underline `page_key`'s chip (`None`, or a page with no chip such
        as Preview, underlines nothing)."""
        for key, chip in self._page_chips.items():
            chip.current = key == page_key
            chip.restyle()

    def set_progress(self, page_key: str, progress):
        """`progress`: a `pages.page_progress.PageProgress`. A run with no
        known total (e.g. training) shows as a busy bar."""
        chip = self._page_chips.get(page_key)
        if chip is None:
            return
        bar = chip.progress
        if not progress.active:
            bar.setVisible(False)
            self._update_tooltip(chip)
            return
        bar.setVisible(True)
        bar.setRange(0, progress.total)
        bar.setValue(progress.done)
        self._update_tooltip(chip)

    def _update_tooltip(self, chip: _Chip):
        tooltip = chip.status_tooltip
        if chip.progress.isVisible():
            bar = chip.progress
            running = f"Running: {bar.value()}/{bar.maximum()}" if bar.maximum() else "Running"
            tooltip = f"{tooltip}\n{running}" if tooltip else running
        chip.button.setToolTip(tooltip)

    # ------------------------------------------------------------------
    # Stage status

    def _reset_chips(self, *_args):
        for spec in stages_core.PIPELINE:
            self._style_stage_chip(spec.key, "empty", active=False, optional=spec.optional)

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
            self._style_stage_chip(state.key, state.status, state.active, state.optional)

    def _style_stage_chip(self, key: str, status: str, active: bool, optional: bool):
        chip = self._stage_chips.get(key)
        if chip is None:
            return
        chip.color = STATUS_COLORS.get(_STATUS_TO_DOT_STATUS.get(status, "unprocessed"), "#8c8c8c")
        chip.source_active = active
        chip.status_tooltip = _STATUS_TOOLTIPS.get(status, status) + (
            " (optional)" if optional else ""
        )
        if active:
            chip.status_tooltip += "\nCurrent Segmentation / Tile Generation source"
        chip.restyle()
        self._update_tooltip(chip)
