"""Single project-tree panel: the one folder picker + batch file-selector
for the whole app, replacing the old per-page quartet (`FolderPanel`,
`EnhanceFolderPanel`, `SegmentationFolderPanel`, `TileFolderPanel`) and the
per-page `FileListPanel` checkbox lists.

Lives once in `main_window.py`, stacked above the page list -- not
duplicated per page -- so the project's folder structure is always visible
(per the project's design goal) and a file's checked state survives
switching pages. Each page reads/writes through this panel's stage-scoped
API (`checked_paths_for_stage`, `mark_result`, ...) instead of owning its
own folder state.

Selecting a leaf (click, or arrow-key nav) only emits `file_selected` --
cheap, and does not by itself load anything into a page. `main_window.py`
turns that into a list of valid next actions (see `selection_actions.py`)
shown in the `SelectionActionsPanel` docked below this tree; a page only
loads the file when one of those action buttons is clicked. This is the
only way a file gets loaded into a page -- there is no separate
double-click behavior.

There is exactly one folder to pick: the project root, which IS the folder
holding the raw 3D stacks (see core/project.py's module docstring) --
there's no separate "raw input folder" step. Picking a folder that already
has raw files in it opens straight into a working project; picking an
empty folder starts a fresh one that raw files can be dropped into later.

Tree shape:

    <project root>
      Raw input                 (raw_pattern glob directly on the project root)
      01 - Reduced (2D)         (01_reduced/*.tiff)
      02 - Denoised             (02_denoised/*.tiff, optional stage)
      03 - Deconvolved          (03_deconvolved/*.tiff, optional stage)
      05 - Tiles                (05_tiles/<fov>/*.tif -- summary only, no
                                 checkboxes: Tile Generation reads its
                                 input from whichever of 01/02/03 is the
                                 active segmentation source, not from here)

Segmentation has no numbered folder of its own (see core/project.py's
module docstring) -- the "Segmentation source" combo picks which of
01/02/03 currently holds `_seg.npy` sidecars; leaves under that stage show
a small mask badge for files that already have one.
"""

from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core import project as project_core
from yeastprep.core import stages as stages_core

from . import settings
from .status_icons import status_icon
from .worker import ProjectScanController

RAW_STAGE = stages_core.STAGE_RAW

SUPPORTED_RAW_PATTERNS = ("*.ims", "*.czi", "*.nd2")

# Numbered display labels aren't part of core.stages' generic labels (those
# are the short breadcrumb-chip text), so they're kept here, but the order
# and set of stages themselves come from the one shared sequence --
# core.stages.PIPELINE, minus the virtual "segmentation" stage, which has
# no tree node (see the module docstring above).
_STAGE_ORDER = tuple(
    spec.key for spec in stages_core.PIPELINE if spec.key != stages_core.STAGE_SEGMENTATION
)
_STAGE_LABELS = {
    RAW_STAGE: "Raw input",
    project_core.STAGE_REDUCED: "01 · Reduced (2D)",
    project_core.STAGE_DENOISED: "02 · Denoised",
    project_core.STAGE_DECONVOLVED: "03 · Deconvolved",
    project_core.STAGE_TILES: "05 · Tiles",
}
_2D_STAGES = (
    project_core.STAGE_REDUCED,
    project_core.STAGE_DENOISED,
    project_core.STAGE_DECONVOLVED,
)


class ProjectTreePanel(QWidget):
    project_root_changed = Signal(str)
    # stage, path -- fires on click (or arrow-key nav); cheap, just updates
    # SelectionActionsPanel. Loading a file into a page only happens when
    # one of that panel's action buttons is clicked, never from this signal
    # directly -- see main_window.py's _on_tree_selection.
    file_selected = Signal(str, str)
    checked_changed = Signal(str)  # stage -- fires when a leaf's checkbox is toggled
    segmentation_source_changed = Signal()
    refreshed = Signal()  # emitted at the end of refresh() -- e.g. for PipelineBreadcrumb

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root = ""
        self._stage_items: dict[str, QTreeWidgetItem] = {}
        self._last_snapshot = None  # most recent project_scan.ProjectScanSnapshot, or None

        self._scan_controller = ProjectScanController()
        self._scan_controller.result_ready.connect(self._on_scan_ready)
        self._scan_controller.error.connect(self._on_scan_error)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._build_project_group())
        layout.addWidget(self._build_source_group())

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        # Every row is a single line of text plus a small fixed-size icon --
        # true everywhere in this tree -- so this is safe, and without it Qt
        # only discovers row heights lazily as they're painted. Leaves get
        # torn down and rebuilt on every refresh() (see _apply_stage), so
        # that stale height cache is exactly what made clicking any item
        # jump the whole tree back to the top.
        self.tree.setUniformRowHeights(True)
        # Column 1 is a narrow "Seg" status dot -- a 2D-stage file's own
        # processing state (column 0) and whether it's been segmented are
        # two independent facts about the same row, so they get separate
        # slots instead of overloading one icon or a text suffix.
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self.tree.setColumnWidth(1, 20)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.currentItemChanged.connect(self._on_current_changed)
        self.tree.setToolTip("Click a file to see available actions below.")
        layout.addWidget(self.tree, 1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        layout.addWidget(refresh_btn)

        self._rebuild_stage_items()

    # ------------------------------------------------------------------
    # Project root -- the folder holding the raw 3D stacks directly; no
    # separate "raw input folder" step.

    def _build_project_group(self) -> QGroupBox:
        group = QGroupBox("Project (folder with raw images)")
        v = QVBoxLayout(group)

        self.project_recent = QComboBox()
        self.project_recent.addItem("Recent projects...")
        self.project_recent.addItems(settings.get_recent_project_roots())
        self.project_recent.activated.connect(self._on_project_recent_picked)
        v.addWidget(self.project_recent)

        row = QHBoxLayout()
        self.project_edit = QLineEdit()
        self.project_edit.setReadOnly(True)
        self.project_edit.setPlaceholderText("No project open")
        row.addWidget(self.project_edit, 1)
        open_btn = QPushButton("Open...")
        open_btn.clicked.connect(self._browse_project)
        row.addWidget(open_btn)
        v.addLayout(row)

        pattern_row = QHBoxLayout()
        pattern_row.addWidget(QLabel("Raw file type:"))
        self.raw_pattern_combo = QComboBox()
        self.raw_pattern_combo.addItems(SUPPORTED_RAW_PATTERNS)
        self.raw_pattern_combo.currentTextChanged.connect(self._on_raw_pattern_changed)
        pattern_row.addWidget(self.raw_pattern_combo)
        pattern_row.addStretch(1)
        v.addLayout(pattern_row)

        return group

    def _browse_project(self):
        start_dir = self._root or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(
            self, "Open project folder (contains raw images)", start_dir
        )
        if chosen:
            self.set_project_root(chosen)

    def _on_project_recent_picked(self, index):
        if index <= 0:
            return
        self.set_project_root(self.project_recent.itemText(index))
        self.project_recent.setCurrentIndex(0)

    def _on_raw_pattern_changed(self, _pattern):
        self._persist_project_field(raw_pattern=self.raw_pattern())
        self.refresh()

    def set_project_root(self, root: str):
        root = str(root)
        Path(root).mkdir(parents=True, exist_ok=True)
        self._root = root
        self.project_edit.setText(root)
        self.project_edit.setToolTip(root)
        settings.add_recent_project_root(root)
        self._refresh_recent(
            self.project_recent, "Recent projects...", settings.get_recent_project_roots()
        )

        config = project_core.load_project_config(root)
        if config:
            self.raw_pattern_combo.setCurrentText(config.raw_pattern)
            self._set_segmentation_source_silent(config.segmentation_source_stage)
        else:
            # First time this folder's been opened as a project -- persist
            # today's raw-pattern choice so it round-trips next time.
            self._persist_project_field(raw_pattern=self.raw_pattern())

        self.refresh()
        self.project_root_changed.emit(root)

    def project_root(self) -> str:
        return self._root

    def project_paths(self) -> project_core.ProjectPaths | None:
        if not self._root:
            return None
        return project_core.ProjectPaths(self._root)

    def raw_pattern(self) -> str:
        return self.raw_pattern_combo.currentText()

    # ------------------------------------------------------------------
    # Segmentation source stage

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Segmentation / Tile Generation source")
        v = QVBoxLayout(group)
        self.source_combo = QComboBox()
        self.source_combo.addItem("Auto (most downstream available)", None)
        for stage in _2D_STAGES:
            self.source_combo.addItem(_STAGE_LABELS[stage], stage)
        self.source_combo.currentIndexChanged.connect(self._on_source_combo_changed)
        v.addWidget(self.source_combo)
        return group

    def _set_segmentation_source_silent(self, stage: str | None):
        self.source_combo.blockSignals(True)
        idx = self.source_combo.findData(stage)
        self.source_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.source_combo.blockSignals(False)

    def _on_source_combo_changed(self, _index):
        self._persist_project_field(segmentation_source_stage=self.segmentation_source_override())
        self.refresh()
        self.segmentation_source_changed.emit()

    def segmentation_source_override(self) -> str | None:
        return self.source_combo.currentData()

    def active_2d_stage(self) -> str | None:
        paths = self.project_paths()
        if paths is None:
            return None
        source = project_core.resolve_2d_source(paths, self.segmentation_source_override())
        if source is None:
            return None
        return source.name

    # ------------------------------------------------------------------
    # Project config persistence (merge-on-write: this panel only owns
    # raw_pattern/segmentation_source_stage -- other
    # fields belong to each stage's own params panel).

    def _persist_project_field(self, **fields):
        if not self._root:
            return
        config = project_core.load_project_config(self._root) or project_core.ProjectConfig()
        for key, value in fields.items():
            setattr(config, key, value)
        project_core.save_project_config(self._root, config)

    # ------------------------------------------------------------------
    # Tree construction

    def _rebuild_stage_items(self):
        self.tree.clear()
        self._stage_items = {}
        for stage in _STAGE_ORDER:
            item = QTreeWidgetItem(self.tree, [_STAGE_LABELS[stage]])
            item.setData(0, Qt.UserRole, ("stage", stage, None))
            item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
            self.tree.addTopLevelItem(item)
            self._stage_items[stage] = item

    def refresh(self):
        """Kicks off a background rescan (`core.project_scan.scan_project`,
        run on `_scan_controller`'s worker thread) and returns immediately
        -- the tree/breadcrumb are populated later, in `_on_scan_ready`,
        once that finishes. Globbing every stage folder and `stat()`-ing
        every file in it used to happen right here, synchronously, on the
        GUI thread -- fine on a local SSD, but on a slow external/network
        drive with hundreds of raw files that walk alone could take long
        enough to make the whole app look hung."""
        if not self._root:
            self._apply_scan(None)
            return
        self._scan_controller.recompute_now(
            self._root,
            self._root,
            (self.raw_pattern(), self.segmentation_source_override(), _STAGE_ORDER),
        )

    def _on_scan_ready(self, result):
        # Discard a scan that's no longer the latest requested (a newer one
        # is already in flight) or that answers a project we've since
        # navigated away from.
        if result.request_id != self._scan_controller.latest_request_id():
            return
        if result.payload != self._root:
            return
        self._last_snapshot = result.scan
        self._apply_scan(result.scan)

    def _on_scan_error(self, message: str):
        self.project_edit.setToolTip(f"{self._root}\n\nLast project scan failed: {message}")

    def _apply_scan(self, snapshot):
        """Populate the tree from an already-computed `ProjectScanSnapshot`
        -- pure Qt-widget bookkeeping, no filesystem access, so this is
        cheap enough to run on the GUI thread. `snapshot` is None only when
        no project is open yet."""
        for stage in _STAGE_ORDER:
            self._apply_stage(stage, snapshot.stages[stage] if snapshot else None)
        self.refreshed.emit()

    def _apply_stage(self, stage: str, result):
        top_item = self._stage_items.get(stage)
        if top_item is None:
            return
        top_item.takeChildren()

        badge = " \U0001f52c" if result is not None and result.is_active_2d else ""
        if result is None or not result.exists:
            empty_msg = result.empty_msg if result is not None else "no project open"
            top_item.setText(0, f"{_STAGE_LABELS[stage]}{badge}  ({empty_msg})")
            top_item.setExpanded(False)
            return

        if stage == project_core.STAGE_TILES:
            top_item.setText(
                0,
                f"{_STAGE_LABELS[stage]}{badge}  ({result.count} FOV(s){result.extra_count_text})",
            )
        else:
            top_item.setText(0, f"{_STAGE_LABELS[stage]}{badge}  ({result.count} file(s))")

        self.tree.blockSignals(True)
        try:
            for leaf_info in result.leaves:
                leaf = QTreeWidgetItem(top_item)
                leaf.setText(0, leaf_info.display_name)
                leaf.setData(0, Qt.UserRole, (leaf_info.kind, leaf_info.stage, leaf_info.path))
                if leaf_info.kind == "file":
                    leaf.setFlags(leaf.flags() | Qt.ItemIsUserCheckable)
                    # Checked by default: batch actions only ever touch
                    # checked files (see checked_paths_for_stage), so
                    # defaulting to unchecked-means-everything would make
                    # "process everything" an invisible fallback rather
                    # than something the tree actually shows you're about
                    # to do.
                    leaf.setCheckState(0, Qt.Checked)
                leaf.setIcon(0, status_icon(leaf_info.icon_state))
                if leaf_info.tooltip:
                    leaf.setToolTip(0, leaf_info.tooltip)
                if leaf_info.seg_icon_state is not None:
                    leaf.setIcon(1, status_icon(leaf_info.seg_icon_state))
                    leaf.setToolTip(1, leaf_info.seg_tooltip)
        finally:
            self.tree.blockSignals(False)
        # Mirrors the old synchronous behavior: every 2D-stage/raw node
        # auto-expands, the tiles summary stays collapsed (it's a per-FOV
        # rollup, not something you page through leaf by leaf).
        top_item.setExpanded(stage != project_core.STAGE_TILES)

    def last_scan_snapshot(self):
        """Most recent completed `project_scan.ProjectScanSnapshot`, or
        None if no scan for the current project has finished yet -- lets
        `PipelineBreadcrumb` reuse this panel's scan instead of redoing its
        own (see that module)."""
        return self._last_snapshot

    # ------------------------------------------------------------------
    # Selection / checkbox API used by pages

    def _on_item_changed(self, item, column):
        if column != 0:
            return
        kind, stage, _path = item.data(0, Qt.UserRole) or (None, None, None)
        if kind == "file":
            self.checked_changed.emit(stage)

    def _on_current_changed(self, current, _previous):
        if current is None:
            return
        kind, stage, path = current.data(0, Qt.UserRole) or (None, None, None)
        if kind in ("file", "tile_fov"):
            self.file_selected.emit(stage, path)

    def all_paths_for_stage(self, stage: str) -> list[str]:
        top_item = self._stage_items.get(stage)
        if top_item is None:
            return []
        return [
            top_item.child(i).data(0, Qt.UserRole)[2]
            for i in range(top_item.childCount())
        ]

    def checked_paths_for_stage(self, stage: str) -> list[str]:
        """Checked leaf paths under `stage`'s node. Leaves start out
        checked (see `_refresh_stage`), so "everything" is the visible
        default rather than an invisible fallback -- unchecking a file
        really does exclude it, including unchecking all of them."""
        top_item = self._stage_items.get(stage)
        if top_item is None:
            return []
        return [
            top_item.child(i).data(0, Qt.UserRole)[2]
            for i in range(top_item.childCount())
            if top_item.child(i).checkState(0) == Qt.Checked
        ]

    def mark_result(self, stage: str, path, success: bool, error: str | None = None):
        top_item = self._stage_items.get(stage)
        if top_item is None:
            return
        path = str(path)
        for i in range(top_item.childCount()):
            leaf = top_item.child(i)
            if leaf.data(0, Qt.UserRole)[2] == path:
                leaf.setIcon(0, status_icon("done" if success else "failed"))
                leaf.setToolTip(0, error or ("Processed" if success else ""))
                return

    # ------------------------------------------------------------------

    @staticmethod
    def _refresh_recent(combo: QComboBox, placeholder: str, entries: list[str]):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(placeholder)
        combo.addItems(entries)
        combo.blockSignals(False)

    def shutdown(self):
        self._scan_controller.shutdown()
