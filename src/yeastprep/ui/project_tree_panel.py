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
      05 - Tiles                (05_tiles/*.tif -- summary only, no
                                 checkboxes: Tile Generation reads its
                                 input from whichever of 01/02/03 is the
                                 active segmentation source, not from here)

Segmentation has no numbered folder of its own (see core/project.py's
module docstring) -- the "Segmentation source" combo picks which of
01/02/03 currently holds `_seg.npy` sidecars; leaves under that stage show
a small mask badge for files that already have one.
"""

import re
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

from yeastprep.core import fs_status as fs_status_core
from yeastprep.core import project as project_core
from yeastprep.core import stages as stages_core
from yeastprep.core import tiles as tiles_core

from . import settings
from .status_icons import status_icon

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

_DIGITS = re.compile(r"(\d+)")


def natsort_key(path: Path):
    parts = _DIGITS.split(path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._build_project_group())
        layout.addWidget(self._build_source_group())

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
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
        self._rebuild_stage_items()
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
        self._rebuild_stage_items()
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

    def _stage_dir(self, stage: str) -> Path | None:
        if stage == RAW_STAGE:
            return Path(self._root) if self._root else None
        paths = self.project_paths()
        if paths is None:
            return None
        return paths.stage_dir(stage)

    def _stage_glob_pattern(self, stage: str) -> str:
        # STAGE_TILES never reaches this -- _refresh_stage short-circuits to
        # _refresh_tiles_children before calling it (see that branch).
        if stage == RAW_STAGE:
            return self.raw_pattern() or "*.ims"
        return "*.tiff"

    def _producer_dir(self, stage: str) -> Path | None:
        """Whichever folder feeds *into* `stage` -- used to flag a leaf as
        stale (producer changed since this stage last ran) rather than
        just "exists"."""
        paths = self.project_paths()
        if stage == project_core.STAGE_REDUCED:
            return Path(self._root) if self._root else None
        if paths is None:
            return None
        if stage == project_core.STAGE_DENOISED:
            return paths.reduced
        if stage == project_core.STAGE_DECONVOLVED:
            config = project_core.load_project_config(self._root) if self._root else None
            override = config.deconvolve_source_stage if config else None
            return project_core.resolve_deconvolve_source(paths, override)
        return None

    def refresh(self):
        active_stage = self.active_2d_stage()
        for stage in _STAGE_ORDER:
            self._refresh_stage(stage, active_stage)
        self.refreshed.emit()

    def _refresh_stage(self, stage: str, active_2d_stage: str | None):
        top_item = self._stage_items.get(stage)
        if top_item is None:
            return
        top_item.takeChildren()

        folder = self._stage_dir(stage)
        badge = " \U0001f52c" if stage == active_2d_stage else ""
        if folder is None or not folder.is_dir():
            empty_msg = "no project open" if stage == RAW_STAGE else "not created yet"
            top_item.setText(0, f"{_STAGE_LABELS[stage]}{badge}  ({empty_msg})")
            top_item.setExpanded(False)
            return

        if stage == project_core.STAGE_TILES:
            # Deliberately never globs 05_tiles/ for its individual crop
            # files -- a project can have hundreds of thousands of those,
            # and the per-FOV summary below already gets everything it
            # needs (count included) from tile_index.csv instead. This is
            # also why the tree never grows a leaf per tile: the file-level
            # viewer for these lives in tileclass's page-by-page grid, not
            # here.
            self._refresh_tiles_children(top_item, folder, active_2d_stage, badge)
            return

        paths = sorted(
            fs_status_core.list_visible(folder, self._stage_glob_pattern(stage)),
            key=natsort_key,
        )
        top_item.setText(0, f"{_STAGE_LABELS[stage]}{badge}  ({len(paths)} file(s))")

        producer_dir = self._producer_dir(stage) if stage in _2D_STAGES else None
        statuses = project_core.stage_file_status(folder, upstream_dir=producer_dir)
        reduced_dir = self._stage_dir(project_core.STAGE_REDUCED) if stage == RAW_STAGE else None

        denoise_channels_done = {}
        if stage == project_core.STAGE_DENOISED and self._root:
            config = project_core.load_project_config(self._root)
            if config:
                denoise_channels_done = config.denoise_channels_done

        # Segmentation masks sit directly beside whichever 2D stage is
        # currently the active segmentation source -- only that stage's
        # rows get a column-1 status dot; the other two 2D stages show
        # nothing there, which is itself informative (masks belong to one
        # stage at a time, never all three).
        seg_statuses = {}
        if stage in _2D_STAGES and stage == active_2d_stage:
            seg_statuses = project_core.segmentation_file_status(folder)

        self.tree.blockSignals(True)
        try:
            for path in paths:
                leaf = QTreeWidgetItem(top_item)
                leaf.setText(0, path.name)
                leaf.setData(0, Qt.UserRole, ("file", stage, str(path)))
                leaf.setFlags(leaf.flags() | Qt.ItemIsUserCheckable)
                # Checked by default: batch actions only ever touch checked
                # files (see checked_paths_for_stage), so defaulting to
                # unchecked-means-everything would make "process everything"
                # an invisible fallback rather than something the tree
                # actually shows you're about to do.
                leaf.setCheckState(0, Qt.Checked)

                if stage in _2D_STAGES:
                    status = statuses.get(path.stem)
                    up_to_date = bool(status and status.up_to_date)
                    upstream_available = bool(status and status.upstream_available)
                    if not upstream_available:
                        icon_state = "archived"
                        leaf.setToolTip(
                            0,
                            "Source folder for this stage isn't present on this computer "
                            "(likely moved to storage) -- can't verify freshness, showing "
                            "as up to date.",
                        )
                    elif stage == project_core.STAGE_DENOISED and up_to_date:
                        n_channels = len(denoise_channels_done.get(path.stem, ()))
                        icon_state = "partial" if n_channels == 1 else "done"
                    else:
                        icon_state = "done" if up_to_date else "stale"
                elif stage == RAW_STAGE:
                    produced = reduced_dir is not None and (reduced_dir / f"{path.stem}.tiff").exists()
                    icon_state = "done" if produced else "unprocessed"
                else:
                    icon_state = "done"
                leaf.setIcon(0, status_icon(icon_state))

                seg_status = seg_statuses.get(path.stem)
                if seg_status is not None:
                    seg_icon_state = "done" if seg_status.up_to_date else "stale"
                    leaf.setIcon(1, status_icon(seg_icon_state))
                    leaf.setToolTip(
                        1,
                        "Segmented, up to date"
                        if seg_status.up_to_date
                        else "Segmented, but the source file changed since",
                    )
        finally:
            self.tree.blockSignals(False)
        top_item.setExpanded(True)

    def _refresh_tiles_children(
        self, top_item, folder: Path, active_2d_stage: str | None, badge: str = ""
    ):
        """Break the "05 · Tiles" summary down into one row per source FOV
        (not per output .tif -- a FOV's cells all export together) --
        reads out_dir's tile_index.csv rather than re-deriving groupings
        from filenames, since that CSV is already the authoritative
        cell_id/fov_id mapping (see core/tiles.append_tile_index). The
        header count below is likewise summed from these per-FOV statuses,
        not a folder glob -- see the caller's note on why 05_tiles/ is
        never listed file-by-file."""
        mask_dir = self._stage_dir(active_2d_stage) if active_2d_stage else None
        fov_statuses = tiles_core.fov_tile_status(folder, mask_dir)
        n_cells = sum(status.n_cells for status in fov_statuses.values())
        top_item.setText(
            0,
            f"{_STAGE_LABELS[project_core.STAGE_TILES]}{badge}  "
            f"({len(fov_statuses)} FOV(s), {n_cells} cell(s))",
        )

        self.tree.blockSignals(True)
        try:
            for fov_id in sorted(fov_statuses, key=lambda stem: natsort_key(Path(stem))):
                status = fov_statuses[fov_id]
                leaf = QTreeWidgetItem(top_item)
                n = status.n_cells
                leaf.setText(0, f"{fov_id}  ({n} cell{'s' if n != 1 else ''})")
                leaf.setData(0, Qt.UserRole, ("tile_fov", project_core.STAGE_TILES, fov_id))
                if not status.mask_available:
                    leaf.setIcon(0, status_icon("archived"))
                    leaf.setToolTip(
                        0,
                        "Source/mask folder for this FOV isn't present on this computer "
                        "(likely moved to storage) -- can't verify freshness, showing as "
                        "up to date.",
                    )
                elif status.up_to_date:
                    leaf.setIcon(0, status_icon("done"))
                else:
                    leaf.setIcon(0, status_icon("stale"))
                    leaf.setToolTip(
                        0, "Segmentation for this FOV changed since these tiles were exported."
                    )
        finally:
            self.tree.blockSignals(False)
        top_item.setExpanded(False)

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
