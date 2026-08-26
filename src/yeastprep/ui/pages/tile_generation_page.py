"""Tile Generation page: crops every segmented cell out of whichever 2D
stage folder is currently the project's active segmentation source (see
the shared ProjectTreePanel's "Segmentation / Tile Generation source"
control), using whichever mask each file already has saved, and writes the
crops into the project's 05_tiles/. Works standalone off whatever's already
on disk -- requires only that Segmentation has run (in this session or a
previous one) on the active source stage.
"""

import subprocess
import sys
from pathlib import Path

from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core import project as project_core
from yeastprep.core.combined_tiff import load_brightfield_channel
from yeastprep.core.segmentation import load_saved_masks, seg_npy_path
from yeastprep.core.tiles import TileParams

from .. import settings
from ..batch_progress_bar import BatchProgressBar
from ..common.preview_source_label import PreviewSourceLabel
from ..diagnostics.tile_generation_preview_panel import TileGenerationPreviewPanel
from ..project_tree_panel import ProjectTreePanel
from ..tile_params_panel import TileParamsPanel
from ..worker import TileBatchWorker
from .page_progress import PageProgress


class TileGenerationPage(QWidget):
    progress_changed = Signal(object)  # PageProgress

    def __init__(self, tree_panel: ProjectTreePanel, parent=None):
        super().__init__(parent)
        self.tree_panel = tree_panel

        self._last_focal_slice = None
        self._last_masks = None
        self._source_stage = None
        self._batch_thread = None
        self._batch_worker = None

        self._build_ui()
        self._wire_up()

        self.tile_params_panel.set_params(settings.get_default_tile_params())
        self._update_batch_button_state()

    # ------------------------------------------------------------------
    # UI construction

    def _build_ui(self):
        outer = QVBoxLayout(self)

        row = QHBoxLayout()
        outer.addLayout(row, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.tile_params_panel = TileParamsPanel()
        left_layout.addWidget(self.tile_params_panel)
        left_layout.addStretch(1)

        left.setMinimumWidth(320)
        left.setMaximumWidth(420)
        row.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_source_label = PreviewSourceLabel()
        right_layout.addWidget(self.preview_source_label)
        self.preview_panel = TileGenerationPreviewPanel()
        right_layout.addWidget(self.preview_panel, 1)

        batch_row = QHBoxLayout()
        self.export_btn = QPushButton("Export Tiles for Selected")
        batch_row.addWidget(self.export_btn)
        self.open_tile_viewer_btn = QPushButton("Open in Tile Viewer")
        batch_row.addWidget(self.open_tile_viewer_btn)
        self.batch_progress = BatchProgressBar()
        batch_row.addWidget(self.batch_progress, 1)
        right_layout.addLayout(batch_row)

        row.addWidget(right, 1)

        self.status_label = QLabel("")
        outer.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Wiring

    def _wire_up(self):
        self.tree_panel.project_root_changed.connect(self._on_project_root_changed)
        self.tree_panel.project_root_changed.connect(self._update_batch_button_state)
        self.tree_panel.checked_changed.connect(self._update_batch_button_state)
        self.tree_panel.refreshed.connect(self._update_batch_button_state)
        self.tree_panel.segmentation_source_changed.connect(self._update_batch_button_state)

        self.tile_params_panel.params_changed.connect(self._on_params_changed)
        self.tile_params_panel.recompute_requested.connect(self._recompute_now)
        self.tile_params_panel.save_defaults_requested.connect(self._save_project_defaults)
        self.tile_params_panel.reset_defaults_requested.connect(self._reset_project_defaults)

        self.export_btn.clicked.connect(self._start_batch)
        self.open_tile_viewer_btn.clicked.connect(lambda _checked=False: self._open_tile_viewer())

    # ------------------------------------------------------------------
    # Project handling

    def _on_project_root_changed(self, root: str):
        config = project_core.load_project_config(root)
        if config:
            self.tile_params_panel.set_params(config.tile_params)
            self.status_label.setText("Loaded tile parameters from this project.")

    # ------------------------------------------------------------------
    # File selection -> live crop-window preview (requires a saved mask;
    # tile export never runs segmentation itself)

    def load_selection(self, stage: str, path: str, mode: str = "live"):
        if mode == "open_viewer_fov":
            self._open_tile_viewer(fov_filter=[path])
            return
        self.preview_source_label.set_path(path)
        masks = load_saved_masks(seg_npy_path(path))
        if masks is None:
            self._last_focal_slice = None
            self._last_masks = None
            self.preview_panel.clear()
            self.status_label.setText(
                f"{Path(path).name}: no saved segmentation found -- run Segmentation first."
            )
            return

        try:
            focal_slice = load_brightfield_channel(path)
        except Exception as exc:
            self.status_label.setText(f"Failed to load {path}: {exc}")
            return

        self._last_focal_slice = focal_slice
        self._last_masks = masks
        self.preview_panel.set_data(focal_slice, masks, self.tile_params_panel.params().size)
        n_cells = int(masks.max())
        self.status_label.setText(
            f"{Path(path).name}: {n_cells} cell{'s' if n_cells != 1 else ''} with saved segmentation"
        )

    def _on_params_changed(self, _params: TileParams):
        if not self.tile_params_panel.is_auto_recompute():
            return
        self._redraw_preview()

    def _recompute_now(self):
        self._redraw_preview()

    def _redraw_preview(self):
        if self._last_masks is None:
            return
        self.preview_panel.set_data(
            self._last_focal_slice, self._last_masks, self.tile_params_panel.params().size
        )

    # ------------------------------------------------------------------
    # Project-scoped defaults

    def _save_project_defaults(self):
        root = self.tree_panel.project_root()
        if not root:
            QMessageBox.warning(self, "yeastprep", "Open a project first.")
            return
        config = project_core.load_project_config(root) or project_core.ProjectConfig()
        config.tile_params = self.tile_params_panel.params()
        project_core.save_project_config(root, config)
        self.status_label.setText("Saved as project defaults.")

    def _reset_project_defaults(self):
        root = self.tree_panel.project_root()
        config = project_core.load_project_config(root) if root else None
        self.tile_params_panel.set_params(
            config.tile_params if config else settings.get_default_tile_params()
        )

    # ------------------------------------------------------------------
    # Batch export

    def _update_batch_button_state(self, *_args):
        paths_root = self.tree_panel.project_paths()
        source_stage = self.tree_panel.active_2d_stage() if paths_root else None
        checked = (
            self.tree_panel.checked_paths_for_stage(source_stage) if source_stage else []
        )
        self.export_btn.setEnabled(bool(checked))
        if not paths_root or not source_stage:
            self.export_btn.setToolTip(
                "No 2D images available yet -- run Data Reduction (and Segmentation) first."
            )
        elif not checked:
            self.export_btn.setToolTip("Check at least one file in the tree to export.")
        else:
            self.export_btn.setToolTip("")

        self.open_tile_viewer_btn.setText(self._tile_viewer_label(source_stage, checked))

    def _tile_viewer_label(self, source_stage, checked: list[str]) -> str:
        if not source_stage:
            return "Open in Tile Viewer"
        total = len(self.tree_panel.all_paths_for_stage(source_stage))
        if checked and len(checked) < total:
            return f"Open in Tile Viewer ({len(checked)} of {total} FOVs)"
        return "Open in Tile Viewer (all FOVs)"

    def _start_batch(self):
        paths_root = self.tree_panel.project_paths()
        source_stage = self.tree_panel.active_2d_stage()
        paths = [Path(p) for p in self.tree_panel.checked_paths_for_stage(source_stage)]
        self._source_stage = source_stage
        outdir = paths_root.tiles

        self._batch_thread = QThread()
        self._batch_worker = TileBatchWorker(paths, outdir, self.tile_params_panel.params())
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.file_result.connect(self._on_batch_file_result)
        self._batch_worker.finished.connect(self._on_batch_finished)

        self.export_btn.setEnabled(False)
        self._emit_progress(PageProgress(active=True, done=0, total=len(paths)))

        self._batch_thread.start()

    def _emit_progress(self, progress: PageProgress):
        self.progress_changed.emit(progress)
        self.batch_progress.apply(progress)

    def _on_batch_progress(self, done: int, total: int, name: str):
        self._emit_progress(PageProgress(active=True, done=done, total=total, message=name))
        self.status_label.setText(f"Exporting tiles from {name} ({done}/{total})")

    def _on_batch_file_result(self, result):
        if self._source_stage:
            self.tree_panel.mark_result(
                self._source_stage, result.path, result.success, result.error
            )
        if result.success:
            self.status_label.setText(f"{result.path.name}: {result.n_cells} tile(s) exported")
        else:
            self.status_label.setText(f"{result.path.name}: {result.error}")

    def _on_batch_finished(self):
        self._emit_progress(PageProgress(active=False))
        self._update_batch_button_state()
        self.status_label.setText("Batch tile export complete.")
        self.tree_panel.refresh()

        self._batch_thread.quit()
        self._batch_thread.wait()
        self._batch_thread = None
        self._batch_worker = None

    def _open_tile_viewer(self, fov_filter: list[str] | None = None):
        """Launch the standalone tile-annotation viewer as a subprocess,
        scoped to `fov_filter` if given (a single-FOV selection action --
        see `load_selection`'s "open_viewer_fov" mode), else to whichever
        FOVs are currently checked under the active 2D stage -- same scope
        `open_tile_viewer_btn`'s label already advertises
        (`_tile_viewer_label`), so what opens always matches what was shown
        before the click.

        Each FOV's crops live in their own `05_tiles/<fov_id>/` subfolder
        (see core/tiles.py's `export_tiles`), so every target FOV is passed
        to tileclass as its own positional folder rather than as a `--fov`
        filter on the shared root -- tileclass's `PooledAnnotations` then
        gives each FOV its own annotation sidecar file instead of one
        shared file covering every FOV."""
        paths_root = self.tree_panel.project_paths()
        if paths_root is None or not paths_root.tiles.is_dir():
            QMessageBox.warning(self, "yeastprep", "No tiles exported yet.")
            return

        if fov_filter is None:
            source_stage = self.tree_panel.active_2d_stage()
            if source_stage:
                all_paths = self.tree_panel.all_paths_for_stage(source_stage)
                checked_paths = self.tree_panel.checked_paths_for_stage(source_stage)
                if checked_paths and len(checked_paths) < len(all_paths):
                    fov_filter = [Path(p).stem for p in checked_paths]

        if fov_filter:
            fov_dirs = [paths_root.tiles / fov_id for fov_id in fov_filter]
        else:
            fov_dirs = sorted(p for p in paths_root.tiles.iterdir() if p.is_dir())
        fov_dirs = [d for d in fov_dirs if d.is_dir()]

        if not fov_dirs:
            QMessageBox.warning(self, "yeastprep", "No tiles exported yet.")
            return

        cmd = [sys.executable, "-m", "tileclass", *[str(d) for d in fov_dirs]]
        subprocess.Popen(cmd, start_new_session=True)

    # ------------------------------------------------------------------

    def shutdown(self):
        settings.set_default_tile_params(self.tile_params_panel.params())
        if self._batch_thread is not None:
            self._batch_thread.quit()
            self._batch_thread.wait()
