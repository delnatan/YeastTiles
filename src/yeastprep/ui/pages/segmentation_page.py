"""Segmentation page: cellpose inference over the brightfield channel of
whichever 2D stage folder is currently the project's active segmentation
source (01_reduced/, 02_denoised/, or 03_deconvolved/ -- see the
"Segmentation / Tile Generation source" control on the shared
ProjectTreePanel). Writes cellpose's native `_seg.npy` sidecar next to each
source tiff, so the real Cellpose GUI can open that same folder directly
for manual correction.
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
from yeastprep.core.segmentation import SegmentationParams, load_saved_masks, seg_npy_path

from .. import settings
from ..batch_progress_bar import BatchProgressBar
from ..common.preview_source_label import PreviewSourceLabel
from ..diagnostics.segmentation_preview_panel import SegmentationPreviewPanel
from ..project_tree_panel import ProjectTreePanel
from ..segmentation_params_panel import SegmentationParamsPanel
from ..worker import SegmentationBatchWorker, SegmentationController
from .page_progress import PageProgress


class SegmentationPage(QWidget):
    progress_changed = Signal(object)  # PageProgress

    def __init__(self, tree_panel: ProjectTreePanel, parent=None):
        super().__init__(parent)
        self.tree_panel = tree_panel

        self._last_focal_slice = None
        self._focal_slice_generation = 0
        self._source_stage = None
        self._batch_thread = None
        self._batch_worker = None

        self._build_ui()
        self._wire_up()

        self.segmentation_params_panel.set_params(settings.get_default_segmentation_params())

    # ------------------------------------------------------------------
    # UI construction

    def _build_ui(self):
        outer = QVBoxLayout(self)

        row = QHBoxLayout()
        outer.addLayout(row, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.segmentation_params_panel = SegmentationParamsPanel()
        left_layout.addWidget(self.segmentation_params_panel)
        left_layout.addStretch(1)

        left.setMinimumWidth(320)
        left.setMaximumWidth(420)
        row.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_source_label = PreviewSourceLabel()
        right_layout.addWidget(self.preview_source_label)
        self.preview_panel = SegmentationPreviewPanel()
        right_layout.addWidget(self.preview_panel, 1)

        batch_row = QHBoxLayout()
        self.segment_btn = QPushButton("Segment && Save Selected")
        batch_row.addWidget(self.segment_btn)
        self.open_cellpose_gui_btn = QPushButton("Open Folder in Cellpose GUI")
        batch_row.addWidget(self.open_cellpose_gui_btn)
        self.batch_progress = BatchProgressBar()
        batch_row.addWidget(self.batch_progress, 1)
        right_layout.addLayout(batch_row)

        row.addWidget(right, 1)

        self.status_label = QLabel("")
        outer.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Wiring

    def _wire_up(self):
        self.tree_panel.file_preview_requested.connect(self._on_file_selected)
        self.tree_panel.segmentation_source_changed.connect(self._on_source_changed)
        self.tree_panel.project_root_changed.connect(self._on_project_root_changed)

        self.controller = SegmentationController()
        self.controller.result_ready.connect(self._on_result_ready)
        self.controller.error.connect(self._on_error)

        self.segmentation_params_panel.params_changed.connect(self._on_params_changed)
        self.segmentation_params_panel.recompute_requested.connect(self._recompute_now)
        self.segmentation_params_panel.save_defaults_requested.connect(
            self._save_project_defaults
        )
        self.segmentation_params_panel.reset_defaults_requested.connect(
            self._reset_project_defaults
        )
        self.segment_btn.clicked.connect(self._start_batch)
        self.open_cellpose_gui_btn.clicked.connect(self._open_cellpose_gui)

    # ------------------------------------------------------------------
    # Source / project handling

    def _on_source_changed(self):
        self._source_stage = self.tree_panel.active_2d_stage()

    def _on_project_root_changed(self, root: str):
        self._source_stage = self.tree_panel.active_2d_stage()
        config = project_core.load_project_config(root)
        if config:
            self.segmentation_params_panel.set_params(config.segmentation_params)
            self.status_label.setText("Loaded segmentation parameters from this project.")

    # ------------------------------------------------------------------
    # File selection -> live preview (from disk, or a saved mask if one
    # already exists)

    def _on_file_selected(self, stage: str, path: str):
        if stage != self.tree_panel.active_2d_stage():
            return
        self.preview_source_label.set_path(path)
        try:
            image = load_brightfield_channel(path)
        except Exception as exc:
            self.status_label.setText(f"Failed to load {path}: {exc}")
            return
        self._set_current_focal_slice(image)

        # Fast inspect path: a saved mask already exists (from a batch
        # run, or corrected in the real Cellpose GUI) -- show it as-is
        # rather than silently re-running cellpose and overwriting the
        # view with a fresh, possibly-different result.
        masks = load_saved_masks(seg_npy_path(path))
        if masks is not None:
            self.preview_panel.set_data(image, masks)
            n_cells = int(masks.max())
            self.status_label.setText(f"Loaded saved segmentation ({n_cells} cells): {Path(path).name}")
            return

        if self.segmentation_params_panel.is_auto_recompute():
            self.controller.schedule(
                self._last_focal_slice,
                self._focal_slice_generation,
                self.segmentation_params_panel.params(),
            )

    def _set_current_focal_slice(self, focal_slice):
        self._last_focal_slice = focal_slice
        self._focal_slice_generation += 1

    def _on_params_changed(self, params: SegmentationParams):
        if self._last_focal_slice is None:
            return
        if not self.segmentation_params_panel.is_auto_recompute():
            return
        self.controller.schedule(self._last_focal_slice, self._focal_slice_generation, params)

    def _recompute_now(self):
        if self._last_focal_slice is None:
            return
        self.controller.recompute_now(
            self._last_focal_slice,
            self._focal_slice_generation,
            self.segmentation_params_panel.params(),
        )

    def _on_result_ready(self, result):
        if result.request_id != self.controller.latest_request_id():
            return  # superseded by a newer request already queued
        self.preview_panel.set_data(result.focal_slice, result.result.masks)

    def _on_error(self, message: str):
        self.status_label.setText(f"Segmentation error: {message}")

    # ------------------------------------------------------------------
    # Project-scoped defaults

    def _save_project_defaults(self):
        root = self.tree_panel.project_root()
        if not root:
            QMessageBox.warning(self, "yeastprep", "Open a project first.")
            return
        config = project_core.load_project_config(root) or project_core.ProjectConfig()
        config.segmentation_params = self.segmentation_params_panel.params()
        project_core.save_project_config(root, config)
        self.status_label.setText("Saved as project defaults.")

    def _reset_project_defaults(self):
        root = self.tree_panel.project_root()
        config = project_core.load_project_config(root) if root else None
        self.segmentation_params_panel.set_params(
            config.segmentation_params if config else settings.get_default_segmentation_params()
        )

    # ------------------------------------------------------------------
    # Batch segmentation

    def _start_batch(self):
        source_stage = self.tree_panel.active_2d_stage()
        if not source_stage:
            QMessageBox.warning(
                self,
                "yeastprep",
                "No 2D images available yet -- run Data Reduction (and optionally "
                "Denoise/Deconvolve) first.",
            )
            return

        paths = [Path(p) for p in self.tree_panel.checked_paths_for_stage(source_stage)]
        if not paths:
            QMessageBox.warning(
                self, "yeastprep", "No files checked -- check at least one file in the tree."
            )
            return

        self._source_stage = source_stage
        self._batch_thread = QThread()
        self._batch_worker = SegmentationBatchWorker(
            paths, self.segmentation_params_panel.params()
        )
        self._batch_worker.moveToThread(self._batch_thread)
        self._thread_started_connection = self._batch_thread.started.connect(
            self._batch_worker.run
        )
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.file_result.connect(self._on_batch_file_result)
        self._batch_worker.finished.connect(self._on_batch_finished)

        self.segment_btn.setEnabled(False)
        self._emit_progress(PageProgress(active=True, done=0, total=len(paths)))

        self._batch_thread.start()

    def _emit_progress(self, progress: PageProgress):
        self.progress_changed.emit(progress)
        self.batch_progress.apply(progress)

    def _on_batch_progress(self, done: int, total: int, name: str):
        self._emit_progress(PageProgress(active=True, done=done, total=total, message=name))
        self.status_label.setText(f"Segmenting {name} ({done}/{total})")

    def _on_batch_file_result(self, result):
        if self._source_stage:
            self.tree_panel.mark_result(
                self._source_stage, result.path, result.success, result.error
            )
        if result.success:
            self.status_label.setText(f"{result.path.name}: {result.n_cells} cells")
        else:
            self.status_label.setText(f"{result.path.name}: {result.error}")

    def _on_batch_finished(self):
        self._emit_progress(PageProgress(active=False))
        self.segment_btn.setEnabled(True)
        self.status_label.setText("Batch segmentation complete.")

        root = self.tree_panel.project_root()
        if root and self._source_stage:
            processed = [
                Path(p).stem
                for p in self.tree_panel.checked_paths_for_stage(self._source_stage)
            ]
            project_core.mark_stage_run(root, "segmentation", processed)
        self.tree_panel.refresh()

        self._batch_thread.quit()
        self._batch_thread.wait()
        self._batch_thread = None
        self._batch_worker = None

    def _open_cellpose_gui(self):
        source_stage = self.tree_panel.active_2d_stage()
        paths_root = self.tree_panel.project_paths()
        if not source_stage or paths_root is None:
            QMessageBox.warning(self, "yeastprep", "No 2D images folder available yet.")
            return
        folder = str(paths_root.stage_dir(source_stage))
        subprocess.Popen(
            [sys.executable, "-m", "cellpose", "--dir", folder],
            start_new_session=True,
        )

    # ------------------------------------------------------------------

    def shutdown(self):
        settings.set_default_segmentation_params(self.segmentation_params_panel.params())
        self.controller.shutdown()
        if self._batch_thread is not None:
            self._batch_thread.quit()
            self._batch_thread.wait()
