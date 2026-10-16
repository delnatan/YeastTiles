"""Segmentation page: cellpose inference over the brightfield channel of
whichever 2D stage folder is currently the project's active segmentation
source (01_reduced/, 02_denoised/, or 03_deconvolved/ -- see the
"Segmentation / Tile Generation source" control on the shared
ProjectTreePanel). Writes cellpose's native `_seg.npy` sidecar next to each
FOV's brightfield-only companion tiff (`combined_tiff.write_brightfield_tiff`),
so the real Cellpose GUI can open that same folder directly for manual
correction.
"""

import subprocess
import sys
from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core import project as project_core
from yeastprep.core.combined_tiff import load_brightfield_channel, write_brightfield_tiff
from yeastprep.core.fs_status import list_visible
from yeastprep.core.segmentation import SegmentationParams, load_saved_masks, seg_npy_path

from .. import settings
from ..common.background_task import BackgroundTask
from ..common.batch_runner import BatchRunner
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
        self._batch_root = None
        self._cellpose_prep = BackgroundTask(self)  # brightfield copies for the GUI
        self._batch_succeeded_stems = []

        self._build_ui()
        self._wire_up()

        self.segmentation_params_panel.set_params(settings.get_default_segmentation_params())
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
        self.batch_runner = BatchRunner("Batch segmentation")
        batch_row.addWidget(self.batch_runner, 1)
        right_layout.addLayout(batch_row)

        row.addWidget(right, 1)

        self.status_label = QLabel("")
        outer.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Wiring

    def _wire_up(self):
        self.tree_panel.segmentation_source_changed.connect(self._on_source_changed)
        self.tree_panel.segmentation_source_changed.connect(self._update_batch_button_state)
        self.tree_panel.project_root_changed.connect(self._on_project_root_changed)
        self.tree_panel.project_root_changed.connect(self._update_batch_button_state)
        self.tree_panel.checked_changed.connect(self._update_batch_button_state)
        self.tree_panel.refreshed.connect(self._update_batch_button_state)

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
        self.batch_runner.progress_changed.connect(self.progress_changed)
        self.batch_runner.progress_changed.connect(self._on_batch_progress)
        self.batch_runner.file_result.connect(self._on_batch_file_result)
        self.batch_runner.finished.connect(self._on_batch_finished)
        self.open_cellpose_gui_btn.clicked.connect(self._open_cellpose_gui)
        self._cellpose_prep.succeeded.connect(self._launch_cellpose_gui)
        self._cellpose_prep.failed.connect(self._on_cellpose_prep_failed)

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

    def load_selection(self, stage: str, path: str, mode: str = "live"):
        # A preview still running (or debounced) for the previous file
        # would otherwise land on top of this one.
        self.controller.invalidate()
        self.preview_source_label.set_path(path)
        try:
            image = load_brightfield_channel(path)
        except Exception as exc:
            self._last_focal_slice = None
            self.preview_panel.clear()
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

        # Not segmented yet: show the image itself rather than leaving the
        # previous file's image and outlines up.
        self.preview_panel.set_data(image, None)
        if self.segmentation_params_panel.is_auto_recompute():
            self.status_label.setText(f"Segmenting {Path(path).name}...")
            self.controller.schedule(
                self._last_focal_slice,
                self._focal_slice_generation,
                self.segmentation_params_panel.params(),
            )
        else:
            self.status_label.setText(
                f"{Path(path).name}: not segmented yet. Click 'Do it' to preview."
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
        self.status_label.setText("Live segmentation preview (not saved).")

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

    def _update_batch_button_state(self, *_args):
        source_stage = self.tree_panel.active_2d_stage()
        checked = (
            self.tree_panel.checked_paths_for_stage(source_stage) if source_stage else []
        )
        running = self.batch_runner.is_running()
        self.segment_btn.setEnabled(bool(checked) and not running)
        if running:
            self.segment_btn.setToolTip("A batch is already running -- see the progress bar.")
        elif not source_stage:
            self.segment_btn.setToolTip(
                "No 2D images available yet -- run Data Reduction (and optionally "
                "Denoise/Deconvolve) first."
            )
        elif not checked:
            self.segment_btn.setToolTip("Check at least one file in the tree to segment.")
        else:
            self.segment_btn.setToolTip("")

    def _start_batch(self):
        if self.batch_runner.is_running():
            return
        source_stage = self.tree_panel.active_2d_stage()
        paths = [Path(p) for p in self.tree_panel.checked_paths_for_stage(source_stage)]
        if not source_stage or not paths:
            return
        self._source_stage = source_stage
        self._batch_root = self.tree_panel.project_root()
        self._batch_succeeded_stems = []
        self.batch_runner.start(
            SegmentationBatchWorker(paths, self.segmentation_params_panel.params())
        )
        self._update_batch_button_state()

    def _on_batch_progress(self, progress: PageProgress):
        if progress.active and progress.message:
            self.status_label.setText(
                f"Segmenting {progress.message} ({progress.done}/{progress.total})"
            )

    def _on_batch_file_result(self, result):
        self.tree_panel.mark_result(self._source_stage, result.path, result.success, result.error)
        if result.success:
            self._batch_succeeded_stems.append(result.path.stem)
            self.status_label.setText(f"{result.path.name}: {result.n_cells} cells")
        else:
            self.status_label.setText(f"{result.path.name}: {result.error}")

    def _on_batch_finished(self, outcome):
        self._update_batch_button_state()
        self.status_label.setText(outcome.summary())
        project_core.mark_stage_run(self._batch_root, "segmentation", self._batch_succeeded_stems)
        self.tree_panel.refresh()

    def busy_description(self) -> str | None:
        return self.batch_runner.describe_running()

    def _open_cellpose_gui(self):
        if self._cellpose_prep.is_running():
            return
        source_stage = self.tree_panel.active_2d_stage()
        paths_root = self.tree_panel.project_paths()
        if not source_stage or paths_root is None:
            QMessageBox.warning(self, "yeastprep", "No 2D images folder available yet.")
            return
        folder = paths_root.stage_dir(source_stage)
        # Cellpose 4's SAM model dropped channel selection, so opening the
        # 2-channel combined tiffs directly would segment/train on the
        # target channel too -- the GUI opens the brightfield-only
        # companion tiffs instead, written here if missing/stale (they live
        # in combined_tiff.CELLPOSE_GUI_SUBDIR, so this glob never sees
        # them). A whole folder's worth takes a while the first time, so
        # it's done off the GUI thread.
        combined_images = sorted(list_visible(folder, "*.tiff"))
        self.open_cellpose_gui_btn.setEnabled(False)
        self.status_label.setText(
            f"Preparing {len(combined_images)} brightfield image(s) for Cellpose..."
        )
        self._cellpose_prep.start(
            lambda: [write_brightfield_tiff(p) for p in combined_images]
        )

    def _launch_cellpose_gui(self, images):
        self.open_cellpose_gui_btn.setEnabled(True)
        # Cellpose 4.x's CLI runs a *headless* batch job whenever --dir/
        # --image_path is given -- passing --dir no longer opens the GUI
        # like it did pre-4.0. Drive cellpose.gui.gui.run() directly
        # instead, preloading the first tiff (and its _seg.npy sidecar, if
        # any); the GUI's own next/prev navigation then covers the rest of
        # the folder.
        launcher = (
            "from cellpose.gui import gui\n"
            "import sys\n"
            "gui.run(image=sys.argv[1] if len(sys.argv) > 1 else None)\n"
        )
        args = [sys.executable, "-c", launcher]
        if images:
            args.append(str(images[0]))
        try:
            subprocess.Popen(args, start_new_session=True)
        except OSError as exc:
            self.status_label.setText(f"Couldn't start Cellpose: {exc}")
            return
        self.status_label.setText("Opening Cellpose GUI (it takes a moment to start)...")

    def _on_cellpose_prep_failed(self, message: str):
        self.open_cellpose_gui_btn.setEnabled(True)
        self.status_label.setText(f"Couldn't prepare images for Cellpose: {message}")

    # ------------------------------------------------------------------

    def shutdown(self):
        settings.set_default_segmentation_params(self.segmentation_params_panel.params())
        self.controller.shutdown()
        self._cellpose_prep.shutdown()
        self.batch_runner.shutdown()
