"""Segmentation page: cellpose inference over the brightfield channel of
already-produced combined-channel tiffs (design.md stage 2b onward).
Self-contained -- owns its own 2D images folder, batch worker, and status
feedback; works standalone off whatever's already on disk, with no
dependency on the Data Reduction page being open (or ever having run in
this session). The one convenience that legitimately needs the *other*
page's state -- "use the Data Reduction output folder" -- is
deliberately not wired here; the shell (which is the only thing that
knows about both pages) connects that button's click.
"""

import re
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

from yeastprep.core.pipeline import load_brightfield_channel
from yeastprep.core.segmentation import SegmentationParams, load_saved_masks, seg_npy_path

from .. import segmentation_folder_config, settings
from ..diagnostics.segmentation_preview_panel import SegmentationPreviewPanel
from ..file_list_panel import FileListPanel
from ..segmentation_folder_panel import SegmentationFolderPanel
from ..segmentation_params_panel import SegmentationParamsPanel
from ..worker import SegmentationBatchWorker, SegmentationController
from .page_progress import PageProgress

_DIGITS = re.compile(r"(\d+)")


def _natsort_key(path: Path):
    parts = _DIGITS.split(path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


class SegmentationPage(QWidget):
    progress_changed = Signal(object)  # PageProgress

    def __init__(self, parent=None):
        super().__init__(parent)

        self._last_focal_slice = None
        self._focal_slice_generation = 0
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

        # Deliberately insulated from the Data Reduction page's raw
        # input/output folders: this points directly at a folder of
        # focal-slice tiffs, so already-processed images can be browsed
        # and segmented without that other page ever being opened.
        self.segmentation_folder_panel = SegmentationFolderPanel()
        left_layout.addWidget(self.segmentation_folder_panel)

        # Click handler wired by the shell, not here -- see module docstring.
        self.use_flatten_output_btn = QPushButton("Use Data Reduction output folder")
        left_layout.addWidget(self.use_flatten_output_btn)

        self.file_list_panel = FileListPanel()
        left_layout.addWidget(self.file_list_panel, 1)

        # No cross-page auto-refresh (pages are insulated -- see module
        # docstring), so this is the way to pick up files that appeared
        # after the folder was first pointed here: a batch run finishing
        # on the Data Reduction page, new corrected masks written by the
        # real Cellpose GUI, files dropped in from elsewhere, etc.
        self.refresh_btn = QPushButton("Refresh file list")
        left_layout.addWidget(self.refresh_btn)

        self.segmentation_params_panel = SegmentationParamsPanel()
        left_layout.addWidget(self.segmentation_params_panel)

        left.setMinimumWidth(320)
        left.setMaximumWidth(420)
        row.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_panel = SegmentationPreviewPanel()
        right_layout.addWidget(self.preview_panel, 1)

        batch_row = QHBoxLayout()
        self.segment_btn = QPushButton("Segment && Save Selected")
        batch_row.addWidget(self.segment_btn)
        self.open_cellpose_gui_btn = QPushButton("Open Folder in Cellpose GUI")
        batch_row.addWidget(self.open_cellpose_gui_btn)
        right_layout.addLayout(batch_row)

        row.addWidget(right, 1)

        self.status_label = QLabel("")
        outer.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Wiring

    def _wire_up(self):
        self.segmentation_folder_panel.folder_changed.connect(self._on_folder_changed)
        self.file_list_panel.file_selected.connect(self._on_file_selected)

        self.controller = SegmentationController()
        self.controller.result_ready.connect(self._on_result_ready)
        self.controller.error.connect(self._on_error)

        self.segmentation_params_panel.params_changed.connect(self._on_params_changed)
        self.segmentation_params_panel.recompute_requested.connect(self._recompute_now)
        self.segmentation_params_panel.save_defaults_requested.connect(
            self._save_folder_defaults
        )
        self.segmentation_params_panel.reset_defaults_requested.connect(
            self._reset_folder_defaults
        )
        self.segment_btn.clicked.connect(self._start_batch)
        self.open_cellpose_gui_btn.clicked.connect(self._open_cellpose_gui)
        self.refresh_btn.clicked.connect(self._rescan_folder)

    # ------------------------------------------------------------------
    # Folder handling

    def _on_folder_changed(self, folder: str):
        self._rescan_folder()
        params = segmentation_folder_config.load_segmentation_folder_config(folder)
        if params:
            self.segmentation_params_panel.set_params(params)
            self.status_label.setText(
                "Loaded segmentation parameters from a previous run in this folder."
            )

    def _rescan_folder(self):
        folder = self.segmentation_folder_panel.folder()
        if not folder or not Path(folder).is_dir():
            self.file_list_panel.set_files([])
            return
        paths = sorted(Path(folder).glob("*.tiff"), key=_natsort_key)
        self.file_list_panel.set_files([str(p) for p in paths])

    # ------------------------------------------------------------------
    # File selection -> live preview (from disk, or a saved mask if one
    # already exists)

    def _on_file_selected(self, path: str):
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
    # Folder-scoped defaults

    def _save_folder_defaults(self):
        folder = self.segmentation_folder_panel.folder()
        if not folder:
            QMessageBox.warning(self, "yeastprep", "Select a 2D images folder first.")
            return
        segmentation_folder_config.save_segmentation_folder_config(
            folder, self.segmentation_params_panel.params()
        )
        self.status_label.setText("Saved as segmentation folder defaults.")

    def _reset_folder_defaults(self):
        folder = self.segmentation_folder_panel.folder()
        params = (
            segmentation_folder_config.load_segmentation_folder_config(folder)
            if folder
            else None
        )
        self.segmentation_params_panel.set_params(
            params or settings.get_default_segmentation_params()
        )

    # ------------------------------------------------------------------
    # Batch segmentation

    def _start_batch(self):
        folder = self.segmentation_folder_panel.folder()
        if not folder:
            QMessageBox.warning(self, "yeastprep", "Select a 2D images folder first.")
            return

        paths = [Path(p) for p in self.file_list_panel.checked_paths()]
        if not paths:
            QMessageBox.warning(
                self,
                "yeastprep",
                f"No focal-slice tiffs found in {folder}.\n"
                "Process some raw stacks into focal slices first (Data Reduction "
                "page), or point this folder at an existing set of them.",
            )
            return

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
        self.progress_changed.emit(PageProgress(active=True, done=0, total=len(paths)))

        self._batch_thread.start()

    def _on_batch_progress(self, done: int, total: int, name: str):
        self.progress_changed.emit(PageProgress(active=True, done=done, total=total, message=name))
        self.status_label.setText(f"Segmenting {name} ({done}/{total})")

    def _on_batch_file_result(self, result):
        self.file_list_panel.mark_result(str(result.path), result.success, result.error)
        if result.success:
            self.status_label.setText(f"{result.path.name}: {result.n_cells} cells")
        else:
            self.status_label.setText(f"{result.path.name}: {result.error}")

    def _on_batch_finished(self):
        self.progress_changed.emit(PageProgress(active=False))
        self.segment_btn.setEnabled(True)
        self.status_label.setText("Batch segmentation complete.")

        self._batch_thread.quit()
        self._batch_thread.wait()
        self._batch_thread = None
        self._batch_worker = None

    def _open_cellpose_gui(self):
        folder = self.segmentation_folder_panel.folder()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "yeastprep", "Select a 2D images folder first.")
            return
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
