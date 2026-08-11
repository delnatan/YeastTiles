"""Data Reduction page: raw multi-channel Z-stacks -> a single combined
2D tiff per FOV (flattened brightfield + sum-projected target channel,
design.md stages 1-2b). Self-contained -- owns its own folder selection,
batch worker, and status feedback; the Segmentation page never reaches
into this one.
"""

import re
from pathlib import Path

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core.channels import infer_channel_selection
from yeastprep.core.pipeline import (
    DEFAULT_CHANNELS,
    find_processed_output,
    load_brightfield_channel,
)

from .. import folder_config, settings
from ..diagnostics.coarse_heatmap_panel import CoarseHeatmapPanel
from ..diagnostics.focal_preview_panel import FocalPreviewPanel
from ..diagnostics.focus_surface_panel import FocusSurfacePanel
from ..diagnostics.raw_slice_panel import RawSlicePanel
from ..diagnostics.tile_variance_panel import TileVarianceInspectorPanel
from ..file_list_panel import FileListPanel
from ..folder_panel import FolderPanel
from ..params_panel import ParamsPanel
from ..rawstack.panel import RawStackViewerPanel
from ..worker import BatchProcessWorker, PipelineController
from .page_progress import PageProgress

_DIGITS = re.compile(r"(\d+)")


def _natsort_key(path: Path):
    parts = _DIGITS.split(path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


class DataReductionPage(QWidget):
    progress_changed = Signal(object)  # PageProgress

    def __init__(self, parent=None):
        super().__init__(parent)

        self._channels_inferred_for_folder = False
        self._last_diagnostics = None
        self._batch_thread = None
        self._batch_worker = None

        self._build_ui()
        self._wire_up()

        default_params = settings.get_default_params()
        self.params_panel.set_params(default_params)
        default_channels = settings.get_default_channels() or DEFAULT_CHANNELS
        self.params_panel.set_channels(default_channels)

    # ------------------------------------------------------------------
    # UI construction

    def _build_ui(self):
        outer = QVBoxLayout(self)

        row = QHBoxLayout()
        outer.addLayout(row, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.folder_panel = FolderPanel()
        left_layout.addWidget(self.folder_panel)

        self.file_list_panel = FileListPanel()
        left_layout.addWidget(self.file_list_panel, 1)

        self.process_btn = QPushButton("Process && Save Selected")
        left_layout.addWidget(self.process_btn)

        left.setMinimumWidth(320)
        left.setMaximumWidth(420)
        row.addWidget(left)

        self.tabs = QTabWidget()
        row.addWidget(self.tabs, 1)

        self.rawstack_panel = RawStackViewerPanel()
        self.tabs.addTab(self.rawstack_panel, "Raw Stack")

        self.flatten_tab = self._build_flatten_tab()
        self.tabs.addTab(self.flatten_tab, "Field Flattening")

        # Page-local status feedback -- deliberately not shared with other
        # pages: a background batch worker on a page you've navigated away
        # from would otherwise stomp on whatever message the page you're
        # looking at is trying to show.
        self.status_label = QLabel("")
        outer.addWidget(self.status_label)

    def _build_flatten_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)

        self.params_panel = ParamsPanel()
        self.params_panel.setMaximumWidth(320)
        layout.addWidget(self.params_panel)

        right = QVBoxLayout()

        grid_splitter_v = QSplitter(Qt.Vertical)
        top_row = QSplitter(Qt.Horizontal)
        self.raw_slice_panel = RawSlicePanel()
        self.coarse_heatmap_panel = CoarseHeatmapPanel()
        top_row.addWidget(self.raw_slice_panel)
        top_row.addWidget(self.coarse_heatmap_panel)

        bottom_row = QSplitter(Qt.Horizontal)
        self.focus_surface_panel = FocusSurfacePanel()
        self.focal_preview_panel = FocalPreviewPanel()
        bottom_row.addWidget(self.focus_surface_panel)
        bottom_row.addWidget(self.focal_preview_panel)

        grid_splitter_v.addWidget(top_row)
        grid_splitter_v.addWidget(bottom_row)
        right.addWidget(grid_splitter_v, 1)

        # Was a QDockWidget (floatable/toggleable via a View menu) back
        # when this lived directly in the QMainWindow; pages are plain
        # QWidgets now, so it's just an embedded panel with a checkbox --
        # same capped height, just without the free-floating capability.
        self.tile_variance_visibility_checkbox = QCheckBox("Show tile variance inspector")
        self.tile_variance_visibility_checkbox.setChecked(True)
        right.addWidget(self.tile_variance_visibility_checkbox)

        self.tile_variance_panel = TileVarianceInspectorPanel()
        self.tile_variance_panel.setMaximumHeight(260)
        right.addWidget(self.tile_variance_panel)
        self.tile_variance_visibility_checkbox.toggled.connect(
            self.tile_variance_panel.setVisible
        )

        layout.addLayout(right, 1)
        return tab

    # ------------------------------------------------------------------
    # Wiring

    def _wire_up(self):
        self.folder_panel.input_folder_changed.connect(self._on_input_folder_changed)
        self.folder_panel.output_folder_changed.connect(self._on_output_folder_changed)
        self.folder_panel.pattern_changed.connect(lambda _p: self._rescan_input_folder())

        self.file_list_panel.file_selected.connect(self._on_file_selected)
        self.process_btn.clicked.connect(self._start_batch)

        self.pipeline = PipelineController()
        self.pipeline.stage_a_done.connect(self._on_stage_a_done)
        self.pipeline.result_ready.connect(self._on_result_ready)
        self.pipeline.error.connect(self._on_pipeline_error)

        self.params_panel.params_changed.connect(self._on_params_or_channels_changed)
        self.params_panel.channels_changed.connect(self._on_params_or_channels_changed)
        self.params_panel.recompute_requested.connect(self._recompute_now)
        self.params_panel.save_defaults_requested.connect(self._save_folder_defaults)
        self.params_panel.reset_defaults_requested.connect(self._reset_folder_defaults)

        self.coarse_heatmap_panel.tile_clicked.connect(self._on_tile_clicked)
        self.focus_surface_panel.tile_clicked.connect(self._on_tile_clicked)

    # ------------------------------------------------------------------
    # Folder handling

    def _on_input_folder_changed(self, _folder: str):
        self._channels_inferred_for_folder = False
        self._rescan_input_folder()

    def _rescan_input_folder(self):
        folder = self.folder_panel.input_folder()
        if not folder or not Path(folder).is_dir():
            self.file_list_panel.set_files([])
            return
        pattern = self.folder_panel.current_pattern()
        paths = sorted(Path(folder).glob(pattern), key=_natsort_key)
        self.file_list_panel.set_files([str(p) for p in paths])

    def _on_output_folder_changed(self, folder: str):
        self.file_list_panel.set_output_folder(folder)
        config = folder_config.load_folder_config(folder)
        if config:
            self.params_panel.set_params(config["params"])
            if config["channels"]:
                self.params_panel.set_channels(config["channels"])
                self._channels_inferred_for_folder = True
            self.status_label.setText("Loaded parameters from a previous run in this folder.")

    # ------------------------------------------------------------------
    # File selection -> raw viewer + flatten pipeline

    def _on_file_selected(self, path: str):
        self.rawstack_panel.load_file(path)
        channels_meta = getattr(self.rawstack_panel, "_meta", None)
        channels_meta = (channels_meta or {}).get("channels") or []
        self.params_panel.set_channels_meta(channels_meta)

        if not self._channels_inferred_for_folder:
            inferred = infer_channel_selection(channels_meta)
            if inferred:
                self.params_panel.set_channels(inferred)
            self._channels_inferred_for_folder = True

        if self._try_load_saved_flatten_output(path):
            return
        self._recompute_now()

    def _try_load_saved_flatten_output(self, path: str) -> bool:
        """Fast inspect path: if this file was already batch-processed and
        the saved focal slice is still up to date, just show it -- no need
        to re-run the full raw-stack pipeline (which touches every pixel
        of every Z slice) just to look at an already-processed file.
        'Recompute Now' (or editing a param with auto-recompute on) still
        forces a live rerun for actual parameter tuning."""
        output_folder = self.folder_panel.output_folder()
        if not output_folder:
            return False
        output_path = find_processed_output(path, output_folder)
        if output_path is None:
            return False
        try:
            focal_slice = load_brightfield_channel(output_path)
        except Exception:
            return False

        # These diagnostics all come from the live pipeline's raw-stack
        # analysis, which this fast path skips -- clear them rather than
        # leaving them showing a previous file's data under a misleading
        # current-file label.
        self._last_diagnostics = None
        self.raw_slice_panel.clear()
        self.coarse_heatmap_panel.clear()
        self.focus_surface_panel.clear()
        self.tile_variance_panel.clear()
        self.focal_preview_panel.set_data(focal_slice)
        self.status_label.setText(
            f"{Path(path).name}: showing saved output. Click 'Recompute Now' to tune live."
        )
        return True

    def _on_params_or_channels_changed(self, _value):
        if not self.params_panel.is_auto_recompute():
            return
        path = self.file_list_panel.current_path()
        if not path:
            return
        self.pipeline.schedule(path, self.params_panel.channels(), self.params_panel.params())

    def _recompute_now(self):
        path = self.file_list_panel.current_path()
        if not path:
            return
        self.pipeline.recompute_now(
            path, self.params_panel.channels(), self.params_panel.params()
        )

    def _on_stage_a_done(self, volume):
        self.raw_slice_panel.set_volume(volume.img3d)
        self.params_panel.set_scale(*volume.scale)
        self.file_list_panel.mark_previewed(str(volume.path))

    def _on_result_ready(self, result):
        if result.request_id != self.pipeline.latest_request_id():
            return  # superseded by a newer request already queued
        diagnostics = result.diagnostics
        self._last_diagnostics = diagnostics
        n_z = result.volume.img3d.shape[0]
        self.coarse_heatmap_panel.set_data(diagnostics.coarse_focal_indices, n_z)
        self.focus_surface_panel.set_data(
            diagnostics.fine_focus_indices, diagnostics.variance_stack.tile_info
        )
        self.focal_preview_panel.set_data(result.focal_slice)
        self.status_label.setText(f"Recomputed diagnostics for {result.volume.path.name}")

    def _on_pipeline_error(self, message: str):
        self.status_label.setText(f"Error: {message}")

    def _on_tile_clicked(self, i: int, j: int):
        if self._last_diagnostics is None:
            return
        focal_index = int(self._last_diagnostics.coarse_focal_indices[i, j])
        self.tile_variance_panel.show_tile(
            i, j, self._last_diagnostics.variance_stack, focal_index
        )

    # ------------------------------------------------------------------
    # Folder-scoped defaults

    def _save_folder_defaults(self):
        folder = self.folder_panel.output_folder()
        if not folder:
            QMessageBox.warning(self, "yeastprep", "Select an output folder first.")
            return
        folder_config.save_folder_config(
            folder, self.params_panel.params(), self.params_panel.channels()
        )
        self.status_label.setText("Saved as folder defaults.")

    def _reset_folder_defaults(self):
        folder = self.folder_panel.output_folder()
        config = folder_config.load_folder_config(folder) if folder else None
        if config:
            self.params_panel.set_params(config["params"])
            if config["channels"]:
                self.params_panel.set_channels(config["channels"])
        else:
            self.params_panel.set_params(settings.get_default_params())
            self.params_panel.set_channels(settings.get_default_channels() or DEFAULT_CHANNELS)

    # ------------------------------------------------------------------
    # Batch processing

    def _start_batch(self):
        input_folder = self.folder_panel.input_folder()
        output_folder = self.folder_panel.output_folder()
        if not input_folder or not output_folder:
            QMessageBox.warning(
                self, "yeastprep", "Select both an input and an output folder first."
            )
            return

        paths = self.file_list_panel.checked_paths()
        if not paths:
            return

        outdir = Path(output_folder) / "processed"

        self._batch_thread = QThread()
        self._batch_worker = BatchProcessWorker(
            paths, outdir, self.params_panel.params(), self.params_panel.channels()
        )
        self._batch_worker.moveToThread(self._batch_thread)
        self._thread_started_connection = self._batch_thread.started.connect(
            self._batch_worker.run
        )
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.file_result.connect(self._on_batch_file_result)
        self._batch_worker.finished.connect(self._on_batch_finished)

        self.process_btn.setEnabled(False)
        self.progress_changed.emit(PageProgress(active=True, done=0, total=len(paths)))

        self._batch_thread.start()

    def _on_batch_progress(self, done: int, total: int, name: str):
        self.progress_changed.emit(PageProgress(active=True, done=done, total=total, message=name))
        self.status_label.setText(f"Processing {name} ({done}/{total})")

    def _on_batch_file_result(self, result):
        self.file_list_panel.mark_result(str(result.path), result.success, result.error)

    def _on_batch_finished(self):
        self.progress_changed.emit(PageProgress(active=False))
        self.process_btn.setEnabled(True)
        self.status_label.setText("Batch processing complete.")
        self.file_list_panel.refresh_statuses()

        output_folder = self.folder_panel.output_folder()
        if output_folder:
            processed = [str(Path(p).name) for p in self.file_list_panel.checked_paths()]
            folder_config.save_folder_config(
                output_folder,
                self.params_panel.params(),
                self.params_panel.channels(),
                files_processed=processed,
            )

        self._batch_thread.quit()
        self._batch_thread.wait()
        self._batch_thread = None
        self._batch_worker = None

    # ------------------------------------------------------------------

    def shutdown(self):
        settings.set_default_params(self.params_panel.params())
        settings.set_default_channels(self.params_panel.channels())
        self.pipeline.shutdown()
        if self._batch_thread is not None:
            self._batch_thread.quit()
            self._batch_thread.wait()
