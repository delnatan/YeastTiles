"""Data Reduction page: raw multi-channel Z-stacks -> a single combined
2D tiff per FOV (flattened brightfield + sum-projected target channel,
design.md stages 1-2b), written into the current project's 01_reduced/.
Reads its raw-input listing/batch-selection through the shared
ProjectTreePanel (see main_window.py) rather than owning its own folder
panel + file list.
"""

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

from yeastprep.core import project as project_core
from yeastprep.core.channels import infer_channel_selection
from yeastprep.core.combined_tiff import load_brightfield_channel
from yeastprep.core.pipeline import DEFAULT_CHANNELS

from .. import settings
from ..batch_progress_bar import BatchProgressBar
from ..common.preview_source_label import PreviewSourceLabel
from ..diagnostics.coarse_heatmap_panel import CoarseHeatmapPanel
from ..diagnostics.focal_preview_panel import FocalPreviewPanel
from ..diagnostics.focus_surface_panel import FocusSurfacePanel
from ..diagnostics.raw_slice_panel import RawSlicePanel
from ..diagnostics.tile_variance_panel import TileVarianceInspectorPanel
from ..params_panel import ParamsPanel
from ..project_tree_panel import ProjectTreePanel
from ..rawstack.panel import RawStackViewerPanel
from ..worker import BatchProcessWorker, PipelineController
from .page_progress import PageProgress


class DataReductionPage(QWidget):
    progress_changed = Signal(object)  # PageProgress

    def __init__(self, tree_panel: ProjectTreePanel, parent=None):
        super().__init__(parent)
        self.tree_panel = tree_panel

        self._channels_inferred_for_folder = False
        self._last_diagnostics = None
        self._current_raw_path = None
        self._batch_thread = None
        self._batch_worker = None

        self._build_ui()
        self._wire_up()

        default_params = settings.get_default_params()
        self.params_panel.set_params(default_params)
        default_channels = settings.get_default_channels() or DEFAULT_CHANNELS
        self.params_panel.set_channels(default_channels)
        self._update_batch_button_state()

    # ------------------------------------------------------------------
    # UI construction

    def _build_ui(self):
        outer = QVBoxLayout(self)

        # Action row sits above the tabs, so its position is fixed
        # regardless of which tab is showing -- it acts on the checked raw
        # files regardless of which diagnostic view is currently open. (The
        # pipeline breadcrumb itself now lives in main_window.py, above
        # every page, not just this one.)
        action_row = QHBoxLayout()
        self.process_btn = QPushButton("Process && Save Selected")
        action_row.addWidget(self.process_btn)
        self.batch_progress = BatchProgressBar()
        action_row.addWidget(self.batch_progress, 1)
        outer.addLayout(action_row)

        self.preview_source_label = PreviewSourceLabel()
        outer.addWidget(self.preview_source_label)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)

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
        self.tree_panel.project_root_changed.connect(self._on_project_root_changed)
        self.tree_panel.project_root_changed.connect(self._update_batch_button_state)
        self.tree_panel.checked_changed.connect(self._update_batch_button_state)
        self.tree_panel.refreshed.connect(self._update_batch_button_state)
        self.process_btn.clicked.connect(self._start_batch)

        self.pipeline = PipelineController()
        self.pipeline.stage_a_done.connect(self._on_stage_a_done)
        self.pipeline.result_ready.connect(self._on_result_ready)
        self.pipeline.error.connect(self._on_pipeline_error)

        self.params_panel.params_changed.connect(self._on_params_or_channels_changed)
        self.params_panel.channels_changed.connect(self._on_params_or_channels_changed)
        self.params_panel.recompute_requested.connect(self._recompute_now)
        self.params_panel.save_defaults_requested.connect(self._save_project_defaults)
        self.params_panel.reset_defaults_requested.connect(self._reset_project_defaults)

        self.coarse_heatmap_panel.tile_clicked.connect(self._on_tile_clicked)
        self.focus_surface_panel.tile_clicked.connect(self._on_tile_clicked)

        self.rawstack_panel.colormap_changed.connect(self._on_raw_colormap_changed)

    # ------------------------------------------------------------------
    # Folder handling

    def _on_project_root_changed(self, root: str):
        self._channels_inferred_for_folder = False
        config = project_core.load_project_config(root)
        if config:
            self.params_panel.set_params(config.flatten_params)
            self.params_panel.set_channels(config.channels)
            self._channels_inferred_for_folder = True
            self.status_label.setText("Loaded parameters from this project.")

    # ------------------------------------------------------------------
    # File selection -> raw viewer + flatten pipeline

    def load_selection(self, stage: str, path: str, mode: str = "live"):
        self._current_raw_path = path
        self.preview_source_label.set_path(path)

        self.rawstack_panel.load_file(path)
        channels_meta = getattr(self.rawstack_panel, "_meta", None)
        channels_meta = (channels_meta or {}).get("channels") or []
        self.params_panel.set_channels_meta(channels_meta)

        if not self._channels_inferred_for_folder:
            inferred = infer_channel_selection(channels_meta)
            if inferred:
                self.params_panel.set_channels(inferred)
            self._channels_inferred_for_folder = True

        # rawstack_panel.load_file() (above) assigns each channel's default
        # colormap and fires colormap_changed for it before this point --
        # i.e. before channel roles are known for *this* file -- so
        # _on_raw_colormap_changed was checking those emissions against the
        # previous file's brightfield/target indices and silently dropping
        # them. Now that roles are resolved, sync the persisted colors
        # explicitly rather than relying on that signal to have caught them.
        self._sync_channel_colormaps_to_settings()

        if self._try_load_saved_flatten_output(path):
            return
        if self.params_panel.is_auto_recompute():
            self._recompute_now()
        else:
            self.status_label.setText(f"{Path(path).name}: click 'Do it' to compute.")

    def _try_load_saved_flatten_output(self, path: str) -> bool:
        """Fast inspect path: if this file was already batch-processed and
        the saved focal slice is still up to date, just show it -- no need
        to re-run the full raw-stack pipeline (which touches every pixel
        of every Z slice) just to look at an already-processed file.
        'Do it' (or editing a param with auto-recompute on) still
        forces a live rerun for actual parameter tuning."""
        paths_root = self.tree_panel.project_paths()
        if paths_root is None:
            return False
        output_path = project_core.find_stage_output(paths_root.reduced, path)
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
            f"{Path(path).name}: showing saved output. Click 'Do it' to tune live."
        )
        return True

    def _on_params_or_channels_changed(self, _value):
        if not self.params_panel.is_auto_recompute():
            return
        if not self._current_raw_path:
            return
        self.pipeline.schedule(
            self._current_raw_path, self.params_panel.channels(), self.params_panel.params()
        )

    def _recompute_now(self):
        if not self._current_raw_path:
            return
        self.pipeline.recompute_now(
            self._current_raw_path, self.params_panel.channels(), self.params_panel.params()
        )

    def _on_stage_a_done(self, volume):
        self.raw_slice_panel.set_volume(volume.img3d)
        self.params_panel.set_scale(*volume.scale)

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

    def _on_raw_colormap_changed(self, channel_idx: int, colormap_name: str):
        """Persist whichever colormap the Raw Stack tab is showing for
        whichever raw channel is currently designated brightfield/target
        (see params_panel.channels()), so PreviewPage's composite
        can match it -- see settings.get_channel_colormap."""
        channels = self.params_panel.channels()
        if channels is None:
            return
        if channel_idx == channels.brightfield:
            settings.set_channel_colormap("brightfield", colormap_name)
        elif channel_idx == channels.projection:
            settings.set_channel_colormap("target", colormap_name)

    def _sync_channel_colormaps_to_settings(self):
        """Read the Raw Stack canvas's *current* brightfield/target colors
        straight off its display state and persist them -- covers the
        default-colormap-on-load case that `_on_raw_colormap_changed`
        alone can't (that signal fires while channel roles still belong to
        the previously loaded file, see the call site in
        `load_selection`). Also fine as a plain re-sync any time roles
        or colors might have changed."""
        display = self.rawstack_panel.canvas.display
        if display is None:
            return
        channels = self.params_panel.channels()
        if 0 <= channels.brightfield < len(display):
            settings.set_channel_colormap(
                "brightfield", display[channels.brightfield].colormap_name
            )
        if 0 <= channels.projection < len(display):
            settings.set_channel_colormap(
                "target", display[channels.projection].colormap_name
            )

    def _on_tile_clicked(self, i: int, j: int):
        if self._last_diagnostics is None:
            return
        focal_index = int(self._last_diagnostics.coarse_focal_indices[i, j])
        self.tile_variance_panel.show_tile(
            i, j, self._last_diagnostics.variance_stack, focal_index
        )

    # ------------------------------------------------------------------
    # Project-scoped defaults

    def _save_project_defaults(self):
        root = self.tree_panel.project_root()
        if not root:
            QMessageBox.warning(self, "yeastprep", "Open a project first.")
            return
        config = project_core.load_project_config(root) or project_core.ProjectConfig()
        config.flatten_params = self.params_panel.params()
        config.channels = self.params_panel.channels()
        config.raw_pattern = self.tree_panel.raw_pattern()
        project_core.save_project_config(root, config)
        self.status_label.setText("Saved as project defaults.")

    def _reset_project_defaults(self):
        root = self.tree_panel.project_root()
        config = project_core.load_project_config(root) if root else None
        if config:
            self.params_panel.set_params(config.flatten_params)
            self.params_panel.set_channels(config.channels)
        else:
            self.params_panel.set_params(settings.get_default_params())
            self.params_panel.set_channels(settings.get_default_channels() or DEFAULT_CHANNELS)

    # ------------------------------------------------------------------
    # Batch processing

    def _update_batch_button_state(self, *_args):
        paths_root = self.tree_panel.project_paths()
        checked = self.tree_panel.checked_paths_for_stage("raw") if paths_root else []
        self.process_btn.setEnabled(bool(checked))
        if paths_root is None:
            self.process_btn.setToolTip("Open a project first.")
        elif not checked:
            self.process_btn.setToolTip("Check at least one raw file in the tree.")
        else:
            self.process_btn.setToolTip("")

    def _start_batch(self):
        paths_root = self.tree_panel.project_paths()
        paths = [Path(p) for p in self.tree_panel.checked_paths_for_stage("raw")]
        outdir = paths_root.reduced

        self._batch_thread = QThread()
        self._batch_worker = BatchProcessWorker(
            paths, outdir, self.params_panel.params(), self.params_panel.channels()
        )
        self._batch_worker.moveToThread(self._batch_thread)
        self._batch_thread.started.connect(self._batch_worker.run)
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.file_result.connect(self._on_batch_file_result)
        self._batch_worker.finished.connect(self._on_batch_finished)

        self.process_btn.setEnabled(False)
        self._emit_progress(PageProgress(active=True, done=0, total=len(paths)))

        self._batch_thread.start()

    def _emit_progress(self, progress: PageProgress):
        self.progress_changed.emit(progress)
        self.batch_progress.apply(progress)

    def _on_batch_progress(self, done: int, total: int, name: str):
        self._emit_progress(PageProgress(active=True, done=done, total=total, message=name))
        self.status_label.setText(f"Processing {name} ({done}/{total})")

    def _on_batch_file_result(self, result):
        self.tree_panel.mark_result("raw", result.path, result.success, result.error)

    def _on_batch_finished(self):
        self._emit_progress(PageProgress(active=False))
        self._update_batch_button_state()
        self.status_label.setText("Batch processing complete.")

        root = self.tree_panel.project_root()
        if root:
            processed = [
                Path(p).stem for p in self.tree_panel.checked_paths_for_stage("raw")
            ]
            project_core.mark_stage_run(root, project_core.STAGE_REDUCED, processed)
        self.tree_panel.refresh()

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
