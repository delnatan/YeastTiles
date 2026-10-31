"""Denoise page: an optional pass over 01_reduced/'s combined-channel
tiffs, writing into 02_denoised/. Each checked channel of each file is
denoised by a network fitted to that image alone (core/denoise.py), so
there is nothing to train or pick beforehand -- check the channels, preview
one on the selected file, then batch.

Reads its input/batch-selection through the shared ProjectTreePanel (see
main_window.py) rather than owning its own folder panel.
"""

from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core import project as project_core
from yeastprep.core.combined_tiff import BRIGHTFIELD_CHANNEL, TARGET_CHANNEL, load_combined_channels
from yeastprep.core.denoise import DenoiseParams

from .. import settings
from ..common.batch_runner import BatchRunner
from ..common.preview_source_label import PreviewSourceLabel
from ..denoise_params_panel import CHANNEL_LABELS, DenoiseParamsPanel
from ..diagnostics.denoise_preview_panel import DenoisePreviewPanel
from ..project_tree_panel import ProjectTreePanel
from ..worker import DenoiseBatchWorker, DenoiseController
from .page_progress import PageProgress


class DenoisePage(QWidget):
    progress_changed = Signal(object)  # PageProgress

    def __init__(self, tree_panel: ProjectTreePanel, parent=None):
        super().__init__(parent)
        self.tree_panel = tree_panel

        # "live": the source channels of a 01_reduced file, denoised on
        # Preview. "saved": a 02_denoised file's channels, shown against
        # its 01_reduced source without recomputing, until Preview.
        self._mode = "live"
        self._source = {}  # channel -> image
        self._output = {}  # channel -> image, saved mode only
        self._source_generation = 0
        self._batch_root = None
        self._batch_channels = ()
        self._batch_succeeded_stems = []

        self._build_ui()
        self._wire_up()

        self.params_panel.set_params(settings.get_default_denoise_params())
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
        self.preview_source_label = PreviewSourceLabel()
        left_layout.addWidget(self.preview_source_label)
        self.params_panel = DenoiseParamsPanel()
        left_layout.addWidget(self.params_panel)
        left_layout.addStretch(1)
        left.setMinimumWidth(320)
        left.setMaximumWidth(420)
        row.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_panel = DenoisePreviewPanel()
        right_layout.addWidget(self.preview_panel, 1)

        batch_group = QGroupBox("Batch: Denoise && Save")
        action_row = QHBoxLayout(batch_group)
        self.denoise_btn = QPushButton("Denoise && Save Selected")
        action_row.addWidget(self.denoise_btn)
        self.batch_runner = BatchRunner("Batch denoise")
        action_row.addWidget(self.batch_runner, 1)
        right_layout.addWidget(batch_group)
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

        self.controller = DenoiseController()
        self.controller.result_ready.connect(self._on_result_ready)
        self.controller.error.connect(self._on_error)

        self.params_panel.params_changed.connect(self._update_batch_button_state)
        self.params_panel.preview_channel_changed.connect(self._on_preview_channel_changed)
        self.params_panel.preview_requested.connect(self._preview_now)
        self.params_panel.save_defaults_requested.connect(self._save_project_defaults)
        self.params_panel.reset_defaults_requested.connect(self._reset_project_defaults)
        self.denoise_btn.clicked.connect(self._start_batch)
        self.batch_runner.progress_changed.connect(self._on_batch_progress)
        self.batch_runner.progress_changed.connect(self.progress_changed)
        self.batch_runner.file_result.connect(self._on_batch_file_result)
        self.batch_runner.finished.connect(self._on_batch_finished)

    def _on_project_root_changed(self, root: str):
        config = project_core.load_project_config(root)
        if config:
            self.params_panel.set_params(config.denoise_params)
            self.status_label.setText("Loaded denoise parameters from this project.")

    # ------------------------------------------------------------------
    # File selection -> preview

    def _channel_label(self) -> str:
        return CHANNEL_LABELS.get(self.params_panel.preview_channel(), "")

    def load_selection(self, stage: str, path: str, mode: str):
        self.preview_source_label.set_path(path)
        if mode == "live":
            self._load_live_source(path)
        elif mode == "saved":
            self._load_saved_output(path)

    def _set_source(self, brightfield, target):
        self._source = {BRIGHTFIELD_CHANNEL: brightfield, TARGET_CHANNEL: target}
        self._source_generation += 1

    def _load_live_source(self, path: str):
        self.controller.invalidate()  # drop any preview of the previous file
        try:
            brightfield, target = load_combined_channels(path)
        except Exception as exc:
            self.status_label.setText(f"Failed to load {path}: {exc}")
            return
        self._mode = "live"
        self._set_source(brightfield, target)
        self._output = {}
        self._render()
        self.status_label.setText(f"{Path(path).name}: click Preview to denoise the shown channel.")

    def _load_saved_output(self, path: str):
        """Show an already-denoised 02_denoised file as-is -- before = the
        matching 01_reduced channel (if it's still there), after = the saved
        output -- without recomputing it."""
        self.controller.invalidate()
        paths_root = self.tree_panel.project_paths()
        source_path = paths_root.reduced / f"{Path(path).stem}.tiff" if paths_root else None
        try:
            out_brightfield, out_target = load_combined_channels(path)
            if source_path is not None and source_path.exists():
                src_brightfield, src_target = load_combined_channels(source_path)
            else:
                src_brightfield, src_target = out_brightfield, out_target
        except Exception as exc:
            self.status_label.setText(f"Failed to load {path}: {exc}")
            return
        self._mode = "saved"
        self._set_source(src_brightfield, src_target)
        self._output = {BRIGHTFIELD_CHANNEL: out_brightfield, TARGET_CHANNEL: out_target}
        self._render()
        self.status_label.setText(f"{Path(path).name}: showing saved output.")

    def _render(self):
        channel = self.params_panel.preview_channel()
        if channel in self._source:
            self.preview_panel.set_data(
                self._source[channel], self._output.get(channel), self._channel_label()
            )

    def _on_preview_channel_changed(self, _channel: int):
        # A live result belongs to the channel it was computed for.
        if self._mode == "live":
            self.controller.invalidate()
            self._output = {}
        self._render()

    def _preview_now(self):
        channel = self.params_panel.preview_channel()
        if channel not in self._source:
            self.status_label.setText("Select a reduced (2D) file in the tree first.")
            return
        self._mode = "live"
        self._output = {}
        self._render()
        self.status_label.setText(f"Denoising {self._channel_label()}...")
        self.controller.recompute_now(
            self._source[channel], self._source_generation, self.params_panel.params()
        )

    def _on_result_ready(self, result):
        if result.request_id != self.controller.latest_request_id():
            return  # superseded by a newer request
        self._output = {self.params_panel.preview_channel(): result.after}
        self.preview_panel.set_data(result.before, result.after, self._channel_label())
        self.status_label.setText("Preview ready.")

    def _on_error(self, message: str):
        self.status_label.setText(f"Denoise error: {message}")

    # ------------------------------------------------------------------
    # Project-scoped defaults

    def _save_project_defaults(self):
        root = self.tree_panel.project_root()
        if not root:
            QMessageBox.warning(self, "yeastprep", "Open a project first.")
            return
        config = project_core.load_project_config(root) or project_core.ProjectConfig()
        config.denoise_params = self.params_panel.params()
        project_core.save_project_config(root, config)
        self.status_label.setText("Saved as project defaults.")

    def _reset_project_defaults(self):
        root = self.tree_panel.project_root()
        config = project_core.load_project_config(root) if root else None
        self.params_panel.set_params(
            config.denoise_params if config else settings.get_default_denoise_params()
        )

    # ------------------------------------------------------------------
    # Batch denoise

    def _update_batch_button_state(self, *_args):
        paths_root = self.tree_panel.project_paths()
        checked = (
            self.tree_panel.checked_paths_for_stage(project_core.STAGE_REDUCED)
            if paths_root
            else []
        )
        channels = self.params_panel.params().channels
        running = self.batch_runner.is_running()
        self.denoise_btn.setEnabled(bool(checked) and bool(channels) and not running)
        if running:
            self.denoise_btn.setToolTip("A batch is already running -- see the progress bar.")
        elif paths_root is None:
            self.denoise_btn.setToolTip("Open a project first.")
        elif not checked:
            self.denoise_btn.setToolTip(
                "Check at least one reduced (2D) file in the tree -- run Data Reduction first "
                "if there are none."
            )
        elif not channels:
            self.denoise_btn.setToolTip("Check at least one channel to denoise.")
        else:
            self.denoise_btn.setToolTip("")

    def _start_batch(self):
        if self.batch_runner.is_running():
            return
        paths_root = self.tree_panel.project_paths()
        paths = [
            Path(p) for p in self.tree_panel.checked_paths_for_stage(project_core.STAGE_REDUCED)
        ]
        params: DenoiseParams = self.params_panel.params()
        if paths_root is None or not paths or not params.channels:
            return

        self._batch_root = str(paths_root.root)
        self._batch_channels = params.channels
        self._batch_succeeded_stems = []
        self.batch_runner.start(DenoiseBatchWorker(paths, paths_root.denoised, params))
        self._update_batch_button_state()

    def _on_batch_progress(self, progress: PageProgress):
        if progress.active and progress.message:
            self.status_label.setText(
                f"Denoising: {progress.message} ({progress.done}/{progress.total})"
            )

    def _on_batch_file_result(self, result):
        self.tree_panel.mark_result(
            project_core.STAGE_REDUCED, result.path, result.success, result.error
        )
        if result.success:
            self._batch_succeeded_stems.append(result.path.stem)
            self.status_label.setText(f"{result.path.name}: saved to {result.output_path}")
        else:
            self.status_label.setText(f"{result.path.name}: {result.error}")

    def _on_batch_finished(self, outcome):
        self._update_batch_button_state()
        self.status_label.setText(outcome.summary())
        # Only files that actually succeeded count as denoised -- a failed
        # file shouldn't get credited, or the tree's half/full-circle
        # indicator would lie about it.
        project_core.mark_stage_run(
            self._batch_root, project_core.STAGE_DENOISED, self._batch_succeeded_stems
        )
        project_core.mark_denoise_channels(
            self._batch_root, self._batch_channels, self._batch_succeeded_stems
        )
        self.tree_panel.refresh()

    def busy_description(self) -> str | None:
        return self.batch_runner.describe_running()

    # ------------------------------------------------------------------

    def shutdown(self):
        settings.set_default_denoise_params(self.params_panel.params())
        self.controller.shutdown()
        self.batch_runner.shutdown()
