"""Denoise page: an optional jssl-denoise pass over 01_reduced/'s
combined-channel tiffs, writing into 02_denoised/. Two tabs -- "Denoising"
(this page's own UI) and "Training" (an embedded TrainDenoisePage) -- since
fitting the checkpoint a run consumes and running that checkpoint are two
sides of the same task, not two separate pipeline stages. A checkpoint the
Training tab finishes drops straight into the Denoising tab's field for
whichever channel it was trained on, and so does one already sitting in a
project that's just been opened (see `_on_project_root_changed` /
core/denoise.find_project_checkpoint) -- both cases mean a whole project can
go straight to batch denoising without a manual Browse first.

Reads its input/batch-selection through the shared ProjectTreePanel (see
main_window.py) rather than owning its own folder panel.
"""

from pathlib import Path

from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core import project as project_core
from yeastprep.core.combined_tiff import BRIGHTFIELD_CHANNEL, TARGET_CHANNEL, load_combined_channels
from yeastprep.core.denoise import DenoiseParams, find_project_checkpoint

from .. import settings
from ..batch_progress_bar import BatchProgressBar
from ..common.preview_source_label import PreviewSourceLabel
from ..denoise_params_panel import DenoiseParamsPanel
from ..diagnostics.denoise_preview_panel import DenoisePreviewPanel
from ..project_tree_panel import ProjectTreePanel
from ..worker import DenoiseBatchWorker, DenoiseController
from .page_progress import PageProgress
from .train_denoise_page import TrainDenoisePage

_CHANNEL_LABELS = {0: "Channel 0 (brightfield)", 1: "Channel 1 (target/fluorescence)"}


class DenoisePage(QWidget):
    progress_changed = Signal(object)  # PageProgress

    def __init__(self, tree_panel: ProjectTreePanel, parent=None):
        super().__init__(parent)
        self.tree_panel = tree_panel

        # "live": _source_* came from a 01_reduced file about to be (or
        # just) denoised in this session -- params changes recompute.
        # "saved": _source_*/_output_* came from inspecting an existing
        # 02_denoised file -- params changes just re-render, until
        # "Do it" switches back to live.
        self._mode = "live"
        self._source_brightfield = None
        self._source_target = None
        self._output_brightfield = None
        self._output_target = None
        self._source_generation = 0
        # Which channel the preview is currently showing, in live mode --
        # lets _on_params_changed tell "you switched channels, the old
        # after-image is now for the wrong channel" apart from "you tweaked
        # tta/checkpoint, the old after-image is still fine to keep looking
        # at until 'Do it'".
        self._displayed_channel = None
        self._batch_thread = None
        self._batch_worker = None

        self._build_ui()
        self._wire_up()

        self.params_panel.set_params(settings.get_default_denoise_params())

    # ------------------------------------------------------------------
    # UI construction

    def _build_ui(self):
        outer = QVBoxLayout(self)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.tabs.addTab(self._build_denoising_tab(), "Denoising")

        self.train_tab = TrainDenoisePage()
        self.tabs.addTab(self.train_tab, "Training")

    def _build_denoising_tab(self) -> QWidget:
        tab = QWidget()
        outer = QVBoxLayout(tab)
        row = QHBoxLayout()
        outer.addLayout(row, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.params_panel = DenoiseParamsPanel()
        left_layout.addWidget(self.params_panel)
        left_layout.addStretch(1)
        left.setMinimumWidth(320)
        left.setMaximumWidth(420)
        row.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_source_label = PreviewSourceLabel()
        right_layout.addWidget(self.preview_source_label)
        self.preview_panel = DenoisePreviewPanel()
        right_layout.addWidget(self.preview_panel, 1)

        batch_group = QGroupBox("Batch: Denoise && Save")
        action_row = QHBoxLayout(batch_group)
        self.denoise_btn = QPushButton("Denoise && Save Selected")
        action_row.addWidget(self.denoise_btn)
        self.batch_progress = BatchProgressBar()
        action_row.addWidget(self.batch_progress, 1)
        right_layout.addWidget(batch_group)

        row.addWidget(right, 1)

        self.status_label = QLabel("")
        outer.addWidget(self.status_label)

        return tab

    # ------------------------------------------------------------------
    # Wiring

    def _wire_up(self):
        self.tree_panel.file_preview_requested.connect(self._on_file_selected)
        self.tree_panel.project_root_changed.connect(self._on_project_root_changed)

        self.controller = DenoiseController()
        self.controller.result_ready.connect(self._on_result_ready)
        self.controller.error.connect(self._on_error)

        self.params_panel.params_changed.connect(self._on_params_changed)
        self.params_panel.recompute_requested.connect(self._recompute_now)
        self.params_panel.save_defaults_requested.connect(self._save_project_defaults)
        self.params_panel.reset_defaults_requested.connect(self._reset_project_defaults)
        self.denoise_btn.clicked.connect(self._start_batch)

        self.train_tab.progress_changed.connect(self.progress_changed)
        self.train_tab.checkpoint_trained.connect(self.params_panel.set_checkpoint_for_channel)
        self.tree_panel.project_root_changed.connect(self.train_tab.set_project_root)

    # ------------------------------------------------------------------
    # Project handling

    def _on_project_root_changed(self, root: str):
        config = project_core.load_project_config(root)
        if config:
            self.params_panel.set_params(config.denoise_params)

        # An explicitly saved project default (config.denoise_checkpoints)
        # wins; for whichever channel has none, fall back to a same-named
        # checkpoint sitting directly in the project root (e.g. one a
        # Training-tab run already saved there by default, or one dropped
        # in by hand) -- so opening a project that already has checkpoints
        # on disk is enough to batch-denoise it, no manual Browse or prior
        # "Save as project defaults" required.
        checkpoints = dict(config.denoise_checkpoints) if config else {}
        discovered = []
        for channel in (BRIGHTFIELD_CHANNEL, TARGET_CHANNEL):
            if channel in checkpoints:
                continue
            found = find_project_checkpoint(root, channel)
            if found:
                checkpoints[channel] = found
                discovered.append(_CHANNEL_LABELS[channel])
        self.params_panel.set_checkpoints_by_channel(checkpoints)

        if discovered:
            found_msg = f"Found checkpoint(s) on disk for {', '.join(discovered)}."
            prefix = "Loaded denoise parameters from this project. " if config else ""
            self.status_label.setText(prefix + found_msg)
        elif config:
            self.status_label.setText("Loaded denoise parameters from this project.")

    # ------------------------------------------------------------------
    # File selection -> live preview, or a saved-output inspect

    def _channel_label(self) -> str:
        return _CHANNEL_LABELS.get(self.params_panel.channel(), "")

    def _current_source_image(self):
        if self.params_panel.channel() == BRIGHTFIELD_CHANNEL:
            return self._source_brightfield
        return self._source_target

    def _on_file_selected(self, stage: str, path: str):
        if stage == project_core.STAGE_REDUCED:
            self.preview_source_label.set_path(path)
            self._load_live_source(path)
        elif stage == project_core.STAGE_DENOISED:
            self.preview_source_label.set_path(path)
            self._load_saved_output(path)

    def _load_live_source(self, path: str):
        try:
            brightfield, target = load_combined_channels(path)
        except Exception as exc:
            self.status_label.setText(f"Failed to load {path}: {exc}")
            return
        self._mode = "live"
        self._source_brightfield = brightfield
        self._source_target = target
        self._source_generation += 1

        image = self._current_source_image()
        self.preview_panel.set_data(image, None, self._channel_label())
        self._displayed_channel = self.params_panel.channel()

        if self.params_panel.is_auto_recompute():
            self.controller.schedule(
                self._current_source_image(), self._source_generation, self.params_panel.params()
            )

    def _load_saved_output(self, path: str):
        """Fast inspect path: show an already-denoised 02_denoised file
        as-is -- before = the matching 01_reduced source channel (if it's
        still there), after = the saved output channel -- without spending
        a live inference pass on it. Mirrors Data Reduction's
        `_try_load_saved_flatten_output`."""
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
        self._source_brightfield, self._source_target = src_brightfield, src_target
        self._output_brightfield, self._output_target = out_brightfield, out_target
        self._source_generation += 1
        self._displayed_channel = self.params_panel.channel()
        self._render_saved_preview()
        self.status_label.setText(
            f"{Path(path).name}: showing saved output. Click 'Do it' to tune live."
        )

    def _render_saved_preview(self):
        channel = self.params_panel.channel()
        before = self._source_brightfield if channel == BRIGHTFIELD_CHANNEL else self._source_target
        after = self._output_brightfield if channel == BRIGHTFIELD_CHANNEL else self._output_target
        self.preview_panel.set_data(before, after, self._channel_label())

    def _on_params_changed(self, params: DenoiseParams):
        if self._source_brightfield is None:
            return
        if self._mode == "saved":
            self._render_saved_preview()
            return
        if params.channel != self._displayed_channel:
            # The channel changed -- whatever "after" was showing belongs to
            # the old channel and would be actively misleading to leave up,
            # so reset to the new channel's raw image (cheap, no inference)
            # even if auto-recompute is off and "Do it" hasn't run yet.
            image = self._current_source_image()
            self.preview_panel.set_data(image, None, self._channel_label())
            self._displayed_channel = params.channel
        if not self.params_panel.is_auto_recompute():
            return
        self.controller.schedule(
            self._current_source_image(), self._source_generation, params
        )

    def _recompute_now(self):
        if self._source_brightfield is None:
            return
        self._mode = "live"
        self._displayed_channel = self.params_panel.channel()
        self.controller.recompute_now(
            self._current_source_image(), self._source_generation, self.params_panel.params()
        )

    def _on_result_ready(self, result):
        if result.request_id != self.controller.latest_request_id():
            return  # superseded by a newer request already queued
        self.preview_panel.set_data(result.before, result.after, self._channel_label())

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
        config.denoise_checkpoints = self.params_panel.checkpoints_by_channel()
        project_core.save_project_config(root, config)
        self.status_label.setText("Saved as project defaults.")

    def _reset_project_defaults(self):
        root = self.tree_panel.project_root()
        config = project_core.load_project_config(root) if root else None
        if config:
            self.params_panel.set_params(config.denoise_params)
            self.params_panel.set_checkpoints_by_channel(config.denoise_checkpoints)
        else:
            self.params_panel.set_params(settings.get_default_denoise_params())

    # ------------------------------------------------------------------
    # Batch denoise

    def _start_batch(self):
        paths_root = self.tree_panel.project_paths()
        if paths_root is None:
            QMessageBox.warning(self, "yeastprep", "Open a project first.")
            return

        if not self.tree_panel.all_paths_for_stage(project_core.STAGE_REDUCED):
            QMessageBox.warning(
                self,
                "yeastprep",
                "No reduced (2D) tiffs found. Run Data Reduction first.",
            )
            return

        paths = [
            Path(p)
            for p in self.tree_panel.checked_paths_for_stage(project_core.STAGE_REDUCED)
        ]
        if not paths:
            QMessageBox.warning(
                self, "yeastprep", "No files checked -- check at least one file in the tree."
            )
            return

        outdir = paths_root.denoised

        self._batch_channel = self.params_panel.channel()
        self._batch_succeeded_stems = []

        self._batch_thread = QThread()
        self._batch_worker = DenoiseBatchWorker(paths, outdir, self.params_panel.params())
        self._batch_worker.moveToThread(self._batch_thread)
        self._thread_started_connection = self._batch_thread.started.connect(
            self._batch_worker.run
        )
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.file_result.connect(self._on_batch_file_result)
        self._batch_worker.finished.connect(self._on_batch_finished)

        self.denoise_btn.setEnabled(False)
        self._emit_progress(PageProgress(active=True, done=0, total=len(paths)))

        self._batch_thread.start()

    def _emit_progress(self, progress: PageProgress):
        self.progress_changed.emit(progress)
        self.batch_progress.apply(progress)

    def _on_batch_progress(self, done: int, total: int, name: str):
        self._emit_progress(PageProgress(active=True, done=done, total=total, message=name))
        self.status_label.setText(f"Denoising {self._channel_label()}: {name} ({done}/{total})")

    def _on_batch_file_result(self, result):
        self.tree_panel.mark_result(
            project_core.STAGE_REDUCED, result.path, result.success, result.error
        )
        if result.success:
            self._batch_succeeded_stems.append(result.path.stem)
            self.status_label.setText(f"{result.path.name}: saved to {result.output_path}")
        else:
            self.status_label.setText(f"{result.path.name}: {result.error}")

    def _on_batch_finished(self):
        self._emit_progress(PageProgress(active=False))
        self.denoise_btn.setEnabled(True)
        self.status_label.setText("Batch denoise complete.")

        root = self.tree_panel.project_root()
        if root:
            # Only files that actually succeeded count as "this channel is
            # now denoised" -- a failed file shouldn't get credited, or the
            # tree's half/full-circle indicator would lie about it.
            project_core.mark_stage_run(
                root, project_core.STAGE_DENOISED, self._batch_succeeded_stems
            )
            project_core.mark_denoise_channels(
                root, self._batch_channel, self._batch_succeeded_stems
            )
        self.tree_panel.refresh()

        self._batch_thread.quit()
        self._batch_thread.wait()
        self._batch_thread = None
        self._batch_worker = None

    # ------------------------------------------------------------------

    def shutdown(self):
        settings.set_default_denoise_params(self.params_panel.params())
        self.controller.shutdown()
        if self._batch_thread is not None:
            self._batch_thread.quit()
            self._batch_thread.wait()
        self.train_tab.shutdown()
