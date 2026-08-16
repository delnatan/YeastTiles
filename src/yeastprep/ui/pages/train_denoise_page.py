"""Train Denoiser page: fits a new jssl-denoise D-Net/N-Net checkpoint on
frames pooled from one or more projects' 01_reduced/ (or 02_denoised/) 2D
tiffs -- the "pool tiles/images from various experiments" requirement from
design.md, applied to training data instead of tile crops. A trained
checkpoint's path can be copied straight into the Denoise page's checkpoint
field.

Pooling legitimately spans multiple projects at once -- selection is still
per-file though, the pool tree shows every pooled project's files (for
whichever stage is selected) as checkable leaves, same "nothing checked
means everything" convenience as `ProjectTreePanel.checked_paths_for_stage`,
just scoped per project instead of to one always-open project. The
checkpoint *output* path, on the other hand, defaults into the single
currently-open project's root (`set_project_root`, called by the embedding
DenoisePage whenever that changes) with a name derived from the channel
being trained, e.g. `denoise_brightfield.pt` -- so there's always an
obvious, non-blank place a freshly trained network lands unless you
deliberately Browse elsewhere.
"""

from pathlib import Path

import numpy as np
import tifffile
from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from jssl_denoise.checkpoint import save_checkpoint

from yeastprep.core import project as project_core
from yeastprep.core.denoise import checkpoint_filename_for_channel

from .. import settings
from ..batch_progress_bar import BatchProgressBar
from ..diagnostics.training_monitor_panel import TrainingMonitorPanel
from ..train_params_panel import TrainParamsPanel
from ..worker import TrainingWorker
from .page_progress import PageProgress

_POOL_STAGES = (project_core.STAGE_REDUCED, project_core.STAGE_DENOISED)
_POOL_STAGE_LABELS = {
    project_core.STAGE_REDUCED: "01 · Reduced (2D)",
    project_core.STAGE_DENOISED: "02 · Denoised",
}


class TrainDenoisePage(QWidget):
    progress_changed = Signal(object)  # PageProgress
    checkpoint_trained = Signal(int, str)  # channel, checkpoint path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pooled_roots: list[str] = []
        self._train_thread = None
        self._train_worker = None
        self._project_root: str | None = None
        self._checkpoint_path_is_default = True

        self._build_ui()
        self._wire_up()
        self._refresh_default_checkpoint_path()

    # ------------------------------------------------------------------
    # UI construction

    def _build_ui(self):
        outer = QVBoxLayout(self)
        row = QHBoxLayout()
        outer.addLayout(row, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(self._build_pool_group())
        left_layout.addWidget(self._build_source_group())
        self.params_panel = TrainParamsPanel()
        left_layout.addWidget(self.params_panel)
        left_layout.addWidget(self._build_checkpoint_group())
        left_layout.addStretch(1)

        # Scrolled rather than packed directly into `row`: the pooled-project
        # tree + training-data group + full hyperparameter form + checkpoint
        # group easily exceed a typical window's height, and without a
        # scroll area the layout has nowhere to put the overflow but to
        # squeeze every row's spacing down -- a scroll area lets each widget
        # render at its natural size and the column scroll instead.
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(360)
        left_scroll.setMaximumWidth(460)
        row.addWidget(left_scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.monitor_panel = TrainingMonitorPanel()
        right_layout.addWidget(self.monitor_panel, 1)

        self.batch_progress = BatchProgressBar()
        right_layout.addWidget(self.batch_progress)

        self.progress_bar_label = QLabel("Ready")
        right_layout.addWidget(self.progress_bar_label)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        buttons.addWidget(self.start_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        buttons.addWidget(self.cancel_btn)
        right_layout.addLayout(buttons)

        row.addWidget(right, 1)

        self.status_label = QLabel("")
        outer.addWidget(self.status_label)

    def _build_pool_group(self) -> QGroupBox:
        group = QGroupBox("Pooled projects")
        v = QVBoxLayout(group)

        self.pool_tree = QTreeWidget()
        self.pool_tree.setHeaderHidden(True)
        # Capped rather than left to expand freely: pooled frame lists are
        # usually short, and letting this grow unbounded starves the
        # hyperparameter form below of the vertical space it needs.
        self.pool_tree.setMaximumHeight(160)
        self.pool_tree.itemChanged.connect(self._on_pool_item_changed)
        v.addWidget(self.pool_tree)

        hint = QLabel("Check individual frames to use a subset -- unchecked means use all.")
        hint.setWordWrap(True)
        v.addWidget(hint)

        buttons = QHBoxLayout()
        self.add_project_btn = QPushButton("Add project...")
        buttons.addWidget(self.add_project_btn)
        self.remove_project_btn = QPushButton("Remove")
        buttons.addWidget(self.remove_project_btn)
        v.addLayout(buttons)
        return group

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Training data")
        v = QVBoxLayout(group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Source stage:"))
        self.stage_combo = QComboBox()
        for stage in _POOL_STAGES:
            self.stage_combo.addItem(_POOL_STAGE_LABELS[stage], stage)
        row1.addWidget(self.stage_combo)
        v.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("Channel 0 (brightfield)", 0)
        self.channel_combo.addItem("Channel 1 (target)", 1)
        self.channel_combo.setToolTip(
            "D-Net is single-channel -- one checkpoint denoises one channel. "
            "Train a separate checkpoint per channel if needed."
        )
        row2.addWidget(self.channel_combo)
        v.addLayout(row2)

        return group

    def _build_checkpoint_group(self) -> QGroupBox:
        group = QGroupBox("Checkpoint output")
        v = QVBoxLayout(group)
        row = QHBoxLayout()
        self.checkpoint_path_edit = QLineEdit()
        self.checkpoint_path_edit.setPlaceholderText("checkpoint.pt")
        self.checkpoint_path_edit.setToolTip(
            "Defaults to denoise_<channel>.pt in the currently open project's "
            "root -- edit or Browse to save somewhere else."
        )
        row.addWidget(self.checkpoint_path_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_checkpoint_save_path)
        row.addWidget(browse_btn)
        v.addLayout(row)
        return group

    # ------------------------------------------------------------------
    # Wiring

    def _wire_up(self):
        self.add_project_btn.clicked.connect(self._add_project)
        self.remove_project_btn.clicked.connect(self._remove_selected_project)
        self.stage_combo.currentIndexChanged.connect(self._refresh_pool_tree)
        self.channel_combo.currentIndexChanged.connect(self._refresh_default_checkpoint_path)
        self.checkpoint_path_edit.textEdited.connect(self._on_checkpoint_path_edited)
        self.start_btn.clicked.connect(self._start_training)
        self.cancel_btn.clicked.connect(self._cancel_training)

    # ------------------------------------------------------------------
    # Pooled project tree -- top-level items are pooled project roots,
    # leaves are individual files under the selected stage, checkable so a
    # subset of frames can be pooled instead of always "every file".

    def _add_project(self):
        chosen = QFileDialog.getExistingDirectory(self, "Add project to training pool")
        if not chosen or chosen in self._pooled_roots:
            return
        self._pooled_roots.append(chosen)
        self._refresh_pool_tree()

    def _remove_selected_project(self):
        item = self.pool_tree.currentItem()
        if item is None:
            return
        while item.parent() is not None:
            item = item.parent()
        root = item.data(0, Qt.UserRole)
        self._pooled_roots = [r for r in self._pooled_roots if r != root]
        self._refresh_pool_tree()

    def _on_pool_item_changed(self, _item, _column):
        pass  # reserved -- frame gathering reads checked state live on demand

    def _refresh_pool_tree(self):
        stage = self.stage_combo.currentData()
        self.pool_tree.blockSignals(True)
        try:
            self.pool_tree.clear()
            for root in self._pooled_roots:
                folder = project_core.ProjectPaths(root).stage_dir(stage)
                paths = sorted(folder.glob("*.tiff")) if folder.is_dir() else []

                top_item = QTreeWidgetItem(self.pool_tree, [f"{root}  ({len(paths)} file(s))"])
                top_item.setData(0, Qt.UserRole, root)
                top_item.setFlags(top_item.flags() & ~Qt.ItemIsUserCheckable)

                for path in paths:
                    leaf = QTreeWidgetItem(top_item, [path.name])
                    leaf.setData(0, Qt.UserRole, str(path))
                    leaf.setFlags(leaf.flags() | Qt.ItemIsUserCheckable)
                    # Checked by default -- see ProjectTreePanel._refresh_stage:
                    # "pool everything" should be what the tree visibly shows,
                    # not an invisible fallback for an empty selection.
                    leaf.setCheckState(0, Qt.Checked)

                top_item.setExpanded(True)
        finally:
            self.pool_tree.blockSignals(False)

    # ------------------------------------------------------------------
    # Checkpoint path -- defaults to denoise_<channel>.pt in the currently
    # open project's root, so there's always an obvious, non-blank save
    # location unless you deliberately pick your own (Browse, or typing
    # into the field, both mark it as no longer auto-managed).

    def set_project_root(self, root: str):
        self._project_root = root
        self._refresh_default_checkpoint_path()

    def _default_checkpoint_filename(self) -> str:
        return checkpoint_filename_for_channel(self.channel_combo.currentData())

    def _default_checkpoint_path(self) -> str:
        filename = self._default_checkpoint_filename()
        if self._project_root:
            return str(Path(self._project_root) / filename)
        return filename

    def _refresh_default_checkpoint_path(self):
        if not self._checkpoint_path_is_default:
            return
        # setText() doesn't fire textEdited (only user keystrokes do), so
        # this can't accidentally flip the field over to "manually edited".
        self.checkpoint_path_edit.setText(self._default_checkpoint_path())

    def _on_checkpoint_path_edited(self, _text):
        self._checkpoint_path_is_default = False

    def _browse_checkpoint_save_path(self):
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save Checkpoint", self.checkpoint_path_edit.text() or "checkpoint.pt", "Checkpoint (*.pt)"
        )
        if path:
            if not path.endswith(".pt"):
                path += ".pt"
            self.checkpoint_path_edit.setText(path)
            self._checkpoint_path_is_default = False

    # ------------------------------------------------------------------
    # Training

    def _checked_paths_for_project(self, top_item: QTreeWidgetItem) -> list[str]:
        """Checked leaf paths under a pooled project's top-level item.
        Leaves start out checked (see `_refresh_pool_tree`), so pooling
        everything is the visible default rather than an invisible
        fallback -- unchecking a frame really does exclude it."""
        return [
            top_item.child(i).data(0, Qt.UserRole)
            for i in range(top_item.childCount())
            if top_item.child(i).checkState(0) == Qt.Checked
        ]

    def _gather_frames(self) -> list[np.ndarray]:
        channel = self.channel_combo.currentData()
        frames: list[np.ndarray] = []
        for i in range(self.pool_tree.topLevelItemCount()):
            top_item = self.pool_tree.topLevelItem(i)
            for path in self._checked_paths_for_project(top_item):
                arr = tifffile.imread(path)
                frames.append(np.asarray(arr[channel]))
        return frames

    def _start_training(self):
        if self._train_thread is not None:
            return

        frames = self._gather_frames()
        if not frames:
            QMessageBox.warning(
                self,
                "yeastprep",
                "No frames to train on -- add a project with files in the chosen "
                "source stage, and make sure at least one frame is checked.",
            )
            return

        config = self.params_panel.params()
        self._current_config = config
        self.monitor_panel.clear()
        self.monitor_panel.log(
            f"frames={len(frames)}  shape={frames[0].shape}  device={config.device or 'auto'}\n"
            f"epochs={config.epochs}  steps_per_epoch={config.steps_per_epoch}  "
            f"tile_size={config.tile_size}  tiles_per_batch={config.tiles_per_batch}  lr={config.lr}  "
            f"early_stop_patience={config.early_stop_patience or 'off'}"
        )
        self.status_label.setText("Training...")
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        total_steps = max(1, config.epochs * config.steps_per_epoch)
        self._emit_progress(PageProgress(active=True, done=0, total=total_steps))

        self._train_worker = TrainingWorker(frames, config)
        self._train_thread = QThread()
        self._train_worker.moveToThread(self._train_thread)

        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.step_progress.connect(self._on_step)
        self._train_worker.epoch_finished.connect(self._on_epoch)
        self._train_worker.epoch_metrics.connect(self._on_epoch_metrics)
        self._train_worker.early_stopped.connect(self._on_early_stopped)
        self._train_worker.finished.connect(self._on_finished)
        self._train_worker.cancelled.connect(self._on_cancelled)
        self._train_worker.error.connect(self._on_error)

        self._train_thread.start()

    def _emit_progress(self, progress: PageProgress):
        self.progress_changed.emit(progress)
        self.batch_progress.apply(progress)

    def _on_step(self, step, total_steps, epoch, loss):
        done = (epoch - 1) * total_steps + step
        total = max(1, total_steps * self._current_config.epochs)
        self._emit_progress(PageProgress(active=True, done=done, total=total))
        self.progress_bar_label.setText(f"epoch {epoch}  step {step}/{total_steps}  loss={loss:.4f}")

    def _on_epoch(self, epoch, total_epochs, loss, lr):
        self._last_epoch_loss = loss
        self.monitor_panel.log(f"epoch {epoch}/{total_epochs}  loss={loss:.4f}  lr={lr:.2e}")

    def _on_epoch_metrics(self, epoch, metrics):
        mu_mse = metrics.get("mu_mse")
        self.monitor_panel.append_epoch(epoch, getattr(self, "_last_epoch_loss", 0.0), mu_mse)

    def _on_early_stopped(self, epoch, best_epoch, best_mu_mse):
        self.monitor_panel.log(
            f"early stopping at epoch {epoch} -- mu_mse plateaued since epoch "
            f"{best_epoch} (best mu_mse={best_mu_mse:.4f}); using that epoch's weights"
        )

    def _finish(self, checkpoint, cancelled: bool):
        path = self.checkpoint_path_edit.text().strip()
        if path:
            try:
                save_checkpoint(path, checkpoint)
                settings.add_recent_denoise_checkpoint(path)
                self.monitor_panel.log(f"saved checkpoint to {path}")
                self.checkpoint_trained.emit(self.channel_combo.currentData(), path)
            except Exception as exc:
                self.monitor_panel.log(f"ERROR saving checkpoint: {exc}")
        self.status_label.setText(
            "Training stopped early -- checkpoint saved." if cancelled else "Training completed."
        )
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._emit_progress(PageProgress(active=False))
        self._train_thread.quit()
        self._train_thread.wait()
        self._train_thread = None
        self._train_worker = None

    def _on_finished(self, checkpoint):
        self._finish(checkpoint, cancelled=False)

    def _on_cancelled(self, checkpoint):
        self._finish(checkpoint, cancelled=True)

    def _on_error(self, message: str):
        self.monitor_panel.log(f"ERROR: {message}")
        self.status_label.setText(f"Training error: {message.splitlines()[0] if message else ''}")
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._emit_progress(PageProgress(active=False))
        self._train_thread.quit()
        self._train_thread.wait()
        self._train_thread = None
        self._train_worker = None

    def _cancel_training(self):
        if self._train_worker is not None:
            self._train_worker.cancel()
            self.status_label.setText("Cancelling after current step...")
            self.cancel_btn.setEnabled(False)

    # ------------------------------------------------------------------

    def shutdown(self):
        if self._train_thread is not None:
            self._cancel_training()
            self._train_thread.quit()
            self._train_thread.wait()
