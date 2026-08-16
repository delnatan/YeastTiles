"""Denoise parameter controls: a channel selector + that channel's
jssl-denoise checkpoint + TTA toggle. Brightfield and the target/
fluorescence channel are different imaging modalities and need different
trained networks, so denoising is done one channel at a time -- this panel
remembers each channel's last-used checkpoint path (in-session, and via
ProjectConfig.denoise_checkpoints across project loads) so switching the
channel selector doesn't lose the other channel's checkpoint."""

from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core.denoise import BRIGHTFIELD_CHANNEL, TARGET_CHANNEL, DenoiseParams


class DenoiseParamsPanel(QWidget):
    params_changed = Signal(object)  # DenoiseParams
    recompute_requested = Signal()
    save_defaults_requested = Signal()
    reset_defaults_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._emitting = False
        self._checkpoint_by_channel: dict[int, str] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._build_group())
        layout.addLayout(self._build_actions_row())
        layout.addStretch(1)

    def _build_group(self) -> QGroupBox:
        group = QGroupBox("Denoise (jssl-denoise)")
        form = QFormLayout(group)
        defaults = DenoiseParams()

        self.channel_combo = QComboBox()
        self.channel_combo.addItem("Channel 0 (brightfield)", BRIGHTFIELD_CHANNEL)
        self.channel_combo.addItem("Channel 1 (target/fluorescence)", TARGET_CHANNEL)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)
        form.addRow("Channel", self.channel_combo)

        checkpoint_row = QHBoxLayout()
        self.checkpoint_path_edit = QLineEdit()
        self.checkpoint_path_edit.setPlaceholderText("(no checkpoint selected)")
        self.checkpoint_path_edit.editingFinished.connect(self._emit_params)
        checkpoint_row.addWidget(self.checkpoint_path_edit, 1)
        self.browse_checkpoint_btn = QPushButton("Browse...")
        self.browse_checkpoint_btn.clicked.connect(self._browse_checkpoint)
        checkpoint_row.addWidget(self.browse_checkpoint_btn)
        form.addRow("Checkpoint", checkpoint_row)

        self.tta = QCheckBox("Test-time augmentation (8-way, slower)")
        self.tta.setChecked(defaults.tta)
        self.tta.stateChanged.connect(self._emit_params)
        form.addRow(self.tta)

        self.auto_recompute = QCheckBox("Auto-recompute on change")
        self.auto_recompute.setChecked(False)
        form.addRow(self.auto_recompute)

        return group

    def _build_actions_row(self) -> QVBoxLayout:
        column = QVBoxLayout()

        recompute_row = QHBoxLayout()
        self.recompute_btn = QPushButton("Do it")
        self.recompute_btn.clicked.connect(self.recompute_requested)
        recompute_row.addWidget(self.recompute_btn)
        column.addLayout(recompute_row)

        defaults_row = QHBoxLayout()
        self.save_defaults_btn = QPushButton("Save as project defaults")
        self.save_defaults_btn.clicked.connect(self.save_defaults_requested)
        defaults_row.addWidget(self.save_defaults_btn)
        self.reset_defaults_btn = QPushButton("Reset to project defaults")
        self.reset_defaults_btn.clicked.connect(self.reset_defaults_requested)
        defaults_row.addWidget(self.reset_defaults_btn)
        column.addLayout(defaults_row)

        return column

    # ------------------------------------------------------------------
    # Public API

    def channel(self) -> int:
        return self.channel_combo.currentData()

    def params(self) -> DenoiseParams:
        return DenoiseParams(
            channel=self.channel(),
            checkpoint_path=self.checkpoint_path_edit.text().strip() or None,
            tta=self.tta.isChecked(),
        )

    def set_params(self, params: DenoiseParams):
        self._checkpoint_by_channel[params.channel] = params.checkpoint_path or ""
        self._emitting = True
        idx = self.channel_combo.findData(params.channel)
        self.channel_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.checkpoint_path_edit.setText(params.checkpoint_path or "")
        self.tta.setChecked(params.tta)
        self._emitting = False

    def is_auto_recompute(self) -> bool:
        return self.auto_recompute.isChecked()

    def checkpoints_by_channel(self) -> dict[int, str]:
        """Every channel's last-known checkpoint path, including whatever's
        currently in the checkpoint field for the active channel -- for
        persisting into ProjectConfig.denoise_checkpoints."""
        checkpoints = dict(self._checkpoint_by_channel)
        checkpoints[self.channel()] = self.checkpoint_path_edit.text().strip()
        return {channel: path for channel, path in checkpoints.items() if path}

    def set_checkpoints_by_channel(self, checkpoints: dict[int, str]):
        """Restore the per-channel checkpoint cache -- e.g. when a project
        loads -- without disturbing which channel is currently selected."""
        self._checkpoint_by_channel.update(checkpoints)
        path = self._checkpoint_by_channel.get(self.channel(), "")
        if path:
            self.checkpoint_path_edit.setText(path)
            self._emit_params()

    def set_checkpoint_for_channel(self, channel: int, path: str):
        """Called when the Training tab finishes a run, so a freshly
        trained checkpoint is immediately usable for its channel here
        without a manual Browse step."""
        self._checkpoint_by_channel[channel] = path
        if channel == self.channel():
            self.checkpoint_path_edit.setText(path)
            self._emit_params()

    # ------------------------------------------------------------------

    def _on_channel_changed(self, _index):
        path = self._checkpoint_by_channel.get(self.channel(), "")
        self._emitting = True
        self.checkpoint_path_edit.setText(path)
        self._emitting = False
        self._emit_params()

    def _browse_checkpoint(self):
        start_dir = self.checkpoint_path_edit.text().strip() or str(Path.home())
        path, _filter = QFileDialog.getOpenFileName(
            self, "Select jssl-denoise checkpoint", start_dir, "Checkpoint (*.pt)"
        )
        if path:
            self.checkpoint_path_edit.setText(path)
            self._emit_params()

    def _emit_params(self, _value=None):
        if self._emitting:
            return
        params = self.params()
        self._checkpoint_by_channel[params.channel] = params.checkpoint_path or ""
        self.params_changed.emit(params)
