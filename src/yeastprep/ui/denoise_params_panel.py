"""Denoise parameter controls: which channels a batch run denoises, and
which channel the preview shows. There is nothing to train or load -- each
channel of each image gets its own network, fitted to that image (see
core/denoise.py) -- so this is the whole of the method's UI."""

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core.denoise import BRIGHTFIELD_CHANNEL, TARGET_CHANNEL, DenoiseParams

CHANNEL_LABELS = {
    BRIGHTFIELD_CHANNEL: "Channel 0 (brightfield)",
    TARGET_CHANNEL: "Channel 1 (target/fluorescence)",
}


class DenoiseParamsPanel(QWidget):
    params_changed = Signal(object)  # DenoiseParams
    preview_channel_changed = Signal(int)
    preview_requested = Signal()
    save_defaults_requested = Signal()
    reset_defaults_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._emitting = False
        # Not exposed: the default is what was validated (see core/denoise.py),
        # but a project's saved value is kept as-is.
        self._steps = DenoiseParams().steps

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._build_group())
        layout.addLayout(self._build_actions())
        layout.addStretch(1)

    def _build_group(self) -> QGroupBox:
        group = QGroupBox("Denoise")
        form = QFormLayout(group)

        about = QLabel(
            "A small network is fitted to each channel of each image on its own "
            "(about 30 s per channel on a laptop GPU) -- nothing to train or load."
        )
        about.setWordWrap(True)
        about.setStyleSheet("color: #a0a0a0;")
        form.addRow(about)

        self.channel_checks = {}
        for channel, label in CHANNEL_LABELS.items():
            check = QCheckBox(label)
            check.setChecked(True)
            check.toggled.connect(self._emit_params)
            self.channel_checks[channel] = check
            form.addRow(check)

        self.preview_combo = QComboBox()
        for channel, label in CHANNEL_LABELS.items():
            self.preview_combo.addItem(label, channel)
        self.preview_combo.currentIndexChanged.connect(
            lambda _i: self.preview_channel_changed.emit(self.preview_channel())
        )
        form.addRow("Preview", self.preview_combo)
        return group

    def _build_actions(self) -> QVBoxLayout:
        column = QVBoxLayout()
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setToolTip("Denoise the previewed channel of the selected file.")
        self.preview_btn.clicked.connect(self.preview_requested)
        column.addWidget(self.preview_btn)

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

    def params(self) -> DenoiseParams:
        return DenoiseParams(
            channels=tuple(c for c, check in self.channel_checks.items() if check.isChecked()),
            steps=self._steps,
        )

    def set_params(self, params: DenoiseParams):
        self._steps = params.steps
        self._emitting = True
        for channel, check in self.channel_checks.items():
            check.setChecked(channel in params.channels)
        self._emitting = False
        self._emit_params()

    def preview_channel(self) -> int:
        return self.preview_combo.currentData()

    def _emit_params(self, _value=None):
        if not self._emitting:
            self.params_changed.emit(self.params())
