"""Deconvolve parameter controls: target-channel-only PSF + solver
settings. Split out of the old enhance_params_panel.py's "Deconvolve" group
box so deconvolution is its own independently toggleable stage (see
denoise_params_panel.py for the other half)."""

from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core.deconvolve import DeconvolveParams


class DeconvolveParamsPanel(QWidget):
    params_changed = Signal(object)  # DeconvolveParams
    recompute_requested = Signal()
    save_defaults_requested = Signal()
    reset_defaults_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._emitting = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._build_group())
        layout.addLayout(self._build_actions_row())
        layout.addStretch(1)

    def _build_group(self) -> QGroupBox:
        group = QGroupBox("Deconvolve (target/fluorescence channel only)")
        form = QFormLayout(group)
        defaults = DeconvolveParams()

        self.enabled = QCheckBox("Enable deconvolution")
        self.enabled.stateChanged.connect(self._emit_params)
        form.addRow(self.enabled)

        psf_row = QHBoxLayout()
        self.psf_path_edit = QLineEdit()
        self.psf_path_edit.setPlaceholderText("(no PSF selected)")
        self.psf_path_edit.editingFinished.connect(self._emit_params)
        psf_row.addWidget(self.psf_path_edit, 1)
        self.browse_psf_btn = QPushButton("Browse...")
        self.browse_psf_btn.clicked.connect(self._browse_psf)
        psf_row.addWidget(self.browse_psf_btn)
        form.addRow("PSF file", psf_row)

        self.zoom = QDoubleSpinBox()
        self.zoom.setRange(1.0, 8.0)
        self.zoom.setSingleStep(0.5)
        self.zoom.setDecimals(2)
        self.zoom.setValue(defaults.zoom)
        self.zoom.valueChanged.connect(self._emit_params)
        form.addRow("Zoom", self.zoom)

        self.background = QDoubleSpinBox()
        self.background.setRange(0.0, 1e6)
        self.background.setDecimals(2)
        self.background.setValue(defaults.background)
        self.background.valueChanged.connect(self._emit_params)
        form.addRow("Background", self.background)

        self.num_iter = QSpinBox()
        self.num_iter.setRange(1, 1000)
        self.num_iter.setValue(defaults.num_iter)
        self.num_iter.valueChanged.connect(self._emit_params)
        form.addRow("Iterations", self.num_iter)

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

    def params(self) -> DeconvolveParams:
        return DeconvolveParams(
            enabled=self.enabled.isChecked(),
            psf_path=self.psf_path_edit.text().strip() or None,
            zoom=self.zoom.value(),
            background=self.background.value(),
            num_iter=self.num_iter.value(),
        )

    def set_params(self, params: DeconvolveParams):
        self._emitting = True
        self.enabled.setChecked(params.enabled)
        self.psf_path_edit.setText(params.psf_path or "")
        self.zoom.setValue(params.zoom)
        self.background.setValue(params.background)
        self.num_iter.setValue(params.num_iter)
        self._emitting = False

    def is_auto_recompute(self) -> bool:
        return self.auto_recompute.isChecked()

    def set_psf_path(self, path: str):
        """Called when the PSF Calculator tab saves a new PSF file, so the
        freshly computed PSF is immediately usable here without a manual
        Browse step."""
        self.psf_path_edit.setText(path)
        self._emit_params()

    # ------------------------------------------------------------------

    def _browse_psf(self):
        start_dir = self.psf_path_edit.text().strip() or str(Path.home())
        path, _filter = QFileDialog.getOpenFileName(
            self, "Select PSF tiff", start_dir, "TIFF (*.tif *.tiff)"
        )
        if path:
            self.psf_path_edit.setText(path)
            self._emit_params()

    def _emit_params(self, _value=None):
        if self._emitting:
            return
        self.params_changed.emit(self.params())
