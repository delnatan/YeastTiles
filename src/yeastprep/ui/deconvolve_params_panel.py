"""Deconvolve parameter controls: target-channel-only PSF + solver
settings."""

from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core.deconvolve import DeconvolveParams, effective_background


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

        self.camera_offset = QDoubleSpinBox()
        self.camera_offset.setRange(0.0, 65535.0)
        self.camera_offset.setDecimals(1)
        self.camera_offset.setValue(defaults.camera_offset)
        self.camera_offset.setToolTip(
            "The camera's dark offset in raw counts per z-slice. The target "
            "channel is a raw z-sum, so the solver's background is this "
            "times each file's z-slice count, plus the extra background."
        )
        self.camera_offset.valueChanged.connect(self._emit_params)
        form.addRow("Camera offset / slice", self.camera_offset)

        self.background = QDoubleSpinBox()
        self.background.setRange(0.0, 1e9)
        self.background.setDecimals(2)
        self.background.setValue(defaults.background)
        self.background.setToolTip("Any further flat background, in summed counts.")
        self.background.valueChanged.connect(self._emit_params)
        form.addRow("Extra background", self.background)

        # Filled in by the page for the selected file (`show_background`).
        self.total_background_label = QLabel("--")
        form.addRow("Total background", self.total_background_label)

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
            camera_offset=self.camera_offset.value(),
            background=self.background.value(),
            num_iter=self.num_iter.value(),
        )

    def set_params(self, params: DeconvolveParams):
        self._emitting = True
        self.enabled.setChecked(params.enabled)
        self.psf_path_edit.setText(params.psf_path or "")
        self.zoom.setValue(params.zoom)
        self.camera_offset.setValue(params.camera_offset)
        self.background.setValue(params.background)
        self.num_iter.setValue(params.num_iter)
        self._emitting = False

    def show_background(self, n_z: int | None):
        """Show the solver background the current settings give a file
        whose target channel sums `n_z` raw z-slices."""
        params = self.params()
        try:
            total = effective_background(params, n_z)
        except ValueError:
            self.total_background_label.setText("unknown -- file has no z-slice count")
            return
        if params.camera_offset and n_z is not None:
            text = f"{total:g}  ({params.camera_offset:g} x {n_z} slices"
            text += f" + {params.background:g})" if params.background else ")"
        else:
            text = f"{total:g}"
        self.total_background_label.setText(text)

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
