"""Training hyperparameter controls for the Train Denoiser page, wrapping
`jssl_denoise.config.TrainingConfig` -- field set mirrors the Train tab of
jssl_denoise's own (pyvistra-plugin-only) DenoiseDialog."""

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from jssl_denoise.config import TrainingConfig


def _dspin(value, minimum, maximum, step=0.01, decimals=4) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(value)
    return box


def _ispin(value, minimum, maximum) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    return box


_DEFAULT_EARLY_STOP_PATIENCE = 15


def _early_stop_spin() -> QSpinBox:
    box = _ispin(_DEFAULT_EARLY_STOP_PATIENCE, 0, 500)
    box.setSpecialValueText("Off")
    box.setToolTip(
        "Stop once mu_mse (the actual denoising-quality metric, not the raw "
        "loss) hasn't improved for this many epochs, and use that best "
        "epoch's weights rather than the final epoch's -- raw loss keeps "
        "drifting well after mu_mse plateaus (N-Net's noise model wandering "
        "once D-Net has converged), so training past the plateau is wasted "
        "compute. 0 disables early stopping."
    )
    return box


def _device_combo() -> QComboBox:
    import torch

    combo = QComboBox()
    combo.addItem("Auto", None)
    combo.addItem("CPU", "cpu")
    cuda_idx = combo.count()
    combo.addItem("CUDA", "cuda")
    if not torch.cuda.is_available():
        combo.model().item(cuda_idx).setEnabled(False)
    mps_idx = combo.count()
    combo.addItem("MPS", "mps")
    if not torch.backends.mps.is_available():
        combo.model().item(mps_idx).setEnabled(False)
    return combo


class TrainParamsPanel(QWidget):
    params_changed = Signal(object)  # TrainingConfig

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._build_group())
        layout.addStretch(1)

    def _build_group(self) -> QGroupBox:
        group = QGroupBox("Training")
        form = QFormLayout(group)
        form.setVerticalSpacing(8)
        defaults = TrainingConfig()

        self.epochs_spin = _ispin(60, 1, 5000)
        self.steps_per_epoch_spin = _ispin(100, 1, 5000)
        self.tile_size_spin = _ispin(defaults.tile_size, 16, 1024)
        self.tiles_per_batch_spin = _ispin(32, 1, 512)
        self.lr_spin = _dspin(defaults.lr, 1e-6, 1.0, 1e-4, decimals=6)
        self.base_filters_spin = _ispin(defaults.base_filters, 4, 256)
        self.nll_beta_spin = _dspin(defaults.nll_beta, 0.0, 1.0, 0.05, decimals=3)
        self.augment_cb = QCheckBox("Augment (flips/rotations)")
        self.augment_cb.setChecked(defaults.augment)
        self.device_combo = _device_combo()
        self.early_stop_spin = _early_stop_spin()

        form.addRow("Epochs:", self.epochs_spin)
        form.addRow("Steps/epoch:", self.steps_per_epoch_spin)
        form.addRow("Tile size:", self.tile_size_spin)
        form.addRow("Tiles/batch:", self.tiles_per_batch_spin)
        form.addRow("Learning rate:", self.lr_spin)
        form.addRow("Base filters:", self.base_filters_spin)
        form.addRow("NLL beta:", self.nll_beta_spin)
        form.addRow(self.augment_cb)
        form.addRow("Device:", self.device_combo)
        form.addRow("Early stop patience:", self.early_stop_spin)

        for box in (
            self.epochs_spin,
            self.steps_per_epoch_spin,
            self.tile_size_spin,
            self.tiles_per_batch_spin,
            self.base_filters_spin,
            self.early_stop_spin,
        ):
            box.valueChanged.connect(self._emit_params)
        self.lr_spin.valueChanged.connect(self._emit_params)
        self.nll_beta_spin.valueChanged.connect(self._emit_params)
        self.augment_cb.stateChanged.connect(self._emit_params)
        self.device_combo.currentIndexChanged.connect(self._emit_params)

        return group

    def params(self) -> TrainingConfig:
        return TrainingConfig(
            lr=self.lr_spin.value(),
            epochs=self.epochs_spin.value(),
            steps_per_epoch=self.steps_per_epoch_spin.value(),
            tile_size=self.tile_size_spin.value(),
            tiles_per_batch=self.tiles_per_batch_spin.value(),
            nll_beta=self.nll_beta_spin.value(),
            augment=self.augment_cb.isChecked(),
            base_filters=self.base_filters_spin.value(),
            device=self.device_combo.currentData(),
            early_stop_patience=self.early_stop_spin.value() or None,
        )

    def _emit_params(self, _value=None):
        self.params_changed.emit(self.params())
