"""Cellpose segmentation parameter controls: model path, diameter,
flow/cellprob thresholds, edge-mask removal.

Structurally mirrors params_panel.py -- same params_changed/auto-recompute
wiring, same Recompute Now / Save & Reset folder-defaults actions.
"""

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
    QVBoxLayout,
    QWidget,
)

from yeastprep.core.segmentation import SegmentationParams


class SegmentationParamsPanel(QWidget):
    params_changed = Signal(object)  # SegmentationParams
    recompute_requested = Signal()
    save_defaults_requested = Signal()
    reset_defaults_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._emitting = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_params_group())
        layout.addLayout(self._build_actions_row())
        layout.addStretch(1)

    # ------------------------------------------------------------------

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("Cellpose model")
        form = QFormLayout(group)

        model_row = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("(built-in default)")
        self.model_path_edit.editingFinished.connect(self._emit_params)
        model_row.addWidget(self.model_path_edit, 1)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_model)
        model_row.addWidget(self.browse_btn)
        form.addRow("Model", model_row)

        return group

    def _build_params_group(self) -> QGroupBox:
        group = QGroupBox("Segmentation parameters")
        form = QFormLayout(group)
        defaults = SegmentationParams()

        self.diameter = QDoubleSpinBox()
        self.diameter.setRange(1.0, 500.0)
        self.diameter.setDecimals(1)
        self.diameter.setValue(defaults.diameter)
        self.diameter.valueChanged.connect(self._emit_params)
        form.addRow("Cell diameter (px)", self.diameter)

        self.flow_threshold = QDoubleSpinBox()
        self.flow_threshold.setRange(0.0, 3.0)
        self.flow_threshold.setSingleStep(0.05)
        self.flow_threshold.setDecimals(2)
        self.flow_threshold.setValue(defaults.flow_threshold)
        self.flow_threshold.valueChanged.connect(self._emit_params)
        form.addRow("Flow threshold", self.flow_threshold)

        self.cellprob_threshold = QDoubleSpinBox()
        self.cellprob_threshold.setRange(-6.0, 6.0)
        self.cellprob_threshold.setSingleStep(0.1)
        self.cellprob_threshold.setDecimals(2)
        self.cellprob_threshold.setValue(defaults.cellprob_threshold)
        self.cellprob_threshold.valueChanged.connect(self._emit_params)
        form.addRow("Cellprob threshold", self.cellprob_threshold)

        self.remove_edge_masks = QCheckBox("Remove edge-touching cells")
        self.remove_edge_masks.setChecked(defaults.remove_edge_masks)
        self.remove_edge_masks.stateChanged.connect(self._emit_params)
        form.addRow(self.remove_edge_masks)

        self.auto_recompute = QCheckBox("Auto-recompute on change")
        self.auto_recompute.setChecked(True)
        form.addRow(self.auto_recompute)

        return group

    def _build_actions_row(self) -> QVBoxLayout:
        column = QVBoxLayout()

        recompute_row = QHBoxLayout()
        self.recompute_btn = QPushButton("Recompute Now")
        self.recompute_btn.clicked.connect(self.recompute_requested)
        recompute_row.addWidget(self.recompute_btn)
        column.addLayout(recompute_row)

        defaults_row = QHBoxLayout()
        self.save_defaults_btn = QPushButton("Save as folder defaults")
        self.save_defaults_btn.clicked.connect(self.save_defaults_requested)
        defaults_row.addWidget(self.save_defaults_btn)
        self.reset_defaults_btn = QPushButton("Reset to folder defaults")
        self.reset_defaults_btn.clicked.connect(self.reset_defaults_requested)
        defaults_row.addWidget(self.reset_defaults_btn)
        column.addLayout(defaults_row)

        return column

    # ------------------------------------------------------------------
    # Public API

    def params(self) -> SegmentationParams:
        return SegmentationParams(
            model_path=self.model_path_edit.text().strip() or None,
            diameter=self.diameter.value(),
            flow_threshold=self.flow_threshold.value(),
            cellprob_threshold=self.cellprob_threshold.value(),
            remove_edge_masks=self.remove_edge_masks.isChecked(),
        )

    def set_params(self, params: SegmentationParams):
        self._emitting = True
        self.model_path_edit.setText(params.model_path or "")
        self.diameter.setValue(params.diameter)
        self.flow_threshold.setValue(params.flow_threshold)
        self.cellprob_threshold.setValue(params.cellprob_threshold)
        self.remove_edge_masks.setChecked(params.remove_edge_masks)
        self._emitting = False

    def is_auto_recompute(self) -> bool:
        return self.auto_recompute.isChecked()

    # ------------------------------------------------------------------

    def _browse_model(self):
        start_dir = self.model_path_edit.text().strip() or str(Path.home())
        path, _filter = QFileDialog.getOpenFileName(
            self, "Select cellpose model file", start_dir
        )
        if path:
            self.model_path_edit.setText(path)
            self._emit_params()

    def _emit_params(self, _value=None):
        if self._emitting:
            return
        self.params_changed.emit(self.params())
