"""Tile export parameter controls: crop size, contrast percentile clip.

Structurally mirrors segmentation_params_panel.py -- same
params_changed/auto-recompute wiring, same Recompute Now / Save & Reset
folder-defaults actions. "Recompute" here just redraws the crop-window
preview (core.tiles.cell_geometry is cheap, no model forward pass), unlike
segmentation's live cellpose recompute.
"""

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QSpinBox,
    QVBoxLayout,
    QPushButton,
    QWidget,
)

from yeastprep.core.tiles import TileParams


class TileParamsPanel(QWidget):
    params_changed = Signal(object)  # TileParams
    recompute_requested = Signal()
    save_defaults_requested = Signal()
    reset_defaults_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._emitting = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._build_params_group())
        layout.addLayout(self._build_actions_row())
        layout.addStretch(1)

    # ------------------------------------------------------------------

    def _build_params_group(self) -> QGroupBox:
        group = QGroupBox("Tile parameters")
        form = QFormLayout(group)
        defaults = TileParams()

        self.size = QSpinBox()
        self.size.setRange(8, 512)
        self.size.setSingleStep(8)
        self.size.setValue(defaults.size)
        self.size.valueChanged.connect(self._emit_params)
        form.addRow("Tile size (px)", self.size)

        self.contrast_lo_pct = QDoubleSpinBox()
        self.contrast_lo_pct.setRange(0.0, 49.0)
        self.contrast_lo_pct.setSingleStep(0.1)
        self.contrast_lo_pct.setDecimals(2)
        self.contrast_lo_pct.setValue(defaults.contrast_lo_pct)
        self.contrast_lo_pct.valueChanged.connect(self._emit_params)
        form.addRow("Contrast low percentile", self.contrast_lo_pct)

        self.contrast_hi_pct = QDoubleSpinBox()
        self.contrast_hi_pct.setRange(51.0, 100.0)
        self.contrast_hi_pct.setSingleStep(0.1)
        self.contrast_hi_pct.setDecimals(2)
        self.contrast_hi_pct.setValue(defaults.contrast_hi_pct)
        self.contrast_hi_pct.valueChanged.connect(self._emit_params)
        form.addRow("Contrast high percentile", self.contrast_hi_pct)

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

    def params(self) -> TileParams:
        return TileParams(
            size=self.size.value(),
            contrast_lo_pct=self.contrast_lo_pct.value(),
            contrast_hi_pct=self.contrast_hi_pct.value(),
        )

    def set_params(self, params: TileParams):
        self._emitting = True
        self.size.setValue(params.size)
        self.contrast_lo_pct.setValue(params.contrast_lo_pct)
        self.contrast_hi_pct.setValue(params.contrast_hi_pct)
        self._emitting = False

    def is_auto_recompute(self) -> bool:
        return self.auto_recompute.isChecked()

    # ------------------------------------------------------------------

    def _emit_params(self, _value=None):
        if self._emitting:
            return
        self.params_changed.emit(self.params())
