"""Field-flattening parameter controls: tile grid, peak-finding
prominence, polynomial degree, defocus offset, and channel selection.

Emits `params_changed`/`channels_changed` on every edit for the
auto-recompute wiring in ui/worker.py, plus explicit
`recompute_requested`/`save_defaults_requested`/`reset_defaults_requested`
signals for the manual-control buttons.
"""

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core.channels import ChannelSelection
from yeastprep.core.pipeline import FlattenFieldParams


class ParamsPanel(QWidget):
    params_changed = Signal(object)  # FlattenFieldParams
    channels_changed = Signal(object)  # ChannelSelection
    recompute_requested = Signal()
    save_defaults_requested = Signal()
    reset_defaults_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._channels_meta: list[dict] = []
        self._emitting = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._build_scale_group())
        layout.addWidget(self._build_params_group())
        layout.addWidget(self._build_channels_group())
        layout.addLayout(self._build_actions_row())
        layout.addStretch(1)

    # ------------------------------------------------------------------

    def _build_scale_group(self) -> QGroupBox:
        group = QGroupBox("Acquisition scale (read-only)")
        form = QFormLayout(group)
        self.dz_label = QLabel("-")
        self.dy_label = QLabel("-")
        self.dx_label = QLabel("-")
        form.addRow("dz (um)", self.dz_label)
        form.addRow("dy (um)", self.dy_label)
        form.addRow("dx (um)", self.dx_label)
        return group

    def _build_params_group(self) -> QGroupBox:
        group = QGroupBox("Field-flattening parameters")
        form = QFormLayout(group)
        defaults = FlattenFieldParams()

        self.num_tiles_y = QSpinBox()
        self.num_tiles_y.setRange(2, 64)
        self.num_tiles_y.setValue(defaults.num_tiles_y)
        self.num_tiles_y.valueChanged.connect(self._emit_params)
        form.addRow("Tiles (Y)", self.num_tiles_y)

        self.num_tiles_x = QSpinBox()
        self.num_tiles_x.setRange(2, 64)
        self.num_tiles_x.setValue(defaults.num_tiles_x)
        self.num_tiles_x.valueChanged.connect(self._emit_params)
        form.addRow("Tiles (X)", self.num_tiles_x)

        self.prominence = QDoubleSpinBox()
        self.prominence.setRange(0.0, 1e9)
        self.prominence.setDecimals(1)
        self.prominence.setValue(defaults.inverted_variance_prominence)
        self.prominence.valueChanged.connect(self._emit_params)
        form.addRow("Peak prominence", self.prominence)

        degree_row = QHBoxLayout()
        self.poly_degree_x = QSpinBox()
        self.poly_degree_x.setRange(1, 6)
        self.poly_degree_x.setValue(defaults.poly_degree[0])
        self.poly_degree_x.valueChanged.connect(self._emit_params)
        degree_row.addWidget(self.poly_degree_x)
        self.poly_degree_y = QSpinBox()
        self.poly_degree_y.setRange(1, 6)
        self.poly_degree_y.setValue(defaults.poly_degree[1])
        self.poly_degree_y.valueChanged.connect(self._emit_params)
        degree_row.addWidget(self.poly_degree_y)
        form.addRow("Poly degree (x, y)", degree_row)

        self.offset_um = QDoubleSpinBox()
        self.offset_um.setRange(-50.0, 50.0)
        self.offset_um.setSingleStep(0.1)
        self.offset_um.setDecimals(3)
        self.offset_um.setValue(defaults.offset_um)
        self.offset_um.valueChanged.connect(self._emit_params)
        form.addRow("Defocus offset (um)", self.offset_um)

        self.auto_recompute = QCheckBox("Auto-recompute on change")
        self.auto_recompute.setChecked(False)
        form.addRow(self.auto_recompute)

        return group

    def _build_channels_group(self) -> QGroupBox:
        group = QGroupBox("Channel selection")
        form = QFormLayout(group)

        # Dropdowns listing the file's actual channels (by name/wavelength),
        # not raw index spin-boxes -- there's nothing to mis-index this way,
        # and picking the same channel for both is immediately visible
        # instead of a silent (0, 0) default. See `set_channels_meta`.
        self.brightfield_channel = QComboBox()
        self.brightfield_channel.currentIndexChanged.connect(self._on_channel_edited)
        form.addRow("Brightfield ch.", self.brightfield_channel)

        self.target_channel = QComboBox()
        self.target_channel.currentIndexChanged.connect(self._on_channel_edited)
        form.addRow("Target (DAPI) ch.", self.target_channel)

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

    def params(self) -> FlattenFieldParams:
        return FlattenFieldParams(
            num_tiles_y=self.num_tiles_y.value(),
            num_tiles_x=self.num_tiles_x.value(),
            inverted_variance_prominence=self.prominence.value(),
            poly_degree=(self.poly_degree_x.value(), self.poly_degree_y.value()),
            offset_um=self.offset_um.value(),
        )

    def set_params(self, params: FlattenFieldParams):
        self._emitting = True
        self.num_tiles_y.setValue(params.num_tiles_y)
        self.num_tiles_x.setValue(params.num_tiles_x)
        self.prominence.setValue(params.inverted_variance_prominence)
        self.poly_degree_x.setValue(params.poly_degree[0])
        self.poly_degree_y.setValue(params.poly_degree[1])
        self.offset_um.setValue(params.offset_um)
        self._emitting = False

    def channels(self) -> ChannelSelection:
        brightfield = self.brightfield_channel.currentData()
        target = self.target_channel.currentData()
        return ChannelSelection(
            brightfield=brightfield if brightfield is not None else 0,
            projection=target if target is not None else 0,
        )

    def set_channels(self, channels: ChannelSelection):
        self._emitting = True
        self._set_combo_channel(self.brightfield_channel, channels.brightfield)
        self._set_combo_channel(self.target_channel, channels.projection)
        self._emitting = False

    def set_scale(self, dz: float, dy: float, dx: float):
        self.dz_label.setText(f"{dz:.4f}")
        self.dy_label.setText(f"{dy:.4f}")
        self.dx_label.setText(f"{dx:.4f}")

    def set_channels_meta(self, channels_meta: list[dict]):
        """Rebuild the brightfield/target dropdowns for a newly loaded
        file's `meta['channels']`, listing every channel by name (or
        wavelength, or just its index if neither is known). Keeps the
        current brightfield/target selection if it's still a valid, distinct
        pair for the new channel count; otherwise falls back to (0, 1)
        rather than silently collapsing both to the same channel."""
        self._channels_meta = channels_meta or []
        n = len(self._channels_meta)

        prev_bf = self.brightfield_channel.currentData()
        prev_tg = self.target_channel.currentData()

        self._emitting = True
        for combo in (self.brightfield_channel, self.target_channel):
            combo.blockSignals(True)
            combo.clear()
            for idx in range(max(n, 1)):
                combo.addItem(self._channel_label(idx), idx)
            combo.blockSignals(False)

        valid_prev = (
            prev_bf is not None
            and prev_tg is not None
            and prev_bf < max(n, 1)
            and prev_tg < max(n, 1)
            and (prev_bf != prev_tg or n <= 1)
        )
        bf, tg = (prev_bf, prev_tg) if valid_prev else ((0, 1) if n >= 2 else (0, 0))
        self._set_combo_channel(self.brightfield_channel, bf)
        self._set_combo_channel(self.target_channel, tg)
        self._emitting = False

    def is_auto_recompute(self) -> bool:
        return self.auto_recompute.isChecked()

    # ------------------------------------------------------------------

    def _emit_params(self, _value=None):
        if self._emitting:
            return
        self.params_changed.emit(self.params())

    def _on_channel_edited(self, _index):
        if self._emitting:
            return
        self.channels_changed.emit(self.channels())

    def _channel_label(self, idx: int) -> str:
        meta = self._channels_meta[idx] if idx < len(self._channels_meta) else {}
        name = meta.get("name")
        if name:
            return f"{idx}: {name}"
        wavelength = meta.get("emission_wavelength") or meta.get("excitation_wavelength")
        if wavelength:
            return f"{idx}: {float(wavelength):.0f}nm"
        return f"Channel {idx}"

    @staticmethod
    def _set_combo_channel(combo: QComboBox, channel_idx: int):
        idx = combo.findData(channel_idx)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
