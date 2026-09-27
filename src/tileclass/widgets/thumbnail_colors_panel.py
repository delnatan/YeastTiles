"""Per-channel color/blend-mode/opacity/contrast/visibility panel for the
thumbnail grid, editing the viewer's ``ChannelStateList``
(``data/channel_state.py``).

Min/max is a plain range rather than percentile-based: label/mask images
(binary segmentation, small-integer label maps) have almost no dynamic
range, and a percentile bracket over them can clip real label values --
plain min/max never does.
"""

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..data.channel_state import ADDITIVE, OVERLAY


class _ColorRow(QWidget):
    """One row: visibility, swatch, Additive/Overlay radios, min/max,
    opacity slider."""

    colorChanged = Signal(int, str)  # channel_idx, hex
    blendModeChanged = Signal(int, str)  # channel_idx, mode
    opacityChanged = Signal(int, float)  # channel_idx, opacity
    visibleChanged = Signal(int, bool)  # channel_idx, visible
    climChanged = Signal(int, float, float)  # channel_idx, vmin, vmax

    def __init__(self, channel_idx, state, clim, parent=None):
        super().__init__(parent)
        self.channel_idx = channel_idx
        self._color_hex = state.color_hex

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(2)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)

        self.visible_check = QCheckBox()
        self.visible_check.setToolTip("Toggle channel visibility")
        self.visible_check.toggled.connect(
            lambda checked: self.visibleChanged.emit(self.channel_idx, checked)
        )
        top.addWidget(self.visible_check)

        top.addWidget(QLabel(f"Ch {channel_idx + 1}"))

        self.swatch = QPushButton()
        self.swatch.setFixedSize(24, 24)
        self.swatch.clicked.connect(self._pick_color)
        top.addWidget(self.swatch)

        self.additive_radio = QRadioButton("Additive")
        self.overlay_radio = QRadioButton("Overlay")
        group = QButtonGroup(self)
        group.addButton(self.additive_radio)
        group.addButton(self.overlay_radio)
        self.additive_radio.toggled.connect(self._on_mode_toggled)
        top.addWidget(self.additive_radio)
        top.addWidget(self.overlay_radio)
        top.addStretch()
        outer.addLayout(top)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)

        bottom.addWidget(QLabel("Min:"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setDecimals(1)
        self.min_spin.setRange(-1e9, 1e9)
        self.min_spin.setSingleStep(1.0)
        self.min_spin.setKeyboardTracking(False)
        self.min_spin.setFixedWidth(70)
        self.min_spin.setToolTip("Minimum intensity")
        self.min_spin.valueChanged.connect(self._on_min_changed)
        bottom.addWidget(self.min_spin)

        bottom.addWidget(QLabel("Max:"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setDecimals(1)
        self.max_spin.setRange(-1e9, 1e9)
        self.max_spin.setSingleStep(1.0)
        self.max_spin.setKeyboardTracking(False)
        self.max_spin.setFixedWidth(70)
        self.max_spin.setToolTip("Maximum intensity")
        self.max_spin.valueChanged.connect(self._on_max_changed)
        bottom.addWidget(self.max_spin)

        bottom.addSpacing(12)
        bottom.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setFixedWidth(90)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        bottom.addWidget(self.opacity_slider)
        self.opacity_label = QLabel("100%")
        self.opacity_label.setFixedWidth(35)
        bottom.addWidget(self.opacity_label)
        bottom.addStretch()
        outer.addLayout(bottom)

        self.set_state(state, clim)

    def set_state(self, state, clim):
        """Reflect *state* (a ChannelState) and the displayed *clim*
        (vmin, vmax) without re-emitting signals for this programmatic
        update."""
        visible = state.visible
        self._color_hex = state.color_hex
        self.swatch.setStyleSheet(
            f"background-color: {state.color_hex}; border: 1px solid #555;"
        )
        radio = (
            self.overlay_radio
            if state.blend_mode == OVERLAY
            else self.additive_radio
        )
        radio.blockSignals(True)
        radio.setChecked(True)
        radio.blockSignals(False)
        pct = round(state.opacity * 100)
        self.opacity_slider.blockSignals(True)
        self.opacity_slider.setValue(pct)
        self.opacity_slider.blockSignals(False)
        self.opacity_label.setText(f"{pct}%")

        self.visible_check.blockSignals(True)
        self.visible_check.setChecked(visible)
        self.visible_check.blockSignals(False)

        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.min_spin.setValue(clim[0])
        self.max_spin.setValue(clim[1])
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)

    def _pick_color(self):
        from qtpy.QtGui import QColor

        chosen = QColorDialog.getColor(QColor(self._color_hex), self, "Channel Color")
        if chosen.isValid():
            hex_color = chosen.name()
            self._color_hex = hex_color
            self.swatch.setStyleSheet(
                f"background-color: {hex_color}; border: 1px solid #555;"
            )
            self.colorChanged.emit(self.channel_idx, hex_color)

    def _on_mode_toggled(self, additive_checked):
        mode = ADDITIVE if additive_checked else OVERLAY
        self.blendModeChanged.emit(self.channel_idx, mode)

    def _on_opacity_changed(self, value):
        self.opacity_label.setText(f"{value}%")
        self.opacityChanged.emit(self.channel_idx, value / 100.0)

    def _on_min_changed(self, value):
        max_val = self.max_spin.value()
        if value < max_val:
            self.climChanged.emit(self.channel_idx, value, max_val)

    def _on_max_changed(self, value):
        min_val = self.min_spin.value()
        if value > min_val:
            self.climChanged.emit(self.channel_idx, min_val, value)


class ThumbnailColorsPanel(QWidget):
    """One `_ColorRow` per channel of `viewer.channel_state`."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.states = viewer.channel_state

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        note = QLabel(
            "<i>Additive brightens on top of other channels (default). "
            "Overlay tints translucently over the composite -- handy for "
            "checking a segmentation mask against the image beneath it. "
            "Min/Max is a plain range (no percentile clipping), so label "
            "or binary masks with very little dynamic range still show "
            "up correctly.</i>"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(note)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.rows_layout = QVBoxLayout(scroll_content)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(2)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        self.rows = []
        self._unsubscribe = self.states.subscribe(lambda _idx: self.refresh_ui())
        self.refresh_ui()

    def _displayed_clim(self, c):
        """The channel's explicit contrast override if set, otherwise a
        plain min/max suggestion sampled across the decoded tiles."""
        clim = self.states[c].clim
        if clim is not None:
            return clim
        data = self.viewer.thumbnail_grid.aggregate_channel_data(c)
        if data is not None and data.size > 0:
            return (float(data.min()), float(data.max()))
        return (0.0, 1.0)

    def refresh_ui(self):
        """Rebuild rows if the channel count changed, else sync values."""
        if len(self.rows) != len(self.states):
            while self.rows_layout.count():
                item = self.rows_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self.rows = []
            for c in range(len(self.states)):
                row = _ColorRow(c, self.states[c], self._displayed_clim(c))
                row.colorChanged.connect(lambda i, v: self.states.set(i, color_hex=v))
                row.blendModeChanged.connect(lambda i, v: self.states.set(i, blend_mode=v))
                row.opacityChanged.connect(lambda i, v: self.states.set(i, opacity=v))
                row.visibleChanged.connect(lambda i, v: self.states.set(i, visible=v))
                row.climChanged.connect(
                    lambda i, vmin, vmax: self.states.set(i, clim=(vmin, vmax))
                )
                self.rows.append(row)
                self.rows_layout.addWidget(row)
            self.rows_layout.addStretch()
        else:
            for c, row in enumerate(self.rows):
                row.set_state(self.states[c], self._displayed_clim(c))

    def closeEvent(self, event):
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        super().closeEvent(event)
