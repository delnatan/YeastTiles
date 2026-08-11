"""Per-channel visibility/colormap/contrast controls for the raw-stack
canvas. Vendored (simplified) from pyvistra/widgets/channel_panel.py's
ChannelRow: no histogram widget or colormap-picker popup, just a checkbox,
a colormap dropdown, and min/max spinboxes -- enough for a "decent"
visual check, not a full contrast-editing suite.
"""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from . import colormaps as _colormaps
from .channel_state import ChannelDisplayList


class ChannelRow(QWidget):
    """One row per channel, bound directly to a shared ChannelDisplayList."""

    def __init__(
        self,
        channel_idx: int,
        channel_name: str,
        display: ChannelDisplayList,
        clim_bounds: tuple[float, float] = (0.0, 65535.0),
        parent=None,
    ):
        super().__init__(parent)
        self.channel_idx = channel_idx
        self.display = display

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)

        top = QHBoxLayout()
        self.chk_visible = QCheckBox()
        self.chk_visible.setChecked(display[channel_idx].visible)
        self.chk_visible.toggled.connect(
            lambda v: display.set_visible(channel_idx, v)
        )
        top.addWidget(self.chk_visible)

        self.name_label = QLabel(channel_name)
        self.name_label.setToolTip(channel_name)
        top.addWidget(self.name_label, 1)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(_colormaps.names())
        self.cmap_combo.setCurrentText(display[channel_idx].colormap_name)
        self.cmap_combo.currentTextChanged.connect(
            lambda name: display.set_colormap_name(channel_idx, name)
        )
        top.addWidget(self.cmap_combo)
        layout.addLayout(top)

        bottom = QHBoxLayout()
        bottom.addWidget(QLabel("min"))
        lo, hi = clim_bounds
        vmin, vmax = display[channel_idx].clim
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(lo, hi)
        self.min_spin.setValue(vmin)
        self.min_spin.valueChanged.connect(self._on_clim_edited)
        bottom.addWidget(self.min_spin)

        bottom.addWidget(QLabel("max"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(lo, hi)
        self.max_spin.setValue(vmax)
        self.max_spin.valueChanged.connect(self._on_clim_edited)
        bottom.addWidget(self.max_spin)
        layout.addLayout(bottom)

        self._unsubscribe = display.subscribe(self._on_display_changed)

    def _on_clim_edited(self, _value):
        self.display.set_clim(
            self.channel_idx, self.min_spin.value(), self.max_spin.value()
        )

    def _on_display_changed(self, idx, field):
        if idx != self.channel_idx:
            return
        state = self.display[idx]
        if field == "clim":
            self.min_spin.blockSignals(True)
            self.max_spin.blockSignals(True)
            self.min_spin.setValue(state.clim[0])
            self.max_spin.setValue(state.clim[1])
            self.min_spin.blockSignals(False)
            self.max_spin.blockSignals(False)
        elif field == "colormap_name":
            self.cmap_combo.blockSignals(True)
            self.cmap_combo.setCurrentText(state.colormap_name)
            self.cmap_combo.blockSignals(False)
        elif field == "visible":
            self.chk_visible.blockSignals(True)
            self.chk_visible.setChecked(state.visible)
            self.chk_visible.blockSignals(False)

    def disconnect_from_display(self):
        """Call explicitly before discarding this row (e.g. when rebuilding
        the row list for a newly loaded file) -- closeEvent doesn't fire
        reliably for a widget that's only ever removed from a layout."""
        self._unsubscribe()
