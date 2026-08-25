"""Per-channel visibility/colormap/contrast controls for the raw-stack
canvas. Vendored (simplified) from pyvistra/widgets/channel_panel.py's
ChannelRow: a checkbox, a colormap dropdown, and min/max spinboxes flanking
pyvistra's own `CompactHistogramWidget` (imported directly, not
re-vendored -- it's already a self-contained QWidget with no dependency on
pyvistra's viewer/renderer classes, just numpy + Qt) -- enough for a
"decent" visual check, not a full contrast-editing suite (no gamma, no
colormap-picker popup).
"""

from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from pyvistra.widgets.histogram import CompactHistogramWidget

from . import colormaps as _colormaps
from .channel_state import ChannelDisplayList

_FALLBACK_HISTOGRAM_COLOR = "#d0d0d0"  # for colormaps with no single display color (e.g. "gray")


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

        lo, hi = clim_bounds
        vmin, vmax = display[channel_idx].clim

        # Its own row rather than sandwiched between the min/max spinboxes --
        # squeezed onto one row with two labels and two spinboxes, the
        # histogram was only ever getting a sliver of this column's width to
        # actually render a distribution in.
        self.histogram = CompactHistogramWidget()
        self.histogram.set_clim(vmin, vmax)
        self.histogram.climChanged.connect(self._on_histogram_clim_changed)
        layout.addWidget(self.histogram, 1)

        clim_row = QHBoxLayout()
        clim_row.addWidget(QLabel("min"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(lo, hi)
        self.min_spin.setValue(vmin)
        self.min_spin.valueChanged.connect(self._on_clim_edited)
        clim_row.addWidget(self.min_spin)

        clim_row.addStretch(1)

        clim_row.addWidget(QLabel("max"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(lo, hi)
        self.max_spin.setValue(vmax)
        self.max_spin.valueChanged.connect(self._on_clim_edited)
        clim_row.addWidget(self.max_spin)
        layout.addLayout(clim_row)

        self._unsubscribe = display.subscribe(self._on_display_changed)

    def set_data(self, data_slice):
        """Feed the histogram this channel's currently displayed plane --
        called by the owning viewer (e.g. RawStackViewerPanel) on every
        slice change, since the histogram has no way to pull that data
        itself."""
        color = self.display[self.channel_idx].display_color() or _FALLBACK_HISTOGRAM_COLOR
        self.histogram.set_data(data_slice, color)

    def _on_clim_edited(self, _value):
        self.display.set_clim(
            self.channel_idx, self.min_spin.value(), self.max_spin.value()
        )

    def _on_histogram_clim_changed(self, vmin, vmax):
        self.display.set_clim(self.channel_idx, vmin, vmax)

    def _on_display_changed(self, idx, field):
        if idx != self.channel_idx:
            return
        state = self.display[idx]
        if field == "clim":
            self.min_spin.blockSignals(True)
            self.max_spin.blockSignals(True)
            self.histogram.blockSignals(True)
            self.min_spin.setValue(state.clim[0])
            self.max_spin.setValue(state.clim[1])
            self.histogram.set_clim(*state.clim)
            self.min_spin.blockSignals(False)
            self.max_spin.blockSignals(False)
            self.histogram.blockSignals(False)
        elif field == "colormap_name":
            self.cmap_combo.blockSignals(True)
            self.cmap_combo.setCurrentText(state.colormap_name)
            self.cmap_combo.blockSignals(False)
            self.histogram.color = QColor(state.display_color() or _FALLBACK_HISTOGRAM_COLOR)
            self.histogram.update()
        elif field == "visible":
            self.chk_visible.blockSignals(True)
            self.chk_visible.setChecked(state.visible)
            self.chk_visible.blockSignals(False)

    def disconnect_from_display(self):
        """Call explicitly before discarding this row (e.g. when rebuilding
        the row list for a newly loaded file) -- closeEvent doesn't fire
        reliably for a widget that's only ever removed from a layout."""
        self._unsubscribe()
