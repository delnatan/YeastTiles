"""Simple per-channel color/blend-mode/opacity/contrast panel for the
fast thumbnail grid.

Deliberately minimal compared to ``ChannelPanel``/``ChannelRow``
(``widgets/channel_panel.py``): no histogram, no gamma, no colormap
picker -- those stay dialog-only for the full per-tile vispy viewer.
This is the *only* contrast/visibility control fast mode offers
(``show_channel_panel`` refuses to open the heavy panel in fast mode --
see ``TiledViewer.show_channel_panel``), so on top of what
``composite_to_rgb`` (``data/thumbnail_cache.py``) needs -- flat color,
additive-vs-overlay blend mode, opacity -- it also exposes a plain
min/max contrast range and a visibility toggle per channel, backed by
``viewer.visual_proxy`` (the same ``TiledVisualProxy`` the old panel
used). Min/max is deliberately *not* percentile-based: label/mask
images (binary segmentation, small-integer label maps) have almost no
dynamic range, and a percentile bracket over them can clip real label
values -- plain min/max never does. Pure Qt -- no vispy, colors come
from ``QColorDialog``, not a named colormap registry.
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

from ..data.overlay_state import ADDITIVE, OVERLAY


class _ColorRow(QWidget):
    """One row: visibility, swatch, Additive/Overlay radios, min/max,
    opacity slider."""

    colorChanged = Signal(int, str)  # channel_idx, hex
    blendModeChanged = Signal(int, str)  # channel_idx, mode
    opacityChanged = Signal(int, float)  # channel_idx, opacity
    visibleChanged = Signal(int, bool)  # channel_idx, visible
    climChanged = Signal(int, float, float)  # channel_idx, vmin, vmax

    def __init__(self, channel_idx, overlay_state, clim, visible, parent=None):
        super().__init__(parent)
        self.channel_idx = channel_idx
        self._color_hex = overlay_state.color_hex

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

        self.set_state(overlay_state, clim, visible)

    def set_state(self, overlay_state, clim, visible):
        """Reflect *overlay_state* (an OverlayChannelState), *clim*
        (vmin, vmax), and *visible* without re-emitting signals for
        this programmatic update."""
        self._color_hex = overlay_state.color_hex
        self.swatch.setStyleSheet(
            f"background-color: {overlay_state.color_hex}; border: 1px solid #555;"
        )
        radio = (
            self.overlay_radio
            if overlay_state.blend_mode == OVERLAY
            else self.additive_radio
        )
        radio.blockSignals(True)
        radio.setChecked(True)
        radio.blockSignals(False)
        pct = round(overlay_state.opacity * 100)
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
    """Per-channel color/blend-mode/opacity/contrast/visibility control
    for a TiledViewer running in fast mode. Color/blend/opacity read
    from ``viewer.overlay_state``; contrast/visibility read from
    ``viewer.visual_proxy`` (shared with the general per-tile viewer,
    but this is the only UI that reaches it in fast mode)."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.overlay_state = viewer.overlay_state
        self.proxy = viewer.visual_proxy

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
        self._unsubscribe = self.overlay_state.subscribe(self._on_state_changed)
        self.refresh_ui()

    def _suggested_state(self, c):
        """(clim, visible) for channel *c*: the user's explicit override
        if they've set one via this panel, otherwise a plain min/max
        suggestion sampled across all currently-decoded tiles."""
        visible = self.proxy.get_channel_visible(c)
        if c in self.proxy.custom_clim:
            return self.proxy.display[c].clim, visible
        data = self.viewer.thumbnail_grid.aggregate_channel_data(c)
        if data is not None and data.size > 0:
            return (float(data.min()), float(data.max())), visible
        return self.proxy.display[c].clim, visible

    def refresh_ui(self):
        """Rebuild rows if the channel count changed, else sync values."""
        if len(self.rows) != len(self.overlay_state):
            while self.rows_layout.count():
                item = self.rows_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            self.rows = []
            for c in range(len(self.overlay_state)):
                clim, visible = self._suggested_state(c)
                row = _ColorRow(c, self.overlay_state[c], clim, visible)
                row.colorChanged.connect(self._on_color_changed)
                row.blendModeChanged.connect(self._on_blend_mode_changed)
                row.opacityChanged.connect(self._on_opacity_changed)
                row.visibleChanged.connect(self._on_visible_changed)
                row.climChanged.connect(self._on_clim_changed)
                self.rows.append(row)
                self.rows_layout.addWidget(row)
            self.rows_layout.addStretch()
        else:
            for c, row in enumerate(self.rows):
                clim, visible = self._suggested_state(c)
                row.set_state(self.overlay_state[c], clim, visible)

    # User input -> overlay_state / visual_proxy.

    def _on_color_changed(self, idx, hex_color):
        self.overlay_state.set_color(idx, hex_color)

    def _on_blend_mode_changed(self, idx, mode):
        self.overlay_state.set_blend_mode(idx, mode)

    def _on_opacity_changed(self, idx, opacity):
        self.overlay_state.set_opacity(idx, opacity)

    def _on_visible_changed(self, idx, visible):
        self.proxy.set_channel_visible(idx, visible)

    def _on_clim_changed(self, idx, vmin, vmax):
        self.proxy.set_clim(idx, vmin, vmax)

    # overlay_state -> UI (external mutation, e.g. loaded from the
    # annotation sidecar file on folder open). visual_proxy changes
    # (new tiles decoded, channel count grows, Auto Contrast All) reach
    # this panel via explicit refresh_ui() calls from TiledViewer
    # instead of a subscription -- update_max_channels replaces
    # visual_proxy.display wholesale on a channel-count change, which
    # would otherwise leave a display.subscribe() callback bound to a
    # stale, abandoned list.

    def _on_state_changed(self, channel_idx, field):
        self.refresh_ui()

    def closeEvent(self, event):
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        super().closeEvent(event)
