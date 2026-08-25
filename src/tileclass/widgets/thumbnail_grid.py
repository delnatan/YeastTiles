"""Fast, pure-Qt thumbnail grid for browsing many small images.

Renders the whole grid as a single widget with a custom ``paintEvent``
instead of one canvas per tile -- no GPU context. Decoding happens on a
background thread (see ``data/thumbnail_cache.py``); compositing
(clim/gamma -> RGB) happens lazily on the GUI thread from already-decoded
planes, so it's numpy-only work, never disk I/O, and safe to do inline in
``paintEvent``.
"""

from pathlib import Path

import numpy as np
from qtpy.QtCore import QRect, Qt, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from qtpy.QtWidgets import QMenu, QWidget

from ..data.overlay_state import default_channel_state
from ..data.thumbnail_cache import (
    DecodeCache,
    ThumbnailDecodeWorker,
    composite_to_rgb,
)


def _readable_text_color(hex_color):
    """Pick black or white text for readability against hex_color."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "black" if luminance > 0.6 else "white"


def _hex_to_rgb01(hex_color):
    """``"#rrggbb"`` -> ``(r, g, b)`` floats in ``[0, 1]``."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b)


class ThumbnailGridWidget(QWidget):
    """One widget draws the whole grid as cached ``QPixmap``\\ s.

    Owns a background ``DecodeCache``/``ThumbnailDecodeWorker`` keyed by
    path. ``set_items`` takes the *entire* folder's path list --
    pagination/paging elsewhere is expressed via ``set_priority`` (which
    paths to decode first), not by handing the widget a smaller list, so
    revisiting an earlier page is always a cache hit.
    """

    selectionChanged = Signal(list)  # list[str] of selected paths
    annotateRequested = Signal()  # act on current selection
    openInViewerRequested = Signal(str)  # path
    showInfoRequested = Signal(str)  # path
    decoded = Signal(str)  # internal: worker thread -> GUI thread relay

    LABEL_HEIGHT = 28
    SPACING = 6

    def __init__(self, cache_bytes=256 * 1024 * 1024, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        self._paths = []
        self._tile_size = 200
        self._show_info = True
        self._cols = 1

        self._selected = []  # list[str], insertion order
        self._selected_set = set()
        self._anchor = None

        self._annotations = {}  # path -> (label, color_hex, confidence)

        self._display = None  # ChannelDisplayList (global settings)
        self._custom_clim = set()  # live ref to VisualState._custom_clim
        self._overlay_state = None  # OverlayStateList (color/blend/opacity)
        self._overlay_unsubscribe = None

        self._pixmap_cache = {}  # path -> QPixmap

        self._cache = DecodeCache(max_bytes=cache_bytes)
        self._worker = ThumbnailDecodeWorker(self._cache, self._deliver_decoded)
        self.decoded.connect(self._on_decoded, Qt.QueuedConnection)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_items(self, paths, dims=None):
        self._paths = list(paths)
        self._selected = []
        self._selected_set.clear()
        self._anchor = None
        self._worker.set_dims(dims)
        self._relayout()
        self.update()

    def set_order(self, paths):
        """Reorder the currently displayed paths (same set as before) --
        selection, decode cache, and worker priority are untouched."""
        self._paths = list(paths)
        self._relayout()
        self.update()

    @property
    def paths(self):
        return list(self._paths)

    def set_priority(self, paths_in_priority_order):
        """Reprioritize background decoding, e.g. current page first."""
        self._worker.set_priority(paths_in_priority_order)

    def set_tile_size(self, px):
        self._tile_size = px
        self._pixmap_cache.clear()
        self._relayout()
        self.update()

    def set_show_info(self, show):
        self._show_info = show
        self._relayout()
        self.update()

    def set_channel_display(self, display, custom_clim):
        """``display``: a ``ChannelDisplayList``. ``custom_clim``: the
        live ``set[int]`` of channel indices explicitly overridden via
        the Colors panel (the same object the owner mutates -- read
        live here, never copied).

        A no-op if both are already the current references -- callers
        (e.g. a per-decoded-image refresh) may call this far more often
        than the display actually changes, and clearing the pixmap
        cache on every call would force a full recomposite each time.
        """
        if display is self._display and custom_clim is self._custom_clim:
            return
        self._display = display
        self._custom_clim = custom_clim
        self._pixmap_cache.clear()
        self.update()

    def set_overlay_state(self, overlay_state):
        """``overlay_state``: an ``OverlayStateList`` (per-channel color /
        blend-mode / opacity, see ``data/overlay_state.py``). Stored as a
        live reference and subscribed to once -- unlike ``display``, this
        list is resized/mutated in place rather than replaced, so this
        only needs calling once."""
        if overlay_state is self._overlay_state:
            return
        self._overlay_state = overlay_state
        self._overlay_unsubscribe = overlay_state.subscribe(
            self._on_overlay_state_changed
        )
        self.invalidate_pixmaps()

    def _on_overlay_state_changed(self, channel_idx, field):
        self.invalidate_pixmaps()

    def set_annotations(self, annotations):
        """``annotations``: ``{path: (label, color_hex, confidence)}``,
        ``confidence`` a float in [0, 1] for an unreviewed AI prediction,
        or ``None`` for a manually-set (or already-reviewed) tag."""
        self._annotations = annotations
        self.update()

    def invalidate_pixmaps(self):
        """Drop cached composited pixmaps so the next paint recomposites
        from already-decoded raw planes (no re-decode, no disk I/O)."""
        self._pixmap_cache.clear()
        self.update()

    @property
    def selected_paths(self):
        return list(self._selected)

    def max_channels(self):
        """Highest channel count among currently-decoded images."""
        best = 1
        for path in self._paths:
            entry = self._cache.get(path)
            if entry is not None:
                best = max(best, entry[0].shape[0])
        return best

    def aggregate_channel_data(self, channel_idx, max_samples_per_tile=2000):
        """Sampled raw intensity values for *channel_idx* across every
        currently-decoded tile -- lets the Colors panel suggest a plain
        min/max contrast range."""
        samples = []
        for path in self._paths:
            entry = self._cache.get(path)
            if entry is None:
                continue
            plane = entry[0]
            if channel_idx >= plane.shape[0]:
                continue
            data = plane[channel_idx]
            total = data.size
            if total > max_samples_per_tile:
                step = max(1, int(np.sqrt(total / max_samples_per_tile)))
                samples.append(data[::step, ::step].ravel())
            else:
                samples.append(data.ravel())
        if samples:
            return np.concatenate(samples)
        return None

    def full_resolution_rgb(self, path):
        """Composited ``(H, W, 3)`` uint8 array at full resolution (no
        tile-size downscale) for *path*, or ``None`` if not yet decoded.
        For callers that want to inspect a tile beyond its thumbnail
        size (e.g. a zoom popup)."""
        entry = self._cache.get(path)
        if entry is None:
            return None
        plane, default_clims = entry
        params = self._channel_params(default_clims)
        return composite_to_rgb(plane, params)

    def close(self):
        self._worker.close()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _cell_size(self):
        h = self._tile_size + (self.LABEL_HEIGHT if self._show_info else 0)
        return self._tile_size, h

    def _relayout(self):
        cell_w, cell_h = self._cell_size()
        avail = max(cell_w, self.width())
        cols = max(1, (avail + self.SPACING) // (cell_w + self.SPACING))
        self._cols = int(cols)
        rows = (len(self._paths) + self._cols - 1) // self._cols if self._paths else 0
        total_h = rows * (cell_h + self.SPACING) + self.SPACING
        self.setMinimumHeight(int(total_h))

    def resizeEvent(self, event):
        self._relayout()
        super().resizeEvent(event)

    def _cell_rect(self, index):
        cell_w, cell_h = self._cell_size()
        col = index % self._cols
        row = index // self._cols
        x = self.SPACING + col * (cell_w + self.SPACING)
        y = self.SPACING + row * (cell_h + self.SPACING)
        return QRect(x, y, cell_w, cell_h)

    def _index_at(self, pos):
        cell_w, cell_h = self._cell_size()
        col = (pos.x() - self.SPACING) // (cell_w + self.SPACING)
        row = (pos.y() - self.SPACING) // (cell_h + self.SPACING)
        if col < 0 or row < 0 or col >= self._cols:
            return None
        index = int(row) * self._cols + int(col)
        if index >= len(self._paths):
            return None
        if not self._cell_rect(index).contains(pos):
            return None
        return index

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(event.rect(), QColor("#1a1a1a"))

        if self._paths:
            cell_h = self._cell_size()[1]
            row_stride = cell_h + self.SPACING
            first_row = max(0, event.rect().top() // row_stride - 1)
            last_row = event.rect().bottom() // row_stride + 1
            first_index = int(first_row) * self._cols
            last_index = min(len(self._paths), (int(last_row) + 1) * self._cols)
            for index in range(max(0, first_index), last_index):
                self._paint_cell(painter, index)

        painter.end()

    def _paint_cell(self, painter, index):
        path = self._paths[index]
        rect = self._cell_rect(index)
        image_rect = QRect(rect.x(), rect.y(), self._tile_size, self._tile_size)

        pixmap = self._get_pixmap(path)
        painter.fillRect(image_rect, QColor("#111111"))
        if pixmap is not None:
            x = image_rect.x() + (image_rect.width() - pixmap.width()) // 2
            y = image_rect.y() + (image_rect.height() - pixmap.height()) // 2
            painter.drawPixmap(x, y, pixmap)

        selected = path in self._selected_set
        pen = QPen(QColor("#4da6ff") if selected else QColor("#333333"))
        pen.setWidth(2 if selected else 1)
        painter.setPen(pen)
        # drawRect fills with the *current brush*, not just strokes the
        # outline -- without this, a leftover brush from the previous
        # cell's annotation badge (see _paint_badge) paints this cell's
        # entire border rect solid, hiding the pixmap underneath.
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(image_rect.adjusted(0, 0, -1, -1))

        annotation = self._annotations.get(path)
        if annotation:
            label, color_hex, confidence = annotation
            self._paint_badge(painter, image_rect, label, color_hex, confidence)

        if self._show_info:
            painter.setPen(QColor("#aaaaaa"))
            name = Path(path).stem
            text_rect = QRect(
                rect.x(),
                image_rect.bottom() + 2,
                rect.width(),
                self.LABEL_HEIGHT - 2,
            )
            metrics = painter.fontMetrics()
            elided = metrics.elidedText(name, Qt.ElideMiddle, text_rect.width())
            painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignTop, elided)

    def _paint_badge(self, painter, image_rect, label, color_hex, confidence=None):
        """confidence: None for a manually-set (or already-reviewed) tag;
        a [0, 1] float for an unreviewed AI prediction -- shown appended
        to the label and with a dashed outline, so unreviewed tiles are
        visually distinct from confirmed ones at a glance."""
        # save/restore: this sets a solid brush for the chip fill, which
        # must not leak into whatever the caller draws next.
        painter.save()
        text = label[:4]
        if confidence is not None:
            text += f" {round(confidence * 100)}"
        color = QColor(color_hex or "#666666")
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        metrics = painter.fontMetrics()
        w = metrics.horizontalAdvance(text) + 8
        h = metrics.height() + 2
        badge_rect = QRect(image_rect.x() + 4, image_rect.y() + 4, w, h)
        painter.drawRoundedRect(badge_rect, 3, 3)
        painter.setPen(QColor(_readable_text_color(color.name())))
        painter.drawText(badge_rect, Qt.AlignCenter, text)
        if confidence is not None:
            pen = QPen(QColor("#ffffff"))
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(badge_rect, 3, 3)
        painter.restore()

    # ------------------------------------------------------------------
    # Compositing (numpy work only -- decode already happened async)
    # ------------------------------------------------------------------

    def _get_pixmap(self, path):
        pixmap = self._pixmap_cache.get(path)
        if pixmap is not None:
            return pixmap
        entry = self._cache.get(path)
        if entry is None:
            return None
        plane, default_clims = entry
        params = self._channel_params(default_clims)
        rgb = composite_to_rgb(plane, params)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg).scaled(
            self._tile_size,
            self._tile_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self._pixmap_cache[path] = pixmap
        return pixmap

    def _channel_params(self, default_clims):
        params = []
        for c, default_clim in enumerate(default_clims):
            if self._display is not None and c < len(self._display):
                state = self._display[c]
                clim = state.clim if c in self._custom_clim else default_clim
                gamma = state.gamma
                visible = state.visible
            else:
                clim = default_clim
                gamma = 1.0
                visible = True

            if self._overlay_state is not None and c < len(self._overlay_state):
                ov = self._overlay_state[c]
                color = _hex_to_rgb01(ov.color_hex)
                blend_mode = ov.blend_mode
                opacity = ov.opacity
            else:
                default_state = default_channel_state(c)
                color = _hex_to_rgb01(default_state.color_hex)
                blend_mode = default_state.blend_mode
                opacity = default_state.opacity

            params.append((clim, gamma, color, visible, blend_mode, opacity))
        return params

    # ------------------------------------------------------------------
    # Decode delivery (worker thread -> GUI thread)
    # ------------------------------------------------------------------

    def _deliver_decoded(self, path):
        """Fires on the worker thread -- relay via a queued signal."""
        self.decoded.emit(path)

    def _on_decoded(self, path):
        self._pixmap_cache.pop(path, None)
        self.update()

    # ------------------------------------------------------------------
    # Mouse / selection
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        index = self._index_at(event.pos())
        if index is not None:
            path = self._paths[index]
            mods = event.modifiers()
            ctrl_or_cmd = bool(mods & (Qt.ControlModifier | Qt.MetaModifier))
            shift = bool(mods & Qt.ShiftModifier)
            self._select(path, ctrl_or_cmd, shift)
        super().mousePressEvent(event)

    def _select(self, path, ctrl_or_cmd, shift):
        if shift and self._anchor is not None and self._anchor in self._paths:
            self._select_range(self._anchor, path)
            return
        if ctrl_or_cmd:
            if path in self._selected_set:
                self._selected.remove(path)
                self._selected_set.discard(path)
            else:
                self._selected.append(path)
                self._selected_set.add(path)
            self._anchor = path
        else:
            self._selected = [path]
            self._selected_set = {path}
            self._anchor = path
        self.update()
        self.selectionChanged.emit(list(self._selected))

    def _select_range(self, anchor, path):
        try:
            lo = self._paths.index(anchor)
            hi = self._paths.index(path)
        except ValueError:
            self._selected = [path]
            self._selected_set = {path}
            self.update()
            self.selectionChanged.emit(list(self._selected))
            return
        if lo > hi:
            lo, hi = hi, lo
        self._selected = list(self._paths[lo : hi + 1])
        self._selected_set = set(self._selected)
        self.update()
        self.selectionChanged.emit(list(self._selected))

    def clear_selection(self):
        self._selected = []
        self._selected_set.clear()
        self.update()
        self.selectionChanged.emit([])

    # ------------------------------------------------------------------
    # Context menu
    # ------------------------------------------------------------------

    def _show_context_menu(self, pos):
        index = self._index_at(pos)
        if index is None:
            return
        path = self._paths[index]
        if path not in self._selected_set:
            self._select(path, ctrl_or_cmd=False, shift=False)

        menu = QMenu(self)
        annotate_action = menu.addAction("Annotate...")
        annotate_action.triggered.connect(self.annotateRequested.emit)
        menu.addSeparator()
        open_action = menu.addAction("Open in Viewer")
        open_action.triggered.connect(
            lambda: self.openInViewerRequested.emit(path)
        )
        info_action = menu.addAction("Show Info")
        info_action.triggered.connect(lambda: self.showInfoRequested.emit(path))
        menu.exec_(self.mapToGlobal(pos))
