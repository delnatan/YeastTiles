"""Shared per-file status dot (unprocessed/previewed/done/failed), used by
`project_tree_panel.py`'s per-FOV rows. Pulled out of the old
`file_list_panel.py` so it isn't tied to a single QListWidget-based widget.
"""

from qtpy.QtCore import Qt, QSize
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap

STATUS_COLORS = {
    "unprocessed": "#8c8c8c",
    "previewed": "#e6c229",
    "done": "#4caf50",
    "failed": "#ff5c5c",
    "stale": "#cc9900",
    # Only one of two possible channels denoised for this file -- see
    # ProjectConfig.denoise_channels_done. Same green as "done" since it's
    # not an error state, just incomplete; the half-fill carries the
    # "incomplete" signal instead of a different color.
    "partial": "#4caf50",
}
_ICON_SIZE = QSize(12, 12)


def status_icon(status: str) -> QIcon:
    pixmap = QPixmap(_ICON_SIZE)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    color = QColor(STATUS_COLORS.get(status, "#8c8c8c"))
    rect_x, rect_y = 1, 1
    rect_w, rect_h = _ICON_SIZE.width() - 2, _ICON_SIZE.height() - 2
    if status == "partial":
        # Left half filled in `color`, right half just an outline -- "one of
        # two channels denoised" at a glance, without a second legend color
        # to learn.
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawPie(rect_x, rect_y, rect_w, rect_h, 90 * 16, 180 * 16)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(color)
        painter.drawEllipse(rect_x, rect_y, rect_w, rect_h)
    else:
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(rect_x, rect_y, rect_w, rect_h)
    painter.end()
    return QIcon(pixmap)
