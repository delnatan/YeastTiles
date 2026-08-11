"""Discovered-file list with per-file processing status and checkboxes for
batch "Process & Save" selection.

Status is derived from output-file existence/mtime (no separate tracking
DB needed) plus live results fed back from BatchProcessWorker.
"""

from pathlib import Path

from qtpy.QtCore import Qt, QSize, Signal
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core.pipeline import find_processed_output

_STATUS_COLORS = {
    "unprocessed": "#8c8c8c",
    "previewed": "#e6c229",
    "done": "#4caf50",
    "failed": "#ff5c5c",
}
_ICON_SIZE = QSize(12, 12)


def _status_icon(status: str) -> QIcon:
    pixmap = QPixmap(_ICON_SIZE)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor(_STATUS_COLORS.get(status, "#8c8c8c")))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(1, 1, _ICON_SIZE.width() - 2, _ICON_SIZE.height() - 2)
    painter.end()
    return QIcon(pixmap)


class FileListPanel(QWidget):
    file_selected = Signal(str)  # absolute path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._output_folder = None
        self._pattern = "*.ims"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QHBoxLayout()
        header.addWidget(QLabel("Files"))
        header.addStretch(1)
        select_all_btn = QPushButton("All")
        select_all_btn.setFixedWidth(40)
        select_all_btn.clicked.connect(lambda: self._set_all_checked(True))
        header.addWidget(select_all_btn)
        select_none_btn = QPushButton("None")
        select_none_btn.setFixedWidth(48)
        select_none_btn.clicked.connect(lambda: self._set_all_checked(False))
        header.addWidget(select_none_btn)
        layout.addLayout(header)

        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self._on_current_changed)
        layout.addWidget(self.list_widget, 1)

    # ------------------------------------------------------------------

    def set_files(self, paths: list[str]):
        self.list_widget.clear()
        for path in paths:
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.UserRole, str(path))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)
        self.refresh_statuses()
        if self.list_widget.count():
            self.list_widget.setCurrentRow(0)

    def set_output_folder(self, folder: str | None):
        self._output_folder = folder
        self.refresh_statuses()

    def current_path(self) -> str | None:
        item = self.list_widget.currentItem()
        return item.data(Qt.UserRole) if item else None

    def checked_paths(self) -> list[str]:
        checked = [
            self.list_widget.item(i).data(Qt.UserRole)
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).checkState() == Qt.Checked
        ]
        return checked or self._all_paths()

    def _all_paths(self) -> list[str]:
        return [
            self.list_widget.item(i).data(Qt.UserRole)
            for i in range(self.list_widget.count())
        ]

    def _set_all_checked(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)

    def _on_current_changed(self, current, _previous):
        if current is not None:
            self.file_selected.emit(current.data(Qt.UserRole))

    # ------------------------------------------------------------------
    # Status

    def is_processed(self, path: str) -> bool:
        status, _ = self._compute_status(Path(path))
        return status == "done"

    def refresh_statuses(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            path = Path(item.data(Qt.UserRole))
            status, tooltip = self._compute_status(path)
            item.setIcon(_status_icon(status))
            if tooltip:
                item.setToolTip(tooltip)

    def _compute_status(self, path: Path) -> tuple[str, str]:
        if not self._output_folder or not path.exists():
            return "unprocessed", ""
        if find_processed_output(path, self._output_folder) is not None:
            return "done", "Processed"
        return "unprocessed", ""

    def mark_previewed(self, path: str):
        item = self._find_item(path)
        if item is not None and item.toolTip() != "Processed":
            item.setIcon(_status_icon("previewed"))
            item.setToolTip("Diagnostics previewed (not saved)")

    def mark_result(self, path: str, success: bool, error: str | None = None):
        item = self._find_item(path)
        if item is None:
            return
        if success:
            item.setIcon(_status_icon("done"))
            item.setToolTip("Processed")
        else:
            item.setIcon(_status_icon("failed"))
            item.setToolTip(error or "Failed")

    def _find_item(self, path: str) -> QListWidgetItem | None:
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == str(path):
                return item
        return None
