"""Read-only per-category population stats for a folder's tile annotations."""

from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QShortcut,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class AnnotationStatsPanel(QWidget):
    """Category -> count / percentage table. Call `refresh()` whenever the
    underlying annotations (or the dataset size) may have changed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Category", "Count", "%"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        # Selectable (not the prior NoSelection) so a user can grab just
        # the rows they want; copy-all button covers the common case.
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        layout.addWidget(self.table)

        QShortcut(QKeySequence.Copy, self.table, self._copy_selection)

        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(self._copy_all)
        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(copy_button)
        layout.addLayout(button_row)

    def _table_text(self, rows):
        """Tab-separated text (header + rows) for the given row indices --
        pastes cleanly into a spreadsheet or text editor."""
        headers = [
            self.table.horizontalHeaderItem(c).text()
            for c in range(self.table.columnCount())
        ]
        lines = ["\t".join(headers)]
        for r in rows:
            cells = [
                self.table.item(r, c).text()
                for c in range(self.table.columnCount())
            ]
            lines.append("\t".join(cells))
        return "\n".join(lines)

    def _copy_all(self):
        QApplication.clipboard().setText(
            self._table_text(range(self.table.rowCount()))
        )

    def _copy_selection(self):
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        if not rows:
            self._copy_all()
            return
        QApplication.clipboard().setText(self._table_text(rows))

    def refresh(self, annotations, total_count):
        """Rebuild the table from `annotations` (a TileAnnotations) against
        `total_count` images total (un-annotated = total_count - tagged).
        Percentages are of the tagged (classified) total, not of
        `total_count` — the "Un-annotated" row has no percentage."""
        counts = {}
        for category in annotations.values():
            counts[category] = counts.get(category, 0) + 1
        tagged_total = sum(counts.values())
        untagged = max(total_count - tagged_total, 0)

        vocabulary = annotations.categories()
        rows = [("Un-annotated", untagged, False)]
        rows += [(name, counts.get(name, 0), True) for name in vocabulary]
        # Categories still in use but no longer in the vocabulary (e.g.
        # removed via Manage Categories) still deserve a row.
        rows += [
            (name, count, True)
            for name, count in counts.items()
            if name not in vocabulary
        ]

        self.table.setRowCount(len(rows))
        for r, (name, count, has_pct) in enumerate(rows):
            self.table.setItem(r, 0, QTableWidgetItem(name))

            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 1, count_item)

            if has_pct:
                pct = (count / tagged_total * 100) if tagged_total else 0.0
                pct_text = f"{pct:.1f}%"
            else:
                pct_text = "–"
            pct_item = QTableWidgetItem(pct_text)
            pct_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, 2, pct_item)
