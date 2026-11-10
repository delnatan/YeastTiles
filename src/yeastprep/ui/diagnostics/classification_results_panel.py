"""Results tab of the Classify Tiles page: what classification produced --
tiles per category, one row per checked FOV -- and the buttons that save it.

Read from the annotation sidecars as they are now, not from the last
inference run: a tile's category is its human-set tag if it has one, else
the classifier's prediction. So a saved table includes every correction
made since.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from tileclass.data.annotations import TileAnnotations
from tileclass.tile_container import get_container

from ..common.lazy_refresh import LazyRefresh


@dataclass
class ResultsTable:
    categories: list = field(default_factory=list)
    # Per FOV, in `header()`'s column order.
    rows: list = field(default_factory=list)

    def header(self) -> list:
        return ["project", "fov", "tiles", *self.categories, "unclassified", "human_set"]

    def total(self) -> list:
        sums = [sum(column) for column in zip(*(row[2:] for row in self.rows))]
        return ["", f"All {len(self.rows)} FOV(s)", *sums]


def results_table(snapshot) -> ResultsTable:
    """`snapshot` is a `classifier_pool.PoolSnapshot`. Counts are of every
    tagged tile, human-set and AI-predicted alike; `human_set` says how
    many of a FOV's tags a human set or accepted."""
    tallies = snapshot.fov_tallies()
    table = ResultsTable()
    for _fov_dir, tally in tallies:
        # Vocabulary order first, then anything in use but no longer in it.
        for name in [*tally.vocabulary, *sorted(tally.counts)]:
            if name not in table.categories:
                table.categories.append(name)
    for fov_dir, tally in tallies:
        fov_path = Path(fov_dir)
        table.rows.append(
            [fov_path.parent.parent.name, fov_path.stem, tally.n_tiles]
            + [tally.counts.get(name, 0) for name in table.categories]
            + [tally.untagged, sum(tally.human_counts.values())]
        )
    return table


def write_summary_csv(path, table: ResultsTable) -> None:
    # utf-8-sig: Excel reads plain utf-8 as a legacy codepage.
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(table.header())
        writer.writerows(table.rows)


def write_tile_csv(path, fov_dirs) -> int:
    """One row per tile of `fov_dirs`, classified or not: which cell it is
    (`cell` is its label in the segmentation mask), its category, and
    whether a human set that (`source` "human") or it's an unreviewed
    prediction ("ai", with its confidence). Returns the rows written."""
    n_rows = 0
    # utf-8-sig: Excel reads plain utf-8 as a legacy codepage.
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["project", "fov", "cell", "category", "source", "confidence"])
        for fov_dir in fov_dirs:
            fov_path = Path(fov_dir)
            store = TileAnnotations(str(fov_dir))
            for cell_id, entry in get_container(fov_dir).entries.items():
                category = store.get(cell_id) or ""
                confidence = store.confidence(cell_id)
                source = "" if not category else "human" if confidence is None else "ai"
                writer.writerow(
                    [fov_path.parent.parent.name, fov_path.stem, entry["label"], category, source]
                    + ["" if confidence is None else f"{confidence:.4f}"]
                )
                n_rows += 1
    return n_rows


class ClassificationResultsPanel(QWidget):
    def __init__(self, pool, parent=None):
        super().__init__(parent)
        self.pool = pool
        self._data = ResultsTable()

        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)

        hint = QLabel(
            "Tiles per category in each FOV checked in the pool. A tile's "
            "category is its human-set tag if it has one, otherwise the "
            "classifier's prediction."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)

        self.table = QTableWidget(0, 0)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        v.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.save_summary_btn = QPushButton("Save Table (CSV)...")
        self.save_summary_btn.setToolTip("This table: one row per FOV, tile counts per category.")
        self.save_summary_btn.clicked.connect(
            lambda: self._save(
                "classification_summary.csv", lambda path: write_summary_csv(path, self._data)
            )
        )
        buttons.addWidget(self.save_summary_btn)
        self.save_tiles_btn = QPushButton("Save Per-Tile Table (CSV)...")
        self.save_tiles_btn.setToolTip(
            "One row per cell: FOV, cell number, category, whether a human "
            "or the classifier set it, and the classifier's confidence."
        )
        self.save_tiles_btn.clicked.connect(
            lambda: self._save(
                "classification_tiles.csv",
                lambda path: write_tile_csv(path, self.pool.checked_fov_dirs()),
            )
        )
        buttons.addWidget(self.save_tiles_btn)
        copy_btn = QPushButton("Copy Table")
        copy_btn.setToolTip(
            "Copy the table, totals included, to paste into Excel or Prism."
        )
        copy_btn.clicked.connect(self._copy)
        buttons.addWidget(copy_btn)
        buttons.addStretch(1)
        self.status_label = QLabel("")
        buttons.addWidget(self.status_label)
        v.addLayout(buttons)

        self._refresh = LazyRefresh(
            self, capture=pool.snapshot, compute=results_table, apply=self._show_table
        )
        pool.changed.connect(self._refresh.invalidate)
        self._show_table(self._data)

    def refresh_from_pool(self):
        self._refresh.invalidate()

    def shutdown(self):
        self._refresh.shutdown()

    def _show_table(self, data: ResultsTable):
        self._data = data
        self.save_summary_btn.setEnabled(bool(data.rows))
        self.save_tiles_btn.setEnabled(bool(data.rows))
        self.table.setColumnCount(len(data.header()))
        self.table.setHorizontalHeaderLabels(
            ["Project", "FOV", "Tiles", *data.categories, "Unclassified", "Human-set"]
        )
        rows = [*data.rows, data.total()] if data.rows else []
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                if c >= 2:
                    item.setTextAlignment(Qt.AlignCenter)
                if r == len(data.rows):  # the total
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(r, c, item)

    def _copy(self):
        rows = [self._data.header(), *self._data.rows]
        if self._data.rows:
            rows.append(self._data.total())
        QApplication.clipboard().setText(
            "\n".join("\t".join(str(value) for value in row) for row in rows)
        )

    def _save(self, default_name: str, write):
        roots = self.pool.roots()
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save Classification Results",
            str(Path(roots[0] if roots else "") / default_name),
            "CSV (*.csv);;All files (*)",
        )
        if not path:
            return
        try:
            write(path)
        except OSError as exc:
            QMessageBox.critical(self, "yeastprep", f"Could not save {path}:\n\n{exc}")
            return
        self.status_label.setText(f"Saved {path}")
