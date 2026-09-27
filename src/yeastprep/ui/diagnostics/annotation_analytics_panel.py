"""Annotation analytics over the classification pool: per-category counts,
plus class distribution plotted against an experimental variable (time,
condition, ...) mapped from FOV names. Uses the pool's checked FOVs; a FOV
is plotted once it's also mapped to a variable value (by the regex
extractor or by hand) -- see `_on_plot_clicked`.
"""

import re
from pathlib import Path

import polars as pl
import seaborn as sns
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tileclass.data.pooled_annotations import PooledAnnotations
from tileclass.scan import scan_container
from tileclass.widgets.annotation_stats_panel import AnnotationStatsPanel

_INCLUDE_COL, _PROJECT_COL, _FOV_COL, _INDEX_COL, _VARIABLE_COL = range(5)


class AnnotationAnalyticsPanel(QWidget):
    def __init__(self, pool, parent=None):
        super().__init__(parent)
        self.pool = pool

        # Per-FOV state survives table rebuilds (keyed by FOV path) so an
        # unrelated pool change (checking/unchecking some other FOV)
        # doesn't wipe hand-entered Variable mappings.
        self._fov_state: dict[str, dict] = {}
        self._category_checked: dict[str, bool] = {}
        self._category_checkboxes: dict[str, QCheckBox] = {}
        self._last_counts_df: pl.DataFrame | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_setup_tab(), "Setup")
        self._plot_tab_index = self.tabs.addTab(self._build_plot_tab(), "Plot")
        outer.addWidget(self.tabs)

    def _build_setup_tab(self) -> QWidget:
        content = QWidget()
        v = QVBoxLayout(content)
        v.setContentsMargins(0, 0, 0, 0)

        v.addWidget(self._build_stats_group())

        v.addWidget(self._build_fov_mapping_group())
        v.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        return scroll

    def _build_plot_tab(self) -> QWidget:
        tab = QWidget()
        v = QVBoxLayout(tab)
        v.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self._build_category_group())

        plot_area = QWidget()
        plot_v = QVBoxLayout(plot_area)
        plot_v.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(7, 4.5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, tab)
        plot_v.addWidget(self.toolbar)
        plot_v.addWidget(self.canvas, 1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        plot_v.addWidget(self.status_label)

        button_row = QHBoxLayout()
        self.save_figure_btn = QPushButton("Save Figure...")
        self.save_figure_btn.clicked.connect(self._save_figure)
        button_row.addWidget(self.save_figure_btn)
        self.export_data_btn = QPushButton("Export Data (CSV)...")
        self.export_data_btn.clicked.connect(self._export_data)
        button_row.addWidget(self.export_data_btn)
        button_row.addStretch(1)
        plot_v.addLayout(button_row)

        splitter.addWidget(plot_area)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([220, 500])
        v.addWidget(splitter)

        return tab

    @staticmethod
    def _wrap_group(title: str, widget: QWidget) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(widget)
        return group

    def _build_stats_group(self) -> QGroupBox:
        group = QGroupBox("Annotation Summary (pool)")
        v = QVBoxLayout(group)
        v.setContentsMargins(4, 4, 4, 4)

        self.stats_panel = AnnotationStatsPanel()
        self.stats_panel.setMaximumHeight(180)
        v.addWidget(self.stats_panel)

        refresh_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip(
            "Re-read annotations from disk and rebuild this summary -- "
            "use after annotating tiles elsewhere (e.g. a tile viewer) "
            "while this tab stayed open, since that doesn't trigger the "
            "automatic refresh that runs when switching onto this tab."
        )
        refresh_btn.clicked.connect(self.refresh_from_pool)
        refresh_row.addWidget(refresh_btn)
        refresh_row.addStretch(1)
        v.addLayout(refresh_row)

        return group

    def _build_fov_mapping_group(self) -> QGroupBox:
        group = QGroupBox("FOV -> Variable mapping")
        v = QVBoxLayout(group)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Variable name:"))
        self.variable_name_edit = QLineEdit("time")
        self.variable_name_edit.setToolTip(
            "What the mapped value below means (e.g. 'time', 'condition', "
            "'replicate') -- becomes the plot's x-axis label."
        )
        name_row.addWidget(self.variable_name_edit)
        v.addLayout(name_row)

        regex_row = QHBoxLayout()
        regex_row.addWidget(QLabel("Regex:"))
        self.regex_edit = QLineEdit()
        self.regex_edit.setPlaceholderText(r"e.g. _F(\d+)")
        self.regex_edit.setToolTip(
            "Applied to each included FOV's name to fill the Index column "
            "below -- capture group 1 if the pattern has one, otherwise "
            "the whole match."
        )
        regex_row.addWidget(self.regex_edit, 1)
        self.extract_btn = QPushButton("Extract from FOV name")
        self.extract_btn.setToolTip(
            "Overwrites the Index column for every currently-included row "
            "below with this regex applied to its FOV name. Also fills "
            "Variable with the same value, but only where Variable is "
            "still empty -- an existing hand-typed Variable is never "
            "overwritten by this."
        )
        self.extract_btn.clicked.connect(self._extract_variable_from_names)
        regex_row.addWidget(self.extract_btn)
        v.addLayout(regex_row)

        self.fov_table = QTableWidget(0, 5)
        self.fov_table.setHorizontalHeaderLabels(
            ["Include", "Project", "FOV", "Index", "Variable"]
        )
        self.fov_table.verticalHeader().setVisible(False)
        self.fov_table.horizontalHeader().setSectionResizeMode(
            _PROJECT_COL, QHeaderView.ResizeToContents
        )
        # Interactive (not ResizeToContents) so a user can shrink this
        # column past a long FOV name -- the tooltip set per-item below
        # covers what a tight column elides.
        self.fov_table.horizontalHeader().setSectionResizeMode(
            _FOV_COL, QHeaderView.Interactive
        )
        self.fov_table.setColumnWidth(_FOV_COL, 100)
        self.fov_table.horizontalHeader().setSectionResizeMode(
            _INDEX_COL, QHeaderView.ResizeToContents
        )
        self.fov_table.horizontalHeader().setSectionResizeMode(
            _VARIABLE_COL, QHeaderView.Stretch
        )
        self.fov_table.setMaximumHeight(180)
        self.fov_table.itemChanged.connect(self._on_fov_item_changed)
        v.addWidget(self.fov_table)

        hint = QLabel(
            "Index is the raw token pulled from the FOV name (e.g. via "
            "Extract, above) -- kept for reference. Variable is the "
            "actual value used in the plot: pre-filled from Index the "
            "first time, but edit it directly whenever a FOV's index "
            "doesn't correspond to its real value (e.g. non-uniform time "
            "points). Only FOVs that are checked *and* have a non-empty "
            "Variable are included in the plot."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)

        return group

    def _build_category_group(self) -> QGroupBox:
        group = QGroupBox("Categories to plot")
        v = QVBoxLayout(group)

        buttons = QHBoxLayout()
        all_btn = QPushButton("All")
        all_btn.clicked.connect(lambda: self._set_all_categories(True))
        buttons.addWidget(all_btn)
        none_btn = QPushButton("None")
        none_btn.clicked.connect(lambda: self._set_all_categories(False))
        buttons.addWidget(none_btn)
        buttons.addStretch(1)
        self.confirmed_only_cb = QCheckBox("Human-confirmed only")
        self.confirmed_only_cb.setToolTip(
            "Off (default): plot every tagged tile, human-confirmed or "
            "still-unreviewed AI prediction alike. On: restrict to "
            "human-confirmed/overridden tags only."
        )
        buttons.addWidget(self.confirmed_only_cb)
        v.addLayout(buttons)

        self.category_list_widget = QWidget()
        self.category_list_layout = QVBoxLayout(self.category_list_widget)
        self.category_list_layout.setContentsMargins(0, 0, 0, 0)
        self.category_list_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(80)
        scroll.setWidget(self.category_list_widget)
        v.addWidget(scroll, 1)

        plot_row = QHBoxLayout()
        self.plot_btn = QPushButton("Plot")
        self.plot_btn.setToolTip(
            "Build the plot from the current FOV mapping (Setup tab) and "
            "the category selection above."
        )
        self.plot_btn.clicked.connect(self._on_plot_clicked)
        plot_row.addWidget(self.plot_btn)
        plot_row.addStretch(1)
        v.addLayout(plot_row)

        return group

    # ------------------------------------------------------------------
    # Refresh from pool -- called on ClassifierPool.changed and
    # whenever this tab becomes the visible one (annotation files can
    # change on disk from an external tileclass viewer session).

    def refresh_from_pool(self):
        fov_dirs = self.pool.checked_fov_dirs()
        pooled = self.pool.pooled_annotations()

        self._refresh_fov_table(fov_dirs)
        self._refresh_category_checks(pooled)
        self._refresh_stats(pooled, fov_dirs)

    def _refresh_stats(self, pooled, fov_dirs):
        annotations = pooled if pooled is not None else PooledAnnotations([])
        paths = [p for f in fov_dirs for p in scan_container(f)]
        self.stats_panel.refresh(annotations, paths)

    def _refresh_fov_table(self, fov_dirs: list[str]):
        current_paths = set(fov_dirs)
        for path in list(self._fov_state):
            if path not in current_paths:
                del self._fov_state[path]
        for path in fov_dirs:
            self._fov_state.setdefault(path, {"include": True, "index": "", "variable": ""})

        self.fov_table.blockSignals(True)
        try:
            self.fov_table.setRowCount(len(fov_dirs))
            for row, path in enumerate(fov_dirs):
                state = self._fov_state[path]
                p = Path(path)

                include_item = QTableWidgetItem()
                include_item.setFlags(
                    (include_item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable
                )
                include_item.setCheckState(Qt.Checked if state["include"] else Qt.Unchecked)
                include_item.setData(Qt.UserRole, path)
                self.fov_table.setItem(row, _INCLUDE_COL, include_item)

                project_item = QTableWidgetItem(p.parent.parent.name)
                project_item.setFlags(project_item.flags() & ~Qt.ItemIsEditable)
                self.fov_table.setItem(row, _PROJECT_COL, project_item)

                fov_item = QTableWidgetItem(p.stem)
                fov_item.setFlags(fov_item.flags() & ~Qt.ItemIsEditable)
                fov_item.setToolTip(path)
                self.fov_table.setItem(row, _FOV_COL, fov_item)

                index_item = QTableWidgetItem(state["index"])
                self.fov_table.setItem(row, _INDEX_COL, index_item)

                variable_item = QTableWidgetItem(state["variable"])
                self.fov_table.setItem(row, _VARIABLE_COL, variable_item)
        finally:
            self.fov_table.blockSignals(False)

    def _on_fov_item_changed(self, item: QTableWidgetItem):
        row = item.row()
        path = self.fov_table.item(row, _INCLUDE_COL).data(Qt.UserRole)
        if path not in self._fov_state:
            return
        if item.column() == _INCLUDE_COL:
            self._fov_state[path]["include"] = item.checkState() == Qt.Checked
        elif item.column() == _INDEX_COL:
            self._fov_state[path]["index"] = item.text().strip()
        elif item.column() == _VARIABLE_COL:
            self._fov_state[path]["variable"] = item.text().strip()

    def _extract_variable_from_names(self):
        pattern = self.regex_edit.text()
        if not pattern:
            return
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            QMessageBox.warning(self, "yeastprep", f"Invalid regex: {exc}")
            return

        self.fov_table.blockSignals(True)
        try:
            for row in range(self.fov_table.rowCount()):
                include_item = self.fov_table.item(row, _INCLUDE_COL)
                if include_item.checkState() != Qt.Checked:
                    continue
                path = include_item.data(Qt.UserRole)
                fov_name = self.fov_table.item(row, _FOV_COL).text()
                match = compiled.search(fov_name)
                if not match:
                    continue
                value = match.group(1) if match.groups() else match.group(0)
                self.fov_table.item(row, _INDEX_COL).setText(value)
                self._fov_state[path]["index"] = value

                variable_item = self.fov_table.item(row, _VARIABLE_COL)
                if not variable_item.text().strip():
                    variable_item.setText(value)
                    self._fov_state[path]["variable"] = value
        finally:
            self.fov_table.blockSignals(False)

    def _refresh_category_checks(self, pooled):
        for name, checkbox in self._category_checkboxes.items():
            self._category_checked[name] = checkbox.isChecked()

        while self.category_list_layout.count() > 1:
            item = self.category_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        categories = pooled.categories() if pooled is not None else []
        self._category_checkboxes = {}
        for name in categories:
            checkbox = QCheckBox(name)
            checkbox.setChecked(self._category_checked.get(name, True))
            self.category_list_layout.insertWidget(
                self.category_list_layout.count() - 1, checkbox
            )
            self._category_checkboxes[name] = checkbox

    def _set_all_categories(self, checked: bool):
        for checkbox in self._category_checkboxes.values():
            checkbox.setChecked(checked)

    # ------------------------------------------------------------------
    # Plot

    def _on_plot_clicked(self):
        pooled = self.pool.pooled_annotations()
        if pooled is None:
            QMessageBox.warning(self, "yeastprep", "No FOVs checked in the pool to plot.")
            return

        selected_categories = {
            name for name, cb in self._category_checkboxes.items() if cb.isChecked()
        }
        if not selected_categories:
            QMessageBox.warning(self, "yeastprep", "No categories checked to plot.")
            return

        included_fovs = [
            (path, state["variable"])
            for path, state in self._fov_state.items()
            if state["include"] and state["variable"]
        ]
        n_skipped = sum(
            1 for state in self._fov_state.values() if state["include"] and not state["variable"]
        )
        if not included_fovs:
            QMessageBox.warning(
                self,
                "yeastprep",
                "No FOVs are both included and mapped to a Variable value "
                "-- fill in the Variable column (or use 'Extract from FOV "
                "name') first.",
            )
            return

        confirmed_only = self.confirmed_only_cb.isChecked()
        fov_prefixes = [(path + "/", value) for path, value in included_fovs]

        rows = []
        for tile_path, category, confidence in pooled.tagged_items():
            if confirmed_only and confidence is not None:
                continue
            if category not in selected_categories:
                continue
            for prefix, value in fov_prefixes:
                if tile_path.startswith(prefix):
                    rows.append({"variable_raw": value, "category": category})
                    break

        if not rows:
            QMessageBox.warning(self, "yeastprep", "No tagged tiles match the current filters.")
            return

        df = pl.DataFrame(rows)
        try:
            df = df.with_columns(pl.col("variable_raw").cast(pl.Float64).alias("variable"))
        except Exception:
            df = df.with_columns(pl.col("variable_raw").alias("variable"))

        counts = (
            df.group_by(["variable", "category"])
            .len()
            .with_columns(
                (pl.col("len") / pl.col("len").sum().over("variable") * 100).alias("percentage")
            )
            .sort(["variable", "category"])
        )
        totals = df.group_by("variable").agg(pl.len().alias("total_count")).sort("variable")
        self._last_counts_df = counts

        variable_name = self.variable_name_edit.text().strip() or "variable"

        self.figure.clear()
        ax_total = self.figure.add_subplot(1, 2, 1)
        ax_pct = self.figure.add_subplot(1, 2, 2)

        # Plain dicts of columns rather than `.to_pandas()` -- seaborn
        # accepts any long-form dict-of-array-likes as `data`, and this
        # avoids pulling in pyarrow (polars' `to_pandas()` dependency)
        # just to hand the frame straight back off to seaborn.
        sns.lineplot(
            data=totals.to_dict(as_series=False),
            x="variable",
            y="total_count",
            marker="s",
            color="#2b5c8f",
            ax=ax_total,
        )
        ax_total.set_xlabel(variable_name)
        ax_total.set_ylabel("Tagged tile count")

        sns.lineplot(
            data=counts.to_dict(as_series=False),
            x="variable",
            y="percentage",
            hue="category",
            marker="o",
            ax=ax_pct,
        )
        ax_pct.set_xlabel(variable_name)
        ax_pct.set_ylabel("Percentage (%)")
        ax_pct.legend(loc="best", fontsize="small")

        self.figure.tight_layout()
        self.canvas.draw_idle()

        status = (
            f"Plotted {len(rows)} tile(s) across "
            f"{len({v for _, v in included_fovs})} variable value(s)."
        )
        if n_skipped:
            status += f" ({n_skipped} included FOV(s) skipped -- no Variable value set.)"
        self.status_label.setText(status)
        self.tabs.setCurrentIndex(self._plot_tab_index)

    def _save_figure(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Figure", "", "PNG image (*.png);;All files (*)"
        )
        if not path:
            return
        self.figure.savefig(path)

    def _export_data(self):
        if self._last_counts_df is None:
            QMessageBox.warning(self, "yeastprep", "Nothing to export -- plot first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data", "", "CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        self._last_counts_df.write_csv(path)
