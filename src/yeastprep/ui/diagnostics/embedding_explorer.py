"""Annotate from the embedding map: the Classify Tiles page's Explore
Embeddings scatter (`EmbeddingScatterWidget`) side by side with a tile
grid (tileclass's `ThumbnailGridWidget`), in one loop:

lasso a region -> its tiles fill the grid, all selected -> Ctrl/Cmd-click
the odd ones out -> press 1-9 (or click a category) -> the tags are saved
and the points recolor in place.

Grid selection is ringed on the plot, and hovering a point previews its
tile. Tags are written through a `PooledAnnotations`, the same per-FOV
sidecar files every tile viewer uses -- each write re-reads its sidecar
first (see tileclass's `data/annotations.py`), so labeling here alongside
open viewer windows is safe. Labels set here are human-confirmed, exactly
like the viewer's number keys.
"""

import numpy as np
from qtpy.QtCore import QPoint, Qt, Signal
from qtpy.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from qtpy.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from tileclass.data.channel_state import ChannelStateList
from tileclass.data.palette import category_color, readable_text_color
from tileclass.tile_container import tile_display_name
from tileclass.widgets.manage_categories_dialog import ManageCategoriesDialog
from tileclass.widgets.thumbnail_grid import ThumbnailGridWidget

from .embedding_scatter_widget import UNLABELED_LABEL, EmbeddingScatterWidget

NUMBER_KEY_CATEGORIES = 9  # matches the tile viewer's 1-9 keys
_CATEGORY_COLUMNS = 3
_N_TILE_CHANNELS = 3  # brightfield, target, mask -- see core/tiles.py
_PREVIEW_PX = 160


def _rgb_to_pixmap(rgb: np.ndarray) -> QPixmap:
    h, w, _ = rgb.shape
    image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
    return QPixmap.fromImage(image)


class _HoverPreview(QWidget):
    """Frameless tooltip-style window: one tile plus its name and label."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)
        self.image = QLabel()
        self.caption = QLabel()
        layout.addWidget(self.image)
        layout.addWidget(self.caption)
        self.setStyleSheet("background: #1a1a1a; color: #dddddd;")

    def show_tile(self, rgb, caption: str, at: QPoint):
        self.image.setPixmap(
            _rgb_to_pixmap(rgb).scaled(
                _PREVIEW_PX, _PREVIEW_PX, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        self.caption.setText(caption)
        self.adjustSize()
        self.move(at + QPoint(16, 16))
        self.show()


class EmbeddingExplorerWidget(QWidget):
    # Tags were written -- e.g. so the page can refresh annotation counts.
    annotationsChanged = Signal()
    # "Open in Viewer": the current grid's tile paths, for a full tile viewer.
    openInViewerRequested = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pooled = None
        self._paths: list[str] = []
        self._index: dict[str, int] = {}
        self._labels: list[str] = []
        self._category_colors: dict[str, str] = {}
        self._title = ""

        self.channel_state = ChannelStateList(_N_TILE_CHANNELS)
        self._build_ui()
        self._wire_up()

    # ------------------------------------------------------------------
    # UI

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Horizontal)
        outer.addWidget(splitter)

        self.scatter = EmbeddingScatterWidget()
        splitter.addWidget(self.scatter)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(4, 0, 0, 0)

        self.selection_label = QLabel(
            "Lasso points on the plot to see their tiles here. Hover a point to preview it."
        )
        self.selection_label.setWordWrap(True)
        side_layout.addWidget(self.selection_label)

        self.category_box = QWidget()
        self.category_layout = QGridLayout(self.category_box)
        self.category_layout.setContentsMargins(0, 0, 0, 0)
        self.category_layout.setSpacing(3)
        side_layout.addWidget(self.category_box)

        actions = QHBoxLayout()
        self.clear_tag_btn = QPushButton("Clear Tag")
        self.clear_tag_btn.setToolTip("Remove the selected tiles' tags (0 / Delete).")
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.setToolTip("Select every tile in the grid (Ctrl/Cmd+A).")
        self.categories_btn = QPushButton("Categories...")
        self.categories_btn.setToolTip("Add, rename or remove categories.")
        self.open_viewer_btn = QPushButton("Open in Viewer")
        self.open_viewer_btn.setToolTip("Open these tiles in a full tile viewer window.")
        for btn in (self.clear_tag_btn, self.select_all_btn, self.categories_btn, self.open_viewer_btn):
            actions.addWidget(btn)
        side_layout.addLayout(actions)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Tile size"))
        self.tile_size_slider = QSlider(Qt.Horizontal)
        self.tile_size_slider.setRange(48, 200)
        self.tile_size_slider.setValue(96)
        size_row.addWidget(self.tile_size_slider, 1)
        side_layout.addLayout(size_row)

        self.grid = ThumbnailGridWidget()
        self.grid.set_tile_size(self.tile_size_slider.value())
        self.grid.set_show_info(False)
        self.grid.set_channel_state(self.channel_state)
        grid_scroll = QScrollArea()
        grid_scroll.setWidgetResizable(True)
        grid_scroll.setWidget(self.grid)
        side_layout.addWidget(grid_scroll, 1)

        splitter.addWidget(side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.preview = _HoverPreview(self)
        self._rebuild_category_buttons()

    def _wire_up(self):
        self.scatter.lassoSelected.connect(self._on_lasso)
        self.scatter.pointHovered.connect(self._on_point_hovered)
        self.grid.selectionChanged.connect(self._on_grid_selection_changed)
        self.grid.annotateRequested.connect(self._pick_category_for_selection)
        self.grid.viewFullSizeRequested.connect(self._view_full_size)
        self.clear_tag_btn.clicked.connect(lambda: self.assign_category(""))
        self.select_all_btn.clicked.connect(self.grid.select_all)
        self.categories_btn.clicked.connect(self._manage_categories)
        self.open_viewer_btn.clicked.connect(
            lambda: self.grid.paths and self.openInViewerRequested.emit(self.grid.paths)
        )
        self.tile_size_slider.valueChanged.connect(self.grid.set_tile_size)

        def shortcut(keys, slot):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)

        for n in range(1, NUMBER_KEY_CATEGORIES + 1):
            shortcut(str(n), lambda n=n: self.assign_number(n))
        for keys in ("0", "Delete", "Backspace"):
            shortcut(keys, lambda: self.assign_category(""))
        shortcut(QKeySequence.SelectAll, self.grid.select_all)

    # ------------------------------------------------------------------
    # Public API

    def show_embedding(self, xy, labels, paths, pooled, knn_acc: float | None = None):
        """`labels`: each path's human-confirmed category, or
        `UNLABELED_LABEL`. `pooled`: the `PooledAnnotations` covering
        `paths`, which tags are written through."""
        self._pooled = pooled
        self._paths = list(paths)
        self._index = {p: i for i, p in enumerate(self._paths)}
        self._labels = list(labels)
        self._title = "Backbone embeddings (UMAP)"
        if knn_acc is not None:
            self._title += f"  --  kNN accuracy at Evaluate: {knn_acc:.2f}"
        self._seed_channel_colors()
        self.grid.set_items([])
        self.grid.set_annotations({})
        self._refresh_categories()
        self.scatter.show_points(xy, self._labels, self._category_colors, self._title)
        self.selection_label.setText(
            f"{len(self._paths)} tiles plotted. Lasso points to see their tiles "
            "here; hover a point to preview it."
        )

    def labels_for(self, paths) -> list[str]:
        """Current labels (including tags set here since `show_embedding`)
        for `paths`, e.g. to re-plot without losing them."""
        return [
            self._labels[self._index[p]] if p in self._index else UNLABELED_LABEL
            for p in paths
        ]

    def assign_number(self, number: int):
        categories = self._vocabulary()
        if number <= len(categories):
            self.assign_category(categories[number - 1])

    def assign_category(self, category: str):
        """Tag every tile selected in the grid with `category` ("" clears)."""
        selected = self.grid.selected_paths
        if self._pooled is None or not selected:
            return
        self._pooled.update((p, category) for p in selected)
        for p in selected:
            self._labels[self._index[p]] = category or UNLABELED_LABEL
        self._refresh_categories()
        self.scatter.set_labels(self._labels, self._category_colors)
        self._refresh_badges()
        verb = f"Tagged {len(selected)} tile(s) as {category}" if category else (
            f"Cleared {len(selected)} tile(s)"
        )
        self.selection_label.setText(f"{verb}. {self._selection_summary()}")
        self.annotationsChanged.emit()

    def shutdown(self):
        self.preview.hide()
        self.grid.close()

    # ------------------------------------------------------------------
    # Categories

    def _vocabulary(self) -> list[str]:
        return list(self._pooled.categories()) if self._pooled is not None else []

    def _refresh_categories(self):
        """Colors in vocabulary order (as the tile viewer assigns them),
        then any tag in use that isn't in the vocabulary."""
        names = self._vocabulary()
        for label in self._labels:
            if label != UNLABELED_LABEL and label not in names:
                names.append(label)
        self._category_colors = {name: category_color(i) for i, name in enumerate(names)}
        self._rebuild_category_buttons()

    def _rebuild_category_buttons(self):
        while self.category_layout.count():
            item = self.category_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        vocabulary = self._vocabulary()
        if not vocabulary:
            hint = QLabel(
                "No categories yet -- add some with Categories..."
                if self._pooled is not None
                else ""
            )
            hint.setStyleSheet("color: #888;")
            self.category_layout.addWidget(hint, 0, 0)
            return
        for i, name in enumerate(vocabulary):
            key = f"{i + 1}  " if i < NUMBER_KEY_CATEGORIES else ""
            color = self._category_colors.get(name, category_color(i))
            btn = QPushButton(f"{key}{name}")
            btn.setStyleSheet(
                f"background-color: {color}; color: {readable_text_color(color)}; padding: 3px;"
            )
            btn.setToolTip(f"Tag the selected tiles as {name}")
            btn.clicked.connect(lambda _checked=False, name=name: self.assign_category(name))
            self.category_layout.addWidget(btn, i // _CATEGORY_COLUMNS, i % _CATEGORY_COLUMNS)

    def _manage_categories(self):
        if self._pooled is None:
            return
        ManageCategoriesDialog(self._pooled, parent=self).exec_()
        # A rename rewrites tags on disk; pick those up for plotted tiles.
        for p, i in self._index.items():
            if self._labels[i] != UNLABELED_LABEL:
                tag = self._pooled.get(p)
                confirmed = tag and self._pooled.confidence(p) is None
                self._labels[i] = tag if confirmed else UNLABELED_LABEL
        self._refresh_categories()
        self.scatter.set_labels(self._labels, self._category_colors)
        self._refresh_badges()
        self.annotationsChanged.emit()

    def _pick_category_for_selection(self):
        vocabulary = self._vocabulary()
        if not vocabulary or not self.grid.selected_paths:
            return
        items = ["(clear tag)"] + vocabulary
        choice, ok = QInputDialog.getItem(
            self, "Annotate", f"Category for {len(self.grid.selected_paths)} tile(s):",
            items, 1, editable=False,
        )
        if ok:
            self.assign_category("" if choice == items[0] else choice)

    # ------------------------------------------------------------------
    # Selection

    def _on_lasso(self, indices):
        # Unlabeled first, then grouped by category: the tiles still needing
        # a tag come first, and a stray in a labeled block stands out.
        def key(i):
            label = self._labels[i]
            return (label != UNLABELED_LABEL, label, i)

        paths = [self._paths[i] for i in sorted(indices, key=key)]
        self.grid.set_items(paths)
        self.grid.set_priority(paths)
        self._refresh_badges()
        self.grid.select_all()
        self.grid.setFocus()

    def _on_grid_selection_changed(self, selected):
        self.scatter.set_highlight([self._index[p] for p in selected if p in self._index])
        self.selection_label.setText(self._selection_summary())

    def _selection_summary(self) -> str:
        shown = len(self.grid.paths)
        if not shown:
            return "Lasso points on the plot to see their tiles here."
        return (
            f"{len(self.grid.selected_paths)} of {shown} tiles selected -- "
            "Ctrl/Cmd-click to toggle, then press 1-9 or click a category."
        )

    def _refresh_badges(self):
        if self._pooled is None:
            return
        badges = {}
        for p in self.grid.paths:
            tag = self._pooled.get(p)
            if tag:
                color = self._category_colors.get(tag, "#666666")
                badges[p] = (tag, color, self._pooled.confidence(p))
        self.grid.set_annotations(badges)

    def _seed_channel_colors(self):
        """The pool's saved per-channel colors (the tile viewer's Colors
        panel), so tiles look the same here as there."""
        if self._pooled is None:
            return
        for idx, (color_hex, blend_mode, opacity) in self._pooled.channel_colors.items():
            if idx < len(self.channel_state):
                self.channel_state.set(
                    idx, color_hex=color_hex, blend_mode=blend_mode, opacity=opacity
                )

    # ------------------------------------------------------------------
    # Previews

    def _on_point_hovered(self, index: int, at):
        if index < 0 or index >= len(self._paths):
            self.preview.hide()
            return
        path = self._paths[index]
        try:
            rgb = self.grid.rgb_for(path)
        except Exception:
            self.preview.hide()
            return
        label = self._labels[index]
        caption = tile_display_name(path)
        if label != UNLABELED_LABEL:
            caption += f"\n{label}"
        self.preview.show_tile(rgb, caption, at)

    def _view_full_size(self, path):
        rgb = self.grid.rgb_for(path)
        dlg = QDialog(self)
        dlg.setWindowTitle(tile_display_name(path))
        layout = QVBoxLayout(dlg)
        label = QLabel()
        label.setPixmap(_rgb_to_pixmap(rgb).scaled(
            4 * rgb.shape[1], 4 * rgb.shape[0], Qt.KeepAspectRatio, Qt.FastTransformation
        ))
        layout.addWidget(label)
        dlg.exec_()

    def hideEvent(self, event):
        self.preview.hide()
        super().hideEvent(event)

    def leaveEvent(self, event):
        self.preview.hide()
        super().leaveEvent(event)

