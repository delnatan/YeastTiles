"""UMAP embedding scatter for the Classify Tiles page's "Explore
Embeddings" -- the plot half of `embedding_explorer.EmbeddingExplorerWidget`,
which pairs it with a tile grid for annotating.

Populated on demand via `tileclass.training.embeddings.umap_2d`/
`knn_accuracy`. Unlabeled points (see `core.classify.unlabeled_paths`) are
drawn underneath as small gray dots outside the category colors, so the
often far more numerous unlabeled tiles never bury annotated ones.

Interaction: left-drag lassos points (`lassoSelected`), hovering near a
point reports it (`pointHovered`) for a tile preview, the wheel zooms about
the cursor, right-drag pans, and Reset View refits. Every point lives in
one collection, so relabeling (`set_labels`) just recolors in place -- no
new UMAP layout, and the map the user has been reading doesn't move.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import to_rgba_array
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.widgets import LassoSelector
from qtpy.QtCore import Signal
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

UNLABELED_LABEL = "(unlabeled)"
_UNLABELED_COLOR = "#c8c8c8"
_UNLABELED_SIZE = 5
_LABELED_SIZE = 16
_HOVER_RADIUS_PX = 8
_ZOOM_STEP = 1.25


class EmbeddingScatterWidget(QWidget):
    # Indices (into the arrays given to `show_points`) of lasso-selected points.
    lassoSelected = Signal(list)
    # Index of the point under the cursor, or -1 for none; plus the cursor's
    # global position (a QPoint) for placing a preview next to it.
    pointHovered = Signal(int, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._xy = None
        self._labels = None  # object array, parallel to _xy
        self._points = None  # the one PathCollection holding every point
        self._highlight = None  # ring collection over highlighted points
        self._home_limits = None
        self._hovered = -1
        self._pan = None  # (x0, y0, xlim, ylim, inverse transform) while right-dragging
        self._lasso = None  # kept alive here -- LassoSelector drops its
        # event connections if its only reference is garbage collected

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self._blank_axes()
        layout.addWidget(self.canvas, 1)

        toolbar = QHBoxLayout()
        toolbar.addStretch(1)
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.setToolTip("Zoom back out to every point. (Wheel zooms, right-drag pans.)")
        self.reset_view_btn.clicked.connect(self.reset_view)
        toolbar.addWidget(self.reset_view_btn)
        layout.addLayout(toolbar)

        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("figure_leave_event", lambda _e: self._set_hovered(-1))

    def _blank_axes(self):
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    # ------------------------------------------------------------------
    # Public API

    def show_points(self, xy, labels, category_colors: dict, title: str = "") -> None:
        """`xy`: (N, 2) from `umap_2d`. `labels`: length-N category names,
        `UNLABELED_LABEL` for the rest. `category_colors`: category -> hex,
        in legend order. Axes are unitless by design: UMAP coordinates and
        inter-cluster distances aren't meaningful on their own -- only
        which points cluster together is."""
        self.ax.clear()
        self._blank_axes()
        self._xy = np.asarray(xy, dtype=float)
        self._points = self.ax.scatter(self._xy[:, 0], self._xy[:, 1], zorder=2)
        self._highlight = None
        self._hovered = -1
        self.ax.set_title(title)

        lo, hi = self._xy.min(axis=0), self._xy.max(axis=0)
        pad = 0.05 * np.maximum(hi - lo, 1e-9)
        self._home_limits = ((lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1]))
        self.reset_view()

        if self._lasso is not None:
            self._lasso.disconnect_events()
        self._lasso = LassoSelector(self.ax, onselect=self._on_lasso, button=1)
        self.set_labels(labels, category_colors)

    def set_labels(self, labels, category_colors: dict) -> None:
        """Recolor every point for `labels` (same order as `show_points`)
        without touching the layout or view."""
        if self._points is None:
            return
        self._labels = np.asarray(labels, dtype=object)
        labeled = self._labels != UNLABELED_LABEL
        # A collection draws in offset order: unlabeled first, underneath.
        order = np.argsort(labeled, kind="stable")
        hexes = [
            category_colors.get(label, _UNLABELED_COLOR) if is_labeled else _UNLABELED_COLOR
            for label, is_labeled in zip(self._labels[order], labeled[order])
        ]
        faces = to_rgba_array(hexes)
        edges = np.zeros_like(faces)
        edges[labeled[order]] = (1.0, 1.0, 1.0, 1.0)
        self._points.set_offsets(self._xy[order])
        self._points.set_facecolors(faces)
        self._points.set_edgecolors(edges)
        self._points.set_linewidths(np.where(labeled[order], 0.3, 0.0))
        self._points.set_sizes(np.where(labeled[order], _LABELED_SIZE, _UNLABELED_SIZE))
        self._update_legend(category_colors)
        self.canvas.draw_idle()

    def set_highlight(self, indices) -> None:
        """Ring the given points (e.g. the tiles selected in the grid)."""
        if self._xy is None:
            return
        points = self._xy[list(indices)] if len(indices) else np.empty((0, 2))
        if self._highlight is None:
            self._highlight = self.ax.scatter(
                points[:, 0], points[:, 1], s=70, facecolors="none",
                edgecolors="black", linewidths=1.2, zorder=5,
            )
        else:
            self._highlight.set_offsets(points)
        self.canvas.draw_idle()

    def reset_view(self) -> None:
        if self._home_limits is None:
            return
        self.ax.set_xlim(*self._home_limits[0])
        self.ax.set_ylim(*self._home_limits[1])
        self.canvas.draw_idle()

    # ------------------------------------------------------------------

    def _update_legend(self, category_colors):
        counts = {}
        for label in self._labels:
            counts[label] = counts.get(label, 0) + 1
        handles = [
            Line2D([], [], marker="o", linestyle="", color=color, label=f"{name} ({counts[name]})")
            for name, color in category_colors.items()
            if counts.get(name)
        ]
        if counts.get(UNLABELED_LABEL):
            handles.append(
                Line2D([], [], marker="o", linestyle="", color=_UNLABELED_COLOR,
                       label=f"{UNLABELED_LABEL} ({counts[UNLABELED_LABEL]})")
            )
        # A fixed corner: "best" placement scans every point.
        self.ax.legend(handles=handles, loc="upper right", fontsize="small")

    def _on_lasso(self, vertices) -> None:
        if self._xy is None:
            return
        inside = MplPath(vertices).contains_points(self._xy)
        indices = np.flatnonzero(inside).tolist()
        if indices:
            self.lassoSelected.emit(indices)

    def _set_hovered(self, index: int) -> None:
        if index != self._hovered:
            self._hovered = index
            self.pointHovered.emit(index, QCursor.pos())

    def _nearest_point(self, event) -> int:
        if self._xy is None or event.inaxes is not self.ax:
            return -1
        pixels = self.ax.transData.transform(self._xy)
        d2 = ((pixels - (event.x, event.y)) ** 2).sum(axis=1)
        nearest = int(np.argmin(d2))
        return nearest if d2[nearest] <= _HOVER_RADIUS_PX**2 else -1

    def _on_motion(self, event) -> None:
        if self._pan is not None:
            x0, y0, xlim, ylim, inverse = self._pan
            start = inverse.transform((x0, y0))
            now = inverse.transform((event.x, event.y))
            dx, dy = start - now
            self.ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
            self.ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
            self.canvas.draw_idle()
            return
        if event.button is not None:  # mid-lasso
            return
        self._set_hovered(self._nearest_point(event))

    def _on_press(self, event) -> None:
        self._set_hovered(-1)
        if event.button == 3 and event.inaxes is self.ax:
            self._pan = (
                event.x, event.y, self.ax.get_xlim(), self.ax.get_ylim(),
                self.ax.transData.inverted(),
            )

    def _on_release(self, event) -> None:
        if event.button == 3:
            self._pan = None

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        scale = 1 / _ZOOM_STEP if event.button == "up" else _ZOOM_STEP
        x, y = event.xdata, event.ydata
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.set_xlim(x + (xlim[0] - x) * scale, x + (xlim[1] - x) * scale)
        self.ax.set_ylim(y + (ylim[0] - y) * scale, y + (ylim[1] - y) * scale)
        self.canvas.draw_idle()
