"""Standalone t-SNE embedding scatter with lasso-select -- extracted from
`ClassifierTrainingMonitorPanel`'s old Embeddings tab so it can be reused by
the Classify Tiles page (see that page's "Explore Embeddings" group), which
gives it real room to work rather than a cramped tab.

Populated on demand via `tileclass.training.linear_probe.tsne_2d`/
`knn_accuracy`, with an optional pool of *unlabeled* points (see
`core.classify.sample_unlabeled`) rendered in a fixed neutral color/marker
outside the per-category color cycle -- still selectable via lasso, so an
unlabeled cluster sitting near/inside a labeled one can be checked against
its actual image just as easily as a labeled outlier can.
"""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.widgets import LassoSelector
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

UNLABELED_LABEL = "(unlabeled)"


class EmbeddingScatterWidget(QWidget):
    # Emitted with the list of tile paths lasso-selected on the scatter, so
    # the owning page can open them in a tile viewer.
    pointsSelected = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._embedding_xy = None
        self._embedding_paths = None
        self._lasso = None  # kept alive here -- LassoSelector drops its
        # event connections if its only reference is garbage collected
        self._selection_highlight = None  # scatter artist ringing selected points

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, 1)

        toolbar = QHBoxLayout()
        toolbar.addStretch(1)
        self.clear_selection_btn = QPushButton("Clear Selection")
        self.clear_selection_btn.setEnabled(False)
        self.clear_selection_btn.setToolTip(
            "Clear the ringed lasso selection above (does not close any "
            "tile viewer windows already opened from it)."
        )
        self.clear_selection_btn.clicked.connect(self.clear_selection)
        toolbar.addWidget(self.clear_selection_btn)
        layout.addLayout(toolbar)

    def show_embedding_scatter(
        self, xy, labels, paths: list[str] | None = None, knn_acc: float | None = None
    ) -> None:
        """`xy`: (N, 2) array from `tileclass.training.linear_probe.tsne_2d`.
        `labels`: length-N category names, colored by matplotlib's default
        cycle -- except `UNLABELED_LABEL`, always drawn last in a fixed
        neutral gray outside that cycle, so an unlabeled sample never steals
        a category's color. `paths`: parallel length-N list of each point's
        tile path -- when given, enables lasso-selecting a region of points
        and emitting their paths via `pointsSelected`, so a suspicious
        cluster (or an outlier sitting with the wrong category, or an
        unlabeled point sitting inside a labeled cluster) can be checked
        against its actual image, not just its label. `knn_acc`: optional
        `knn_accuracy` result (computed over confirmed/labeled points only),
        shown in the title as a quick separability readout -- t-SNE and kNN
        are both local-neighborhood notions of separability, so the plot and
        the number tell a consistent story (see `tsne_2d`'s docstring for
        why this replaced a PCA projection). Axes are unitless/unlabeled by
        design: unlike PCA's PC1/PC2, t-SNE coordinates and inter-cluster
        distances aren't meaningful on their own -- only which points
        cluster together is."""
        self.ax.clear()
        categories = sorted(c for c in set(labels) if c != UNLABELED_LABEL)
        for category in categories:
            mask = [label == category for label in labels]
            points = xy[mask]
            self.ax.scatter(points[:, 0], points[:, 1], label=category, s=12)
        if UNLABELED_LABEL in labels:
            mask = [label == UNLABELED_LABEL for label in labels]
            points = xy[mask]
            self.ax.scatter(
                points[:, 0],
                points[:, 1],
                label=UNLABELED_LABEL,
                s=10,
                c="lightgray",
                marker="x",
                zorder=1,
            )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        title = "Backbone embeddings (t-SNE)"
        if knn_acc is not None:
            title += f"  --  kNN accuracy: {knn_acc:.2f}"
        self.ax.set_title(title)
        self.ax.legend(loc="best", fontsize="small")

        self._embedding_xy = xy
        self._embedding_paths = paths
        self._selection_highlight = None  # ax.clear() above already dropped the artist
        self.clear_selection_btn.setEnabled(False)
        if self._lasso is not None:
            self._lasso.disconnect_events()
            self._lasso = None
        if paths is not None:
            self._lasso = LassoSelector(self.ax, onselect=self._on_lasso_select)

        self.canvas.draw_idle()

    def _on_lasso_select(self, vertices) -> None:
        if self._embedding_paths is None or self._embedding_xy is None:
            return
        selected_mask = MplPath(vertices).contains_points(self._embedding_xy)
        selected = [p for p, keep in zip(self._embedding_paths, selected_mask) if keep]
        if not selected:
            return

        if self._selection_highlight is not None:
            self._selection_highlight.remove()
        points = self._embedding_xy[selected_mask]
        self._selection_highlight = self.ax.scatter(
            points[:, 0],
            points[:, 1],
            s=80,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )
        self.clear_selection_btn.setEnabled(True)
        self.canvas.draw_idle()
        self.pointsSelected.emit(selected)

    def clear_selection(self) -> None:
        """Removes the ringed highlight left by the last lasso selection.
        Purely cosmetic -- any tile viewer windows already opened from that
        selection are unaffected."""
        if self._selection_highlight is not None:
            self._selection_highlight.remove()
            self._selection_highlight = None
            self.canvas.draw_idle()
        self.clear_selection_btn.setEnabled(False)
