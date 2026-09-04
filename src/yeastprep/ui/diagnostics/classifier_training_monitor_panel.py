"""Right-side diagnostics for the Classifier Training page -- the "rich
information about the classification and various tasks" a training session
needs to be reviewable before its checkpoint gets deployed. Modeled on
`training_monitor_panel.TrainingMonitorPanel` (same loss-plot-over-log
layout for live progress), extended with two more tabs this task needs that
denoise training doesn't: a pooled-dataset summary (so a bad/empty pool is
obvious before a run is even started) and an embedding scatter (VICReg-only,
populated once via `tileclass.training.linear_probe.tsne_2d`/`knn_accuracy`
after a run finishes -- there's no per-epoch embedding to plot).
"""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.widgets import LassoSelector
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class ClassifierTrainingMonitorPanel(QWidget):
    # Emitted with the list of tile paths lasso-selected on the embedding
    # scatter, so the owning page can open them in a tile viewer.
    pointsSelected = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._epochs: list[int] = []
        self._losses: list[float] = []
        self._secondary: list[float] = []  # val_accuracy (supervised) -- unused for VICReg
        self._secondary_label = "val accuracy"

        self._embedding_xy = None
        self._embedding_paths = None
        self._lasso = None  # kept alive here -- LassoSelector drops its
        # event connections if its only reference is garbage collected
        self._selection_highlight = None  # scatter artist ringing selected points

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.tabs.addTab(self._build_progress_tab(), "Progress")
        self.tabs.addTab(self._build_dataset_tab(), "Dataset")
        self.tabs.addTab(self._build_embeddings_tab(), "Embeddings")

    # ------------------------------------------------------------------
    # Progress tab: loss (+ optional secondary metric) vs epoch, plus a log

    def _build_progress_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax_right = self.ax.twinx()
        self._reset_axes()

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.log_view)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        tab_layout.addWidget(splitter)
        return tab

    def _reset_axes(self, secondary_label: str | None = None):
        if secondary_label is not None:
            self._secondary_label = secondary_label
        self.ax.clear()
        self.ax_right.clear()
        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel("loss")
        self.ax_right.set_ylabel(self._secondary_label)

    def clear(self, secondary_label: str | None = None):
        self._epochs = []
        self._losses = []
        self._secondary = []
        self._reset_axes(secondary_label)
        self.canvas.draw_idle()
        self.log_view.clear()

    def append_supervised_epoch(self, progress) -> None:
        """`progress`: a `tileclass.training.supervised.TrainingProgress`.
        Its `epoch` restarts at 1 for each of the two training stages
        (probe, finetune), so the x-axis position used here is the running
        point index, not `progress.epoch` itself -- otherwise stage 2
        would overplot stage 1."""
        point = len(self._epochs) + 1
        self._epochs.append(point)
        self._losses.append(progress.avg_loss if progress.avg_loss is not None else 0.0)
        if progress.val_accuracy is not None:
            self._secondary.append(progress.val_accuracy)
        self._redraw()
        self.log_view.appendPlainText(
            f"[{progress.stage}] epoch {progress.epoch}/{progress.total_epochs}  "
            f"loss={progress.avg_loss:.4f}  val_acc={progress.val_accuracy:.3f}"
        )

    def append_vicreg_epoch(self, progress) -> None:
        """`progress`: a `tileclass.training.vicreg.VICRegProgress`. VICReg
        has no held-out accuracy metric per epoch (see
        `linear_probe`/`show_embedding_scatter` for the post-hoc
        equivalent), so the right axis instead tracks `std` -- collapsed
        embeddings (VICReg's classic failure mode) show up as `std`
        trending toward zero."""
        self._epochs.append(progress.epoch)
        self._losses.append(progress.avg_loss)
        std = progress.metrics.get("std")
        if std is not None:
            self._secondary.append(std)
        self._redraw()
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in progress.metrics.items())
        self.log_view.appendPlainText(
            f"epoch {progress.epoch}/{progress.total_epochs}  "
            f"loss={progress.avg_loss:.4f}  {metrics_str}"
        )

    def _redraw(self):
        self._reset_axes()
        (loss_line,) = self.ax.plot(self._epochs, self._losses, color="tab:blue", label="loss")
        lines = [loss_line]
        if self._secondary:
            (secondary_line,) = self.ax_right.plot(
                self._epochs[: len(self._secondary)],
                self._secondary,
                color="tab:orange",
                label=self._secondary_label,
            )
            lines.append(secondary_line)
        self.ax.legend(lines, [line.get_label() for line in lines], loc="upper right")
        self.canvas.draw_idle()

    def log(self, text: str):
        self.log_view.appendPlainText(text)

    # ------------------------------------------------------------------
    # Dataset tab: pooled category counts, confirmed vs AI-predicted,
    # singleton-category warnings -- reuses PooledAnnotations' own summary
    # methods directly rather than cross-importing tileclass's
    # AnnotationStatsPanel widget (see module docstring).

    def _build_dataset_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(4, 4, 4, 4)

        self.dataset_table = QTableWidget(0, 3)
        self.dataset_table.setHorizontalHeaderLabels(["Category", "Confirmed", "AI-predicted"])
        self.dataset_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.dataset_table.verticalHeader().setVisible(False)
        tab_layout.addWidget(self.dataset_table)
        return tab

    def set_dataset_summary(self, pooled) -> None:
        """`pooled`: a `tileclass.data.pooled_annotations.PooledAnnotations`
        built from the currently checked FOV folders. Flags a category with
        exactly one confirmed tile as a "singleton" the way
        `tileclass.widgets.annotation_stats_panel.AnnotationStatsPanel`
        does, since a singleton category can't be split into train/val and
        (for VICReg) can't form a same-category pair from two *different*
        crops."""
        confirmed = {}
        predicted = {}
        for _, category, confidence in pooled.tagged_items():
            counts = predicted if confidence is not None else confirmed
            counts[category] = counts.get(category, 0) + 1

        categories = sorted(set(confirmed) | set(predicted))
        self.dataset_table.setRowCount(len(categories))
        for row, category in enumerate(categories):
            n_confirmed = confirmed.get(category, 0)
            label = str(n_confirmed) + (" (singleton)" if n_confirmed == 1 else "")
            self.dataset_table.setItem(row, 0, QTableWidgetItem(category))
            self.dataset_table.setItem(row, 1, QTableWidgetItem(label))
            self.dataset_table.setItem(row, 2, QTableWidgetItem(str(predicted.get(category, 0))))

    # ------------------------------------------------------------------
    # Embeddings tab: VICReg-only, populated once after a run finishes.

    def _build_embeddings_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        self.embedding_figure = Figure(figsize=(5, 4), tight_layout=True)
        self.embedding_canvas = FigureCanvasQTAgg(self.embedding_figure)
        self.embedding_ax = self.embedding_figure.add_subplot(111)
        tab_layout.addWidget(self.embedding_canvas)

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
        tab_layout.addLayout(toolbar)
        return tab

    def show_embedding_scatter(
        self, xy, labels, paths: list[str] | None = None, knn_acc: float | None = None
    ) -> None:
        """`xy`: (N, 2) array from `tileclass.training.linear_probe.tsne_2d`.
        `labels`: length-N category names, colored by matplotlib's default
        cycle. `paths`: parallel length-N list of each point's tile path --
        when given, enables lasso-selecting a region of points and emitting
        their paths via `pointsSelected`, so a suspicious cluster (or an
        outlier sitting with the wrong category) can be checked against its
        actual image, not just its label. `knn_acc`: optional `knn_accuracy`
        result, shown in the title as a quick separability readout -- t-SNE
        and kNN are both local-neighborhood notions of separability, so the
        plot and the number tell a consistent story (see `tsne_2d`'s
        docstring for why this replaced a PCA projection). Axes are
        unitless/unlabeled by design: unlike PCA's PC1/PC2, t-SNE
        coordinates and inter-cluster distances aren't meaningful on their
        own -- only which points cluster together is."""
        self.embedding_ax.clear()
        for category in sorted(set(labels)):
            mask = [label == category for label in labels]
            points = xy[mask]
            self.embedding_ax.scatter(points[:, 0], points[:, 1], label=category, s=12)
        self.embedding_ax.set_xticks([])
        self.embedding_ax.set_yticks([])
        title = "Backbone embeddings (t-SNE)"
        if knn_acc is not None:
            title += f"  --  kNN accuracy: {knn_acc:.2f}"
        self.embedding_ax.set_title(title)
        self.embedding_ax.legend(loc="best", fontsize="small")

        self._embedding_xy = xy
        self._embedding_paths = paths
        self._selection_highlight = None  # ax.clear() above already dropped the artist
        self.clear_selection_btn.setEnabled(False)
        if self._lasso is not None:
            self._lasso.disconnect_events()
            self._lasso = None
        if paths is not None:
            self._lasso = LassoSelector(self.embedding_ax, onselect=self._on_lasso_select)

        self.embedding_canvas.draw_idle()

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
        self._selection_highlight = self.embedding_ax.scatter(
            points[:, 0],
            points[:, 1],
            s=80,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )
        self.clear_selection_btn.setEnabled(True)
        self.embedding_canvas.draw_idle()
        self.pointsSelected.emit(selected)

    def clear_selection(self) -> None:
        """Removes the ringed highlight left by the last lasso selection.
        Purely cosmetic -- any tile viewer windows already opened from that
        selection are unaffected."""
        if self._selection_highlight is not None:
            self._selection_highlight.remove()
            self._selection_highlight = None
            self.embedding_canvas.draw_idle()
        self.clear_selection_btn.setEnabled(False)
