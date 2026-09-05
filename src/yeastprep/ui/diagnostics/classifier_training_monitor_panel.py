"""Right-side diagnostics for the Classifier Training page -- the "rich
information about the classification and various tasks" a training session
needs to be reviewable before its checkpoint gets deployed. Modeled on
`training_monitor_panel.TrainingMonitorPanel` (same loss-plot-over-log
layout for live progress), extended with one more tab this task needs that
denoise training doesn't: a pooled-dataset summary (so a bad/empty pool is
obvious before a run is even started). The embedding scatter that used to
live in a third tab here has moved to the Classify Tiles page's "Explore
Embeddings" group (`diagnostics.embedding_scatter_widget.EmbeddingScatterWidget`)
-- training-time hyperparameters/progress and inference-time exploration are
deliberately separate pages now.
"""

from qtpy.QtWidgets import (
    QHeaderView,
    QPlainTextEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtCore import Qt, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class ClassifierTrainingMonitorPanel(QWidget):
    # Emitted whenever the Dataset tab becomes the visible one -- annotations
    # can change on disk from outside this page entirely (a separately
    # opened tile viewer, another yeastprep window), so the owning page uses
    # this to pull a fresh summary rather than relying only on pool changes.
    datasetTabActivated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._epochs: list[int] = []
        self._losses: list[float] = []
        self._secondary: list[float] = []  # val_accuracy (supervised) -- unused for VICReg
        self._secondary_label = "val accuracy"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.tabs.addTab(self._build_progress_tab(), "Progress")
        self._dataset_tab_index = self.tabs.addTab(self._build_dataset_tab(), "Dataset")
        self.tabs.currentChanged.connect(self._on_current_tab_changed)

    def _on_current_tab_changed(self, index: int) -> None:
        if index == self._dataset_tab_index:
            self.datasetTabActivated.emit()

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
        has no held-out accuracy metric per epoch (see the Classify Tiles
        page's "Explore Embeddings" group for the post-hoc equivalent), so
        the right axis instead tracks `std` -- collapsed
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
    # Dataset tab: pooled human-confirmed category counts, singleton
    # -category warnings -- reuses PooledAnnotations' own summary methods
    # directly rather than cross-importing tileclass's AnnotationStatsPanel
    # widget (see module docstring). Confirmed-only: what a training run
    # actually draws on. AI-predicted counts (a classify-time concern) live
    # on the Classify Tiles page's own results summary instead.

    def _build_dataset_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(4, 4, 4, 4)

        self.dataset_table = QTableWidget(0, 2)
        self.dataset_table.setHorizontalHeaderLabels(["Category", "Confirmed"])
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
        for _, category, confidence in pooled.tagged_items():
            if confidence is None:
                confirmed[category] = confirmed.get(category, 0) + 1

        categories = sorted(confirmed)
        self.dataset_table.setRowCount(len(categories))
        for row, category in enumerate(categories):
            n_confirmed = confirmed[category]
            label = str(n_confirmed) + (" (singleton)" if n_confirmed == 1 else "")
            self.dataset_table.setItem(row, 0, QTableWidgetItem(category))
            self.dataset_table.setItem(row, 1, QTableWidgetItem(label))
