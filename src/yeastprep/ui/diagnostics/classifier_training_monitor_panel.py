"""Right-side diagnostics for the Classifier Training page: live loss
(+ validation accuracy) plots over a log, and a Dataset tab with the
pool's per-category counts, so a bad or empty pool is obvious before a
run starts. Only human-annotated tiles are trained on.
"""

from qtpy.QtWidgets import (
    QPlainTextEdit,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from tileclass.widgets.annotation_stats_panel import AnnotationStatsPanel
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
        self._secondary: list[float] = []  # val_accuracy (supervised) / std (VICReg)
        # VICReg's kNN accuracy: scored only every few epochs, so it keeps
        # its own x positions rather than sharing `_epochs`.
        self._knn_epochs: list[int] = []
        self._knn: list[float] = []
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
        # `Axes.clear()` resets a twin axis's tick/label side back to the
        # default left, undoing what `ax.twinx()` set up -- re-apply it here
        # since this runs on every redraw (see `_redraw`), not just once at
        # construction.
        self.ax_right.yaxis.tick_right()
        self.ax_right.yaxis.set_label_position("right")
        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel("loss")
        self.ax_right.set_ylabel(
            self._secondary_label + (" / kNN acc" if self._knn else "")
        )

    def clear(self, secondary_label: str | None = None):
        self._epochs = []
        self._losses = []
        self._secondary = []
        self._knn_epochs = []
        self._knn = []
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
        """`progress`: a `tileclass.training.vicreg.VICRegProgress`. The
        right axis tracks two things: the `std` loss term -- it's the
        hinge on per-dimension spread, so it rising rather than settling
        near zero is the sign of collapsing embeddings (VICReg's classic
        failure mode) -- and, on the epochs it was scored, the kNN
        accuracy on the annotated tiles (the same score as Explore
        Embeddings), which decides which epoch gets saved."""
        self._epochs.append(progress.epoch)
        self._losses.append(progress.avg_loss)
        std = progress.metrics.get("std")
        if std is not None:
            self._secondary.append(std)
        knn = getattr(progress, "knn_accuracy", None)
        if knn is not None:
            self._knn_epochs.append(progress.epoch)
            self._knn.append(knn)
        self._redraw()
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in progress.metrics.items())
        line = (
            f"epoch {progress.epoch}/{progress.total_epochs}  "
            f"loss={progress.avg_loss:.4f}  {metrics_str}  lr={progress.lr:.2e}"
        )
        if knn is not None:
            line += f"  kNN={knn:.3f}"
        self.log_view.appendPlainText(line)

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
        if self._knn:
            (knn_line,) = self.ax_right.plot(
                self._knn_epochs,
                self._knn,
                color="tab:green",
                marker="o",
                markersize=3,
                label="kNN acc",
            )
            lines.append(knn_line)
        self.ax.legend(lines, [line.get_label() for line in lines], loc="upper right")
        self.canvas.draw_idle()

    def log(self, text: str):
        self.log_view.appendPlainText(text)

    # ------------------------------------------------------------------
    # Dataset tab

    def _build_dataset_tab(self) -> QWidget:
        self.dataset_stats = AnnotationStatsPanel()
        return self.dataset_stats

    def set_dataset_summary(self, summary) -> None:
        """`summary`: the checked FOVs' `pool_data.PoolSummary`."""
        self.dataset_stats.show_counts(
            summary.counts, summary.human_counts, summary.untagged, summary.vocabulary
        )
