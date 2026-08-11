"""Non-modal progress dialog for a background classifier training run:
a coarse progress bar (fraction of total epochs across both the probe and
fine-tune stages) plus a scrolling per-epoch loss/val-accuracy log.

Cancel and Close are independent: Cancel asks the worker to stop (checked
between epochs, not preemptive -- see workers.TrainingWorker), Close just
hides the dialog while training keeps running in the background --
main_window.py holds the worker/thread reference and re-shows this same
dialog if "Train Classifier..." is invoked again while one is in flight.
"""

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QVBoxLayout,
)


class TrainingProgressDialog(QDialog):
    cancel_requested = Signal()

    def __init__(self, total_epochs: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Train Classifier")
        self.resize(480, 360)
        self._total_epochs = total_epochs
        self._done_epochs = 0

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Starting...")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, max(total_epochs, 1))
        layout.addWidget(self.progress_bar)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        buttons = QDialogButtonBox()
        self.cancel_btn = buttons.addButton("Cancel", QDialogButtonBox.ActionRole)
        self.cancel_btn.clicked.connect(self.cancel_requested)
        self.close_btn = buttons.addButton("Close", QDialogButtonBox.RejectRole)
        self.close_btn.clicked.connect(self.hide)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------

    def on_progress(self, progress):
        self._done_epochs += 1
        self.progress_bar.setValue(min(self._done_epochs, self._total_epochs))
        line = f"[{progress.stage}] epoch {progress.epoch}/{progress.total_epochs}"
        if progress.avg_loss is not None:
            line += f"  loss={progress.avg_loss:.4f}"
        if progress.val_accuracy is not None:
            line += f"  val_acc={progress.val_accuracy:.3f}"
        self.log.appendPlainText(line)
        self.status_label.setText(line)

    def on_finished(self, result):
        self.progress_bar.setValue(self._total_epochs)
        self.status_label.setText(f"Done -- val accuracy {result.val_accuracy:.3f}")
        per_class = ", ".join(
            f"{name}={acc:.2f}" for name, acc in result.per_class_accuracy.items()
        )
        self.log.appendPlainText(
            f"\nTraining complete: {result.train_count} train / {result.val_count} val tiles.\n"
            f"Val accuracy: {result.val_accuracy:.3f}\n"
            f"Per-category: {per_class}\n"
            f"Saved to {result.weights_path}"
        )
        self.cancel_btn.setEnabled(False)

    def on_error(self, message):
        self.status_label.setText("Failed")
        self.log.appendPlainText(f"\nTraining failed: {message}")
        self.cancel_btn.setEnabled(False)

    def on_cancelled(self):
        self.status_label.setText("Cancelled")
        self.log.appendPlainText("\nTraining cancelled -- no weights were saved.")
        self.cancel_btn.setEnabled(False)
