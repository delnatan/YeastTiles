"""One stage page's batch run: the thread/worker lifecycle every batch page
shares, plus the in-page progress bar and a Cancel button next to it.

Each page used to carry its own copy of this plumbing, and each copy had
the same gaps: no way to cancel, nothing stopping a second run from being
started on top of the first (whose QThread then got garbage-collected
while still running, which kills the app), and no record of which project
a run belonged to. Pages now only build the worker and react to
`file_result`/`finished`.

`BatchRunner` is the widget pages place in their action row, where the
plain `BatchProgressBar` used to go. It stays hidden while idle.
"""

from dataclasses import dataclass

from qtpy.QtCore import QCoreApplication, QThread, Signal
from qtpy.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QWidget

from ..batch_progress_bar import BatchProgressBar
from ..pages.page_progress import PageProgress


@dataclass
class BatchOutcome:
    label: str  # e.g. "Batch denoise"
    total: int
    done: int = 0
    error: str | None = None  # a failure that stopped the whole run
    cancelled: bool = False

    def summary(self) -> str:
        if self.error:
            return f"{self.label} stopped: {self.error}"
        if self.cancelled:
            return f"{self.label} cancelled after {self.done} of {self.total} file(s)."
        return f"{self.label} complete ({self.total} file(s))."


class BatchRunner(QWidget):
    progress_changed = Signal(object)  # PageProgress, for the page's nav chip
    file_result = Signal(object)  # the stage's per-file result dataclass
    finished = Signal(object)  # BatchOutcome -- always fires, once per run

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self._label = label
        self._thread = None
        self._worker = None
        self._outcome = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.progress_bar = BatchProgressBar()
        layout.addWidget(self.progress_bar, 1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Stop after the file currently being processed.")
        self.cancel_btn.clicked.connect(self.cancel)
        self.cancel_btn.setVisible(False)
        layout.addWidget(self.cancel_btn)

    def is_running(self) -> bool:
        return self._thread is not None

    def describe_running(self) -> str | None:
        """e.g. "Batch denoise (3/40)", or None when idle -- for the
        quit-while-running prompt."""
        if self._outcome is None:
            return None
        return f"{self._label} ({self._outcome.done}/{self._outcome.total})"

    def start(self, worker):
        """Run a `ui.worker.BatchWorker` on a thread of its own. Callers
        check `is_running()` first; a second concurrent run is refused."""
        if self.is_running():
            raise RuntimeError(f"{self._label} is already running")
        self._outcome = BatchOutcome(label=self._label, total=worker.total)
        self._worker = worker
        self._thread = QThread()
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.file_result.connect(self.file_result)
        worker.error.connect(self._on_error)
        worker.cancelled.connect(self._on_cancelled)
        worker.finished.connect(self._on_finished)

        self.cancel_btn.setText("Cancel")
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setVisible(True)
        self._apply(PageProgress(active=True, done=0, total=worker.total))
        self._thread.start()

    def cancel(self):
        if self._worker is None:
            return
        self._worker.cancel()
        self.cancel_btn.setText("Cancelling...")
        self.cancel_btn.setEnabled(False)

    def shutdown(self):
        """Cancel and wait for the current file to finish -- for app exit.
        The worker's last signals are queued for this thread, whose event
        loop is about to stop, so they're delivered here: the page's
        `finished` handler still records what the run got done."""
        if self._thread is None:
            return
        self.cancel()
        self._thread.quit()
        self._thread.wait()
        QCoreApplication.sendPostedEvents()

    # ------------------------------------------------------------------

    def _apply(self, progress: PageProgress):
        self.progress_bar.apply(progress)
        self.progress_changed.emit(progress)

    def _on_progress(self, done: int, total: int, name: str):
        self._outcome.done = done
        self._apply(PageProgress(active=True, done=done, total=total, message=name))

    def _on_error(self, message: str):
        self._outcome.error = message
        QMessageBox.warning(self, "yeastprep", f"{self._label} stopped: {message}")

    def _on_cancelled(self):
        self._outcome.cancelled = True

    def _on_finished(self):
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None
        outcome, self._outcome = self._outcome, None
        self.cancel_btn.setVisible(False)
        self._apply(PageProgress(active=False))
        self.finished.emit(outcome)
