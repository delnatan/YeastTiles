"""Background QThread worker(s) for tileclass. Currently just
`TrainingWorker`, wrapping `training.supervised.train_classifier` --
mirrors yeastprep's `ui/worker.py` batch-worker pattern (progress/
finished/error signals, a `cancel()` flag checked between epochs rather
than anything preemptive).
"""

from qtpy.QtCore import QObject, Signal

from .training.supervised import TrainingCancelled, TrainingParams, train_classifier


class TrainingWorker(QObject):
    progress = Signal(object)  # TrainingProgress
    finished = Signal(object)  # TrainingResult
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, records, params: TrainingParams = TrainingParams()):
        super().__init__()
        self._records = list(records)
        self._params = params
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            result = train_classifier(
                self._records,
                params=self._params,
                progress_callback=self.progress.emit,
                cancel_check=lambda: self._cancel_requested,
            )
        except TrainingCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.finished.emit(result)
