"""Run one slow function off the GUI thread and hear back on it.

For one-shot jobs that used to freeze the window while they ran (a
backbone forward pass plus UMAP, writing a folder's worth of brightfield
copies). Unlike `BatchRunner` there's no per-item progress or cancel --
just a result, or an error message.

    self.task = BackgroundTask(self)
    self.task.succeeded.connect(self._on_done)   # receives fn's return value
    self.task.failed.connect(self._on_failed)    # receives "message"
    self.task.start(lambda: slow_thing(inputs))

`fn` runs on another thread: it must not touch widgets, and should get
everything it needs from the GUI up front (as arguments bound into it).
"""

import traceback

from qtpy.QtCore import QObject, QThread, Signal


class _Call(QObject):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            result = self._fn()
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc) or type(exc).__name__)
            return
        self.done.emit(result)


class BackgroundTask(QObject):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._call = None

    def is_running(self) -> bool:
        return self._thread is not None

    def start(self, fn):
        if self.is_running():
            raise RuntimeError("a background task is already running")
        self._call = _Call(fn)
        self._thread = QThread()
        self._call.moveToThread(self._thread)
        self._thread.started.connect(self._call.run)
        self._call.done.connect(self._on_done)
        self._call.failed.connect(self._on_failed)
        self._thread.start()

    def shutdown(self):
        """Wait for a running call to return (it can't be interrupted) --
        for app exit, where destroying a running QThread would crash. Its
        result is dropped."""
        if self._thread is None:
            return
        self._call.done.disconnect()
        self._call.failed.disconnect()
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._call = None

    def _finish(self):
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._call = None

    def _on_done(self, result):
        self._finish()
        self.succeeded.emit(result)

    def _on_failed(self, message: str):
        self._finish()
        self.failed.emit(message)
