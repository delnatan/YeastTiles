"""Keep a widget's contents in step with something that changes often,
without doing the work until it's wanted.

    self.lazy = LazyRefresh(
        self.stats_panel,
        capture=pool.snapshot,        # GUI thread: read what the job needs
        compute=summarize,            # worker thread: slow, no widgets
        apply=self.stats_panel.show,  # GUI thread: show the result
    )
    pool.changed.connect(self.lazy.invalidate)

`invalidate()` only notes that the widget is out of date. The job runs once
the widget is on screen, a moment after the last invalidation (so ten
checkbox clicks in a row make one run), and a run overtaken by a newer
invalidation is dropped rather than shown. A widget on a tab nobody opens
never triggers any work.
"""

from qtpy.QtCore import QEvent, QObject, QTimer, Signal

from .background_task import BackgroundTask


class LazyRefresh(QObject):
    busyChanged = Signal(bool)  # a run started / finished

    def __init__(self, widget, capture, compute, apply, delay_ms=250, parent=None):
        super().__init__(parent or widget)
        self._widget = widget
        self._capture = capture
        self._compute = compute
        self._apply = apply
        self._dirty = True  # never filled yet
        self._task = BackgroundTask(self)
        self._task.succeeded.connect(self._on_done)
        self._task.failed.connect(self._on_failed)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(delay_ms)
        self._timer.timeout.connect(self._run)
        widget.installEventFilter(self)

    def invalidate(self, *_args):
        """The widget's data may have changed; refresh it when it's visible."""
        self._dirty = True
        if self._widget.isVisible():
            self._timer.start()  # restarts a pending one

    def eventFilter(self, obj, event):
        if obj is self._widget and event.type() == QEvent.Show and self._dirty:
            self._timer.start()
        return False

    def shutdown(self):
        self._timer.stop()
        self._task.shutdown()

    # ------------------------------------------------------------------

    def _run(self):
        if self._task.is_running() or not self._dirty or not self._widget.isVisible():
            return  # _on_done picks up whatever changed meanwhile
        self._dirty = False
        arg = self._capture()
        self.busyChanged.emit(True)
        self._task.start(lambda: self._compute(arg))

    def _on_done(self, result):
        if self._dirty:
            # Overtaken while running: this result is stale. Go again if
            # anyone is looking; otherwise the next Show does.
            if self._widget.isVisible():
                self._timer.start()
            else:
                self.busyChanged.emit(False)
            return
        self.busyChanged.emit(False)
        self._apply(result)

    def _on_failed(self, _message: str):
        # BackgroundTask has printed the traceback; keep what's on screen.
        self.busyChanged.emit(False)
