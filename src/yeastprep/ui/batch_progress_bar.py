"""A visible per-page progress bar for batch pipeline runs. The sidebar's
stage-list bar (see main_window.py) is a compact always-visible sliver next
to the stage name; this renders the same `PageProgress` larger and in-page,
next to the button that started the run, so progress is easy to notice
without hunting for the sidebar.
"""

from qtpy.QtWidgets import QProgressBar

from .pages.page_progress import PageProgress


class BatchProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTextVisible(True)
        self.setVisible(False)

    def apply(self, progress: PageProgress):
        if not progress.active:
            self.setVisible(False)
            return
        self.setVisible(True)
        self.setRange(0, max(progress.total, 1))
        self.setValue(progress.done)
        self.setFormat(f"%v/%m  {progress.message}" if progress.message else "%v/%m")
