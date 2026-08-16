"""Small persistent label naming whichever file is currently driving a
page's live preview -- i.e. what a 'Do it' click or an auto-recompute would
act on. That's always the tree item currently *selected* (clicked/
highlighted), which is independent of which files are *checked* for batch
processing -- easy to conflate, since both live in the same tree. Shared by
Data Reduction/Denoise/Deconvolve/Segmentation/Tile Generation so the
wording is consistent everywhere it appears.
"""

from pathlib import Path

from qtpy.QtWidgets import QLabel

_NO_FILE_TEXT = "Previewing: (no file selected -- click a file in the tree)"


class PreviewSourceLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(_NO_FILE_TEXT, parent)
        self.setStyleSheet("font-style: italic; color: #a0a0a0;")

    def set_path(self, path):
        self.setText(f"Previewing: {Path(path).name}")

    def clear_path(self):
        self.setText(_NO_FILE_TEXT)
