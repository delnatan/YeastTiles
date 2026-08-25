"""Small persistent label naming whichever file is currently loaded on a
page -- i.e. what a 'Do it' click or an auto-recompute would act on. That's
set only when a `SelectionActionsPanel` action button is clicked (see
`ui/selection_actions_panel.py`), which is deliberately independent of
whichever tree item is merely *highlighted* by a click (that only updates
the actions panel, not any page) and of which files are *checked* for
batch processing -- easy to conflate, since all three live on/near the
same tree. Shared by Data Reduction/Denoise/Deconvolve/Segmentation/Tile
Generation so the wording is consistent everywhere it appears.
"""

from pathlib import Path

from qtpy.QtWidgets import QLabel

_NO_FILE_TEXT = "Loaded: none -- click a file in the tree, then an action below it"


class PreviewSourceLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(_NO_FILE_TEXT, parent)
        self.setStyleSheet("font-style: italic; color: #a0a0a0;")

    def set_path(self, path):
        self.setText(f"Loaded: {Path(path).name}")

    def clear_path(self):
        self.setText(_NO_FILE_TEXT)
