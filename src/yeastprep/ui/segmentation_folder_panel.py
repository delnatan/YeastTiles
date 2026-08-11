"""2D-images folder selection for the Segmentation stage.

Deliberately independent of FolderPanel's raw-input/output pair: per
design.md, the pipeline has a hard stage boundary between 3D->2D data
reduction (raw stacks -> combined-channel 2D tiffs) and 2D processing
(those tiffs -> masks -> tiles). Segmentation should work on any folder
of already-produced tiffs -- including ones from a previous session, or
copied in from elsewhere -- without requiring a raw-input folder to ever
be loaded in this app instance.

Each tiff here holds both channels from stage 1 (brightfield + target,
see core/pipeline.py's `combine_channels`) even though segmentation only
ever reads the brightfield one -- one file per FOV, not two matched by
filename stem. This folder doubles as both the read source for those
tiffs and the write destination for cellpose's `_seg.npy` sidecars:
design.md requires masks to live next to the images they came from so the
Cellpose GUI can open the folder directly for correction (and its own
channel picker handles the brightfield/target split when it opens these
same multi-channel files), so there's no separate output field to
insulate here.
"""

from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yeastprep.ui import settings


class SegmentationFolderPanel(QWidget):
    """Emits `folder_changed(str)` whenever the resolved, existing folder
    changes (not on every keystroke -- only on Browse/recent-pick/
    programmatic set), matching FolderPanel's convention."""

    folder_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._folder = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        group = QGroupBox("2D images folder (brightfield + target, combined)")
        v = QVBoxLayout(group)

        self.recent_combo = QComboBox()
        self.recent_combo.addItem("Recent folders...")
        self.recent_combo.addItems(settings.get_recent_segmentation_dirs())
        self.recent_combo.activated.connect(self._on_recent_picked)
        v.addWidget(self.recent_combo)

        row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setPlaceholderText("No folder selected")
        row.addWidget(self.folder_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        row.addWidget(browse_btn)
        v.addLayout(row)

        self.status_label = QLabel(" ")
        v.addWidget(self.status_label)

        layout.addWidget(group)

    # ------------------------------------------------------------------

    def _browse(self):
        start_dir = self._folder or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Select 2D images folder", start_dir)
        if chosen:
            self.set_folder(chosen)

    def _on_recent_picked(self, index):
        if index <= 0:
            return
        self.set_folder(self.recent_combo.itemText(index))
        self.recent_combo.setCurrentIndex(0)

    def set_folder(self, folder: str):
        folder = str(folder)
        self._folder = folder
        self.folder_edit.setText(folder)
        self._revalidate()
        settings.add_recent_segmentation_dir(folder)
        self._refresh_recent()
        self.folder_changed.emit(folder)

    def _revalidate(self):
        folder = self._folder
        if not folder or not Path(folder).is_dir():
            self.status_label.setText("No folder selected")
            self.status_label.setStyleSheet("color: #999;")
            return
        n = len(list(Path(folder).glob("*.tiff")))
        if n:
            self.status_label.setText(f"{n} tiff(s) found")
            self.status_label.setStyleSheet("color: #4caf50;")
        else:
            self.status_label.setText("No '*.tiff' files found here")
            self.status_label.setStyleSheet("color: #ff5c5c;")

    def _refresh_recent(self):
        self.recent_combo.blockSignals(True)
        self.recent_combo.clear()
        self.recent_combo.addItem("Recent folders...")
        self.recent_combo.addItems(settings.get_recent_segmentation_dirs())
        self.recent_combo.blockSignals(False)

    def folder(self) -> str:
        return self._folder
