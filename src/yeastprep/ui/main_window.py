"""YeastPrepWindow: the page-list shell hosting each pipeline stage as an
independent page (see ui/pages/) -- a QListWidget sidebar (one row per
stage, each showing a small progress indicator for that stage's
background work) driving a QStackedWidget. Switching pages doesn't stop a
page's background worker: QStackedWidget only hides the widget, so a
batch run on a page you've navigated away from keeps going and keeps
updating its sidebar progress bar.

Pages don't know about each other (see pages/__init__.py) -- any wiring
that legitimately needs both, like the Segmentation page's "use the
Data-Reduction output folder" shortcut, is connected here, the one place
that holds references to all of them.
"""

from pathlib import Path

from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStackedWidget,
    QWidget,
)

from . import settings
from .pages.data_reduction_page import DataReductionPage
from .pages.page_progress import PageProgress
from .pages.segmentation_page import SegmentationPage
from .pages.tile_generation_page import TileGenerationPage


class YeastPrepWindow(QMainWindow):
    def __init__(self, initial_input_folder: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("yeastprep")
        self.resize(1400, 900)

        self.data_reduction_page = DataReductionPage()
        self.segmentation_page = SegmentationPage()
        self.tile_generation_page = TileGenerationPage()
        self._pages = [
            ("Data Reduction", self.data_reduction_page),
            ("Segmentation", self.segmentation_page),
            ("Tile Generation", self.tile_generation_page),
        ]

        self._build_ui()
        self._wire_up()

        settings.restore_window_geometry(self)

        if initial_input_folder:
            self.data_reduction_page.folder_panel.set_input_folder(initial_input_folder)

    # ------------------------------------------------------------------
    # UI construction

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.page_list = QListWidget()
        self.page_list.setFixedWidth(180)
        layout.addWidget(self.page_list)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)

        self._progress_bars = {}
        for name, page in self._pages:
            item = QListWidgetItem()
            self.page_list.addItem(item)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 6, 8, 6)
            row_layout.addWidget(QLabel(name), 1)

            progress = QProgressBar()
            progress.setFixedWidth(50)
            progress.setFixedHeight(14)
            progress.setTextVisible(False)
            progress.setVisible(False)
            row_layout.addWidget(progress)

            item.setSizeHint(row.sizeHint())
            self.page_list.setItemWidget(item, row)
            self._progress_bars[id(page)] = progress

            self.stack.addWidget(page)

        self.page_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # Wiring

    def _wire_up(self):
        self.page_list.currentRowChanged.connect(self.stack.setCurrentIndex)

        self.data_reduction_page.progress_changed.connect(
            lambda p: self._on_progress_changed(self.data_reduction_page, p)
        )
        self.segmentation_page.progress_changed.connect(
            lambda p: self._on_progress_changed(self.segmentation_page, p)
        )
        self.tile_generation_page.progress_changed.connect(
            lambda p: self._on_progress_changed(self.tile_generation_page, p)
        )

        self.segmentation_page.use_flatten_output_btn.clicked.connect(
            self._use_flatten_output_as_segmentation_folder
        )
        self.tile_generation_page.use_segmentation_folder_btn.clicked.connect(
            self._use_segmentation_folder_as_tile_input
        )

    def _on_progress_changed(self, page, progress: PageProgress):
        bar = self._progress_bars[id(page)]
        if not progress.active:
            bar.setVisible(False)
            return
        bar.setVisible(True)
        bar.setRange(0, max(progress.total, 1))
        bar.setValue(progress.done)
        bar.setToolTip(progress.message)

    def _use_flatten_output_as_segmentation_folder(self):
        output_folder = self.data_reduction_page.folder_panel.output_folder()
        if not output_folder:
            QMessageBox.warning(
                self, "yeastprep", "Select an output folder on the Data Reduction page first."
            )
            return
        self.segmentation_page.segmentation_folder_panel.set_folder(
            str(Path(output_folder) / "processed")
        )

    def _use_segmentation_folder_as_tile_input(self):
        folder = self.segmentation_page.segmentation_folder_panel.folder()
        if not folder:
            QMessageBox.warning(
                self, "yeastprep", "Select a 2D images folder on the Segmentation page first."
            )
            return
        self.tile_generation_page.tile_folder_panel.set_input_folder(folder)

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        settings.save_window_geometry(self)
        self.data_reduction_page.shutdown()
        self.segmentation_page.shutdown()
        self.tile_generation_page.shutdown()
        super().closeEvent(event)
