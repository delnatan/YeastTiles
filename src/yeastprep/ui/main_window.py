"""YeastPrepWindow: the app shell. A single narrow left column stacks the
compact stage list (one row per pipeline stage, each showing a small
progress indicator for that stage's background work) on top of the
persistent `ProjectTreePanel` -- the single project folder picker +
checkable batch-selection tree shared by every page (see
project_tree_panel.py's module docstring) -- rather than putting them in
two side-by-side columns, so the rest of the window is free for image
previews. That column drives a QStackedWidget holding the actual pages.
Switching pages doesn't stop a page's background worker: QStackedWidget
only hides the widget, so a batch run on a page you've navigated away from
keeps going and keeps updating its stage-list progress bar.

Pages read/write the project's folder tree only through `tree_panel` --
there's no per-page folder state, and no hand-wired "use X output"
buttons: every page just asks the shared tree which stage is currently the
active 2D source.

Clicking a file in the tree also switches the stack to whichever page owns
that stage (`_stage_pages`) -- raw files go to Data Reduction, since the
rawstack viewer there already doubles as a passive viewer for that stage.
01_reduced/02_denoised/03_deconvolved files all go to `PreviewPage`
instead of their respective processing pages (Denoise, Deconvolve): every
file under those stages already exists on disk by the time it's clickable,
so the click means "show me what this is," not "take me to the page that
(re)produces it" -- that page's params/batch/live-recompute UI is one more
click away in the sidebar for when reprocessing really is the intent.
"""

from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core import project as project_core

from . import settings
from .common.stage_breadcrumb import PipelineBreadcrumb
from .pages.data_reduction_page import DataReductionPage
from .pages.deconvolve_page import DeconvolvePage
from .pages.denoise_page import DenoisePage
from .pages.page_progress import PageProgress
from .pages.preview_page import PreviewPage
from .pages.segmentation_page import SegmentationPage
from .pages.tile_generation_page import TileGenerationPage
from .project_tree_panel import RAW_STAGE, ProjectTreePanel


class YeastPrepWindow(QMainWindow):
    def __init__(self, initial_input_folder: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("yeastprep")
        self.resize(1600, 950)

        self.tree_panel = ProjectTreePanel()

        self.data_reduction_page = DataReductionPage(self.tree_panel)
        self.preview_page = PreviewPage(self.tree_panel)
        self.denoise_page = DenoisePage(self.tree_panel)
        self.deconvolve_page = DeconvolvePage(self.tree_panel)
        self.segmentation_page = SegmentationPage(self.tree_panel)
        self.tile_generation_page = TileGenerationPage(self.tree_panel)
        self._pages = [
            ("Data Reduction", self.data_reduction_page),
            ("Preview", self.preview_page),
            ("Denoise", self.denoise_page),
            ("Deconvolve", self.deconvolve_page),
            ("Segmentation", self.segmentation_page),
            ("Tile Generation", self.tile_generation_page),
        ]
        # Tree click -> auto-navigate target, per stage. Denoise/Deconvolve
        # are deliberately absent here even though they "own" 02_denoised/
        # 03_deconvolved -- see this module's docstring -- so a click always
        # lands on a passive viewer; those pages are reachable from the
        # sidebar list instead.
        self._stage_pages = {
            RAW_STAGE: self.data_reduction_page,
            project_core.STAGE_REDUCED: self.preview_page,
            project_core.STAGE_DENOISED: self.preview_page,
            project_core.STAGE_DECONVOLVED: self.preview_page,
        }

        self._build_ui()
        self._wire_up()

        settings.restore_window_geometry(self)

        if initial_input_folder:
            self.tree_panel.set_project_root(initial_input_folder)

    # ------------------------------------------------------------------
    # UI construction

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # Spans the full window width, above the sidebar/stack split, so the
        # linear-workflow guide stays visible no matter which page is
        # showing -- previously it only lived inside Data Reduction's own
        # layout and vanished on every other page.
        self.breadcrumb = PipelineBreadcrumb(self.tree_panel)
        outer.addWidget(self.breadcrumb)

        layout = QHBoxLayout()
        outer.addLayout(layout, 1)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar.setMinimumWidth(320)
        sidebar.setMaximumWidth(420)
        layout.addWidget(sidebar)

        self.page_list = QListWidget()
        sidebar_layout.addWidget(self.page_list)

        sidebar_layout.addWidget(self.tree_panel, 1)

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

        # The stage list only ever holds a handful of short rows -- cap its
        # height to what those rows actually need so the tree panel below
        # it (a much longer, more useful list) gets the rest of the column.
        row_height = self.page_list.sizeHintForRow(0)
        frame = 2 * self.page_list.frameWidth()
        self.page_list.setMaximumHeight(row_height * len(self._pages) + frame + 4)

        self.page_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # Wiring

    def _wire_up(self):
        self.page_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.tree_panel.file_selected.connect(self._on_tree_file_selected)

        for _name, page in self._pages:
            page.progress_changed.connect(
                lambda p, page=page: self._on_progress_changed(page, p)
            )

    def _on_tree_file_selected(self, stage: str, _path: str):
        page = self._stage_pages.get(stage)
        if page is None:
            return
        index = self.stack.indexOf(page)
        if index >= 0:
            self.page_list.setCurrentRow(index)

    def _on_progress_changed(self, page, progress: PageProgress):
        bar = self._progress_bars[id(page)]
        if not progress.active:
            bar.setVisible(False)
            return
        bar.setVisible(True)
        bar.setRange(0, max(progress.total, 1))
        bar.setValue(progress.done)
        bar.setToolTip(progress.message)

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        settings.save_window_geometry(self)
        for _name, page in self._pages:
            page.shutdown()
        super().closeEvent(event)
