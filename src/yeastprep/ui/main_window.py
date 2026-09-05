"""YeastPrepWindow: the app shell. A single narrow left column stacks the
compact stage list (one row per pipeline stage, each showing a small
progress indicator for that stage's background work) on top of the
persistent `ProjectTreePanel` -- the single project folder picker +
checkable batch-selection tree shared by every page (see
project_tree_panel.py's module docstring) -- and the `SelectionActionsPanel`
below that, rather than putting the tree and page list in two side-by-side
columns, so the rest of the window is free for image previews. That column
drives a QStackedWidget holding the actual pages. Switching pages doesn't
stop a page's background worker: QStackedWidget only hides the widget, so
a batch run on a page you've navigated away from keeps going and keeps
updating its stage-list progress bar.

Pages read/write the project's folder tree only through `tree_panel` --
there's no per-page folder state, and no hand-wired "use X output"
buttons: every page just asks the shared tree which stage is currently the
active 2D source.

Clicking a file in the tree does not by itself load anything into a page.
It updates `selection_panel` with the list of tasks valid for that file
(`selection_actions.actions_for_selection`, e.g. a 01_reduced file offers
Denoise/Deconvolve/Segment/Preview, filtered to whatever the project's
current stage-resolution state actually supports) and clicking one of
those action buttons is what switches the stack to the owning page *and*
loads the file into it (`_on_action_triggered`) -- one click, and the
actions panel always shows exactly what a click on the currently-selected
item can do next.
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

from . import selection_actions, settings
from .common.stage_breadcrumb import PipelineBreadcrumb
from .pages.classifier_training_page import ClassifierTrainingPage
from .pages.classify_tiles_page import ClassifyTilesPage
from .pages.data_reduction_page import DataReductionPage
from .pages.deconvolve_page import DeconvolvePage
from .pages.denoise_page import DenoisePage
from .pages.page_progress import PageProgress
from .pages.preview_page import PreviewPage
from .pages.segmentation_page import SegmentationPage
from .pages.tile_generation_page import TileGenerationPage
from .project_tree_panel import ProjectTreePanel
from .selection_actions_panel import SelectionActionsPanel


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
        self.classifier_training_page = ClassifierTrainingPage(self.tree_panel)
        self.classify_tiles_page = ClassifyTilesPage(self.tree_panel)
        self._pages = [
            ("Data Reduction", self.data_reduction_page),
            ("Preview", self.preview_page),
            ("Denoise", self.denoise_page),
            ("Deconvolve", self.deconvolve_page),
            ("Segmentation", self.segmentation_page),
            ("Tile Generation", self.tile_generation_page),
            ("Classifier Training", self.classifier_training_page),
            ("Classify Tiles", self.classify_tiles_page),
        ]
        # page_key (see selection_actions.py) -> page instance, used by
        # _on_action_triggered to route a SelectionActionsPanel button
        # click to the page it names.
        self._page_by_key = {
            "data_reduction": self.data_reduction_page,
            "preview": self.preview_page,
            "denoise": self.denoise_page,
            "deconvolve": self.deconvolve_page,
            "segmentation": self.segmentation_page,
            "tile_generation": self.tile_generation_page,
            "classifier_training": self.classifier_training_page,
            "classify_tiles": self.classify_tiles_page,
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

        self.selection_panel = SelectionActionsPanel()
        sidebar_layout.addWidget(self.selection_panel)

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
        self.tree_panel.file_selected.connect(self._on_tree_selection)
        self.selection_panel.action_triggered.connect(self._on_action_triggered)
        self.classifier_training_page.checkpointTrained.connect(
            self.classify_tiles_page.set_default_checkpoint
        )

        for _name, page in self._pages:
            page.progress_changed.connect(
                lambda p, page=page: self._on_progress_changed(page, p)
            )

    def _on_tree_selection(self, stage: str, path: str):
        actions = selection_actions.actions_for_selection(stage, path, self.tree_panel)
        self.selection_panel.set_selection(stage, path, actions)

    def _on_action_triggered(self, page_key: str, stage: str, path: str, mode: str):
        page = self._page_by_key.get(page_key)
        if page is None:
            return
        index = self.stack.indexOf(page)
        if index >= 0:
            self.page_list.setCurrentRow(index)
        page.load_selection(stage, path, mode)

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
        self.tree_panel.shutdown()
        super().closeEvent(event)
