"""YeastPrepWindow: the app shell. Two ways to reach a page:

- the navigation strip across the top (`PipelineBreadcrumb`): one chip per
  pipeline stage plus the classification pages. Clicking a chip just shows
  that page; the strip also underlines the current page and shows each
  page's background-run progress. Switching pages doesn't stop a page's
  worker -- QStackedWidget only hides the widget -- so a batch run on a page
  you've navigated away from keeps going and keeps reporting to its chip.
- the `SelectionActionsPanel` under the project tree: the tasks valid for
  the selected tree item, each of which shows its page *and* loads the file
  into it.

The left column (`ProjectTreePanel` -- the single project folder picker +
checkable batch-selection tree shared by every page, see
project_tree_panel.py's module docstring -- over the `SelectionActionsPanel`)
sits in a QSplitter (`main_splitter`) against a QStackedWidget holding the
pages, so it can be dragged wider for a long project path; the sash position
persists across sessions (see `settings.save_splitter_state`/
`restore_splitter_state`).

Pages read/write the project's folder tree only through `tree_panel` --
there's no per-page folder state, and no hand-wired "use X output"
buttons: every page just asks the shared tree which stage is currently the
active 2D source.

Denoise and Segmentation need the `prep` extra's jssl-denoise/cellpose, so
on an install without them those two slots get a `MissingExtraPage` saying
what to install, and the rest of the window (Classifier Training, Classify
Tiles, ...) still works -- see yeastprep/optional_deps.py.
"""

from qtpy.QtWidgets import (
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core import project as project_core
from yeastprep.core import stages as stages_core
from yeastprep.optional_deps import missing

from . import selection_actions, settings
from .classifier_pool import ClassifierPool
from .common.stage_breadcrumb import PipelineBreadcrumb
from .pages.classifier_training_page import ClassifierTrainingPage
from .pages.classify_tiles_page import ClassifyTilesPage
from .pages.data_reduction_page import DataReductionPage
from .pages.deconvolve_page import DeconvolvePage
from .pages.missing_extra_page import MissingExtraPage
from .pages.preview_page import PreviewPage
from .pages.tile_generation_page import TileGenerationPage
from .project_tree_panel import ProjectTreePanel
from .selection_actions_panel import SelectionActionsPanel
from .tile_viewer import open_tile_viewer


class YeastPrepWindow(QMainWindow):
    def __init__(self, initial_input_folder: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("yeastprep")
        self.resize(1600, 950)

        self.tree_panel = ProjectTreePanel()

        self.data_reduction_page = DataReductionPage(self.tree_panel)
        self.preview_page = PreviewPage(self.tree_panel)
        self.denoise_page = self._build_denoise_page()
        self.deconvolve_page = DeconvolvePage(self.tree_panel)
        self.segmentation_page = self._build_segmentation_page()
        self.tile_generation_page = TileGenerationPage(self.tree_panel)
        self.classifier_pool = ClassifierPool(self)
        self.classifier_training_page = ClassifierTrainingPage(self.classifier_pool)
        self.classify_tiles_page = ClassifyTilesPage(self.classifier_pool)
        # page_key (see selection_actions.py) -> page, in stack order.
        self._pages = {
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

    # The page modules are imported only once their packages are known to
    # be installed, since they (and their params panels) import them at
    # module level.
    def _build_denoise_page(self):
        absent = missing("jssl_denoise")
        if absent:
            return MissingExtraPage("Denoise", absent, extra="prep")
        from .pages.denoise_page import DenoisePage

        return DenoisePage(self.tree_panel)

    def _build_segmentation_page(self):
        absent = missing("cellpose")
        if absent:
            return MissingExtraPage("Segmentation", absent, extra="prep")
        from .pages.segmentation_page import SegmentationPage

        return SegmentationPage(self.tree_panel)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        self.breadcrumb = PipelineBreadcrumb(
            self.tree_panel,
            stage_pages={
                project_core.STAGE_REDUCED: "data_reduction",
                project_core.STAGE_DENOISED: "denoise",
                project_core.STAGE_DECONVOLVED: "deconvolve",
                stages_core.STAGE_SEGMENTATION: "segmentation",
                project_core.STAGE_TILES: "tile_generation",
            },
            extra_pages=[
                ("Classifier Training", "classifier_training"),
                ("Classify Tiles", "classify_tiles"),
            ],
        )
        outer.addWidget(self.breadcrumb)

        self.main_splitter = QSplitter()
        outer.addWidget(self.main_splitter, 1)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar.setMinimumWidth(220)
        sidebar_layout.addWidget(self.tree_panel, 1)
        self.selection_panel = SelectionActionsPanel()
        sidebar_layout.addWidget(self.selection_panel)
        self.main_splitter.addWidget(sidebar)

        self.stack = QStackedWidget()
        for page in self._pages.values():
            self.stack.addWidget(page)
        self.main_splitter.addWidget(self.stack)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        if not settings.restore_splitter_state("main_sidebar", self.main_splitter):
            self.main_splitter.setSizes([280, self.width() - 280])

        self._show_page("data_reduction")

    # ------------------------------------------------------------------
    # Wiring

    def _wire_up(self):
        self.breadcrumb.page_requested.connect(self._show_page)
        self.tree_panel.file_selected.connect(self._on_tree_selection)
        self.selection_panel.action_triggered.connect(self._on_action_triggered)

        for key, page in self._pages.items():
            page.progress_changed.connect(
                lambda progress, key=key: self.breadcrumb.set_progress(key, progress)
            )

    def _show_page(self, page_key: str):
        self.stack.setCurrentWidget(self._pages[page_key])
        self.breadcrumb.set_current_page(page_key)

    def _on_tree_selection(self, stage: str, path: str):
        actions = selection_actions.actions_for_selection(stage, path, self.tree_panel)
        self.selection_panel.set_selection(stage, path, actions)

    def _on_action_triggered(self, page_key: str, stage: str, path: str, mode: str):
        if mode == "open_tile_viewer":
            # `path` is a FOV id for a tiles-stage selection.
            open_tile_viewer([self.tree_panel.project_paths().tiles / f"{path}.tiles"])
            return
        if mode == "add_to_pool":
            self.classifier_pool.add_project(self.tree_panel.project_root())
            self._show_page("classifier_training")
            return
        page = self._pages.get(page_key)
        if page is None:
            return
        self._show_page(page_key)
        page.load_selection(stage, path, mode)

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        settings.save_window_geometry(self)
        settings.save_splitter_state("main_sidebar", self.main_splitter)
        for page in self._pages.values():
            page.shutdown()
        self.tree_panel.shutdown()
        super().closeEvent(event)
