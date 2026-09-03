"""Pooled-projects tree shared by both tabs of the Classifier Training page
(see `classifier_training_page.py`) -- the "cumulative dataset" pooling UX,
adapted from `train_denoise_page.TrainDenoisePage`'s "Add project.../Remove"
pool tree. The one structural difference: denoise pools individual stage
tiffs as checkable leaves, this pools **FOV tile folders** under each
project's `05_tiles/` (`core.classify.discover_fov_dirs`), since a
classifier trains on annotated tile crops, not whole 2D frames.
"""

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tileclass.data.pooled_annotations import PooledAnnotations

from yeastprep.core.classify import discover_fov_dirs


class ClassifierPoolWidget(QWidget):
    pool_changed = Signal()  # a project was added/removed, or a FOV (un)checked

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pooled_roots: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_group())

    def _build_group(self) -> QGroupBox:
        group = QGroupBox("Pooled projects")
        v = QVBoxLayout(group)

        self.pool_tree = QTreeWidget()
        self.pool_tree.setHeaderHidden(True)
        self.pool_tree.setUniformRowHeights(True)
        self.pool_tree.setMaximumHeight(160)
        self.pool_tree.itemChanged.connect(lambda *_: self.pool_changed.emit())
        v.addWidget(self.pool_tree)

        hint = QLabel(
            "Check individual FOVs to use a subset -- unchecked means use all. "
            "Training draws on every human-confirmed annotated tile across the pool."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)

        buttons = QHBoxLayout()
        self.add_project_btn = QPushButton("Add project...")
        self.add_project_btn.clicked.connect(self._prompt_add_project)
        buttons.addWidget(self.add_project_btn)
        self.remove_project_btn = QPushButton("Remove")
        self.remove_project_btn.clicked.connect(self._remove_selected_project)
        buttons.addWidget(self.remove_project_btn)
        v.addLayout(buttons)
        return group

    # ------------------------------------------------------------------

    def add_project(self, root: str):
        """Public entry point for `ClassifierTrainingPage.load_selection`
        pool-adding the project a "Train Classifier..." selection action
        was triggered from -- same effect as the user clicking "Add
        project..." and picking it by hand."""
        root = str(root)
        if root in self._pooled_roots:
            return
        self._pooled_roots.append(root)
        self._refresh_pool_tree()
        self.pool_changed.emit()

    def _prompt_add_project(self):
        chosen = QFileDialog.getExistingDirectory(self, "Add project to training pool")
        if chosen:
            self.add_project(chosen)

    def _remove_selected_project(self):
        item = self.pool_tree.currentItem()
        if item is None:
            return
        while item.parent() is not None:
            item = item.parent()
        root = item.data(0, Qt.UserRole)
        self._pooled_roots = [r for r in self._pooled_roots if r != root]
        self._refresh_pool_tree()
        self.pool_changed.emit()

    def _refresh_pool_tree(self):
        self.pool_tree.blockSignals(True)
        try:
            self.pool_tree.clear()
            for root in self._pooled_roots:
                fov_dirs = discover_fov_dirs(root)

                top_item = QTreeWidgetItem(self.pool_tree, [f"{root}  ({len(fov_dirs)} FOV(s))"])
                top_item.setData(0, Qt.UserRole, root)
                top_item.setFlags(top_item.flags() & ~Qt.ItemIsUserCheckable)

                for fov_dir in fov_dirs:
                    leaf = QTreeWidgetItem(top_item, [fov_dir.name])
                    leaf.setData(0, Qt.UserRole, str(fov_dir))
                    leaf.setFlags(leaf.flags() | Qt.ItemIsUserCheckable)
                    # Checked by default -- "pool everything" is the visible
                    # default, not an invisible fallback for nothing checked
                    # (mirrors TrainDenoisePage._refresh_pool_tree).
                    leaf.setCheckState(0, Qt.Checked)

                top_item.setExpanded(True)
        finally:
            self.pool_tree.blockSignals(False)

    # ------------------------------------------------------------------

    def _checked_paths_for_project(self, top_item: QTreeWidgetItem) -> list[str]:
        return [
            top_item.child(i).data(0, Qt.UserRole)
            for i in range(top_item.childCount())
            if top_item.child(i).checkState(0) == Qt.Checked
        ]

    def checked_fov_dirs(self) -> list[str]:
        fov_dirs: list[str] = []
        for i in range(self.pool_tree.topLevelItemCount()):
            fov_dirs.extend(self._checked_paths_for_project(self.pool_tree.topLevelItem(i)))
        return fov_dirs

    def pooled_annotations(self) -> PooledAnnotations | None:
        """A `PooledAnnotations` over the currently checked FOV folders, or
        `None` if nothing is checked -- callers (training start, dataset
        summary) both need this same "nothing to pool" guard."""
        fov_dirs = self.checked_fov_dirs()
        if not fov_dirs:
            return None
        return PooledAnnotations(fov_dirs)

    def gather_confirmed_records(self) -> list[tuple[str, str]]:
        """(path, category) pairs restricted to human-confirmed tags --
        training on the model's own unreviewed predictions would just
        reinforce its current mistakes (see
        `tileclass.training.supervised.train_classifier`'s own docstring
        for the same point)."""
        pooled = self.pooled_annotations()
        if pooled is None:
            return []
        return [
            (path, category)
            for path, category, confidence in pooled.tagged_items()
            if confidence is None
        ]
