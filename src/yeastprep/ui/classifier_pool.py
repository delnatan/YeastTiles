"""The classification pool: which projects' tiles the Classifier Training and
Classify Tiles pages work on. One `ClassifierPool` is shared by both pages
and saved across sessions (see `settings.get_classifier_pool`); each page
shows it through its own `ClassifierPoolWidget`.

A pooled project contributes every FOV container under its `05_tiles/`
(`core.classify.discover_fov_dirs`). FOVs are included unless unchecked, and
the unchecked set is what's saved -- so FOVs exported later join the pool
automatically.
"""

import os

from qtpy.QtCore import QObject, Qt, Signal
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
from tileclass.scan import scan_container

from yeastprep.core.classify import discover_fov_dirs

from . import settings


class ClassifierPool(QObject):
    changed = Signal()  # a project was added/removed, or a FOV (un)checked

    def __init__(self, parent=None):
        super().__init__(parent)
        roots, excluded = settings.get_classifier_pool()
        self._roots: list[str] = roots
        self._excluded: set[str] = set(excluded)

    def _save_and_notify(self):
        settings.set_classifier_pool(self._roots, sorted(self._excluded))
        self.changed.emit()

    # ------------------------------------------------------------------

    def roots(self) -> list[str]:
        return list(self._roots)

    def add_project(self, root: str):
        root = str(root)
        if root not in self._roots:
            self._roots.append(root)
            self._save_and_notify()

    def remove_project(self, root: str):
        if root in self._roots:
            self._roots.remove(root)
            prefix = os.path.join(root, "")
            self._excluded = {f for f in self._excluded if not f.startswith(prefix)}
            self._save_and_notify()

    def set_fov_included(self, fov_path: str, included: bool):
        if included:
            self._excluded.discard(fov_path)
        else:
            self._excluded.add(fov_path)
        self._save_and_notify()

    def is_fov_included(self, fov_path: str) -> bool:
        return fov_path not in self._excluded

    # ------------------------------------------------------------------

    def checked_fov_dirs(self) -> list[str]:
        return [
            str(fov)
            for root in self._roots
            for fov in discover_fov_dirs(root)
            if str(fov) not in self._excluded
        ]

    def pooled_annotations(self) -> PooledAnnotations | None:
        """A `PooledAnnotations` over the checked FOVs, or `None` if none
        are checked."""
        fov_dirs = self.checked_fov_dirs()
        return PooledAnnotations(fov_dirs) if fov_dirs else None

    def tile_paths(self) -> list[str]:
        return [p for fov in self.checked_fov_dirs() for p in scan_container(fov)]

    def gather_confirmed_records(self) -> list[tuple[str, str]]:
        """(path, category) pairs restricted to human-set tags -- training
        on the model's own unreviewed predictions would just reinforce its
        current mistakes."""
        pooled = self.pooled_annotations()
        if pooled is None:
            return []
        return [
            (path, category)
            for path, category, confidence in pooled.tagged_items()
            if confidence is None
        ]


class ClassifierPoolWidget(QWidget):
    """Tree view of a `ClassifierPool`: projects with checkable FOVs, plus
    Add/Remove project buttons."""

    def __init__(self, pool: ClassifierPool, parent=None):
        super().__init__(parent)
        self.pool = pool
        self._editing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        group = QGroupBox("Classification pool")
        layout.addWidget(group)
        v = QVBoxLayout(group)

        self.pool_tree = QTreeWidget()
        self.pool_tree.setHeaderHidden(True)
        self.pool_tree.setUniformRowHeights(True)
        self.pool_tree.setMaximumHeight(160)
        self.pool_tree.itemChanged.connect(self._on_item_changed)
        v.addWidget(self.pool_tree)

        hint = QLabel(
            "Shared by Classifier Training and Classify Tiles, and kept "
            "between sessions. Uncheck FOVs to leave them out."
        )
        hint.setWordWrap(True)
        v.addWidget(hint)

        buttons = QHBoxLayout()
        add_btn = QPushButton("Add project...")
        add_btn.clicked.connect(self._prompt_add_project)
        buttons.addWidget(add_btn)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_selected_project)
        buttons.addWidget(remove_btn)
        v.addLayout(buttons)

        pool.changed.connect(self._on_pool_changed)
        self._rebuild()

    def _prompt_add_project(self):
        chosen = QFileDialog.getExistingDirectory(self, "Add project to classification pool")
        if chosen:
            self.pool.add_project(chosen)

    def _remove_selected_project(self):
        item = self.pool_tree.currentItem()
        if item is None:
            return
        while item.parent() is not None:
            item = item.parent()
        self.pool.remove_project(item.data(0, Qt.UserRole))

    def _on_item_changed(self, item: QTreeWidgetItem, _column: int):
        if item.parent() is None:
            return
        # This tree already shows the new state; don't rebuild it from
        # inside its own itemChanged.
        self._editing = True
        try:
            self.pool.set_fov_included(
                item.data(0, Qt.UserRole), item.checkState(0) == Qt.Checked
            )
        finally:
            self._editing = False

    def _on_pool_changed(self):
        if not self._editing:
            self._rebuild()

    def _rebuild(self):
        self.pool_tree.blockSignals(True)
        try:
            self.pool_tree.clear()
            for root in self.pool.roots():
                fov_dirs = discover_fov_dirs(root)
                top_item = QTreeWidgetItem(self.pool_tree, [f"{root}  ({len(fov_dirs)} FOV(s))"])
                top_item.setData(0, Qt.UserRole, root)
                top_item.setFlags(top_item.flags() & ~Qt.ItemIsUserCheckable)
                # Long paths get elided in a narrow tree.
                top_item.setToolTip(0, root)
                for fov_dir in fov_dirs:
                    leaf = QTreeWidgetItem(top_item, [fov_dir.stem])
                    leaf.setData(0, Qt.UserRole, str(fov_dir))
                    leaf.setToolTip(0, str(fov_dir))
                    leaf.setFlags(leaf.flags() | Qt.ItemIsUserCheckable)
                    included = self.pool.is_fov_included(str(fov_dir))
                    leaf.setCheckState(0, Qt.Checked if included else Qt.Unchecked)
                top_item.setExpanded(True)
        finally:
            self.pool_tree.blockSignals(False)
