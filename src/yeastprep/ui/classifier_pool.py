"""The classification pool: which projects' tiles the Classifier Training and
Classify Tiles pages work on. One `ClassifierPool` is shared by both pages
and saved across sessions (see `settings.get_classifier_pool`); each page
shows it through its own `ClassifierPoolWidget`.

A pooled project contributes every FOV container under its `05_tiles/`
(`core.classify.discover_fov_dirs`). FOVs are included unless unchecked, and
the unchecked set is what's saved -- so FOVs exported later join the pool
automatically.

The project list is shared, but each page checks FOVs in its own `PoolScope`
(`SCOPES`): narrowing Classify Tiles to a few FOVs to sample a sub-population
for Explore Embeddings must not quietly drop them from the next training run.
"""

import os

from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
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
from .pool_data import PoolDataCache, PoolSummary


SCOPES = ("training", "classify")


class ClassifierPool(QObject):
    # Which projects are pooled changed (added, removed).
    projects_changed = Signal()
    # A FOV was (un)checked in the named scope. This only changes what that
    # scope's page looks at: nothing is loaded or dropped here -- the pages
    # read what they need, when they need it (see `PoolDataCache`).
    selection_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        roots, excluded = settings.get_classifier_pool(SCOPES)
        self._roots: list[str] = roots
        self._excluded: dict[str, set[str]] = {
            scope: set(fovs) for scope, fovs in excluded.items()
        }
        self.cache = PoolDataCache()
        self._scopes: dict[str, PoolScope] = {}

    def _save(self):
        settings.set_classifier_pool(
            self._roots, {scope: sorted(fovs) for scope, fovs in self._excluded.items()}
        )

    def scope(self, name: str) -> "PoolScope":
        """The pool as one page sees it: the shared projects, its own FOV checks."""
        if name not in self._scopes:
            self._scopes[name] = PoolScope(self, name)
        return self._scopes[name]

    # ------------------------------------------------------------------

    def roots(self) -> list[str]:
        return list(self._roots)

    def add_project(self, root: str):
        root = str(root)
        if root not in self._roots:
            self._roots.append(root)
            self._save()
            self.projects_changed.emit()

    def remove_project(self, root: str):
        if root in self._roots:
            self._roots.remove(root)
            prefix = os.path.join(root, "")
            for scope, fovs in self._excluded.items():
                self._excluded[scope] = {f for f in fovs if not f.startswith(prefix)}
            self.cache.forget_project(root)
            self._save()
            self.projects_changed.emit()

    def set_fovs_included(self, scope: str, fov_paths, included: bool):
        excluded = self._excluded[scope]
        before = set(excluded)
        if included:
            excluded.difference_update(fov_paths)
        else:
            excluded.update(fov_paths)
        if excluded != before:
            self._save()
            self.selection_changed.emit(scope)

    def is_fov_included(self, scope: str, fov_path: str) -> bool:
        return fov_path not in self._excluded[scope]

    def excluded(self, scope: str) -> frozenset:
        return frozenset(self._excluded[scope])


class PoolSnapshot:
    """The checked FOVs of one scope at one moment, plus everything that
    reads them. Taken on the GUI thread (`PoolScope.snapshot()`), so a
    worker thread can use it while the user keeps clicking checkboxes."""

    def __init__(self, roots, excluded, cache: PoolDataCache):
        self.roots = tuple(roots)
        self.excluded = frozenset(excluded)
        self._cache = cache

    def checked_fov_dirs(self) -> list[str]:
        return [
            str(fov)
            for root in self.roots
            for fov in discover_fov_dirs(root)
            if str(fov) not in self.excluded
        ]

    def summary(self) -> PoolSummary:
        """Tile and tag counts over the checked FOVs, from the cache."""
        return self._cache.summarize(self.checked_fov_dirs())

    def fov_tallies(self) -> list:
        """(fov_dir, `FovTally`) per checked FOV, from the cache."""
        return [(fov, self._cache.tally(fov)) for fov in self.checked_fov_dirs()]

    def pooled_annotations(self) -> PooledAnnotations | None:
        """A `PooledAnnotations` over the checked FOVs, or `None` if none
        are checked. Always fresh from disk: it's what tags are written
        through."""
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


class PoolScope(QObject):
    """One page's view of a `ClassifierPool`: projects are shared, FOV
    checks are this scope's own. Everything a page reads -- tile paths,
    annotations, training records -- covers only the FOVs checked here."""

    projects_changed = Signal()
    selection_changed = Signal()  # this scope's checks only
    changed = Signal()  # either

    def __init__(self, pool: ClassifierPool, name: str):
        super().__init__(pool)
        self.pool = pool
        self.name = name
        pool.projects_changed.connect(self.projects_changed)
        pool.projects_changed.connect(self.changed)
        pool.selection_changed.connect(self._on_selection_changed)

    def _on_selection_changed(self, scope: str):
        if scope == self.name:
            self.selection_changed.emit()
            self.changed.emit()

    def roots(self) -> list[str]:
        return self.pool.roots()

    def add_project(self, root: str):
        self.pool.add_project(root)

    def remove_project(self, root: str):
        self.pool.remove_project(root)

    def set_fovs_included(self, fov_paths, included: bool):
        self.pool.set_fovs_included(self.name, fov_paths, included)

    def set_fov_included(self, fov_path: str, included: bool):
        self.set_fovs_included([fov_path], included)

    def is_fov_included(self, fov_path: str) -> bool:
        return self.pool.is_fov_included(self.name, fov_path)

    # ------------------------------------------------------------------

    def snapshot(self) -> PoolSnapshot:
        return PoolSnapshot(self.roots(), self.pool.excluded(self.name), self.pool.cache)

    def checked_fov_dirs(self) -> list[str]:
        return self.snapshot().checked_fov_dirs()

    def pooled_annotations(self) -> PooledAnnotations | None:
        return self.snapshot().pooled_annotations()

    def tile_paths(self) -> list[str]:
        return self.snapshot().tile_paths()

    def gather_confirmed_records(self) -> list[tuple[str, str]]:
        return self.snapshot().gather_confirmed_records()


class ClassifierPoolWidget(QWidget):
    """Tree view of a `PoolScope`: projects with checkable FOVs, plus
    Add/Remove project buttons. Right-click for check/uncheck all."""

    def __init__(self, pool: PoolScope, hint: str, parent=None):
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
        self.pool_tree.setMaximumHeight(240)
        self.pool_tree.itemChanged.connect(self._on_item_changed)
        self.pool_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.pool_tree.customContextMenuRequested.connect(self._show_context_menu)
        v.addWidget(self.pool_tree)

        hint_label = QLabel(
            f"{hint} Projects are shared by Classifier Training and Classify "
            "Tiles; FOV checks are each page's own. Right-click to check or "
            "uncheck a project's FOVs at once."
        )
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #a0a0a0;")
        v.addWidget(hint_label)

        buttons = QHBoxLayout()
        add_btn = QPushButton("Add project...")
        add_btn.clicked.connect(self._prompt_add_project)
        buttons.addWidget(add_btn)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_selected_project)
        buttons.addWidget(remove_btn)
        v.addLayout(buttons)

        pool.projects_changed.connect(self._rebuild)
        pool.selection_changed.connect(self._on_selection_changed)
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
        # This tree already shows the new state; don't touch it from
        # inside its own itemChanged.
        self._editing = True
        try:
            self.pool.set_fov_included(
                item.data(0, Qt.UserRole), item.checkState(0) == Qt.Checked
            )
        finally:
            self._editing = False

    def _show_context_menu(self, pos):
        item = self.pool_tree.itemAt(pos)
        if item is None:
            return
        project = item if item.parent() is None else item.parent()
        fovs = [project.child(i).data(0, Qt.UserRole) for i in range(project.childCount())]
        if not fovs:
            return
        menu = QMenu(self)
        menu.addAction("Check all", lambda: self.pool.set_fovs_included(fovs, True))
        menu.addAction("Uncheck all", lambda: self.pool.set_fovs_included(fovs, False))
        if item is not project:
            this = item.data(0, Qt.UserRole)

            def only_this():
                self.pool.set_fovs_included([f for f in fovs if f != this], False)
                self.pool.set_fov_included(this, True)

            menu.addAction(f"Check only {item.text(0)}", only_this)
        menu.exec_(self.pool_tree.viewport().mapToGlobal(pos))

    def _on_selection_changed(self):
        # A bulk check/uncheck (context menu), or the same scope's other
        # widget. Only the checkmarks move; the project list is untouched.
        if self._editing:
            return
        self.pool_tree.blockSignals(True)
        try:
            for i in range(self.pool_tree.topLevelItemCount()):
                project = self.pool_tree.topLevelItem(i)
                for j in range(project.childCount()):
                    leaf = project.child(j)
                    included = self.pool.is_fov_included(leaf.data(0, Qt.UserRole))
                    leaf.setCheckState(0, Qt.Checked if included else Qt.Unchecked)
        finally:
            self.pool_tree.blockSignals(False)

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
