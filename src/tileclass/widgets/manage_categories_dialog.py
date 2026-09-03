"""Dialog for managing a folder's predefined tile annotation categories."""

from collections import defaultdict

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ..data.annotations import normalized_category_key


class ManageCategoriesDialog(QDialog):
    """Add/rename/delete the predefined category vocabulary for a
    `TileAnnotations` store. Each action applies and saves immediately,
    matching how tile tagging itself works — there is no separate
    OK/Cancel commit step."""

    def __init__(self, annotations, parent=None):
        super().__init__(parent)
        self.annotations = annotations
        self.setWindowTitle("Manage Categories")
        self.resize(320, 360)
        self._setup_ui()
        self._refresh_list()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Predefined categories for this folder:"))

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self.list_widget)

        button_row = QHBoxLayout()
        self.add_btn = QPushButton("Add...")
        self.rename_btn = QPushButton("Rename...")
        self.delete_btn = QPushButton("Delete")
        self.find_duplicates_btn = QPushButton("Find Duplicates...")
        self.add_btn.clicked.connect(self._add)
        self.rename_btn.clicked.connect(self._rename)
        self.delete_btn.clicked.connect(self._delete)
        self.find_duplicates_btn.clicked.connect(self._find_duplicates)
        button_row.addWidget(self.add_btn)
        button_row.addWidget(self.rename_btn)
        button_row.addWidget(self.delete_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.find_duplicates_btn)
        layout.addLayout(button_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.accept)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.accept)
        layout.addWidget(buttons)

    def _refresh_list(self, select=None):
        self.list_widget.clear()
        for name in self.annotations.categories():
            self.list_widget.addItem(name)
        if select is not None:
            matches = self.list_widget.findItems(select, Qt.MatchExactly)
            if matches:
                self.list_widget.setCurrentItem(matches[0])

    def _selected_name(self):
        item = self.list_widget.currentItem()
        return item.text() if item is not None else None

    def _existing_names_ci(self):
        return {normalized_category_key(name) for name in self.annotations.categories()}

    def _add(self):
        name, ok = QInputDialog.getText(self, "Add Category", "Category name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if normalized_category_key(name) in self._existing_names_ci():
            QMessageBox.information(
                self, "Add Category", f"'{name}' already exists."
            )
            return
        self.annotations.add_category(name)
        self._refresh_list(select=name)

    def _rename(self):
        old = self._selected_name()
        if old is None:
            return
        new, ok = QInputDialog.getText(
            self, "Rename Category", "New name:", text=old
        )
        if not ok:
            return
        new = new.strip()
        if not new or new == old:
            return
        if normalized_category_key(new) in self._existing_names_ci():
            QMessageBox.information(
                self, "Rename Category", f"'{new}' already exists."
            )
            return
        self.annotations.rename_category(old, new)
        self._refresh_list(select=new)

    def _find_duplicates(self):
        """Scan every pooled folder's raw category strings (not just the
        already-collapsed `categories()` view -- see
        `PooledAnnotations.raw_category_names`) for case/whitespace variants
        of the same category, and let the user pick a canonical spelling to
        rewrite them to via `rename_category` (which persists to every
        pooled folder's sidecar file)."""
        groups = defaultdict(list)
        for name in self.annotations.raw_category_names():
            groups[normalized_category_key(name)].append(name)
        duplicate_groups = [variants for variants in groups.values() if len(variants) > 1]

        if not duplicate_groups:
            QMessageBox.information(
                self, "Find Duplicates", "No near-duplicate categories found."
            )
            return

        for variants in duplicate_groups:
            canonical, ok = QInputDialog.getItem(
                self,
                "Merge Duplicate Categories",
                "These look like the same category (case/whitespace differs):\n"
                f"{', '.join(variants)}\n\nMerge into which spelling?",
                variants,
                0,
                editable=False,
            )
            if not ok:
                continue
            for variant in variants:
                if variant != canonical:
                    self.annotations.rename_category(variant, canonical)

        self._refresh_list()

    def _delete(self):
        name = self._selected_name()
        if name is None:
            return
        count = self.annotations.usage_count(name)
        if count:
            message = (
                f"{count} image(s) are tagged '{name}'. Remove it from the "
                "category list anyway? Existing tags will be kept but the "
                "category will no longer be selectable."
            )
        else:
            message = f"Remove category '{name}'?"
        if QMessageBox.question(self, "Delete Category", message) != QMessageBox.Yes:
            return
        self.annotations.remove_category(name)
        self._refresh_list()
