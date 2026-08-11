"""Dialog for managing a folder's predefined tile annotation categories."""

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
        self.add_btn.clicked.connect(self._add)
        self.rename_btn.clicked.connect(self._rename)
        self.delete_btn.clicked.connect(self._delete)
        button_row.addWidget(self.add_btn)
        button_row.addWidget(self.rename_btn)
        button_row.addWidget(self.delete_btn)
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
        return {name.lower() for name in self.annotations.categories()}

    def _add(self):
        name, ok = QInputDialog.getText(self, "Add Category", "Category name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name.lower() in self._existing_names_ci():
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
        if new.lower() in self._existing_names_ci():
            QMessageBox.information(
                self, "Rename Category", f"'{new}' already exists."
            )
            return
        self.annotations.rename_category(old, new)
        self._refresh_list(select=new)

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
