"""Read-only JSON tree viewer dialog -- for peeking at a checkpoint's
sibling `meta.json` (training params, category list, provenance, ...)
without leaving the app or opening the file in a text editor. Deliberately
generic (takes any already-parsed JSON value, not just checkpoint meta) so
it isn't tied to `tileclass.training`'s particular meta.json shape and
survives that shape changing.
"""

from qtpy.QtWidgets import QDialog, QTreeWidget, QTreeWidgetItem, QVBoxLayout


def _add_json_node(parent_item: QTreeWidgetItem, key: str, value) -> None:
    if isinstance(value, dict):
        node = QTreeWidgetItem(parent_item, [key, f"{{{len(value)}}}"])
        for child_key, child_value in value.items():
            _add_json_node(node, str(child_key), child_value)
    elif isinstance(value, list):
        node = QTreeWidgetItem(parent_item, [key, f"[{len(value)}]"])
        for index, child_value in enumerate(value):
            _add_json_node(node, str(index), child_value)
    else:
        QTreeWidgetItem(parent_item, [key, "null" if value is None else str(value)])


class JsonTreeDialog(QDialog):
    """Non-modal-friendly (callers still typically `.exec_()` it) viewer:
    one collapsible tree, "Key" / "Value" columns, expanded two levels deep
    by default so a checkpoint's top-level fields (`categories`, `params`,
    `last_trained`, ...) are visible immediately without clicking through
    every branch."""

    def __init__(self, title: str, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(520, 480)

        layout = QVBoxLayout(self)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Key", "Value"])
        self.tree.setColumnWidth(0, 220)
        layout.addWidget(self.tree)

        root = self.tree.invisibleRootItem()
        _add_json_node(root, "(root)", data)
        self.tree.expandToDepth(1)
