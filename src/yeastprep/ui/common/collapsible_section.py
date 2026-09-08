"""A titled, collapsible content section -- the standard "Advanced options"
disclosure pattern, for grouping rarely-touched controls (extra
hyperparameters, tuning knobs) out of a form's initial view without
deleting them or splitting them into a separate dialog."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    def __init__(
        self, title: str, content: QWidget, start_expanded: bool = False, parent=None
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._toggle = QToolButton()
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(start_expanded)
        self._toggle.setText(title)
        self._toggle.setAutoRaise(True)
        self._toggle.clicked.connect(self._on_toggled)
        layout.addWidget(self._toggle)

        self._content = content
        layout.addWidget(self._content)

        self._on_toggled(start_expanded)

    def _on_toggled(self, checked: bool):
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)
