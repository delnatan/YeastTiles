"""Persistent "what can I do with the currently selected tree item" panel,
docked directly under `ProjectTreePanel` in the sidebar. Replaces the old
single-click auto-navigate (which silently routed some stages to a
read-only page and others nowhere) and the double-click-loads-a-page
pattern: every task a tree selection can trigger is listed here by name --
see `selection_actions.actions_for_selection` for how the list is derived
-- so there's exactly one place to look for "what happens if I click this."
"""

from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget

from yeastprep.core import project as project_core
from yeastprep.core import stages as stages_core

from .selection_actions import Action

_STAGE_LABELS = {spec.key: spec.label for spec in stages_core.PIPELINE}

_PLACEHOLDER_TEXT = "Click a file in the tree to see available actions."


class SelectionActionsPanel(QWidget):
    action_triggered = Signal(str, str, str, str)  # page_key, stage, path, mode

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.group = QGroupBox("Selection")
        outer.addWidget(self.group)

        layout = QVBoxLayout(self.group)

        self.header_label = QLabel(_PLACEHOLDER_TEXT)
        self.header_label.setWordWrap(True)
        layout.addWidget(self.header_label)

        self.stage_label = QLabel("")
        self.stage_label.setStyleSheet("color: #a0a0a0;")
        layout.addWidget(self.stage_label)

        self._buttons: list[QPushButton] = []
        self._buttons_layout = QVBoxLayout()
        self._buttons_layout.setContentsMargins(0, 4, 0, 0)
        layout.addLayout(self._buttons_layout)

    def set_selection(self, stage: str, path: str, actions: list[Action]):
        name = f"FOV {path}" if stage == project_core.STAGE_TILES else Path(path).name
        self.header_label.setText(f"Selected: {name}")
        self.stage_label.setText(f"Stage: {_STAGE_LABELS.get(stage, stage)}")
        self._rebuild_buttons(stage, path, actions)

    def clear_selection(self):
        self.header_label.setText(_PLACEHOLDER_TEXT)
        self.stage_label.setText("")
        self._rebuild_buttons(None, None, [])

    def _rebuild_buttons(self, stage, path, actions: list[Action]):
        for btn in self._buttons:
            self._buttons_layout.removeWidget(btn)
            btn.deleteLater()
        self._buttons = []

        for action in actions:
            btn = QPushButton(action.label)
            btn.clicked.connect(
                lambda _checked=False, a=action, s=stage, p=path: self.action_triggered.emit(
                    a.page_key, s, p, a.mode
                )
            )
            self._buttons_layout.addWidget(btn)
            self._buttons.append(btn)
