"""Input/output folder selection -- the highest-priority UX element per
the project's design.md scoping decision. Validates the input folder has
loadable files, and makes proposing/remembering an output folder low
friction.
"""

from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yeastprep.ui import settings

SUPPORTED_PATTERNS = ("*.ims", "*.czi", "*.nd2")


class FolderPanel(QWidget):
    """Emits `input_folder_changed(str)` / `output_folder_changed(str)`
    whenever the resolved, existing folder changes (not on every
    keystroke -- only on Browse/recent-pick/programmatic set)."""

    input_folder_changed = Signal(str)
    output_folder_changed = Signal(str)
    pattern_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._input_dir = ""
        self._output_dir = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(self._build_input_group())
        layout.addWidget(self._build_output_group())
        layout.addStretch(1)

    # ------------------------------------------------------------------
    # Input group
    # ------------------------------------------------------------------

    def _build_input_group(self) -> QGroupBox:
        group = QGroupBox("Input folder (raw stacks)")
        v = QVBoxLayout(group)

        self.input_recent = QComboBox()
        self.input_recent.addItem("Recent folders...")
        self.input_recent.addItems(settings.get_recent_input_dirs())
        self.input_recent.activated.connect(self._on_input_recent_picked)
        v.addWidget(self.input_recent)

        row = QHBoxLayout()
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        self.input_edit.setPlaceholderText("No folder selected")
        row.addWidget(self.input_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_input)
        row.addWidget(browse_btn)
        v.addLayout(row)

        pattern_row = QHBoxLayout()
        pattern_row.addWidget(QLabel("File type:"))
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(SUPPORTED_PATTERNS)
        self.pattern_combo.currentTextChanged.connect(self._on_pattern_changed)
        pattern_row.addWidget(self.pattern_combo)
        pattern_row.addStretch(1)
        v.addLayout(pattern_row)

        self.input_status = QLabel(" ")
        v.addWidget(self.input_status)

        return group

    def _browse_input(self):
        start_dir = self._input_dir or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Select input folder", start_dir)
        if chosen:
            self.set_input_folder(chosen)

    def _on_input_recent_picked(self, index):
        if index <= 0:
            return
        self.set_input_folder(self.input_recent.itemText(index))
        self.input_recent.setCurrentIndex(0)

    def _on_pattern_changed(self, pattern):
        self.pattern_changed.emit(pattern)
        self._revalidate_input()

    def set_input_folder(self, folder: str):
        folder = str(folder)
        self._input_dir = folder
        self.input_edit.setText(folder)
        self._revalidate_input()
        settings.add_recent_input_dir(folder)
        self._refresh_recent(self.input_recent, settings.get_recent_input_dirs())
        self.input_folder_changed.emit(folder)

    def _revalidate_input(self):
        folder = self._input_dir
        if not folder or not Path(folder).is_dir():
            self.input_status.setText("No folder selected")
            self.input_status.setStyleSheet("color: #999;")
            return
        n_files = len(list(Path(folder).glob(self.current_pattern())))
        if n_files:
            self.input_status.setText(f"{n_files} file(s) found")
            self.input_status.setStyleSheet("color: #4caf50;")
        else:
            self.input_status.setText(
                f"No '{self.current_pattern()}' files found here"
            )
            self.input_status.setStyleSheet("color: #ff5c5c;")

    def current_pattern(self) -> str:
        return self.pattern_combo.currentText()

    # ------------------------------------------------------------------
    # Output group
    # ------------------------------------------------------------------

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox("Output folder (flattened slices + projections)")
        v = QVBoxLayout(group)

        self.output_recent = QComboBox()
        self.output_recent.addItem("Recent folders...")
        self.output_recent.addItems(settings.get_recent_output_dirs())
        self.output_recent.activated.connect(self._on_output_recent_picked)
        v.addWidget(self.output_recent)

        row = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setPlaceholderText("No folder selected")
        row.addWidget(self.output_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output)
        row.addWidget(browse_btn)
        v.addLayout(row)

        suggest_btn = QPushButton("Suggest output folder from input")
        suggest_btn.clicked.connect(self._suggest_output)
        v.addWidget(suggest_btn)

        self.output_status = QLabel(" ")
        v.addWidget(self.output_status)

        return group

    def _browse_output(self):
        start_dir = self._output_dir or self._input_dir or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Select output folder", start_dir)
        if chosen:
            self.set_output_folder(chosen)

    def _on_output_recent_picked(self, index):
        if index <= 0:
            return
        self.set_output_folder(self.output_recent.itemText(index))
        self.output_recent.setCurrentIndex(0)

    def _suggest_output(self):
        if not self._input_dir:
            return
        input_dir = Path(self._input_dir)
        suggestion = input_dir.parent / "yeastprep_output" / input_dir.name
        self.set_output_folder(str(suggestion))

    def set_output_folder(self, folder: str, create_if_missing: bool = True):
        folder = str(folder)
        if create_if_missing:
            Path(folder).mkdir(parents=True, exist_ok=True)
        self._output_dir = folder
        self.output_edit.setText(folder)
        exists = Path(folder).is_dir()
        self.output_status.setText("Ready" if exists else "Will be created")
        self.output_status.setStyleSheet("color: #4caf50;" if exists else "color: #cc9900;")
        settings.add_recent_output_dir(folder)
        self._refresh_recent(self.output_recent, settings.get_recent_output_dirs())
        self.output_folder_changed.emit(folder)

    # ------------------------------------------------------------------

    @staticmethod
    def _refresh_recent(combo: QComboBox, entries: list[str]):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Recent folders...")
        combo.addItems(entries)
        combo.blockSignals(False)

    def input_folder(self) -> str:
        return self._input_dir

    def output_folder(self) -> str:
        return self._output_dir
