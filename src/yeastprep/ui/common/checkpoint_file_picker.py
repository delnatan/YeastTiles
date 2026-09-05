"""Shared checkpoint-weights-file picker, used by both the Classifier
Training page (Deploy, "Starting Point" backbone) and the Classify Tiles
page (Run Inference weights, Explore Embeddings backbone) -- previously
private to `classifier_training_page.py`, pulled out here so a second page
can reuse it without reaching into that module's internals.
"""

import json
from pathlib import Path

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from .json_tree_dialog import JsonTreeDialog


def looks_like_vicreg_backbone(meta_path: Path) -> bool:
    """Whether `meta_path` was written by `tileclass.training.vicreg`'s
    `_save_backbone` rather than `tileclass.training.supervised`'s
    `_save_weights` -- both save a `meta.json` with a `categories` key, so
    that alone can't tell them apart, but only the VICReg one ever writes
    a `pairing` key (see `training/vicreg.py`'s `_save_backbone`). Used to
    reject a VICReg backbone picked (by mistake, via Browse) as an
    inference checkpoint before it fails deep inside a background thread
    with a raw PyTorch state_dict-mismatch error -- a headless backbone
    has no classification head to load into `YeastEfficientNetClassifier`."""
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return False
    return "pairing" in meta


class CheckpointFilePicker(QWidget):
    """A checkpoint-weights-file field + Browse + Deployed + View Metadata
    row, with "still showing its default until the user edits/Browses/picks
    Deployed" tracking -- the shape shared by the Classifier Training page's
    "Starting point" backbone field and both tabs' Deploy field, and the
    Classify Tiles page's inference-weights and embeddings-backbone fields.
    `resolve()` centralizes validation so callers just get back a
    ready-to-use (weights_path, meta_path) pair or `None` (a QMessageBox has
    already explained why, in that case).

    "View Metadata..." opens the sibling meta.json (whatever's currently
    typed/Browsed/Deployed into the field, not necessarily a validated
    pair) in a `JsonTreeDialog` -- so a user can inspect what a checkpoint
    was trained on and with what parameters before committing to using it
    for inference, further training, or deploying it live."""

    pathChanged = Signal(str)

    def __init__(
        self,
        placeholder: str,
        tooltip: str,
        deployed_path: Path,
        deployed_tooltip: str,
        parent=None,
    ):
        super().__init__(parent)
        self._deployed_path = deployed_path
        self._is_default = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setToolTip(tooltip)
        self.edit.textEdited.connect(self._on_edited)
        layout.addWidget(self.edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        layout.addWidget(browse_btn)
        deployed_btn = QPushButton("Deployed")
        deployed_btn.setToolTip(deployed_tooltip)
        deployed_btn.clicked.connect(self._use_deployed)
        layout.addWidget(deployed_btn)
        metadata_btn = QPushButton("View Metadata...")
        metadata_btn.setToolTip(
            "Open this checkpoint's sibling meta.json in a browsable tree -- "
            "what it was trained on, with what parameters, and when."
        )
        metadata_btn.clicked.connect(self._view_metadata)
        layout.addWidget(metadata_btn)

    def _on_edited(self, text):
        self._is_default = False
        self.pathChanged.emit(text)

    def _browse(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "Checkpoint", self.edit.text(), "PyTorch weights (*.pth)"
        )
        if path:
            self.edit.setText(path)
            self._is_default = False
            self.pathChanged.emit(path)

    def _use_deployed(self):
        path = str(self._deployed_path)
        self.edit.setText(path)
        self._is_default = False
        self.pathChanged.emit(path)

    def _view_metadata(self):
        text = self.edit.text().strip()
        if not text:
            QMessageBox.information(self, "yeastprep", "Choose a checkpoint first.")
            return
        meta_path = Path(text).with_name("meta.json")
        if not meta_path.is_file():
            QMessageBox.warning(
                self, "yeastprep", f"No meta.json found next to {text}."
            )
            return
        try:
            data = json.loads(meta_path.read_text())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "yeastprep", f"Could not read {meta_path}: {exc}")
            return
        dialog = JsonTreeDialog(f"Metadata -- {meta_path}", data, parent=self)
        dialog.exec_()

    def set_default_path(self, path: Path):
        """Called once a training run in this session produces a fresh
        checkpoint -- only takes effect while the user hasn't Browsed/typed/
        picked Deployed themselves, same "obvious default until you
        deliberately override it" convention as the checkpoint-output field."""
        if self._is_default:
            self.edit.setText(str(path))
            self.pathChanged.emit(str(path))

    def resolve(self, *, expect_vicreg: bool, wrong_kind_message: str) -> tuple[Path, Path] | None:
        """Validates the current text as a (weights.pth, sibling meta.json)
        pair of the expected kind -- `expect_vicreg=False` for a
        classifier checkpoint, `True` for a headless VICReg backbone (see
        `looks_like_vicreg_backbone`). Shows a `QMessageBox.warning` and
        returns `None` for anything wrong; `wrong_kind_message` (a
        `str.format`-style template taking `meta_path`) supplies the
        caller-specific "pick a checkpoint from the other tab/page instead"
        hint."""
        text = self.edit.text().strip()
        if not text:
            QMessageBox.warning(self, "yeastprep", "Choose a checkpoint first.")
            return None
        weights_path = Path(text)
        meta_path = weights_path.with_name("meta.json")
        if not weights_path.is_file() or not meta_path.is_file():
            QMessageBox.warning(
                self,
                "yeastprep",
                f"Expected both {weights_path.name} and a sibling meta.json at "
                f"{weights_path.parent} -- one or both are missing.",
            )
            return None
        if looks_like_vicreg_backbone(meta_path) != expect_vicreg:
            QMessageBox.warning(self, "yeastprep", wrong_kind_message.format(meta_path=meta_path))
            return None
        return weights_path, meta_path
