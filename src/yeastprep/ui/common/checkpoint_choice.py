"""Dropdown for choosing a checkpoint: the deployed one (the default),
ImageNet weights (for a backbone starting point), or any `.pth` picked via
"Browse...". A browsed file is checked up front to have a sibling
`meta.json` of the right kind, so a mistake can't surface later as a
state-dict error deep inside a background thread.

Two kinds: "classifier" (`weights.pth`, with a classification head) and
"backbone" (a headless VICReg `backbone.pth`). The deployed item's label
is refreshed whenever the list is opened, so it never goes stale after a
Deploy.
"""

import json
from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QFileDialog, QMessageBox

from tileclass.classifiers.yeast_efficientnet import WEIGHTS_PATH as CLASSIFIER_WEIGHTS_PATH
from tileclass.training.vicreg import VICREG_WEIGHTS_PATH as BACKBONE_WEIGHTS_PATH

DEPLOYED_PATHS = {"classifier": CLASSIFIER_WEIGHTS_PATH, "backbone": BACKBONE_WEIGHTS_PATH}
_KIND_NAMES = {"classifier": "classifier", "backbone": "VICReg backbone"}

_DEPLOYED = "deployed"
_IMAGENET = "imagenet"
_BROWSE = "browse"


def read_meta(weights_path) -> dict | None:
    try:
        return json.loads(Path(weights_path).with_name("meta.json").read_text())
    except (OSError, ValueError):
        return None


def checkpoint_kind(weights_path) -> str | None:
    """"backbone", "classifier", or None if there's no readable sibling
    meta.json. Both kinds save `categories`; only VICReg's
    `_save_backbone` writes `pairing`."""
    meta = read_meta(weights_path)
    if meta is None:
        return None
    return "backbone" if "pairing" in meta else "classifier"


def describe_checkpoint(weights_path) -> str:
    """One line for a tooltip/label: when it was trained and on what."""
    meta = read_meta(weights_path)
    if meta is None:
        return str(weights_path)
    trained = (meta.get("last_trained") or "unknown date")[:10]
    counts = meta.get("category_counts") or {}
    categories = ", ".join(
        f"{c} ({counts[c]})" if c in counts else c for c in meta.get("categories") or []
    )
    return f"trained {trained}; categories: {categories or 'n/a'}"


def pick_checkpoint_file(parent, kind: str) -> Path | None:
    """File dialog for a `kind` checkpoint, validated. None if cancelled or
    rejected (a message box has explained why)."""
    chosen, _filter = QFileDialog.getOpenFileName(
        parent, f"Choose {_KIND_NAMES[kind]} checkpoint", "", "Checkpoints (*.pth)"
    )
    if not chosen:
        return None
    found = checkpoint_kind(chosen)
    if found is None:
        QMessageBox.warning(parent, "yeastprep", f"No readable meta.json next to {chosen}.")
        return None
    if found != kind:
        QMessageBox.warning(
            parent,
            "yeastprep",
            f"{chosen} is a {_KIND_NAMES[found]} checkpoint, not a {_KIND_NAMES[kind]}.",
        )
        return None
    return Path(chosen)


class CheckpointChoice(QComboBox):
    def __init__(self, kind: str, allow_imagenet: bool = False, parent=None):
        super().__init__(parent)
        self._kind = kind
        self._deployed_path = DEPLOYED_PATHS[kind]
        self._last_index = 0

        self.addItem("", _DEPLOYED)
        if allow_imagenet:
            self.addItem("ImageNet weights (no pretraining)", _IMAGENET)
        self.addItem("Browse...", _BROWSE)
        self.activated.connect(self._on_activated)
        self._refresh_deployed()
        if not self._deployed_path.exists() and allow_imagenet:
            self.setCurrentIndex(1)
            self._last_index = 1

    def showPopup(self):
        self._refresh_deployed()
        super().showPopup()

    def _refresh_deployed(self):
        exists = self._deployed_path.exists()
        name = _KIND_NAMES[self._kind]
        self.setItemText(0, f"Deployed {name}" if exists else f"Deployed {name} (none yet)")
        self.setItemData(
            0,
            describe_checkpoint(self._deployed_path) if exists else "Nothing deployed yet.",
            Qt.ToolTipRole,
        )
        self.model().item(0).setEnabled(exists)

    def _on_activated(self, index: int):
        if self.itemData(index) != _BROWSE:
            self._last_index = index
            return
        chosen = pick_checkpoint_file(self, self._kind)
        if chosen is None:
            self.setCurrentIndex(self._last_index)
            return
        existing = self.findData(str(chosen))
        if existing < 0:
            existing = self.count() - 1  # just above "Browse..."
            self.insertItem(existing, chosen.parent.name + "/" + chosen.name, str(chosen))
            self.setItemData(
                existing, f"{chosen}\n{describe_checkpoint(chosen)}", Qt.ToolTipRole
            )
        self.setCurrentIndex(existing)
        self._last_index = existing

    def weights_path(self) -> Path | None:
        """The chosen checkpoint, or None for ImageNet weights -- or for the
        deployed item when nothing is deployed."""
        data = self.currentData()
        if data == _DEPLOYED:
            return self._deployed_path if self._deployed_path.exists() else None
        if data == _IMAGENET:
            return None
        return Path(data)

    def describe(self) -> str:
        """For a log line: what the current choice resolves to."""
        path = self.weights_path()
        if path is None:
            return "ImageNet weights"
        prefix = "deployed " if self.currentData() == _DEPLOYED else ""
        return f"{prefix}{path}"
