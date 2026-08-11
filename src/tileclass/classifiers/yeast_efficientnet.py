"""Yeast cell-crop classifier: brightfield + fluorescence + mask crops,
EfficientNet-B0 with a stem modified for 2 input channels.

Self-contained inference port of BurgessLab/imaging/Bri/yeastVIC.py +
03_inference.py -- only what's needed to load the trained weights and
classify crops, none of the training / self-supervised-pretraining
machinery those scripts also carry.

torch/torchvision are imported lazily (inside methods, not at module
scope) so listing available classifiers never requires them -- only
actually running one does. See the ``ai`` extra in pyproject.toml.
"""

import json
from pathlib import Path

import numpy as np
import tifffile

from .base import TileClassifier
from .device import select_device

WEIGHTS_DIR = Path(__file__).parent / "weights" / "yeast_efficientnet"
WEIGHTS_PATH = WEIGHTS_DIR / "weights.pth"
META_PATH = WEIGHTS_DIR / "meta.json"

BATCH_SIZE = 64


def _load_masked_crop(path):
    """Read a (brightfield, fluorescence, mask) crop TIFF into a
    ``(2, H, W)`` float32 array in [0, 1], zeroing pixels outside the
    mask (255 = valid). Matches ``MaskedMicroscopyDataset.__getitem__``."""
    img = tifffile.imread(path)
    if img.shape[-1] == 3:
        img = img.transpose(2, 0, 1)

    brightfield, fluorescence, mask = img[0], img[1], img[2]
    valid = mask == 255
    bf = np.where(valid, brightfield, 0.0)
    fl = np.where(valid, fluorescence, 0.0)
    return np.stack([bf, fl], axis=0).astype(np.float32) / 255.0


class YeastEfficientNetClassifier(TileClassifier):
    name = "Yeast (bf + fluorescence + mask)"

    def __init__(self):
        meta = json.loads(META_PATH.read_text())
        self.categories = meta["categories"]
        self._model = None
        self._device = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch

        from ..training.model import build_yeast_efficientnet

        device = select_device()

        model = build_yeast_efficientnet(len(self.categories), pretrained=False)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        model.to(device)
        model.eval()

        self._model = model
        self._device = device

    def predict(self, paths):
        """Return a list of (label, confidence) pairs, one per path."""
        self._ensure_loaded()
        import torch

        results = []
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start : start + BATCH_SIZE]
            batch = np.stack([_load_masked_crop(p) for p in batch_paths])
            x = torch.from_numpy(batch).to(self._device)
            with torch.no_grad():
                probs = torch.softmax(self._model(x), dim=1)
            conf, pred = probs.max(dim=1)
            for label_idx, confidence in zip(
                pred.cpu().tolist(), conf.cpu().tolist()
            ):
                results.append((self.categories[label_idx], confidence))
        return results
