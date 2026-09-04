"""Yeast cell-crop classifier: brightfield + fluorescence + mask crops,
EfficientNet-B0 with a stem modified for 2 input channels.

Self-contained inference port of BurgessLab/imaging/Bri/yeastVIC.py +
03_inference.py -- only what's needed to load the trained weights and
classify crops, none of the training / self-supervised-pretraining
machinery those scripts also carry.

torch/torchvision are imported lazily (inside methods, not at module
scope) so listing available classifiers never requires them -- only
actually running one does. See the ``classification`` extra in pyproject.toml.
"""

import json
from pathlib import Path

import numpy as np

from .base import TileClassifier
from .device import select_device

WEIGHTS_DIR = Path(__file__).parent / "weights" / "yeast_efficientnet"
WEIGHTS_PATH = WEIGHTS_DIR / "weights.pth"
META_PATH = WEIGHTS_DIR / "meta.json"

BATCH_SIZE = 64


class YeastEfficientNetClassifier(TileClassifier):
    name = "Yeast (bf + fluorescence + mask)"

    def __init__(self, weights_path=None, meta_path=None):
        """Defaults to the live-deployed slot -- what every existing
        caller (the `CLASSIFIERS` registry, tileclass's Auto-Annotate)
        wants. `weights_path`/`meta_path` let a caller point at some other
        (weights.pth, meta.json) pair instead -- e.g. yeastprep's
        Classifier Training page running inference with a just-trained,
        not-yet-deployed session checkpoint (see `core.classify.classify_pool`).
        Resolved here rather than as parameter defaults so a test (or any
        caller) that monkeypatches the module-level `WEIGHTS_PATH`/
        `META_PATH` still takes effect -- a parameter default binds once at
        module-import time, before any monkeypatch could run."""
        self._weights_path = Path(weights_path) if weights_path is not None else WEIGHTS_PATH
        meta_path = Path(meta_path) if meta_path is not None else META_PATH
        meta = json.loads(meta_path.read_text())
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
        model.load_state_dict(torch.load(self._weights_path, map_location=device))
        model.to(device)
        model.eval()

        self._model = model
        self._device = device

    def predict(self, paths):
        """Return a list of (label, confidence) pairs, one per path."""
        self._ensure_loaded()
        import torch

        from ..training.dataset import load_masked_crop

        results = []
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start : start + BATCH_SIZE]
            batch = np.stack([load_masked_crop(p) for p in batch_paths])
            x = torch.from_numpy(batch).to(self._device)
            with torch.no_grad():
                probs = torch.softmax(self._model(x), dim=1)
            conf, pred = probs.max(dim=1)
            for label_idx, confidence in zip(
                pred.cpu().tolist(), conf.cpu().tolist()
            ):
                results.append((self.categories[label_idx], confidence))
        return results
