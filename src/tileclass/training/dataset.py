"""Training-time dataset + label-set utilities, ported from
NN_workflow/yeastVIC.py -- only the supervised-training pieces (the
VICReg-pair transform and `load_annotations`'s txt-file parsing aren't
needed here: records come from `PooledAnnotations.tagged_items()`
in-process, not from re-reading files off disk; see
`training/supervised.py`).
"""

import random
from collections import Counter

import numpy as np
import tifffile
from torch.utils.data import Dataset


class RandomGaussianNoise:
    def __init__(self, std=0.02, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x):
        import torch

        if torch.rand(1) < self.p:
            return x + torch.randn_like(x) * self.std
        return x


class ClampTensor:
    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x):
        import torch

        return torch.clamp(x, min=self.min_val, max=self.max_val)


class ClassificationTransform:
    """Light augmentation for supervised fine-tuning -- `train=False`
    (validation) is a no-op, matching NN_workflow/yeastVIC.py exactly."""

    def __init__(self, train=True):
        self._train = train
        self._transform = None

    def _build(self):
        import torchvision.transforms.v2 as T

        return T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=45),
                RandomGaussianNoise(std=0.02, p=0.5),
                ClampTensor(min_val=0.0, max_val=1.0),
            ]
        )

    def __call__(self, x):
        if not self._train:
            return x
        if self._transform is None:
            self._transform = self._build()
        return self._transform(x)


def load_masked_crop(path):
    """Read a (brightfield, fluorescence, mask) crop TIFF into a (2, H, W)
    float32 array in [0, 1], zeroing pixels outside the mask (255 =
    valid). Shared by `MaskedMicroscopyDataset`,
    `classifiers.yeast_efficientnet`, and `vicreg.ClassPairDataset` so
    the three can't silently diverge on how a crop is decoded."""
    img = tifffile.imread(path)
    if img.shape[-1] == 3:
        img = img.transpose(2, 0, 1)

    brightfield, fluorescence, mask = img[0], img[1], img[2]
    valid = mask == 255
    bf = np.where(valid, brightfield, 0.0)
    fl = np.where(valid, fluorescence, 0.0)
    return np.stack([bf, fl], axis=0).astype(np.float32) / 255.0


class MaskedMicroscopyDataset(Dataset):
    """Wraps `load_masked_crop` in the `Dataset` interface, optionally
    paired with per-item labels and a transform."""

    def __init__(self, file_paths, labels=None, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        import torch

        tensor_img = torch.from_numpy(load_masked_crop(self.file_paths[idx]))

        if self.transform:
            tensor_img = self.transform(tensor_img)

        if self.labels is not None:
            return tensor_img, self.labels[idx]
        return tensor_img


def stratified_split(labels, val_frac=0.2, seed=0):
    """Split indices into (train_idx, val_idx), splitting each class
    independently so rare classes still show up in both sets -- except a
    class with only one example, which goes entirely to train (one
    training example teaches the model a little; one validation-only
    example teaches nothing and just makes that class's reported val
    accuracy a coin flip). Early in an annotate-and-train loop, most
    classes will be exactly this rare."""
    rng = random.Random(seed)
    by_class = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)

    train_idx, val_idx = [], []
    for idxs in by_class.values():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = max(1, round(len(idxs) * val_frac)) if len(idxs) > 1 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


def class_weights(labels, num_classes):
    """Inverse-frequency weights for nn.CrossEntropyLoss(weight=...)."""
    import torch

    counts = Counter(labels)
    total = len(labels)
    return torch.tensor(
        [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)],
        dtype=torch.float32,
    )
