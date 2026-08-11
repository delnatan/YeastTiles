import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset


# 1. Define picklable classes for your custom math operations
class RandomGaussianNoise(nn.Module):
    def __init__(self, std=0.02, p=0.5):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            return x + torch.randn_like(x) * self.std
        return x


class ClampTensor(nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, min=self.min_val, max=self.max_val)


class MaskedMicroscopyDataset(Dataset):
    def __init__(self, file_paths, labels=None, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Read TIFF: assumes shape (C, H, W) or (H, W, C)
        img = tifffile.imread(self.file_paths[idx])
        if img.shape[-1] == 3:
            img = img.transpose(2, 0, 1)  # Ensure (3, 64, 64)

        brightfield = img[0]
        fluorescence = img[1]
        mask = img[2]

        # Zero out invalid regions using the mask (255 is valid)
        valid_mask = mask == 255
        bf_masked = np.where(valid_mask, brightfield, 0.0)
        fl_masked = np.where(valid_mask, fluorescence, 0.0)

        # Stack 2 channels and normalize from 0-255 to 0.0-1.0
        img_2ch = (
            np.stack([bf_masked, fl_masked], axis=0).astype(np.float32) / 255.0
        )
        tensor_img = torch.from_numpy(img_2ch)

        if self.transform:
            tensor_img = self.transform(tensor_img)

        if self.labels is not None:
            return tensor_img, self.labels[idx]
        return tensor_img


# VICReg requires two differently augmented views (x, x') of the same image
class VICRegTransform:
    def __init__(self):
        # Spatial transformations that work safely on 2-channel scientific data
        self.transform = T.Compose(
            [
                T.RandomResizedCrop(
                    size=(64, 64), scale=(0.6, 1.0), antialias=True
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=45),
                # Add light Gaussian noise to simulate sensor noise
                RandomGaussianNoise(std=0.02, p=0.5),
                ClampTensor(min_val=0.0, max_val=1.0),
            ]
        )

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# Phase 2 (supervised) only needs one augmented view per image, not a
# correlated pair like VICRegTransform.
class ClassificationTransform:
    def __init__(self, train=True):
        if train:
            self.transform = T.Compose(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomRotation(degrees=45),
                    RandomGaussianNoise(std=0.02, p=0.5),
                    ClampTensor(min_val=0.0, max_val=1.0),
                ]
            )
        else:
            self.transform = None

    def __call__(self, x):
        return self.transform(x) if self.transform is not None else x


def load_annotations(txt_paths, label_alias=None):
    """Parse `crops_F*.txt` annotation files into (crop_path, label) pairs.

    Each txt file's crop directory is assumed to share its stem, e.g.
    `crops_F0.txt` -> `crops_F0/`. Lines starting with '#' are header/comment
    lines and are skipped. Only annotated cells appear in these files, so no
    filtering for missing labels is needed. `label_alias` remaps raw label
    strings (e.g. {"binucleate": "two"}) so files that used different
    vocabulary for the same class still merge into one label set.
    """
    label_alias = label_alias or {}
    records = []
    for txt_path in txt_paths:
        txt_path = Path(txt_path)
        crop_dir = txt_path.with_suffix("")
        for line in txt_path.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            filename, label = line.split("\t")
            label = label_alias.get(label, label)
            records.append((str(crop_dir / filename), label))
    return records


def stratified_split(labels, val_frac=0.2, seed=0):
    """Split indices into (train_idx, val_idx), splitting each class
    independently so rare classes still show up in both sets."""
    rng = random.Random(seed)
    by_class = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)

    train_idx, val_idx = [], []
    for idxs in by_class.values():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_val = max(1, round(len(idxs) * val_frac))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    return train_idx, val_idx


def class_weights(labels, num_classes):
    """Inverse-frequency weights for nn.CrossEntropyLoss(weight=...)."""
    counts = Counter(labels)
    total = len(labels)
    return torch.tensor(
        [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)],
        dtype=torch.float32,
    )


def get_modified_efficientnet(num_classes=None, pretrained=True):
    # Start from ImageNet weights so the SSL pre-training isn't learning
    # low-level filters from scratch on a comparatively small dataset.
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Target the first Conv2d layer in the stem
    old_conv = model.features[0][0]

    # Modify for 2 channels and stride=1 (keeps 64x64 spatial resolution intact)
    new_conv = nn.Conv2d(
        in_channels=2,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=(1, 1),  # Crucial change: was (2, 2)
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    if pretrained:
        # The pretrained stem expects 3 RGB channels; ours has 2. Average the
        # RGB filters into a single filter and duplicate it across both new
        # channels, rescaling by 3/2 so the expected activation magnitude
        # (mean_weight * num_input_channels) matches the original stem.
        with torch.no_grad():
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(mean_weight.repeat(1, 2, 1, 1) * (3 / 2))
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

    model.features[0][0] = new_conv

    # If num_classes is None, strip classifier to output raw 1280-dim feature vector
    if num_classes is None:
        model.classifier = nn.Identity()
    else:
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    return model


import time


class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        batch_size, num_features = z1.shape

        sim_loss = nn.functional.mse_loss(z1, z2)

        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-04)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-04)
        std_loss = torch.mean(nn.functional.relu(1.0 - std_z1)) + torch.mean(
            nn.functional.relu(1.0 - std_z2)
        )

        z1_centered = z1 - z1.mean(dim=0)
        z2_centered = z2 - z2.mean(dim=0)
        cov_z1 = (z1_centered.T @ z1_centered) / (batch_size - 1)
        cov_z2 = (z2_centered.T @ z2_centered) / (batch_size - 1)

        cov_loss = (self.off_diagonal(cov_z1).pow(2).sum() / num_features) + (
            self.off_diagonal(cov_z2).pow(2).sum() / num_features
        )

        total_loss = (
            self.sim_coeff * sim_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        # Return total loss for backprop, plus a dict of individual metrics for logging
        metrics = {
            "sim": sim_loss.item(),
            "std": std_loss.item(),
            "cov": cov_loss.item(),
            "total": total_loss.item(),
        }
        return total_loss, metrics


class VICRegModel(nn.Module):
    def __init__(self, backbone, feature_dim=1280, proj_dim=1024):
        super().__init__()
        self.backbone = backbone
        # 3-layer MLP projector head (discarded after pre-training)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        y = self.backbone(x)
        z = self.projector(y)
        return y, z
