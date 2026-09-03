"""Self-supervised VICReg backbone pretraining (design.md stage 4's other
half; see `training/supervised.py` for the supervised classifier head
that gets built on top of this backbone afterwards).

Departs from vanilla VICReg -- which forms each positive pair from two
augmented views of the *same* image -- by drawing the pair from two
*different* crops sharing the same annotated category instead
(`ClassPairDataset` below). Both views are still independently
augmented, but the network is pushed to pull together the true
within-class variation actually present in this dataset (different
cells, same morphology) rather than just different crops/noise of one
photograph. This needs the annotations supervised training would
otherwise consume, so it's a weakly-supervised, category-conditioned
VICReg, not a fully unlabeled pretraining step -- appropriate here since
every crop that ever enters training already gets hand-annotated by
design.md's workflow.

A category with only one annotated example can't produce two distinct
crops, so it falls back to the classic same-image pair for that one
category (see `ClassPairDataset`) -- expected early in an
annotate-and-train loop, same situation `dataset.stratified_split`
already handles for the supervised side.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..classifiers.device import select_device
from .dataset import ClampTensor, RandomGaussianNoise, load_masked_crop
from .model import build_yeast_efficientnet
from .supervised import TrainingCancelled

VICREG_WEIGHTS_DIR = Path(__file__).parent / "weights" / "vicreg_backbone"
VICREG_WEIGHTS_PATH = VICREG_WEIGHTS_DIR / "backbone.pth"
VICREG_META_PATH = VICREG_WEIGHTS_DIR / "meta.json"


class VICRegAugmentation:
    """One augmented view: spatial jitter (crop/flip/rotation) plus light
    sensor-noise simulation. Unlike the classic VICReg transform (called
    once per image to produce a *pair*), this is called independently on
    each of the two class-mates `ClassPairDataset` already drew, so it
    returns a single tensor, not a tuple."""

    def __init__(self, crop_size=64, crop_scale=(0.6, 1.0)):
        import torchvision.transforms.v2 as T

        self._transform = T.Compose(
            [
                T.RandomResizedCrop(
                    size=(crop_size, crop_size), scale=crop_scale, antialias=True
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=45),
                RandomGaussianNoise(std=0.02, p=0.5),
                ClampTensor(min_val=0.0, max_val=1.0),
            ]
        )

    def __call__(self, x):
        return self._transform(x)


def worker_init_fn(_worker_id):
    """DataLoader workers fork with Python's global `random`/`numpy.random`
    state already seeded from the parent process, so left alone every
    worker draws the identical sequence of class-pair choices (a fork
    copies, it doesn't reseed). Reseed both from the per-worker seed
    torch already assigned via its own generator, so workers diverge."""
    import numpy as np

    seed = torch.utils.data.get_worker_info().seed % (2**32)
    random.seed(seed)
    np.random.seed(seed)


class ClassPairDataset(Dataset):
    """VICReg positive pairs drawn from two different annotated crops in
    the same category, each independently augmented by `transform`
    (falls back to one crop paired with itself for a singleton category
    -- see module docstring).

    `__getitem__` ignores `idx` and re-samples a category and two of its
    members fresh on every call -- standard practice for contrastive
    datasets, since there's no fixed notion of "the i-th pair". This
    decouples epoch length from raw tile count: `epoch_length` defaults
    to `len(paths)` so an epoch still looks like "one pass through the
    data" by default, but can be set independently.

    `balanced=True` (default) samples the category uniformly rather than
    by frequency, so a rare category gets exercised in as many pairs as
    the majority one over an epoch. The annotation counts here are
    unavoidably skewed, and unlike supervised fine-tuning's
    inverse-frequency loss weighting, VICReg's loss has no per-sample
    label to reweight -- uniform category sampling is the only lever
    available to keep minority categories from being drowned out during
    pretraining.
    """

    def __init__(
        self, paths, labels, transform, epoch_length=None, balanced=True, seed=0
    ):
        if len(paths) != len(labels):
            raise ValueError("paths and labels must be the same length")
        if not paths:
            raise ValueError("no labeled paths given")

        self.paths = list(paths)
        self.transform = transform
        self.balanced = balanced

        self.by_class = {}
        for idx, label in enumerate(labels):
            self.by_class.setdefault(label, []).append(idx)
        self.classes = sorted(self.by_class)
        self.singleton_classes = sorted(
            c for c, idxs in self.by_class.items() if len(idxs) < 2
        )

        self._rng = random.Random(seed)
        self.epoch_length = (
            epoch_length if epoch_length is not None else len(self.paths)
        )

    def __len__(self):
        return self.epoch_length

    def _sample_pair_indices(self):
        if self.balanced:
            cls = self._rng.choice(self.classes)
        else:
            weights = [len(self.by_class[c]) for c in self.classes]
            cls = self._rng.choices(self.classes, weights=weights, k=1)[0]

        members = self.by_class[cls]
        if len(members) >= 2:
            return tuple(self._rng.sample(members, 2))
        return members[0], members[0]

    def __getitem__(self, _idx):
        i, j = self._sample_pair_indices()
        img1 = torch.from_numpy(load_masked_crop(self.paths[i]))
        img2 = torch.from_numpy(load_masked_crop(self.paths[j]))
        return self.transform(img1), self.transform(img2)


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
        # 3-layer MLP projector head (discarded after pretraining -- only
        # `backbone` gets saved/reused downstream).
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


@dataclass
class VICRegParams:
    epochs: int = 1000
    batch_size: int = 128
    epoch_length: int | None = None  # None -> one pair per tile per epoch
    balanced_sampling: bool = True
    lr: float = 1e-4
    weight_decay: float = 5e-5
    proj_dim: int = 1024
    sim_coeff: float = 25.0
    std_coeff: float = 25.0
    cov_coeff: float = 1.0
    num_workers: int = 4
    seed: int = 0
    # Whether to warm-start from the live-slot backbone (VICREG_WEIGHTS_PATH)
    # if one exists, vs. always starting cold from ImageNet -- see
    # `pretrain_vicreg`'s docstring for why warm-starting is the deliberate
    # default here (unlike `training.supervised.train_classifier`, which
    # never warm-starts at all).
    warm_start: bool = True


@dataclass
class VICRegProgress:
    epoch: int  # 1-based
    total_epochs: int
    avg_loss: float
    metrics: dict = field(default_factory=dict)  # averaged sim/std/cov/total


@dataclass
class VICRegResult:
    categories: list = field(default_factory=list)
    singleton_categories: list = field(default_factory=list)
    pair_count: int = 0
    final_loss: float = 0.0
    weights_path: Path = VICREG_WEIGHTS_PATH


def load_backbone(weights_path=VICREG_WEIGHTS_PATH, device=None):
    """Build a headless (num_classes=None) EfficientNet backbone and load
    pretrained VICReg weights into it -- for embedding extraction, not
    further training (caller should `.eval()` and wrap forward passes in
    `torch.no_grad()`)."""
    device = device or select_device()
    backbone = build_yeast_efficientnet(num_classes=None, pretrained=False)
    backbone.load_state_dict(torch.load(weights_path, map_location=device))
    return backbone.to(device)


def _save_backbone(
    model,
    categories,
    singleton_categories,
    params: VICRegParams,
    trained_on_paths,
    output_dir: Path | None = None,
) -> Path:
    """Write `backbone.pth`/`meta.json` into `output_dir` if given, else
    into the live warm-start slot (`VICREG_WEIGHTS_DIR`) -- backing up
    whatever's already there first, so a bad pretraining run can't destroy
    the last good backbone irreversibly (mirrors
    `training/supervised.py`'s `_save_weights`). Returns the directory
    written to.

    `trained_on_paths`: every crop path this run pretrained on -- recorded
    in meta.json as `trained_on_paths` so a later run can tell (via
    `warm_start_overlap`) how much of a newly pooled dataset this backbone
    has already been exposed to before deciding whether/how hard to
    warm-start from it again."""
    import json
    import shutil

    weights_dir = Path(output_dir) if output_dir is not None else VICREG_WEIGHTS_DIR
    weights_path = weights_dir / "backbone.pth"
    meta_path = weights_dir / "meta.json"

    weights_dir.mkdir(parents=True, exist_ok=True)
    if weights_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = weights_dir / f"backbone.pth.bak-{stamp}"
        shutil.copy2(weights_path, backup_path)

    torch.save(model.backbone.state_dict(), weights_path)

    meta = {
        "categories": categories,
        "singleton_categories": singleton_categories,
        "pairing": "class-conditioned (different crops, same category)",
        "trained_on_paths": sorted(str(p) for p in trained_on_paths),
        "params": {
            "epochs": params.epochs,
            "batch_size": params.batch_size,
            "balanced_sampling": params.balanced_sampling,
            "lr": params.lr,
            "weight_decay": params.weight_decay,
            "proj_dim": params.proj_dim,
            "sim_coeff": params.sim_coeff,
            "std_coeff": params.std_coeff,
            "cov_coeff": params.cov_coeff,
            "warm_start": params.warm_start,
        },
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "description": (
            "VICReg-pretrained EfficientNet-B0 backbone (brightfield + "
            "fluorescence + mask, stem modified for 2 input channels), "
            "positive pairs drawn from distinct crops sharing an "
            "annotated category."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return weights_dir


def warm_start_overlap(paths, meta_path=VICREG_META_PATH) -> tuple[int, int] | None:
    """How much of `paths` (crop paths a caller is about to pretrain on)
    the backbone recorded at `meta_path` has already seen, as
    `(already_seen, total)` -- or `None` if `meta_path` doesn't exist, or
    exists but predates this field (an older checkpoint saved before
    `trained_on_paths` was added). Used by the training UI to warn before
    a warm-started run: repeatedly exposing the backbone to the same crops
    across rounds isn't the validation-leakage bug that motivated always
    training the supervised classifier from scratch (VICReg has no
    held-out split to corrupt -- see `training/supervised.py`'s module
    docstring), but a user deliberately trying to broaden a backbone's
    exposure still wants to know how much of a newly pooled dataset is
    actually new to it."""
    import json

    meta_path = Path(meta_path)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None
    trained_on = meta.get("trained_on_paths")
    if trained_on is None:
        return None
    seen = set(trained_on)
    paths = list(paths)
    already_seen = sum(1 for p in paths if str(p) in seen)
    return already_seen, len(paths)


def pretrain_vicreg(
    records,
    params: VICRegParams = VICRegParams(),
    progress_callback=None,
    cancel_check=None,
    output_dir: Path | None = None,
) -> VICRegResult:
    """`records`: list of (crop_path, category) pairs, e.g. from
    `PooledAnnotations.tagged_items()` filtered to human-confirmed tags.
    Categories are only used to group crops into pairs -- the resulting
    backbone has no classification head and doesn't need a fixed
    category vocabulary the way `supervised.train_classifier` does, so a
    later run with a different (or larger) set of categories can freely
    warm-start from whatever backbone is currently saved.

    `params.warm_start` (default `True`): whether to initialize from the
    live-slot backbone (`VICREG_WEIGHTS_PATH`) if one exists, vs. always
    starting cold from an ImageNet-pretrained stem. Warm-starting is safe
    to default on here in a way it isn't for `supervised.train_classifier`
    (which never warm-starts, full stop): VICReg pretraining has no
    held-out validation split, so there's no "the model already trained on
    what's now supposed to be held out" leakage bug to worry about --
    repeat exposure to the same crops across pretraining rounds is just
    more self-supervised training, same as more epochs. Set it `False`
    for a clean, from-scratch run when that's what's wanted instead (e.g.
    comparing against a known baseline, or deliberately discarding a
    backbone trained on since-corrected annotations). Either way, this
    always reads from the live slot regardless of `output_dir` -- only
    where the *result* lands changes (see `train_classifier`'s matching
    parameter for the rationale). Use `warm_start_overlap` to check how
    much of `records` a given live backbone has already seen before
    deciding.

    Raises `ValueError` for too little data, `TrainingCancelled` if
    `cancel_check()` goes true between epochs.
    """
    if len(records) < 2:
        raise ValueError(
            "Not enough annotated tiles to pretrain VICReg -- need at least "
            "2 annotated crops."
        )

    paths = [p for p, _ in records]
    labels = [label for _, label in records]

    dataset = ClassPairDataset(
        paths,
        labels,
        transform=VICRegAugmentation(),
        epoch_length=params.epoch_length,
        balanced=params.balanced_sampling,
        seed=params.seed,
    )

    device = select_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Mirror supervised.train_classifier's guard against an empty loader
    # when the annotated set is smaller than one batch.
    batch_size = min(params.batch_size, len(dataset))
    drop_last = len(dataset) > batch_size

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=params.num_workers,
        worker_init_fn=worker_init_fn if params.num_workers > 0 else None,
        persistent_workers=params.num_workers > 0,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
    )

    if params.warm_start and VICREG_WEIGHTS_PATH.exists():
        backbone = build_yeast_efficientnet(num_classes=None, pretrained=False)
        backbone.load_state_dict(
            torch.load(VICREG_WEIGHTS_PATH, map_location="cpu")
        )
    else:
        backbone = build_yeast_efficientnet(num_classes=None, pretrained=True)

    vicreg = VICRegModel(backbone, feature_dim=1280, proj_dim=params.proj_dim).to(
        device
    )
    optimizer = torch.optim.AdamW(
        vicreg.parameters(), lr=params.lr, weight_decay=params.weight_decay
    )
    criterion = VICRegLoss(
        sim_coeff=params.sim_coeff,
        std_coeff=params.std_coeff,
        cov_coeff=params.cov_coeff,
    )

    vicreg.train()
    final_avg_loss = 0.0
    for epoch in range(params.epochs):
        if cancel_check is not None and cancel_check():
            raise TrainingCancelled()

        epoch_totals = {"sim": 0.0, "std": 0.0, "cov": 0.0, "total": 0.0}
        num_batches = 0
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()

            _, z1 = vicreg(x1)
            _, z2 = vicreg(x2)

            loss, metrics = criterion(z1, z2)
            loss.backward()
            optimizer.step()

            for key, value in metrics.items():
                epoch_totals[key] += value
            num_batches += 1

        num_batches = max(num_batches, 1)
        avg_metrics = {k: v / num_batches for k, v in epoch_totals.items()}
        final_avg_loss = avg_metrics["total"]

        if progress_callback is not None:
            progress_callback(
                VICRegProgress(
                    epoch=epoch + 1,
                    total_epochs=params.epochs,
                    avg_loss=final_avg_loss,
                    metrics=avg_metrics,
                )
            )

    categories = dataset.classes
    weights_dir = _save_backbone(
        vicreg, categories, dataset.singleton_classes, params, paths, output_dir=output_dir
    )

    return VICRegResult(
        categories=categories,
        singleton_categories=dataset.singleton_classes,
        pair_count=len(dataset) * params.epochs,
        final_loss=final_avg_loss,
        weights_path=weights_dir / "backbone.pth",
    )
