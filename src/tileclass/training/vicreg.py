"""Self-supervised VICReg backbone pretraining (design.md stage 4's other
half; see `training/supervised.py` for the supervised classifier head
that gets built on top of this backbone afterwards).

Standard VICReg: each positive pair is two independently augmented views
of the *same* tile (`TileViewPairDataset`), so no labels are involved and
every tile in the pool can be used, annotated or not. That keeps the
backbone honest to build on: it has never seen a label, so the supervised
stage's held-out validation split, and the Explore Embeddings kNN score,
aren't inflated by it.

The augmentations (`VICRegAugmentation`) define what the embedding learns
to ignore: orientation, a little framing, per-channel intensity scaling,
blur and sensor noise. Anything they don't touch -- shape, texture, where
the fluorescence sits in the cell -- is what it keeps.
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
from .supervised import TrainingCancelled, _clone_state_dict
from ..tile_container import group_training_provenance, training_provenance_contains

VICREG_WEIGHTS_DIR = Path(__file__).parent / "weights" / "vicreg_backbone"
VICREG_WEIGHTS_PATH = VICREG_WEIGHTS_DIR / "backbone.pth"
VICREG_META_PATH = VICREG_WEIGHTS_DIR / "meta.json"


class RandomRot90:
    """Rotate by a random multiple of 90 degrees -- exact, no
    interpolation, so it costs nothing in sharpness the way an arbitrary
    angle does."""

    def __call__(self, x):
        return torch.rot90(x, k=random.randint(0, 3), dims=(-2, -1))


class RandomChannelIntensity:
    """Independently rescale each channel by a random gain and gamma.
    Tiles are already percentile-stretched per crop at export
    (core/tiles.py), so absolute brightness carries little meaning; this
    makes the embedding robust to what's left of it. Pixels outside the
    cell mask are zero and stay zero (0 * gain, 0 ** gamma). Ranges kept
    mild in case relative fluorescence level matters to a category."""

    def __init__(self, gain=(0.8, 1.2), gamma=(0.8, 1.25), p=0.8):
        self.gain = gain
        self.gamma = gamma
        self.p = p

    def __call__(self, x):
        if random.random() >= self.p:
            return x
        n = x.shape[0]
        gain = torch.empty(n, 1, 1).uniform_(*self.gain)
        gamma = torch.empty(n, 1, 1).uniform_(*self.gamma)
        return gain * x.clamp(min=0.0) ** gamma


class VICRegAugmentation:
    """One augmented view; `TileViewPairDataset` calls it twice per tile.
    Spatial: a mild random-resized crop (cells are centered, so a deep
    crop would cut them off), flips plus exact 90-degree turns, and a
    further small rotation -- together any orientation. Then per-channel
    intensity jitter, occasional blur, sensor noise."""

    def __init__(self, crop_size=64, crop_scale=(0.7, 1.0)):
        import torchvision.transforms.v2 as T

        self._transform = T.Compose(
            [
                T.RandomResizedCrop(
                    size=(crop_size, crop_size),
                    scale=crop_scale,
                    ratio=(0.9, 1.1),
                    antialias=True,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                RandomRot90(),
                T.RandomRotation(degrees=45),
                RandomChannelIntensity(),
                T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.3),
                RandomGaussianNoise(std=0.02, p=0.5),
                ClampTensor(min_val=0.0, max_val=1.0),
            ]
        )

    def __call__(self, x):
        return self._transform(x)


def worker_init_fn(_worker_id):
    """DataLoader workers fork with Python's global `random` state already
    seeded from the parent process, so left alone every worker draws the
    identical sequence from it (a fork copies, it doesn't reseed) -- and
    `RandomRot90`/`RandomChannelIntensity` use it. Reseed from the
    per-worker seed torch already assigned via its own generator."""
    random.seed(torch.utils.data.get_worker_info().seed % (2**32))


class TileViewPairDataset(Dataset):
    """VICReg positive pairs: two independent `transform` draws of the
    same tile. Unlabeled -- just tile paths."""

    def __init__(self, paths, transform):
        if not paths:
            raise ValueError("no tile paths given")
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = torch.from_numpy(load_masked_crop(self.paths[idx]))
        return self.transform(img), self.transform(img)


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
    # An epoch is one pass over every pooled tile, annotated or not -- far
    # more than the annotated subset this used to train on, so fewer
    # epochs than before.
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 5e-5
    proj_dim: int = 1024
    sim_coeff: float = 25.0
    std_coeff: float = 25.0
    cov_coeff: float = 1.0
    num_workers: int = 4
    seed: int = 0


@dataclass
class VICRegProgress:
    epoch: int  # 1-based
    total_epochs: int
    avg_loss: float
    metrics: dict = field(default_factory=dict)  # averaged sim/std/cov/total


@dataclass
class VICRegResult:
    tile_count: int = 0
    final_loss: float = 0.0
    weights_path: Path = VICREG_WEIGHTS_PATH
    # True when this result came from a cancelled run that still had at
    # least one completed epoch's improvement to fall back on -- see
    # `pretrain_vicreg`'s "best weights" tracking.
    cancelled: bool = False


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
    params: VICRegParams,
    trained_on_paths,
    output_dir: Path | None = None,
    started_from: Path | None = None,
) -> Path:
    """Write `backbone.pth`/`meta.json` into `output_dir` if given, else
    into the live warm-start slot (`VICREG_WEIGHTS_DIR`) -- backing up
    whatever's already there first, so a bad pretraining run can't destroy
    the last good backbone irreversibly (mirrors
    `training/supervised.py`'s `_save_weights`). Returns the directory
    written to.

    `trained_on_paths`: every tile path this run pretrained on -- recorded
    in meta.json as `trained_on` (see `tile_container.group_training_provenance`)
    so a later run can tell (via `warm_start_overlap`) how much of a newly
    pooled dataset this backbone has already been exposed to before
    deciding whether/how hard to warm-start from it again.

    `started_from`: the backbone this run was initialized from, or `None`
    for the ImageNet-pretrained stem -- recorded as `started_from`.

    meta.json's `pairing` key is also what marks a checkpoint as a
    backbone rather than a classifier (yeastprep's
    `checkpoint_choice.checkpoint_kind`)."""
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

    trained_on_paths = list(trained_on_paths)
    meta = {
        "pairing": "self-supervised (two augmented views of the same tile)",
        "tile_count": len(trained_on_paths),
        "trained_on": group_training_provenance(trained_on_paths),
        "params": {
            "epochs": params.epochs,
            "batch_size": params.batch_size,
            "lr": params.lr,
            "weight_decay": params.weight_decay,
            "proj_dim": params.proj_dim,
            "sim_coeff": params.sim_coeff,
            "std_coeff": params.std_coeff,
            "cov_coeff": params.cov_coeff,
        },
        "started_from": str(started_from) if started_from is not None else "imagenet",
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "description": (
            "VICReg-pretrained EfficientNet-B0 backbone (brightfield + "
            "fluorescence, mask applied, stem modified for 2 input "
            "channels), positive pairs from two augmented views of the "
            "same tile; no labels."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return weights_dir


def warm_start_overlap(paths, meta_path=VICREG_META_PATH) -> tuple[int, int] | None:
    """How much of `paths` (crop paths a caller is about to pretrain on)
    the backbone recorded at `meta_path` has already seen, as
    `(already_seen, total)` -- or `None` if `meta_path` doesn't exist, or
    exists but predates this field (an older checkpoint saved before
    `trained_on` was added). Used by the training UI to warn before a warm-started run: repeatedly
    exposing the backbone to the same crops across rounds isn't the
    validation-leakage bug that motivated always training the supervised
    classifier from scratch (VICReg never sees labels, so there's no
    held-out split to corrupt -- see `training/supervised.py`'s module
    docstring), but a user
    deliberately trying to broaden a backbone's exposure still wants to
    know how much of a newly pooled dataset is actually new to it."""
    import json

    meta_path = Path(meta_path)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None
    trained_on = meta.get("trained_on")
    if trained_on is None:
        return None
    paths = list(paths)
    already_seen = sum(
        1 for p in paths if training_provenance_contains(trained_on, p)
    )
    return already_seen, len(paths)


def pretrain_vicreg(
    paths,
    params: VICRegParams = VICRegParams(),
    progress_callback=None,
    cancel_check=None,
    output_dir: Path | None = None,
    backbone_weights_path: Path | None = None,
) -> VICRegResult:
    """`paths`: tile paths to pretrain on -- e.g. every tile in the pool
    (yeastprep's `ClassifierPool.tile_paths()`), annotated or not; labels
    are never used. The resulting backbone has no classification head, so
    any later supervised run, with any set of categories, can start from
    it.

    `backbone_weights_path`: a backbone to continue from (e.g. the deployed
    one), or `None` to start from an ImageNet-pretrained stem -- the same
    choice `supervised.train_classifier` takes. Continuing from a backbone
    is safe here in a way warm-starting the classifier head isn't: VICReg
    never sees labels, so repeat exposure to the same tiles is just more
    self-supervised training. Use `warm_start_overlap` to check how much
    of `paths` a backbone has already seen.

    Raises `ValueError` for too little data. If `cancel_check()` goes true
    between epochs: raises `TrainingCancelled` if no epoch has finished yet
    (nothing to save), otherwise returns a `VICRegResult` (with
    `cancelled=True`) built from whichever epoch had the lowest total VICReg
    loss so far, saved the same way a completed run's result would be.
    """
    paths = list(paths)
    if len(paths) < 2:
        raise ValueError(
            "Not enough tiles to pretrain VICReg -- need at least 2 (its "
            "variance and covariance terms are batch statistics)."
        )

    dataset = TileViewPairDataset(paths, transform=VICRegAugmentation())
    random.seed(params.seed)
    torch.manual_seed(params.seed)

    device = select_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Mirror supervised.train_classifier's guard against an empty loader
    # when there are fewer tiles than one batch.
    batch_size = min(params.batch_size, len(dataset))
    drop_last = len(dataset) > batch_size

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=params.num_workers,
        worker_init_fn=worker_init_fn if params.num_workers > 0 else None,
        persistent_workers=params.num_workers > 0,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
    )

    if backbone_weights_path is not None:
        backbone = build_yeast_efficientnet(num_classes=None, pretrained=False)
        backbone.load_state_dict(
            torch.load(backbone_weights_path, map_location="cpu")
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
    # Tracks the lowest-total-loss epoch seen so far -- recovered on
    # cancellation instead of discarding the whole run (mirrors
    # `supervised.train_classifier`'s val_accuracy-based tracking; VICReg
    # has no held-out split, so total loss is the closest analog here).
    best = {"loss": float("inf"), "state": None}

    try:
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
            if final_avg_loss < best["loss"]:
                best["loss"] = final_avg_loss
                best["state"] = _clone_state_dict(vicreg)

            if progress_callback is not None:
                progress_callback(
                    VICRegProgress(
                        epoch=epoch + 1,
                        total_epochs=params.epochs,
                        avg_loss=final_avg_loss,
                        metrics=avg_metrics,
                    )
                )
    except TrainingCancelled:
        # Nothing to fall back to (cancelled before a single epoch
        # finished) -- same as before, discard the run entirely.
        if best["state"] is None:
            raise
        vicreg.load_state_dict(best["state"])
        weights_dir = _save_backbone(
            vicreg,
            params,
            paths,
            output_dir=output_dir,
            started_from=backbone_weights_path,
        )
        return VICRegResult(
            tile_count=len(paths),
            final_loss=best["loss"],
            weights_path=weights_dir / "backbone.pth",
            cancelled=True,
        )

    weights_dir = _save_backbone(
        vicreg,
        params,
        paths,
        output_dir=output_dir,
        started_from=backbone_weights_path,
    )

    return VICRegResult(
        tile_count=len(paths),
        final_loss=final_avg_loss,
        weights_path=weights_dir / "backbone.pth",
    )
