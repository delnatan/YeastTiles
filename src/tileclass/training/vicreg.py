"""Self-supervised VICReg backbone pretraining (design.md stage 4's other
half; see `training/supervised.py` for the supervised classifier head
that gets built on top of this backbone afterwards).

Standard VICReg: each positive pair is two independently augmented views
of the *same* tile (`TileViewPairDataset`), so no labels are involved and
every tile in the pool can be used, annotated or not. The backbone never
trains on a label. Annotated tiles, when given, are used only to *score*
checkpoints (a kNN vote, every few epochs) and pick which epoch to save --
see `pretrain_vicreg` for what that does and doesn't leak.

The augmentations (`VICRegAugmentation`) define what the embedding learns
to ignore: orientation, a little framing, the exact cell outline,
per-channel intensity scaling, blur, sensor noise and small patches of
brightfield. Anything they don't touch -- where the fluorescence sits in
the cell, how many foci there are -- is what it keeps.

Optimization: AdamW with linear warmup then cosine decay to zero
(`_lr_lambda`), no weight decay on BatchNorm parameters or biases
(`_param_groups`).
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..classifiers.device import select_device
from .dataset import ClampTensor, RandomGaussianNoise, load_crop_with_mask
from .embeddings import extract_embeddings, knn_accuracy
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


class RandomMaskJitter:
    """Dilate or erode the (1, H, W) 0/1 cell mask by 1..`max_px` pixels.
    The two views of a tile otherwise share an identical silhouette, so the
    network could match them on outline and size alone without ever looking
    at the fluorescence pattern -- jittering the outline per view takes that
    shortcut away. Dilation reveals real pixels just outside the segmented
    cell (tiles store the image unmasked), erosion trims the edge; neither
    touches the cell's interior, so nuclei stay put."""

    def __init__(self, max_px=2, p=0.5):
        self.max_px = max_px
        self.p = p

    def __call__(self, mask):
        if random.random() >= self.p:
            return mask
        r = random.randint(1, self.max_px)
        k = 2 * r + 1
        m = mask.unsqueeze(0)
        if random.random() < 0.5:
            m = nn.functional.max_pool2d(m, k, stride=1, padding=r)
        else:
            m = -nn.functional.max_pool2d(-m, k, stride=1, padding=r)
        return m.squeeze(0)


class VICRegAugmentation:
    """One augmented view; `TileViewPairDataset` calls it twice per tile.

    Takes a (3, H, W) [brightfield, fluorescence, mask] crop with the
    intensity channels *unmasked* (`dataset.load_crop_with_mask`) and
    returns the (2, H, W) masked view the backbone sees -- the same format
    as `dataset.load_masked_crop` at inference.

    1. Spatial, applied jointly to image and mask: a mild random-resized
       crop (cells are centered, so a deep crop would cut them off), flips,
       exact 90-degree turns and a further small rotation -- together any
       orientation. Resampling is bilinear throughout: nearest-neighbour
       (torchvision v2's default for rotation) makes jagged edges that can
       merge or split small DAPI foci on a 64x64 tile.
    2. Mask jitter (`RandomMaskJitter`), then the mask is applied.
    3. Per-channel intensity jitter, occasional blur, sensor noise -- then
       the mask is applied *again*, so the background is exact zeros as at
       inference rather than blurred or noisy.
    4. Occasional small erased patches in the brightfield channel only, so
       the embedding can't lean on one patch of brightfield texture.
       Fluorescence is never erased: wiping a DAPI focus would change what
       the tile *is* (a double into a single), not just how it looks.
    """

    def __init__(self, crop_size=64, crop_scale=(0.6, 1.0)):
        import torchvision.transforms.v2 as T

        bilinear = T.InterpolationMode.BILINEAR
        self._spatial = T.Compose(
            [
                T.RandomResizedCrop(
                    size=(crop_size, crop_size),
                    scale=crop_scale,
                    ratio=(0.9, 1.1),
                    interpolation=bilinear,
                    antialias=True,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                RandomRot90(),
                T.RandomRotation(degrees=45, interpolation=bilinear),
            ]
        )
        self._mask_jitter = RandomMaskJitter(max_px=2, p=0.5)
        self._photometric = T.Compose(
            [
                RandomChannelIntensity(),
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.3
                ),
                RandomGaussianNoise(std=0.02, p=0.5),
            ]
        )
        self._erase_bf = T.RandomErasing(
            p=0.25, scale=(0.01, 0.05), ratio=(0.5, 2.0), value=0.0
        )
        self._clamp = ClampTensor(min_val=0.0, max_val=1.0)

    def __call__(self, x):
        x = self._spatial(x)
        image = x[:2]
        # Bilinear resampling leaves fractional mask edges; re-binarize.
        mask = (x[2:3] > 0.5).to(image.dtype)
        mask = self._mask_jitter(mask)

        image = self._photometric(image * mask) * mask
        bf = self._erase_bf(image[:1])
        image = torch.cat([bf, image[1:]], dim=0)
        return self._clamp(image)


def worker_init_fn(_worker_id):
    """DataLoader workers fork with Python's global `random` state already
    seeded from the parent process, so left alone every worker draws the
    identical sequence from it (a fork copies, it doesn't reseed) -- and
    `RandomRot90`/`RandomChannelIntensity` use it. Reseed from the
    per-worker seed torch already assigned via its own generator."""
    random.seed(torch.utils.data.get_worker_info().seed % (2**32))


class TileViewPairDataset(Dataset):
    """VICReg positive pairs: two independent `transform` draws of the
    same tile. Unlabeled -- just tile paths. `transform` receives the
    (3, H, W) unmasked-image-plus-mask crop from `load_crop_with_mask`
    and is responsible for applying the mask (as `VICRegAugmentation`
    does)."""

    def __init__(self, paths, transform):
        if not paths:
            raise ValueError("no tile paths given")
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = torch.from_numpy(load_crop_with_mask(self.paths[idx]))
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
    # An epoch is one pass over every pooled tile, annotated or not. With
    # the cosine schedule the run length sets how long the LR decays over;
    # ~200 epochs of a ~30k-tile pool is where the kNN score typically
    # levels off -- watch it and shorten if it plateaus early.
    epochs: int = 200
    # 128 peaks at ~7 GiB on a 12 GB card (EfficientNet-B0, stride-1 stem,
    # two views); 192 is ~10.5 GiB, too tight, and 256 runs out of memory.
    # Scale `lr` roughly in proportion if you change this.
    batch_size: int = 128
    # Peak learning rate: reached after `warmup_epochs` of linear warmup,
    # then cosine-decayed to zero by the last epoch (`_lr_lambda`).
    lr: float = 5e-4
    warmup_epochs: int = 10
    # Applied to weights only -- never to BatchNorm parameters or biases
    # (`_param_groups`).
    weight_decay: float = 5e-5
    proj_dim: int = 1024
    sim_coeff: float = 25.0
    std_coeff: float = 25.0
    cov_coeff: float = 1.0
    num_workers: int = 4
    seed: int = 0
    # Every this many epochs (and after the last), score the backbone with
    # a kNN vote over the annotated tiles passed as `eval_records` -- see
    # `pretrain_vicreg`. 0 turns the monitor off.
    eval_every: int = 5


@dataclass
class VICRegProgress:
    epoch: int  # 1-based
    total_epochs: int
    avg_loss: float
    metrics: dict = field(default_factory=dict)  # averaged sim/std/cov/total
    lr: float = 0.0  # learning rate at the end of the epoch
    # Held-out kNN accuracy of the backbone's features on the annotated
    # tiles (`embeddings.knn_accuracy`, the same score as Explore
    # Embeddings) -- None on epochs where it wasn't computed.
    knn_accuracy: float | None = None


@dataclass
class VICRegResult:
    tile_count: int = 0
    final_loss: float = 0.0
    weights_path: Path = VICREG_WEIGHTS_PATH
    # Which epoch's weights were saved (1-based) and how they were chosen:
    # "knn" (best kNN accuracy), "loss" (lowest loss; cancelled run with no
    # kNN score) or "last" (completed run with no kNN score).
    selected_epoch: int = 0
    selected_by: str = "last"
    knn_accuracy: float | None = None  # of the saved weights, if scored
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
    selection: dict | None = None,
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

    `selection`: which epoch was saved and why, plus the run's kNN
    history -- recorded as `selection`.

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
            "warmup_epochs": params.warmup_epochs,
            "schedule": "linear warmup, cosine decay",
            "weight_decay": params.weight_decay,
            "proj_dim": params.proj_dim,
            "sim_coeff": params.sim_coeff,
            "std_coeff": params.std_coeff,
            "cov_coeff": params.cov_coeff,
        },
        "started_from": str(started_from) if started_from is not None else "imagenet",
        "selection": selection or {},
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


def _lr_lambda(warmup_steps: int, total_steps: int):
    """Per-step LR multiplier: linear warmup from ~0 to 1 over
    `warmup_steps`, then cosine decay to 0 at `total_steps`. A constant LR
    tends to leave self-supervised training at a worse point; warmup keeps
    the first, largest updates from wrecking the ImageNet (or warm-start)
    features before the projector has settled."""
    import math

    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return factor


def _param_groups(model, weight_decay):
    """Weight decay on conv/linear weights only. BatchNorm scales/shifts
    and biases (every parameter with fewer than 2 dims) are excluded --
    decaying them just pulls normalization toward zero for no
    regularization benefit."""
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _knn_score(backbone, eval_records, device):
    """Held-out kNN accuracy of `backbone`'s 1280-dim features on
    `eval_records` -- `embeddings.knn_accuracy`, the same number Explore
    Embeddings reports. None if the records can't form a split (fewer
    than two categories, or too few tiles). Leaves `backbone` in eval
    mode; the caller switches back to train."""
    paths = [p for p, _ in eval_records]
    labels = [label for _, label in eval_records]
    if len(set(labels)) < 2:
        return None
    embeddings = extract_embeddings(paths, backbone, device)
    try:
        return knn_accuracy(embeddings, labels)
    except ValueError:
        return None


def pretrain_vicreg(
    paths,
    params: VICRegParams = VICRegParams(),
    progress_callback=None,
    cancel_check=None,
    output_dir: Path | None = None,
    backbone_weights_path: Path | None = None,
    eval_records=None,
) -> VICRegResult:
    """`paths`: tile paths to pretrain on -- e.g. every tile in the pool
    (yeastprep's `PoolScope.tile_paths()`), annotated or not; labels
    are never used for training. The resulting backbone has no
    classification head, so any later supervised run, with any set of
    categories, can start from it.

    `backbone_weights_path`: a backbone to continue from (e.g. the deployed
    one), or `None` to start from an ImageNet-pretrained stem -- the same
    choice `supervised.train_classifier` takes. Continuing from a backbone
    is safe here in a way warm-starting the classifier head isn't: VICReg
    never trains on labels, so repeat exposure to the same tiles is just
    more self-supervised training. Use `warm_start_overlap` to check how
    much of `paths` a backbone has already seen.

    `eval_records`: optional (path, category) pairs -- the human-annotated
    tiles -- scored every `params.eval_every` epochs (and after the last)
    with a kNN vote over the backbone's features. The VICReg loss itself
    says little about whether the embedding separates the categories you
    care about; this does, and it's what picks which epoch gets saved.
    Caveat: picking an epoch by a score on the annotated tiles lets those
    labels nudge the choice of backbone, so a supervised run on top of it
    may report a slightly optimistic validation accuracy. It's one choice
    among a handful of checkpoints, not training, so the effect is small;
    pass `eval_records=None` (or `eval_every=0`) to keep the backbone
    strictly label-free -- it then saves the last epoch.

    Raises `ValueError` for too little data. If `cancel_check()` goes true
    between epochs: raises `TrainingCancelled` if no epoch has finished yet
    (nothing to save), otherwise returns a `VICRegResult` (with
    `cancelled=True`) built from the best epoch so far -- by kNN accuracy
    if any epoch was scored, else by lowest total VICReg loss -- saved the
    same way a completed run's result would be.
    """
    paths = list(paths)
    if len(paths) < 2:
        raise ValueError(
            "Not enough tiles to pretrain VICReg -- need at least 2 (its "
            "variance and covariance terms are batch statistics)."
        )
    eval_records = list(eval_records or [])
    monitor = params.eval_every > 0 and len({c for _, c in eval_records}) >= 2

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
        _param_groups(vicreg, params.weight_decay), lr=params.lr
    )
    steps_per_epoch = max(1, len(loader))
    warmup_steps = min(params.warmup_epochs, params.epochs) * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _lr_lambda(max(1, warmup_steps), max(1, params.epochs * steps_per_epoch)),
    )
    criterion = VICRegLoss(
        sim_coeff=params.sim_coeff,
        std_coeff=params.std_coeff,
        cov_coeff=params.cov_coeff,
    )

    vicreg.train()
    final_avg_loss = 0.0
    knn_history = []  # [epoch, accuracy] pairs, for meta.json
    # Best epoch so far, by kNN accuracy when scored, else by total loss
    # (VICReg has no held-out split of its own). Saved on completion when
    # kNN-scored; recovered on cancellation either way.
    best_knn = {"score": -1.0, "loss": 0.0, "epoch": 0, "state": None}
    best_loss = {"loss": float("inf"), "epoch": 0, "state": None}

    def selection(selected_by, epoch, knn):
        return {
            "selected_by": selected_by,
            "selected_epoch": epoch,
            "knn_accuracy": knn,
            "knn_history": knn_history,
            "eval_tile_count": len(eval_records) if monitor else 0,
        }

    def save(selected_by, epoch, loss, knn, cancelled=False):
        weights_dir = _save_backbone(
            vicreg,
            params,
            paths,
            output_dir=output_dir,
            started_from=backbone_weights_path,
            selection=selection(selected_by, epoch, knn),
        )
        return VICRegResult(
            tile_count=len(paths),
            final_loss=loss,
            weights_path=weights_dir / "backbone.pth",
            cancelled=cancelled,
            selected_epoch=epoch,
            selected_by=selected_by,
            knn_accuracy=knn,
        )

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
                scheduler.step()

                for key, value in metrics.items():
                    epoch_totals[key] += value
                num_batches += 1

            num_batches = max(num_batches, 1)
            avg_metrics = {k: v / num_batches for k, v in epoch_totals.items()}
            final_avg_loss = avg_metrics["total"]
            if final_avg_loss < best_loss["loss"]:
                best_loss.update(
                    loss=final_avg_loss, epoch=epoch + 1, state=_clone_state_dict(vicreg)
                )

            knn = None
            last_epoch = epoch + 1 == params.epochs
            if monitor and ((epoch + 1) % params.eval_every == 0 or last_epoch):
                knn = _knn_score(vicreg.backbone, eval_records, device)
                vicreg.train()
                if knn is not None:
                    knn_history.append([epoch + 1, knn])
                    if knn > best_knn["score"]:
                        best_knn.update(
                            score=knn,
                            loss=final_avg_loss,
                            epoch=epoch + 1,
                            state=_clone_state_dict(vicreg),
                        )

            if progress_callback is not None:
                progress_callback(
                    VICRegProgress(
                        epoch=epoch + 1,
                        total_epochs=params.epochs,
                        avg_loss=final_avg_loss,
                        metrics=avg_metrics,
                        lr=scheduler.get_last_lr()[0],
                        knn_accuracy=knn,
                    )
                )
    except TrainingCancelled:
        if best_knn["state"] is not None:
            vicreg.load_state_dict(best_knn["state"])
            return save("knn", best_knn["epoch"], best_knn["loss"], best_knn["score"], True)
        # Nothing to fall back to (cancelled before a single epoch
        # finished) -- discard the run entirely.
        if best_loss["state"] is None:
            raise
        vicreg.load_state_dict(best_loss["state"])
        return save("loss", best_loss["epoch"], best_loss["loss"], None, True)

    if best_knn["state"] is not None:
        vicreg.load_state_dict(best_knn["state"])
        return save("knn", best_knn["epoch"], best_knn["loss"], best_knn["score"])
    return save("last", params.epochs, final_avg_loss, None)
