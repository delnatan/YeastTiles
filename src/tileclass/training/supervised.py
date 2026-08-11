"""Supervised fine-tuning of the deployed yeast-tile classifier.

Ported from NN_workflow/02_supervised_training.py's two-stage procedure
(frozen-backbone linear probe, then full unfreeze at a low LR) into a
plain, Qt-free function callable from a background QThread (see
`tileclass/workers.py`) -- `progress_callback`/`cancel_check` stand in for
that script's `print()`s and its lack of any way to stop early.

Differences from the original script, both deliberate:

- Records come from `PooledAnnotations.tagged_items()` in-process (a list
  of already-resolved (path, label) pairs), not by re-parsing
  `crops_F*.txt` files off disk -- tileclass already has this data loaded.
- The backbone starts from whatever's currently deployed
  (`classifiers/yeast_efficientnet.py`'s `weights.pth`) if present, rather
  than requiring a separate VICReg-only checkpoint -- this is continual
  fine-tuning of the production classifier, not a from-scratch retrain
  each time, and VICReg pretraining isn't wired into this app yet (see
  training/__init__.py). Falls back to an ImageNet-pretrained stem for a
  first-ever training run with nothing deployed yet.
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..classifiers.device import select_device
from ..classifiers.yeast_efficientnet import META_PATH, WEIGHTS_DIR, WEIGHTS_PATH
from .dataset import (
    ClassificationTransform,
    MaskedMicroscopyDataset,
    class_weights,
    stratified_split,
)
from .model import build_yeast_efficientnet


class TrainingCancelled(Exception):
    pass


@dataclass
class TrainingParams:
    val_frac: float = 0.2
    seed: int = 0
    batch_size: int = 64
    probe_epochs: int = 10
    finetune_epochs: int = 20
    probe_lr: float = 1e-3
    finetune_backbone_lr: float = 1e-5
    finetune_head_lr: float = 1e-4
    weight_decay: float = 1e-4


@dataclass
class TrainingProgress:
    stage: str  # "probe" | "finetune"
    epoch: int  # 1-based
    total_epochs: int
    avg_loss: float | None = None
    val_accuracy: float | None = None


@dataclass
class TrainingResult:
    val_accuracy: float
    per_class_accuracy: dict = field(default_factory=dict)  # category -> accuracy
    categories: list = field(default_factory=list)
    train_count: int = 0
    val_count: int = 0
    weights_path: Path = WEIGHTS_PATH


def resolve_target_categories(records) -> list[str]:
    """The fixed label vocabulary this training run will use: the
    currently deployed classifier's categories if one exists (so the
    classifier head can be warm-started), else every distinct category
    name present in `records` (a first-ever training run, nothing
    deployed yet)."""
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())["categories"]
    return sorted({label for _, label in records})


def _evaluate(model, loader, device, num_classes):
    import torch

    model.eval()
    correct, total = 0, 0
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
            for c in range(num_classes):
                mask = y == c
                class_total[c] += mask.sum().item()
                class_correct[c] += (preds[mask] == c).sum().item()
    model.train()
    per_class_acc = torch.where(
        class_total > 0, class_correct / class_total.clamp(min=1), class_total
    )
    return correct / total, per_class_acc


def _run_stage(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_classes,
    num_epochs,
    stage_name,
    progress_callback,
    cancel_check,
):
    val_acc, per_class_acc = 0.0, None
    for epoch in range(num_epochs):
        if cancel_check is not None and cancel_check():
            raise TrainingCancelled()

        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        val_acc, per_class_acc = _evaluate(model, val_loader, device, num_classes)
        if progress_callback is not None:
            progress_callback(
                TrainingProgress(
                    stage=stage_name,
                    epoch=epoch + 1,
                    total_epochs=num_epochs,
                    avg_loss=avg_loss,
                    val_accuracy=val_acc,
                )
            )
    return val_acc, per_class_acc


def _init_model(num_classes, categories, device):
    """Warm-start from the currently deployed classifier if its category
    vocabulary matches exactly (so the head's shape lines up); otherwise
    cold-start from an ImageNet-pretrained stem with a fresh random head."""
    import torch

    if WEIGHTS_PATH.exists() and META_PATH.exists():
        deployed_categories = json.loads(META_PATH.read_text())["categories"]
        if deployed_categories == categories:
            model = build_yeast_efficientnet(num_classes, pretrained=False)
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
            return model.to(device)
    return build_yeast_efficientnet(num_classes, pretrained=True).to(device)


def _save_weights(model, categories: list[str]):
    """Back up whatever's currently deployed (if anything) before
    overwriting -- a bad training run shouldn't be able to destroy the
    last good model irreversibly."""
    import torch

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if WEIGHTS_PATH.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = WEIGHTS_DIR / f"weights.pth.bak-{stamp}"
        shutil.copy2(WEIGHTS_PATH, backup_path)

    torch.save(model.state_dict(), WEIGHTS_PATH)

    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    meta["categories"] = categories
    meta["last_trained"] = datetime.now(timezone.utc).isoformat()
    meta.setdefault(
        "description",
        "Yeast cell-crop classifier (brightfield + fluorescence + mask, "
        "EfficientNet-B0 stem modified for 2 input channels).",
    )
    META_PATH.write_text(json.dumps(meta, indent=2))


def train_classifier(
    records,
    params: TrainingParams = TrainingParams(),
    progress_callback=None,
    cancel_check=None,
) -> TrainingResult:
    """`records`: list of (crop_path, category) pairs -- typically
    `PooledAnnotations.tagged_items()` filtered to human-confirmed tags
    only (see `workers.TrainingWorker`; training on the model's own
    unreviewed predictions would just reinforce its current mistakes).
    Raises `ValueError` for too little data or an unrecognized category,
    `TrainingCancelled` if `cancel_check()` goes true between epochs."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    categories = resolve_target_categories(records)
    label_to_idx = {name: i for i, name in enumerate(categories)}

    unknown = sorted({label for _, label in records if label not in label_to_idx})
    if unknown:
        raise ValueError(f"Unrecognized categories: {unknown}")

    paths = [p for p, _ in records]
    labels = [label_to_idx[label] for _, label in records]

    train_idx, val_idx = stratified_split(labels, val_frac=params.val_frac, seed=params.seed)
    if not train_idx or not val_idx:
        raise ValueError(
            "Not enough annotated tiles to form a train/validation split -- "
            "annotate more tiles (ideally at least a couple per category) first."
        )

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    train_dataset = MaskedMicroscopyDataset(
        train_paths, labels=train_labels, transform=ClassificationTransform(train=True)
    )
    val_dataset = MaskedMicroscopyDataset(
        val_paths, labels=val_labels, transform=ClassificationTransform(train=False)
    )

    device = select_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # A small early-annotation dataset shouldn't silently train on zero
    # batches: with drop_last=True unconditionally, a train set smaller
    # than one batch would produce an empty loader. Only drop the
    # trailing partial batch once there's more than one full batch's
    # worth of data (a size-1 trailing batch would also break BatchNorm,
    # which needs batch > 1).
    batch_size = min(params.batch_size, len(train_dataset))
    drop_last = len(train_dataset) > batch_size

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    val_loader = DataLoader(
        val_dataset, batch_size=min(params.batch_size, len(val_dataset)), shuffle=False
    )

    num_classes = len(categories)
    model = _init_model(num_classes, categories, device)

    weights = class_weights(train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Stage 1: linear probe -- freeze the backbone, train only the head.
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=params.probe_lr,
        weight_decay=params.weight_decay,
    )
    model.train()
    _run_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        num_classes,
        params.probe_epochs,
        "probe",
        progress_callback,
        cancel_check,
    )

    # Stage 2: unfreeze everything, fine-tune end-to-end at a low LR so
    # existing backbone features aren't wrecked by the small labeled set.
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": params.finetune_backbone_lr},
            {"params": model.classifier.parameters(), "lr": params.finetune_head_lr},
        ],
        weight_decay=params.weight_decay,
    )
    val_acc, per_class_acc = _run_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        num_classes,
        params.finetune_epochs,
        "finetune",
        progress_callback,
        cancel_check,
    )

    _save_weights(model, categories)

    return TrainingResult(
        val_accuracy=val_acc,
        per_class_accuracy={
            categories[c]: float(acc) for c, acc in enumerate(per_class_acc)
        },
        categories=categories,
        train_count=len(train_idx),
        val_count=len(val_idx),
        weights_path=WEIGHTS_PATH,
    )
