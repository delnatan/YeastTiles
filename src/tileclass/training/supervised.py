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
- Every run trains from scratch -- an ImageNet-pretrained stem (or a
  VICReg backbone via `backbone_weights_path`) with a freshly initialized
  head, never a warm start from whatever's currently deployed. This is a
  deliberate correctness choice, not an efficiency one: `train_classifier`
  recomputes the train/val split fresh from whatever's currently annotated
  every run (see `stratified_split` below), and annotations accumulate
  between runs in this app's annotate-and-retrain loop. A crop that landed
  in *train* last run can land in *val* this run once the pool has grown
  -- if the model were warm-started from last run's weights, that crop's
  validation accuracy this run would be silently optimistic, since the
  model already trained on it. Retraining from scratch every time keeps
  each run's validation split honestly held-out. (VICReg pretraining, in
  `training/vicreg.py`, has no such split and is warm-started deliberately
  -- see that module's docstring.)
"""

import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..classifiers.device import select_device
from ..classifiers.yeast_efficientnet import WEIGHTS_DIR, WEIGHTS_PATH
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
    probe_epochs: int = 100
    finetune_epochs: int = 500
    probe_lr: float = 5e-4
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
    """The label vocabulary this training run will use: every distinct
    category name present in `records`. Every run trains a freshly
    initialized head (see module docstring), so there's no existing
    classifier vocabulary to match shapes against -- pass an explicit
    `categories` to `train_classifier` instead if a run needs a fixed
    vocabulary wider than what's currently annotated."""
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


def _init_model(num_classes, device, backbone_weights_path=None):
    """Every run starts from a freshly initialized classifier head (see
    module docstring) -- never warm-started from whatever's currently
    deployed. Two starting points for the backbone itself:

    1. `backbone_weights_path`, if given -- a headless (no classifier
       head) backbone checkpoint such as `training.vicreg.pretrain_vicreg`
       produces. A fresh, randomly-initialized head is attached, since a
       VICReg backbone was never trained with one.
    2. Cold-start from an ImageNet-pretrained stem with a fresh random
       head, if `backbone_weights_path` isn't given (or doesn't exist).
    """
    import torch

    if backbone_weights_path is not None and Path(backbone_weights_path).exists():
        model = build_yeast_efficientnet(num_classes, pretrained=False)
        backbone_state = torch.load(backbone_weights_path, map_location=device)
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        expected_missing = {"classifier.1.weight", "classifier.1.bias"}
        if set(missing) != expected_missing or unexpected:
            raise ValueError(
                f"Backbone checkpoint at {backbone_weights_path} doesn't match "
                f"this model's architecture (missing={missing}, "
                f"unexpected={unexpected})"
            )
        return model.to(device)

    return build_yeast_efficientnet(num_classes, pretrained=True).to(device)


def _save_weights(
    model,
    categories: list[str],
    trained_on_paths,
    category_counts: dict[str, int] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Write `weights.pth`/`meta.json` into `output_dir` if given, else
    into the live inference slot (`WEIGHTS_DIR`) -- backing up whatever's
    already there first, so a bad training run can't destroy the last good
    model irreversibly. Returns the directory written to, so callers that
    default to `output_dir=None` can still report where the result landed.

    `trained_on_paths`: the crop paths this run actually trained on (the
    train split, not val) -- recorded in meta.json as provenance. Every
    supervised run starts from scratch, so nothing here needs to *warn*
    about repeat exposure the way `training/vicreg.py`'s equivalent field
    does for its warm-startable backbone; this is just an honest record of
    what went into this specific checkpoint.

    `category_counts`: number of annotated crops per category across this
    run's whole pool (train + val, unlike `trained_on_paths`) -- lets
    someone peeking at meta.json later (see `_CheckpointFilePicker`'s
    "View Metadata..." in yeastprep's Classifier Training page) see how
    lopsided the training data was without cross-referencing
    `trained_on_paths` by hand.
    """
    import torch

    weights_dir = Path(output_dir) if output_dir is not None else WEIGHTS_DIR
    weights_path = weights_dir / "weights.pth"
    meta_path = weights_dir / "meta.json"

    weights_dir.mkdir(parents=True, exist_ok=True)
    if weights_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = weights_dir / f"weights.pth.bak-{stamp}"
        shutil.copy2(weights_path, backup_path)

    torch.save(model.state_dict(), weights_path)

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["categories"] = categories
    if category_counts is not None:
        meta["category_counts"] = {c: category_counts.get(c, 0) for c in categories}
    meta["trained_on_paths"] = sorted(str(p) for p in trained_on_paths)
    meta["last_trained"] = datetime.now(timezone.utc).isoformat()
    meta.setdefault(
        "description",
        "Yeast cell-crop classifier (brightfield + fluorescence + mask, "
        "EfficientNet-B0 stem modified for 2 input channels).",
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    return weights_dir


def train_classifier(
    records,
    params: TrainingParams = TrainingParams(),
    progress_callback=None,
    cancel_check=None,
    backbone_weights_path=None,
    categories=None,
    output_dir: Path | None = None,
) -> TrainingResult:
    """`records`: list of (crop_path, category) pairs -- typically
    `PooledAnnotations.tagged_items()` filtered to human-confirmed tags
    only (see `workers.TrainingWorker`; training on the model's own
    unreviewed predictions would just reinforce its current mistakes).
    `backbone_weights_path`: optional headless VICReg backbone checkpoint
    (see `training.vicreg.pretrain_vicreg`) to start from instead of an
    ImageNet-pretrained stem -- see `_init_model`. The classifier head
    itself is always freshly initialized either way (module docstring).
    `categories`: explicit target vocabulary, overriding
    `resolve_target_categories`'s records-derived one -- for training a
    classifier over a wider/different vocabulary than what's currently
    annotated (e.g. matching a deployed classifier's category list even
    though this run's records don't cover every one of them). Pass
    `sorted(set(label for _, label in records))` for "just use whatever
    categories these records contain" (the default).
    `output_dir`: where to write the resulting `weights.pth`/`meta.json`.
    Defaults to the live inference slot (today's behavior: training
    auto-deploys). Callers that want an explicit, reviewable deploy step
    instead (e.g. yeastprep's Classifier Training page, which always
    trains on a pooled multi-project dataset that shouldn't silently
    overwrite tileclass's currently deployed model) should pass a
    project-local session folder here and promote it later via
    `tileclass.checkpoint_import.import_checkpoint`.
    Raises `ValueError` for too little data, an unrecognized category, or
    a `backbone_weights_path` that doesn't match this model's
    architecture; `TrainingCancelled` if `cancel_check()` goes true
    between epochs."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    categories = categories if categories is not None else resolve_target_categories(records)
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
    model = _init_model(num_classes, device, backbone_weights_path=backbone_weights_path)

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

    category_counts = Counter(label for _, label in records)
    weights_dir = _save_weights(
        model, categories, train_paths, category_counts=category_counts, output_dir=output_dir
    )

    return TrainingResult(
        val_accuracy=val_acc,
        per_class_accuracy={
            categories[c]: float(acc) for c, acc in enumerate(per_class_acc)
        },
        categories=categories,
        train_count=len(train_idx),
        val_count=len(val_idx),
        weights_path=weights_dir / "weights.pth",
    )
