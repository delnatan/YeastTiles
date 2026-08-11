import os
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yeastVIC import (
    ClassificationTransform,
    MaskedMicroscopyDataset,
    class_weights,
    get_modified_efficientnet,
    load_annotations,
    stratified_split,
)

BACKBONE_WEIGHTS = "vicreg_efficientnet_stem_modified.pth"
OUT_WEIGHTS = "yeast_classifier_efficientnet.pth"

# crops_F3.txt used "binucleate" where every other file used "two" for the
# same class (confirmed: F3 has zero "two" labels, and this is the only file
# with "binucleate"), so merge it in.
LABEL_ALIAS = {"binucleate": "two"}
CATEGORIES = ["single", "tetrad", "two", "junk", "weird", "empty", "missegment"]
LABEL_TO_IDX = {name: idx for idx, name in enumerate(CATEGORIES)}


def evaluate(model, loader, device, num_classes):
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


def train_stage(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_classes,
    num_epochs,
    stage_name,
    log_interval=10,
):
    total_batches = len(train_loader)
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"[{stage_name}] epoch {epoch + 1}/{num_epochs} "
                    f"[{batch_idx + 1}/{total_batches}] loss={loss.item():.4f}"
                )

        val_acc, per_class_acc = evaluate(model, val_loader, device, num_classes)
        per_class_str = ", ".join(
            f"{CATEGORIES[c]}={acc:.2f}" for c, acc in enumerate(per_class_acc)
        )
        print(
            f"[{stage_name}] epoch {epoch + 1}/{num_epochs} done, "
            f"avg loss={epoch_loss / total_batches:.4f} val_acc={val_acc:.4f}\n"
            f"  per-class val acc: {per_class_str}"
        )


if __name__ == "__main__":
    txt_files = sorted(Path(".").glob("crops_F*.txt"))
    records = load_annotations(txt_files, label_alias=LABEL_ALIAS)
    print(f"Loaded {len(records)} annotated crops across {len(txt_files)} files")

    unknown = {label for _, label in records if label not in LABEL_TO_IDX}
    if unknown:
        raise ValueError(f"Unrecognized labels found: {unknown}")

    paths = [p for p, _ in records]
    labels = [LABEL_TO_IDX[label] for _, label in records]
    print(
        "Class distribution:",
        {CATEGORIES[c]: n for c, n in sorted(Counter(labels).items())},
    )

    train_idx, val_idx = stratified_split(labels, val_frac=0.2, seed=0)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

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

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    num_workers = min(8, os.cpu_count() or 0)
    batch_size = 64

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
    )

    num_classes = len(CATEGORIES)
    model = get_modified_efficientnet(num_classes=num_classes, pretrained=False)

    backbone_state = torch.load(BACKBONE_WEIGHTS, map_location="cpu")
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    assert not unexpected, f"Unexpected keys when loading backbone: {unexpected}"
    assert set(missing) == {"classifier.1.weight", "classifier.1.bias"}, (
        f"Unexpected missing keys when loading backbone: {missing}"
    )
    print(f"Loaded VICReg backbone from {BACKBONE_WEIGHTS}")
    model = model.to(device)

    weights = class_weights(train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Stage 1: linear probe -- freeze the VICReg backbone, train only the
    # randomly-initialized classifier head.
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4
    )
    model.train()
    start_time = time.time()
    train_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        num_classes,
        num_epochs=10,
        stage_name="probe",
    )
    print(f"Linear probe done in {time.time() - start_time:.1f}s")

    # Stage 2: unfreeze everything, fine-tune end-to-end at a low LR so the
    # SSL-pretrained backbone features aren't wrecked by the small labeled set.
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )
    start_time = time.time()
    train_stage(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        num_classes,
        num_epochs=20,
        stage_name="finetune",
    )
    print(f"Fine-tuning done in {time.time() - start_time:.1f}s")

    torch.save(model.state_dict(), OUT_WEIGHTS)
    print(f"Saved classifier weights to {OUT_WEIGHTS}")
