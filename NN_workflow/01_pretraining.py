import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from yeastVIC import (
    MaskedMicroscopyDataset,
    VICRegLoss,
    VICRegModel,
    VICRegTransform,
    get_modified_efficientnet,
)

if __name__ == "__main__":
    # SSL pre-training doesn't need labels, so pool crops from every FOV folder
    crop_dirs = sorted(p for p in Path(".").glob("crops_F*") if p.is_dir())
    unlabeled_files = [
        str(p) for d in crop_dirs for p in d.glob("[!.]*.tif")
    ]
    print(f"Found {len(unlabeled_files)} images across {len(crop_dirs)} folders")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Data loading is CPU-bound (TIFF read + numpy masking), so use several
    # worker processes when multiple cores are available.
    num_workers = min(8, os.cpu_count() or 0)

    ssl_dataset = MaskedMicroscopyDataset(
        unlabeled_files, transform=VICRegTransform()
    )
    # VICReg requires a decent batch size to calculate reliable variance/covariance statistics
    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    num_epochs = 50
    log_interval = 5
    total_batches = len(ssl_loader)

    backbone = get_modified_efficientnet(num_classes=None)
    vicreg = VICRegModel(backbone).to(device)

    optimizer = torch.optim.AdamW(
        vicreg.parameters(), lr=1e-3, weight_decay=1e-4
    )
    criterion = VICRegLoss()

    # Training loop
    vicreg.train()
    start_time = time.time()

    for epoch in range(num_epochs):  # 50-100 epochs is typical for domain-specific SSL
        epoch_loss = 0.0
        for batch_idx, (x1, x2) in enumerate(ssl_loader):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()

            _, z1 = vicreg(x1)
            _, z2 = vicreg(x2)

            loss, metrics = criterion(z1, z2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch + 1}/{num_epochs} "
                    f"[{batch_idx + 1}/{total_batches}] "
                    f"loss={metrics['total']:.4f} "
                    f"(sim={metrics['sim']:.4f} std={metrics['std']:.4f} cov={metrics['cov']:.4f}) "
                    f"elapsed={elapsed:.1f}s"
                )

        print(
            f"Epoch {epoch + 1}/{num_epochs} done, "
            f"avg loss={epoch_loss / total_batches:.4f}"
        )

    # Save only the backbone weights for Phase 2
    torch.save(
        vicreg.backbone.state_dict(), "vicreg_efficientnet_stem_modified.pth"
    )
    print("Saved backbone weights to vicreg_efficientnet_stem_modified.pth")
