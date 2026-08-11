"""Run the trained classifier on crop TIFFs.

Usage:
    python3 03_inference.py crops_F0                       # every crop in a folder -> CSV
    python3 03_inference.py crops_F0/ndj1_01_cell00001.tif  # a single crop -> printed
"""

import sys
from pathlib import Path

import polars as pl
import torch
from torch.utils.data import DataLoader

from yeastVIC import ClassificationTransform, MaskedMicroscopyDataset, get_modified_efficientnet

WEIGHTS = "yeast_classifier_efficientnet.pth"
CATEGORIES = ["single", "tetrad", "two", "junk", "weird", "empty", "missegment"]


def load_model(device):
    model = get_modified_efficientnet(num_classes=len(CATEGORIES), pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def classify(model, paths, device, batch_size=64):
    """Return (predicted_label, confidence) for each path in `paths`."""
    dataset = MaskedMicroscopyDataset(paths, transform=ClassificationTransform(train=False))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    labels, confidences = [], []
    for x in loader:
        x = x.to(device)
        probs = torch.softmax(model(x), dim=1)
        conf, pred = probs.max(dim=1)
        labels.extend(CATEGORIES[p] for p in pred.cpu().tolist())
        confidences.extend(conf.cpu().tolist())
    return labels, confidences


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("crops_F0")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = load_model(device)

    if target.is_dir():
        paths = sorted(str(p) for p in target.glob("[!.]*.tif"))
        print(f"Classifying {len(paths)} crops in {target}...")
        labels, confidences = classify(model, paths, device)

        out_csv = f"{target.name}_predictions.csv"
        pl.DataFrame(
            {"crop_path": paths, "predicted_label": labels, "confidence": confidences}
        ).write_csv(out_csv)
        print(f"Saved predictions to {out_csv}")
    else:
        labels, confidences = classify(model, [str(target)], device)
        print(f"{target}: {labels[0]} (confidence={confidences[0]:.3f})")
