"""Check whether the class-paired VICReg backbone (04_vicreg_class_pairs.py)
produces embeddings that separate the annotated categories *without* any
further backbone training: extract frozen embeddings, then (a) a
gradient-free k-NN vote and (b) a lightly trained linear head on top of
those embeddings -- both diagnostics never let the backbone itself
adapt, so a good score here is evidence the information lives in the
embedding, not in something the probe stage papered over.

Saves a 2D PCA scatter of the embedding space (colored by category) next
to the backbone weights for a quick visual read.

Usage:
    python 05_embedding_probe.py [--tiles-dir PATH] [--weights PATH]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wildtype_annotations import load_tagged_records

from tileclass.training.linear_probe import (
    LinearProbeParams,
    extract_embeddings,
    knn_accuracy,
    pca_2d,
    train_linear_probe,
)
from tileclass.training.vicreg import VICREG_WEIGHTS_PATH, load_backbone

DEFAULT_TILES_DIR = "/home/starrluxton/BurgessLab/wild-type/40X A/05_tiles"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-dir", default=DEFAULT_TILES_DIR)
    parser.add_argument("--weights", default=str(VICREG_WEIGHTS_PATH))
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument(
        "--plot-out",
        default=str(Path(__file__).parent / "embedding_pca.png"),
    )
    return parser.parse_args()


def plot_pca(coords, labels, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    categories = sorted(set(labels))
    cmap = plt.get_cmap("tab10" if len(categories) <= 10 else "tab20")
    labels = list(labels)
    for i, category in enumerate(categories):
        mask = [label == category for label in labels]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=12,
            alpha=0.7,
            color=cmap(i % cmap.N),
            label=f"{category} (n={sum(mask)})",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("VICReg embedding space (class-conditioned pairs)")
    ax.legend(loc="best", fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved PCA scatter to {out_path}")


if __name__ == "__main__":
    args = parse_args()

    records = load_tagged_records(args.tiles_dir)
    print(f"Loaded {len(records)} annotated crops from {args.tiles_dir}")
    counts = Counter(label for _, label in records)
    print("Class distribution:", dict(sorted(counts.items())))

    paths = [p for p, _ in records]
    labels = [label for _, label in records]
    categories = sorted(set(labels))

    backbone = load_backbone(weights_path=args.weights)
    device = next(backbone.parameters()).device
    embeddings = extract_embeddings(paths, backbone, device)
    print(f"Extracted embeddings: {embeddings.shape}")

    knn_acc = knn_accuracy(embeddings, labels, k=args.knn_k)
    print(f"k-NN (k={args.knn_k}) held-out accuracy: {knn_acc:.4f}")

    probe_params = LinearProbeParams(epochs=args.probe_epochs)
    probe_result = train_linear_probe(
        embeddings, labels, categories, params=probe_params
    )
    print(
        f"Linear probe val accuracy: {probe_result.val_accuracy:.4f} "
        f"(train={probe_result.train_count}, val={probe_result.val_count})"
    )
    print("Per-class val accuracy:")
    for category in categories:
        acc = probe_result.per_class_accuracy.get(category)
        acc_str = f"{acc:.2f}" if acc is not None else "n/a (no val examples)"
        print(f"  {category}: {acc_str}")

    coords = pca_2d(embeddings)
    plot_pca(coords, labels, args.plot_out)
