"""Pretrain the VICReg backbone with class-conditioned positive pairs:
instead of augmenting one image twice (classic VICReg), each pair is two
*different* annotated crops from the same category, each independently
augmented (see `tileclass.training.vicreg.ClassPairDataset`). Run this
after annotating tiles in the tiled-viewer GUI and before
05_embedding_probe.py, which checks whether the resulting embedding
space actually separates those categories.

Usage:
    python 04_vicreg_class_pairs.py [--tiles-dir PATH] [--epochs N]
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wildtype_annotations import load_tagged_records

from tileclass.training.vicreg import VICRegParams, pretrain_vicreg

DEFAULT_TILES_DIR = "/home/starrluxton/BurgessLab/wild-type/40X A/05_tiles"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-dir", default=DEFAULT_TILES_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--frequency-sampling",
        action="store_true",
        help="Sample pairs by category frequency instead of uniformly "
        "(default is uniform, so rare categories aren't drowned out).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    records = load_tagged_records(args.tiles_dir)
    print(f"Loaded {len(records)} annotated crops from {args.tiles_dir}")

    counts = Counter(label for _, label in records)
    print("Class distribution:", dict(sorted(counts.items())))
    singleton = sorted(c for c, n in counts.items() if n < 2)
    if singleton:
        print(
            f"Note: {singleton} have only 1 annotated example each -- "
            "their pairs fall back to the same crop augmented twice."
        )

    params = VICRegParams(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        balanced_sampling=not args.frequency_sampling,
    )

    def report(progress):
        m = progress.metrics
        print(
            f"Epoch {progress.epoch}/{progress.total_epochs} "
            f"loss={progress.avg_loss:.4f} "
            f"(sim={m['sim']:.4f} std={m['std']:.4f} cov={m['cov']:.4f})"
        )

    start_time = time.time()
    result = pretrain_vicreg(records, params=params, progress_callback=report)
    elapsed = time.time() - start_time

    print(f"Pretraining done in {elapsed:.1f}s")
    print(f"Saved backbone weights to {result.weights_path}")
    print(f"Categories seen: {result.categories}")
