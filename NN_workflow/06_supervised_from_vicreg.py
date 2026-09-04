"""Train the deployed yeast-tile classifier warm-started from the
class-paired VICReg backbone (04_vicreg_class_pairs.py), on the same
wild-type annotations. This is the wild-type-data equivalent of
02_supervised_training.py, which is hardwired to a different dataset
(Data/Daniel/Bri's crops_F*.txt convention and its own category list) --
this script instead uses `tileclass.training.supervised.train_classifier`,
the general version that already backs the tiled-viewer GUI's "Train"
button, so a run here also updates what the GUI classifies with.

Usage:
    python 06_supervised_from_vicreg.py [--tiles-dir PATH] [--backbone PATH]
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wildtype_annotations import load_tagged_records

from tileclass.training.supervised import TrainingParams, train_classifier
from tileclass.training.vicreg import VICREG_WEIGHTS_PATH

DEFAULT_TILES_DIR = "/home/starrluxton/BurgessLab/wild-type/40X A/05_tiles"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-dir", default=DEFAULT_TILES_DIR)
    parser.add_argument("--backbone", default=str(VICREG_WEIGHTS_PATH))
    parser.add_argument("--probe-epochs", type=int, default=10)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    records = load_tagged_records(args.tiles_dir)
    print(f"Loaded {len(records)} annotated crops from {args.tiles_dir}")
    print("Class distribution:", dict(sorted(Counter(l for _, l in records).items())))

    params = TrainingParams(
        probe_epochs=args.probe_epochs, finetune_epochs=args.finetune_epochs
    )

    def report(progress):
        print(
            f"[{progress.stage}] epoch {progress.epoch}/{progress.total_epochs} "
            f"loss={progress.avg_loss:.4f} val_acc={progress.val_accuracy:.4f}"
        )

    # Explicit, rather than inferred from whatever classifier happens to
    # already be deployed -- see train_classifier's `categories` docstring:
    # a previously-deployed classifier for an unrelated experiment/
    # vocabulary shouldn't dictate this run's category list.
    categories = sorted({label for _, label in records})

    start_time = time.time()
    result = train_classifier(
        records,
        params=params,
        progress_callback=report,
        backbone_weights_path=args.backbone,
        categories=categories,
    )
    elapsed = time.time() - start_time

    print(f"Training done in {elapsed:.1f}s")
    print(f"Val accuracy: {result.val_accuracy:.4f}")
    print("Per-class val accuracy:")
    for category, acc in result.per_class_accuracy.items():
        print(f"  {category}: {acc:.2f}")
    print(f"Saved classifier weights to {result.weights_path}")
