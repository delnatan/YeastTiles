"""Classify every tile in a tiles directory with the currently deployed
classifier (`tileclass.classifiers.yeast_efficientnet`), tagging every
untagged tile and logging a run summary so classifier performance can be
tracked over time as more training data is collected.

- Tiles that already carry a tag (human-confirmed *or* a still-standing
  AI prediction) are never overwritten -- run 07_strip_ai_predictions.py
  first if you want a newer model to re-predict everything that isn't
  human-confirmed.
- Every tile is still run through the model, though, so that tiles with
  a *human-confirmed* tag can be used as a held-out check: comparing the
  model's prediction against that ground truth gives an agreement rate
  for this run, appended to a CSV history log alongside confidence
  stats and the predicted category distribution -- since every AI tag
  carries a confidence score, this trend is easy to build up run over
  run without any extra bookkeeping.

Usage:
    python 08_classify_tile_set.py [--tiles-dir PATH] [--history PATH]
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wildtype_annotations import pooled_annotations

from tileclass.classifiers.yeast_efficientnet import META_PATH, YeastEfficientNetClassifier
from tileclass.scan import scan_folder

DEFAULT_TILES_DIR = "/home/starrluxton/BurgessLab/wild-type/40X A/05_tiles"
DEFAULT_HISTORY_PATH = Path(__file__).parent / "classification_history.csv"

HISTORY_FIELDS = [
    "timestamp",
    "tiles_dir",
    "model_last_trained",
    "n_total",
    "n_newly_tagged",
    "n_human_confirmed",
    "n_agree_with_human",
    "accuracy_vs_human",
    "mean_confidence",
    "median_confidence",
    "category_counts",
    "per_category_accuracy_vs_human",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-dir", default=DEFAULT_TILES_DIR)
    parser.add_argument("--history", default=str(DEFAULT_HISTORY_PATH))
    return parser.parse_args()


def append_history_row(history_path, row):
    history_path = Path(history_path)
    is_new = not history_path.exists()
    with open(history_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    args = parse_args()

    pooled = pooled_annotations(args.tiles_dir)
    # Normalized to match PooledAnnotations.folders' own normalization
    # (os.path.normpath(os.path.abspath(...))) so `pooled.get`/`.confidence`
    # can match each tile to its owning folder.
    paths = [os.path.normpath(os.path.abspath(p)) for p in scan_folder(args.tiles_dir)]
    print(f"Found {len(paths)} tiles under {args.tiles_dir}")

    classifier = YeastEfficientNetClassifier()
    predictions = classifier.predict(paths)

    updates = []
    category_counts = Counter()
    confidences = []
    n_human_confirmed = 0
    n_agree = 0
    per_category_total = Counter()
    per_category_agree = Counter()

    for path, (label, confidence) in zip(paths, predictions):
        category_counts[label] += 1
        confidences.append(confidence)

        existing_category = pooled.get(path)
        if existing_category is None:
            updates.append((path, label, confidence))
            continue

        if pooled.confidence(path) is None:
            # Human-confirmed tag -- a free held-out check of this run's
            # predictions against ground truth.
            n_human_confirmed += 1
            per_category_total[existing_category] += 1
            if label == existing_category:
                n_agree += 1
                per_category_agree[existing_category] += 1

    pooled.update_with_confidence(updates)

    accuracy = n_agree / n_human_confirmed if n_human_confirmed else None
    per_category_accuracy = {
        category: per_category_agree[category] / total
        for category, total in per_category_total.items()
    }
    model_last_trained = (
        json.loads(META_PATH.read_text()).get("last_trained") if META_PATH.exists() else None
    )

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tiles_dir": str(args.tiles_dir),
        "model_last_trained": model_last_trained,
        "n_total": len(paths),
        "n_newly_tagged": len(updates),
        "n_human_confirmed": n_human_confirmed,
        "n_agree_with_human": n_agree,
        "accuracy_vs_human": accuracy,
        "mean_confidence": statistics.fmean(confidences) if confidences else None,
        "median_confidence": statistics.median(confidences) if confidences else None,
        "category_counts": json.dumps(dict(sorted(category_counts.items()))),
        "per_category_accuracy_vs_human": json.dumps(
            dict(sorted(per_category_accuracy.items()))
        ),
    }
    append_history_row(args.history, row)

    print(f"Tagged {len(updates)} previously-untagged tile(s).")
    print(f"Category distribution (all tiles): {dict(sorted(category_counts.items()))}")
    print(
        f"Mean confidence: {row['mean_confidence']:.4f}"
        if confidences
        else "Mean confidence: n/a (no tiles)"
    )
    if n_human_confirmed:
        print(
            f"Agreement with {n_human_confirmed} human-confirmed tile(s): "
            f"{accuracy:.4f} ({n_agree}/{n_human_confirmed})"
        )
        print(f"  per-category: {per_category_accuracy}")
    else:
        print("No human-confirmed tiles found to check agreement against.")
    print(f"Run summary appended to {args.history}")
