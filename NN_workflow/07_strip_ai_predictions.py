"""Clear stale AI predictions from a tiles directory's annotation sidecar
files, leaving human-confirmed tags completely untouched.

As the deployed classifier improves with more training data, its old
unreviewed predictions (tagged via `update_with_confidence`, carrying a
confidence score -- see `tileclass.data.annotations`) become stale. Run
this before 08_classify_tile_set.py so the newer model gets to
re-predict every tile that hasn't been manually reviewed, instead of
leaving it skipped because it already carries an old AI tag.

Tags a human has set or confirmed (confidence is None) are never
touched, regardless of how this script is run.

Usage:
    python 07_strip_ai_predictions.py [--tiles-dir PATH]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from wildtype_annotations import pooled_annotations

DEFAULT_TILES_DIR = "/home/starrluxton/BurgessLab/wild-type/40X A/05_tiles"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-dir", default=DEFAULT_TILES_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pooled = pooled_annotations(args.tiles_dir)
    removed = pooled.clear_unconfirmed()
    print(f"Removed {removed} unreviewed AI prediction(s) from {args.tiles_dir}")
    print("Human-confirmed tags were left untouched.")
