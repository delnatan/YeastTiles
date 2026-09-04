"""Load (crop_path, category) records for the class-paired VICReg
scripts (04_vicreg_class_pairs.py / 05_embedding_probe.py) from a
`tileclass`-annotated tiles directory -- the same `TileAnnotations`
sidecar format the tiled-viewer GUI reads and writes (see
`tileclass.data.annotations`), so these standalone scripts see exactly
what a human annotated in the app: a tiles directory whose immediate
subdirectories are per-FOV crop folders, each with a sibling
`<fov_name>.txt` annotation file.
"""

from pathlib import Path

from tileclass.data.pooled_annotations import PooledAnnotations


def pooled_annotations(tiles_dir):
    """Build the `PooledAnnotations` covering every per-FOV crop folder
    directly under `tiles_dir` -- shared by every NN_workflow script that
    needs to read or write tags for a whole tiles directory at once."""
    folders = sorted(p for p in Path(tiles_dir).iterdir() if p.is_dir())
    return PooledAnnotations(folders)


def load_tagged_records(tiles_dir):
    """Only human-confirmed tags (confidence is None) -- matches
    `tileclass.training.supervised`'s stated rationale for the same
    filter: training on the model's own unreviewed predictions would
    just reinforce its current mistakes."""
    pooled = pooled_annotations(tiles_dir)
    return [
        (path, category)
        for path, category, confidence in pooled.tagged_items()
        if confidence is None
    ]
