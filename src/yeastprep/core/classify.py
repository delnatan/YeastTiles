"""Filesystem conventions for classifier-training sessions run from
yeastprep's Classifier Training page. Training itself (`train_classifier`,
`pretrain_vicreg`) lives in `tileclass.training` -- already fully decoupled
from tileclass's Qt/annotation UI, so this module only supplies the
yeastprep-side path conventions (which FOV folders a project offers to pool,
and where a training session's non-live output checkpoint goes), mirroring
`core/denoise.py`'s `checkpoint_filename_for_channel`/`find_project_checkpoint`.

Unlike denoise, a classifier checkpoint is never written to a project folder
directly by the training run -- see `checkpoint_dir_for_project` below --
because a single pooled training session spans multiple projects at once, so
there's no one project it "belongs" to; the session output directory is just
a convenient, discoverable default, not project state.
"""

import os
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from tileclass.scan import scan_container

from .fs_status import list_visible
from .project import STAGE_TILES

CHECKPOINT_STAGE = "06_classifier"


def discover_fov_dirs(project_root) -> list[Path]:
    """FOV containers under `project_root/05_tiles/`, one per exported field
    of view (see `core/tiles.py`'s `export_tiles`) -- the checkable leaves
    a pool tree offers for a given pooled project."""
    tiles_dir = Path(project_root) / STAGE_TILES
    return sorted(p for p in list_visible(tiles_dir, "*.tiles") if p.is_file())


def checkpoint_dir_for_project(project_root) -> Path:
    """Per-project home for training-session outputs, sibling to
    `05_tiles/` rather than inside it so exported tile crops and trained
    checkpoints never mix in the same directory listing."""
    return Path(project_root) / CHECKPOINT_STAGE


def supervised_output_dir(project_root) -> Path:
    """Directory `train_classifier(..., output_dir=...)` writes
    `weights.pth`/`meta.json` into (matches its own live-slot filenames,
    see `tileclass.classifiers.yeast_efficientnet.WEIGHTS_PATH`/
    `META_PATH`) -- kept in its own subfolder, not the vicreg one below,
    so a project that runs both kinds of training doesn't have one
    overwrite the other's same-named files."""
    return checkpoint_dir_for_project(project_root) / "classifier"


def vicreg_output_dir(project_root) -> Path:
    """Directory `pretrain_vicreg(..., output_dir=...)` writes
    `backbone.pth`/`meta.json` into (matches
    `tileclass.training.vicreg`'s own `VICREG_WEIGHTS_PATH`/
    `VICREG_META_PATH` filenames)."""
    return checkpoint_dir_for_project(project_root) / "vicreg_backbone"


def default_supervised_checkpoint_paths(project_root) -> tuple[Path, Path]:
    output_dir = supervised_output_dir(project_root)
    return output_dir / "weights.pth", output_dir / "meta.json"


def default_vicreg_checkpoint_paths(project_root) -> tuple[Path, Path]:
    output_dir = vicreg_output_dir(project_root)
    return output_dir / "backbone.pth", output_dir / "meta.json"


_DEFAULT_PATHS_BY_KIND = {
    "supervised": default_supervised_checkpoint_paths,
    "vicreg": default_vicreg_checkpoint_paths,
}


def find_project_checkpoint(project_root, kind: str) -> tuple[Path, Path] | None:
    """The `kind` ("supervised" or "vicreg") checkpoint pair sitting in
    `project_root`'s session output folder, if a previous run already
    produced one -- lets the training page offer a project's last session
    output (e.g. for redeploy) without the user re-Browsing to it."""
    weights_path, meta_path = _DEFAULT_PATHS_BY_KIND[kind](project_root)
    if weights_path.is_file() and meta_path.is_file():
        return weights_path, meta_path
    return None


@dataclass
class ClassifyPoolResult:
    n_total: int = 0
    n_newly_tagged: int = 0
    n_human_confirmed: int = 0
    n_agree_with_human: int = 0
    category_counts: dict = field(default_factory=dict)
    mean_confidence: float | None = None

    @property
    def accuracy_vs_human(self) -> float | None:
        return self.n_agree_with_human / self.n_human_confirmed if self.n_human_confirmed else None


def classify_pool(pooled, classifier) -> ClassifyPoolResult:
    """Run `classifier` (a `tileclass.classifiers.base.TileClassifier`,
    e.g. a `YeastEfficientNetClassifier` pointed at a just-trained,
    not-yet-deployed session checkpoint) over every tile crop under
    `pooled`'s folders (a `tileclass.data.pooled_annotations.PooledAnnotations`),
    tagging any tile that has no existing tag yet -- ported from
    `NN_workflow/08_classify_tile_set.py`'s standalone script into a
    reusable, Qt-free function.

    A tile that already carries a tag -- human-confirmed *or* a
    still-standing AI prediction -- is never overwritten, matching
    `tileclass`'s own Auto-Annotate convention: re-running after a newer
    model is trained only fills in what's still blank, unless the sidecar
    files are stripped of unreviewed predictions first
    (`PooledAnnotations.clear_unconfirmed`). Every tile is still run
    through the model regardless, so a human-confirmed tile doubles as a
    free held-out accuracy check (`n_agree_with_human`/`accuracy_vs_human`)
    for whatever checkpoint `classifier` wraps.
    """
    paths = []
    for folder in pooled.folders:
        paths.extend(scan_container(folder))
    # PooledAnnotations dispatches per-tile calls by exact string match
    # against its own normalized folder roots -- see its module docstring.
    paths = [os.path.normpath(os.path.abspath(p)) for p in paths]

    predictions = classifier.predict(paths)

    updates = []
    category_counts = Counter()
    confidences = []
    n_human_confirmed = 0
    n_agree = 0
    for path, (label, confidence) in zip(paths, predictions):
        category_counts[label] += 1
        confidences.append(confidence)

        existing_category = pooled.get(path)
        if existing_category is None:
            updates.append((path, label, confidence))
            continue

        if pooled.confidence(path) is None:
            n_human_confirmed += 1
            if label == existing_category:
                n_agree += 1

    pooled.update_with_confidence(updates)

    return ClassifyPoolResult(
        n_total=len(paths),
        n_newly_tagged=len(updates),
        n_human_confirmed=n_human_confirmed,
        n_agree_with_human=n_agree,
        category_counts=dict(category_counts),
        mean_confidence=statistics.fmean(confidences) if confidences else None,
    )
