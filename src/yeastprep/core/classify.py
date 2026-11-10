"""Filesystem conventions for classifier-training sessions run from
yeastprep's Classifier Training page. Training itself (`train_classifier`,
`pretrain_vicreg`) lives in `tileclass.training` -- already fully decoupled
from tileclass's Qt/annotation UI, so this module only supplies the
yeastprep-side path conventions (which FOV folders a project offers to pool,
and where a training session's non-live output checkpoint goes).

A classifier checkpoint is never written to a project folder directly by
the training run -- see `checkpoint_dir_for_project` below --
because a single pooled training session spans multiple projects at once, so
there's no one project it "belongs" to; the session output directory is just
a convenient, discoverable default, not project state.
"""

import os
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from tileclass.scan import scan_container
from tileclass.tile_container import training_provenance_contains

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


# Tiles handed to `classifier.predict` at a time: the grain of
# `classify_pool`'s progress reports.
INFERENCE_CHUNK = 256


@dataclass
class ClassifyPoolResult:
    n_total: int = 0  # tiles scored
    n_newly_tagged: int = 0  # had no tag before
    n_replaced: int = 0  # an earlier AI prediction, re-predicted
    n_human_confirmed: int = 0
    n_agree_with_human: int = 0
    # The human-confirmed tiles the checkpoint was *not* trained on, and how
    # many of those it agrees with -- None when the checkpoint's training
    # record can't be matched to these files (see `_usable_provenance`).
    n_human_unseen: int | None = None
    n_agree_unseen: int = 0
    category_counts: dict = field(default_factory=dict)
    mean_confidence: float | None = None

    @property
    def accuracy_vs_human(self) -> float | None:
        return self.n_agree_with_human / self.n_human_confirmed if self.n_human_confirmed else None

    @property
    def accuracy_unseen(self) -> float | None:
        return self.n_agree_unseen / self.n_human_unseen if self.n_human_unseen else None


def _usable_provenance(classifier):
    """`classifier.trained_on` (`tile_container.group_training_provenance`'s
    `{container path: [cell labels]}`), or None if it can't say which of
    this machine's tiles were trained on: it records absolute paths, so a
    checkpoint trained elsewhere (or before a project moved) matches
    nothing here, and "not trained on" would wrongly mean every tile."""
    trained_on = getattr(classifier, "trained_on", None)
    if not trained_on or not any(os.path.exists(path) for path in trained_on):
        return None
    return {path: set(labels) for path, labels in trained_on.items()}


def classify_pool(pooled, classifier, progress_callback=None) -> ClassifyPoolResult:
    """Run `classifier` (a `tileclass.classifiers.base.TileClassifier`)
    over every tile under `pooled`'s folders (a
    `tileclass.data.pooled_annotations.PooledAnnotations`), one FOV at a
    time, writing each FOV's predictions as it finishes.

    Every tile without a human-set or accepted tag gets this classifier's
    prediction -- including one still carrying an earlier, unreviewed AI
    prediction, so after a run every prediction in the pool comes from the
    one checkpoint rather than a mix of whichever models ran before. A
    human-set tag is never overwritten.

    The human-confirmed tiles are scored too, as an accuracy check. Most of
    them are usually the checkpoint's own training tiles, which it has
    memorized -- `n_human_unseen`/`accuracy_unseen` count only the ones it
    wasn't trained on (its validation split, plus anything annotated
    since), which is the number that says how far to trust the rest.

    `progress_callback(done, total)` is called every `INFERENCE_CHUNK`
    tiles.
    """
    per_folder = [
        # PooledAnnotations dispatches per-tile calls by exact string match
        # against its own normalized folder roots -- see its module docstring.
        (folder, [os.path.normpath(os.path.abspath(p)) for p in scan_container(folder)])
        for folder in pooled.folders
    ]
    total = sum(len(paths) for _folder, paths in per_folder)
    trained_on = _usable_provenance(classifier)

    result = ClassifyPoolResult(n_total=total, n_human_unseen=0 if trained_on else None)
    category_counts = Counter()
    confidence_sum = 0.0
    done = 0

    for _folder, paths in per_folder:
        predictions = []
        for start in range(0, len(paths), INFERENCE_CHUNK):
            chunk = paths[start : start + INFERENCE_CHUNK]
            predictions.extend(classifier.predict(chunk))
            done += len(chunk)
            if progress_callback is not None:
                progress_callback(done, total)

        updates = []
        for path, (label, confidence) in zip(paths, predictions):
            category_counts[label] += 1
            confidence_sum += confidence

            existing_category = pooled.get(path)
            if existing_category is None:
                result.n_newly_tagged += 1
                updates.append((path, label, confidence))
            elif pooled.confidence(path) is not None:
                result.n_replaced += 1
                updates.append((path, label, confidence))
            else:
                agrees = label == existing_category
                result.n_human_confirmed += 1
                result.n_agree_with_human += agrees
                if trained_on and not training_provenance_contains(trained_on, path):
                    result.n_human_unseen += 1
                    result.n_agree_unseen += agrees
        pooled.update_with_confidence(updates)

    result.category_counts = dict(category_counts)
    result.mean_confidence = confidence_sum / total if total else None
    return result


def unlabeled_paths(pooled) -> list[str]:
    """Every not-yet-human-reviewed tile path in `pooled`'s checked folders
    -- for the Classify Tiles page's "Explore Embeddings" group, which
    plots them (in gray) alongside the annotated ones. Same path-gathering
    as `classify_pool` (`scan_container` per folder, normalized abspath),
    filtered to whatever `pooled` has no *human-confirmed* tag for -- a
    still-standing AI prediction still counts as "unlabeled" here:
    otherwise, running "Run Inference on Pool" first -- which tags every
    untagged tile with an AI prediction -- would leave nothing unlabeled
    to show."""
    paths = []
    for folder in pooled.folders:
        paths.extend(scan_container(folder))
    paths = [os.path.normpath(os.path.abspath(p)) for p in paths]
    return [p for p in paths if pooled.get(p) is None or pooled.confidence(p) is not None]


def sample_unlabeled(pooled, n: int, rng: random.Random | None = None) -> list[str]:
    """A random `n` of `unlabeled_paths(pooled)` (all of them if there are
    no more than `n`) -- for a pool too big to embed whole."""
    rng = rng or random.Random()
    untagged = unlabeled_paths(pooled)
    if len(untagged) <= n:
        return untagged
    return rng.sample(untagged, n)
