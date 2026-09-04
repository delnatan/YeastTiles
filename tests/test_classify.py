"""Tests for yeastprep.core.classify's pure filesystem-convention helpers
and its pooled-inference driver."""

import numpy as np
from tileclass.data.pooled_annotations import PooledAnnotations
from tileclass.tile_container import write_container

from yeastprep.core.classify import (
    ClassifyPoolResult,
    checkpoint_dir_for_project,
    classify_pool,
    default_supervised_checkpoint_paths,
    default_vicreg_checkpoint_paths,
    discover_fov_dirs,
    find_project_checkpoint,
)


def test_discover_fov_dirs_lists_only_visible_containers(tmp_path):
    tiles_dir = tmp_path / "05_tiles"
    tiles_dir.mkdir(parents=True)
    (tiles_dir / "fov_001.tiles").write_bytes(b"")
    (tiles_dir / "fov_002.tiles").write_bytes(b"")
    (tiles_dir / ".hidden_fov.tiles").write_bytes(b"")
    (tiles_dir / "tile_index.csv").write_text("")

    assert discover_fov_dirs(tmp_path) == [
        tiles_dir / "fov_001.tiles",
        tiles_dir / "fov_002.tiles",
    ]


def test_discover_fov_dirs_missing_tiles_stage_returns_empty(tmp_path):
    assert discover_fov_dirs(tmp_path) == []


def test_checkpoint_dir_for_project_is_sibling_of_tiles_stage(tmp_path):
    assert checkpoint_dir_for_project(tmp_path) == tmp_path / "06_classifier"


def test_default_checkpoint_paths_are_distinct_per_kind(tmp_path):
    sup_weights, sup_meta = default_supervised_checkpoint_paths(tmp_path)
    vic_weights, vic_meta = default_vicreg_checkpoint_paths(tmp_path)

    checkpoint_dir = checkpoint_dir_for_project(tmp_path)
    assert checkpoint_dir in sup_weights.parents
    assert checkpoint_dir in vic_weights.parents
    assert sup_weights.parent != vic_weights.parent  # separate subfolders
    assert {sup_weights, sup_meta} != {vic_weights, vic_meta}


def test_find_project_checkpoint_none_when_not_yet_trained(tmp_path):
    assert find_project_checkpoint(tmp_path, "supervised") is None
    assert find_project_checkpoint(tmp_path, "vicreg") is None


def test_find_project_checkpoint_found_once_both_files_exist(tmp_path):
    weights_path, meta_path = default_supervised_checkpoint_paths(tmp_path)
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"fake weights")
    meta_path.write_text("{}")

    found = find_project_checkpoint(tmp_path, "supervised")
    assert found == (weights_path, meta_path)


def test_find_project_checkpoint_none_when_only_one_file_exists(tmp_path):
    weights_path, _meta_path = default_vicreg_checkpoint_paths(tmp_path)
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"fake weights")
    # meta.json not written yet

    assert find_project_checkpoint(tmp_path, "vicreg") is None


class _FakeClassifier:
    """Always predicts `label` with `confidence` -- enough to exercise
    `classify_pool`'s tagging/agreement bookkeeping without a real model."""

    categories = ["single", "junk"]

    def __init__(self, label="single", confidence=0.9):
        self.label = label
        self.confidence = confidence

    def predict(self, paths):
        return [(self.label, self.confidence) for _ in paths]


def _make_fov(tmp_path, n_tiles=4):
    tiles_dir = tmp_path / "proj1" / "05_tiles"
    tiles_dir.mkdir(parents=True)
    container = tiles_dir / "fov_001.tiles"
    cells = [
        (f"fov_001_cell{i:05d}", i, np.zeros((3, 4, 4), dtype=np.uint8)) for i in range(n_tiles)
    ]
    write_container(container, cells)
    return container


def test_classify_pool_tags_only_untagged_tiles(tmp_path):
    fov = _make_fov(tmp_path, n_tiles=4)
    sidecar = fov.parent / "fov_001.txt"
    sidecar.write_text(
        "#categories\tsingle\tjunk\n#dims\tCYX\n"
        "fov_001_cell00000.tif\tsingle\n"  # human-confirmed
        "fov_001_cell00001.tif\tjunk\t0.5\n"  # existing AI prediction
    )
    pooled = PooledAnnotations([str(fov)])

    result = classify_pool(pooled, _FakeClassifier(label="single", confidence=0.9))

    assert isinstance(result, ClassifyPoolResult)
    assert result.n_total == 4
    assert result.n_newly_tagged == 2  # cell00002, cell00003 only
    tagged = dict((path, (category, confidence)) for path, category, confidence in pooled.tagged_items())
    assert tagged[f"{fov}/fov_001_cell00000.tif"] == ("single", None)  # untouched
    assert tagged[f"{fov}/fov_001_cell00001.tif"] == ("junk", 0.5)  # untouched
    assert tagged[f"{fov}/fov_001_cell00002.tif"] == ("single", 0.9)  # newly tagged
    assert tagged[f"{fov}/fov_001_cell00003.tif"] == ("single", 0.9)  # newly tagged


def test_classify_pool_computes_agreement_with_human_confirmed(tmp_path):
    fov = _make_fov(tmp_path, n_tiles=2)
    sidecar = fov.parent / "fov_001.txt"
    sidecar.write_text(
        "#categories\tsingle\tjunk\n#dims\tCYX\n"
        "fov_001_cell00000.tif\tsingle\n"
        "fov_001_cell00001.tif\tjunk\n"
    )
    pooled = PooledAnnotations([str(fov)])

    # Predicts "single" for everything -- agrees with cell00000, disagrees with cell00001.
    result = classify_pool(pooled, _FakeClassifier(label="single", confidence=1.0))

    assert result.n_human_confirmed == 2
    assert result.n_agree_with_human == 1
    assert result.accuracy_vs_human == 0.5
    assert result.n_newly_tagged == 0  # both tiles were already tagged


def test_classify_pool_empty_pool(tmp_path):
    fov = _make_fov(tmp_path, n_tiles=0)
    pooled = PooledAnnotations([str(fov)])

    result = classify_pool(pooled, _FakeClassifier())

    assert result.n_total == 0
    assert result.mean_confidence is None
    assert result.accuracy_vs_human is None
