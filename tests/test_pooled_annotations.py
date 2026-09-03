"""Tests for tileclass.data.pooled_annotations.PooledAnnotations -- pure
logic, no Qt needed. Uses real TileAnnotations sidecar files on tmp_path,
matching the project's convention of exercising real dependencies rather
than mocking them.
"""

from pathlib import Path

from tileclass.data.pooled_annotations import PooledAnnotations


def _make_folder(tmp_path, name) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    (folder / "a.tif").touch()
    (folder / "b.tif").touch()
    return folder


def test_get_and_update_route_to_the_owning_folder(tmp_path):
    folder1 = _make_folder(tmp_path, "expA")
    folder2 = _make_folder(tmp_path, "expB")
    pool = PooledAnnotations([folder1, folder2])

    path1 = str(folder1 / "a.tif")
    path2 = str(folder2 / "a.tif")

    pool.update([(path1, "single"), (path2, "tetrad")])

    assert pool.get(path1) == "single"
    assert pool.get(path2) == "tetrad"
    # Each folder's own sidecar only knows about its own tile.
    assert (tmp_path / "expA.txt").exists()
    assert (tmp_path / "expB.txt").exists()
    assert "a.tif\tsingle" in (tmp_path / "expA.txt").read_text()
    assert "a.tif\ttetrad" in (tmp_path / "expB.txt").read_text()


def test_update_batches_one_save_per_touched_folder(tmp_path):
    folder1 = _make_folder(tmp_path, "expA")
    folder2 = _make_folder(tmp_path, "expB")
    pool = PooledAnnotations([folder1, folder2])

    pool.update(
        [
            (str(folder1 / "a.tif"), "single"),
            (str(folder1 / "b.tif"), "tetrad"),
            (str(folder2 / "a.tif"), "junk"),
        ]
    )

    assert pool.get(str(folder1 / "a.tif")) == "single"
    assert pool.get(str(folder1 / "b.tif")) == "tetrad"
    assert pool.get(str(folder2 / "a.tif")) == "junk"


def test_categories_returns_union_across_folders(tmp_path):
    folder1 = _make_folder(tmp_path, "expA")
    folder2 = _make_folder(tmp_path, "expB")
    pool = PooledAnnotations([folder1, folder2])

    pool.add_category("single")
    # Simulate folder2 already having its own vocabulary before pooling.
    from tileclass.data.annotations import TileAnnotations

    store2 = TileAnnotations(str(folder2))
    store2.add_category("weird")

    pool2 = PooledAnnotations([folder1, folder2])
    assert pool2.categories() == ["single", "weird"]


def test_add_category_propagates_to_every_pooled_folder(tmp_path):
    folder1 = _make_folder(tmp_path, "expA")
    folder2 = _make_folder(tmp_path, "expB")
    pool = PooledAnnotations([folder1, folder2])

    pool.add_category("junk")

    from tileclass.data.annotations import TileAnnotations

    assert "junk" in TileAnnotations(str(folder1)).categories()
    assert "junk" in TileAnnotations(str(folder2)).categories()


def test_usage_count_sums_across_folders(tmp_path):
    folder1 = _make_folder(tmp_path, "expA")
    folder2 = _make_folder(tmp_path, "expB")
    pool = PooledAnnotations([folder1, folder2])

    pool.update(
        [
            (str(folder1 / "a.tif"), "single"),
            (str(folder1 / "b.tif"), "single"),
            (str(folder2 / "a.tif"), "single"),
        ]
    )

    assert pool.usage_count("single") == 3


def test_tagged_items_returns_absolute_paths_with_confidence(tmp_path):
    folder1 = _make_folder(tmp_path, "expA")
    pool = PooledAnnotations([folder1])

    pool.update([(str(folder1 / "a.tif"), "single")])
    pool.update_with_confidence([(str(folder1 / "b.tif"), "junk", 0.42)])

    items = {(Path(p).name, c, conf) for p, c, conf in pool.tagged_items()}
    assert items == {("a.tif", "single", None), ("b.tif", "junk", 0.42)}


def test_clear_unconfirmed_drops_ai_tags_but_keeps_human_tags(tmp_path):
    folder1 = _make_folder(tmp_path, "expA")
    folder2 = _make_folder(tmp_path, "expB")
    pool = PooledAnnotations([folder1, folder2])

    pool.update([(str(folder1 / "a.tif"), "single")])
    pool.update_with_confidence(
        [
            (str(folder1 / "b.tif"), "junk", 0.42),
            (str(folder2 / "a.tif"), "tetrad", 0.91),
        ]
    )

    removed = pool.clear_unconfirmed()

    assert removed == 2
    assert pool.get(str(folder1 / "a.tif")) == "single"
    assert pool.get(str(folder1 / "b.tif")) is None
    assert pool.get(str(folder2 / "a.tif")) is None
    # Reload from disk to confirm the removal was actually persisted.
    reloaded = PooledAnnotations([folder1, folder2])
    assert reloaded.get(str(folder1 / "a.tif")) == "single"
    assert reloaded.get(str(folder1 / "b.tif")) is None


def test_categories_collapses_case_and_whitespace_variants(tmp_path):
    from tileclass.data.annotations import TileAnnotations

    folder1 = _make_folder(tmp_path, "expA")
    folder2 = _make_folder(tmp_path, "expB")

    TileAnnotations(str(folder1)).add_category("two")
    TileAnnotations(str(folder2)).add_category(" Two ")

    pool = PooledAnnotations([folder1, folder2])
    assert pool.categories() == ["two"]  # first-seen spelling wins


def test_add_category_no_ops_for_existing_case_or_whitespace_variant(tmp_path):
    from tileclass.data.annotations import TileAnnotations

    folder = _make_folder(tmp_path, "expA")
    store = TileAnnotations(str(folder))
    store.add_category("two")
    store.add_category(" TWO")

    assert store.categories() == ["two"]


def test_raw_category_names_surfaces_divergence_categories_hides(tmp_path):
    from tileclass.data.annotations import TileAnnotations

    folder1 = _make_folder(tmp_path, "expA")
    folder2 = _make_folder(tmp_path, "expB")

    TileAnnotations(str(folder1)).add_category("two")
    TileAnnotations(str(folder2)).add_category(" Two ")

    pool = PooledAnnotations([folder1, folder2])
    assert pool.categories() == ["two"]
    assert pool.raw_category_names() == ["Two", "two"]


def test_add_folders_extends_pool_and_skips_already_pooled(tmp_path):
    import os

    folder1 = _make_folder(tmp_path, "expA")
    folder2 = _make_folder(tmp_path, "expB")
    pool = PooledAnnotations([folder1])

    added = pool.add_folders([folder1, folder2])

    norm_folder2 = os.path.normpath(os.path.abspath(str(folder2)))
    assert added == [norm_folder2]  # folder1 skipped, already pooled
    assert norm_folder2 in pool.folders

    pool.update([(str(folder2 / "a.tif"), "single")])
    assert pool.get(str(folder2 / "a.tif")) == "single"


def test_single_folder_pool_matches_plain_tile_annotations_behavior(tmp_path):
    folder = _make_folder(tmp_path, "solo")
    pool = PooledAnnotations([folder])

    assert pool.label() == "solo"
    assert pool.root_dir == str(folder)

    pool.update([(str(folder / "a.tif"), "single")])
    assert pool.get(str(folder / "a.tif")) == "single"
    assert list(pool.values()) == ["single"]


def test_get_unknown_path_raises_keyerror(tmp_path):
    folder1 = _make_folder(tmp_path, "expA")
    other = tmp_path / "not_pooled"
    other.mkdir()
    pool = PooledAnnotations([folder1])

    try:
        pool.get(str(other / "a.tif"))
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for a path outside every pooled folder")
