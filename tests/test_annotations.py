"""Tests for tileclass.data.annotations.TileAnnotations: the sidecar-file
naming contract, migration of old `.tif`-suffixed keys, concurrent
writers sharing one sidecar, and the per-folder `categories.tsv`
vocabulary shared by every `.tiles` container in a project.
"""

from tileclass.data.annotations import CATEGORIES_FILE_NAME, TileAnnotations
from tileclass.data.pooled_annotations import PooledAnnotations


def test_container_sidecar_is_named_after_the_fov(tmp_path):
    """`05_tiles/<fov>.tiles` keeps its tags in `05_tiles/<fov>.txt`."""
    store = TileAnnotations(str(tmp_path / "fov1.tiles"))
    assert store.file_path == str(tmp_path / "fov1.txt")


def test_legacy_tif_keys_are_migrated_on_load(tmp_path):
    sidecar = tmp_path / "fov1.txt"
    sidecar.write_text(
        "#categories\tsingle\tbudded\n"
        "fov1_cell00001.tif\tsingle\n"
        "fov1_cell00002.tif\tbudded\t0.9000\n"
    )

    store = TileAnnotations(str(tmp_path / "fov1.tiles"))
    assert store.get("fov1_cell00001") == "single"
    assert store.confidence("fov1_cell00002") == 0.9
    assert ".tif" not in sidecar.read_text()
    # The rewrite moved the header's vocabulary into the shared file.
    assert "#categories" not in sidecar.read_text()
    assert (tmp_path / CATEGORIES_FILE_NAME).read_text() == "single\nbudded\n"


def test_concurrent_writer_does_not_erase_other_writers_tags(tmp_path):
    """Two stores open on the same sidecar (e.g. a tile viewer and a
    background inference pass, each loaded before the other saved): a
    save from one must not drop what the other already saved."""
    viewer = TileAnnotations(str(tmp_path / "fov1.tiles"))
    inference = TileAnnotations(str(tmp_path / "fov1.tiles"))

    viewer.update([("fov1_cell00001", "single")])
    inference.update_with_confidence([("fov1_cell00002", "budded", 0.9)])

    reopened = TileAnnotations(str(tmp_path / "fov1.tiles"))
    assert reopened.get("fov1_cell00001") == "single"
    assert reopened.get("fov1_cell00002") == "budded"


def test_prediction_never_replaces_a_human_tag_saved_meanwhile(tmp_path):
    viewer = TileAnnotations(str(tmp_path / "fov1.tiles"))
    inference = TileAnnotations(str(tmp_path / "fov1.tiles"))

    viewer.update([("fov1_cell00001", "single")])
    inference.update_with_confidence([("fov1_cell00001", "budded", 0.9)])

    reopened = TileAnnotations(str(tmp_path / "fov1.tiles"))
    assert reopened.get("fov1_cell00001") == "single"
    assert reopened.confidence("fov1_cell00001") is None


def test_containers_in_one_folder_share_one_vocabulary(tmp_path):
    TileAnnotations(str(tmp_path / "fov1.tiles")).add_category("single")
    TileAnnotations(str(tmp_path / "fov1.tiles")).add_category("budded")

    assert TileAnnotations(str(tmp_path / "fov2.tiles")).categories() == ["single", "budded"]
    assert (tmp_path / CATEGORIES_FILE_NAME).read_text() == "single\nbudded\n"
    assert not (tmp_path / "fov1.txt").exists()  # no tags, so no sidecar


def test_shared_vocabulary_is_built_from_sibling_headers(tmp_path):
    for fov in ("fov1", "fov2"):
        (tmp_path / f"{fov}.tiles").touch()
    (tmp_path / "fov1.txt").write_text("#categories\tsingle\tbudded\nfov1_cell00001\tsingle\n")
    (tmp_path / "fov2.txt").write_text("#categories\tBudded\tjunk\nfov2_cell00001\tjunk\n")

    store = TileAnnotations(str(tmp_path / "fov2.tiles"))
    assert store.categories() == ["single", "budded", "junk"]

    store.update([("fov2_cell00002", "single")])
    assert "#categories" not in (tmp_path / "fov2.txt").read_text()
    assert TileAnnotations(str(tmp_path / "fov1.tiles")).categories() == ["single", "budded", "junk"]


def test_rename_rewrites_tags_in_every_sibling_sidecar(tmp_path):
    for fov in ("fov1", "fov2"):
        (tmp_path / f"{fov}.tiles").touch()
    TileAnnotations(str(tmp_path / "fov1.tiles")).update([("fov1_cell00001", "budded")])
    TileAnnotations(str(tmp_path / "fov2.tiles")).update([("fov2_cell00001", "budded")])
    store = TileAnnotations(str(tmp_path / "fov1.tiles"))
    store.add_category("budded")

    assert store.usage_count("budded") == 2
    store.rename_category("budded", "budding")

    assert store.categories() == ["budding"]
    assert store.get("fov1_cell00001") == "budding"
    assert TileAnnotations(str(tmp_path / "fov2.tiles")).get("fov2_cell00001") == "budding"


def test_pool_sees_a_category_removed_from_the_shared_vocabulary(tmp_path):
    TileAnnotations(str(tmp_path / "fov1.tiles")).add_category("single")
    TileAnnotations(str(tmp_path / "fov1.tiles")).add_category("junk")
    pool = PooledAnnotations([tmp_path / "fov1.tiles", tmp_path / "fov2.tiles"])

    pool.remove_category("junk")

    assert pool.categories() == ["single"]
    assert (tmp_path / CATEGORIES_FILE_NAME).read_text() == "single\n"
