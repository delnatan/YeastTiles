"""Tests for tileclass.data.annotations.TileAnnotations: the sidecar-file
naming contract, migration of old `.tif`-suffixed keys, and concurrent
writers sharing one sidecar.
"""

from tileclass.data.annotations import TileAnnotations


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
    assert sidecar.read_text().startswith("#categories\tsingle\tbudded\n")


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
