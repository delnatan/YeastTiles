"""Tests for tileclass.data.annotations.TileAnnotations -- specifically the
sidecar-file naming contract that must hold regardless of whether a FOV's
cells live in a loose `<fov_id>/` folder or a packed `<fov_id>.tiles`
container (see tile_container.py), since a project's existing annotations
were written back when only the folder form existed.
"""

from tileclass.data.annotations import TileAnnotations


def test_container_root_reuses_same_sidecar_name_as_folder_root(tmp_path):
    """An already-annotated project's sidecar (written when 05_tiles/<fov>/
    was a real folder) must still be found after that folder is packed
    into 05_tiles/<fov>.tiles -- otherwise packing silently orphans every
    existing tag."""
    folder_root = tmp_path / "fov1"
    container_root = tmp_path / "fov1.tiles"

    assert TileAnnotations(str(folder_root)).file_path == TileAnnotations(
        str(container_root)
    ).file_path


def test_tags_written_against_the_folder_are_visible_through_the_container(tmp_path):
    folder_root = tmp_path / "fov1"
    folder_root.mkdir()
    TileAnnotations(str(folder_root))["fov1_cell00001.tif"] = "single"

    reopened = TileAnnotations(str(tmp_path / "fov1.tiles"))
    assert reopened.get("fov1_cell00001.tif") == "single"


def test_concurrent_writer_does_not_erase_other_writers_tags(tmp_path):
    """Two stores open on the same sidecar (e.g. a tile viewer and a
    background inference pass, each loaded before the other saved): a
    save from one must not drop what the other already saved."""
    viewer = TileAnnotations(str(tmp_path / "fov1.tiles"))
    inference = TileAnnotations(str(tmp_path / "fov1.tiles"))

    viewer.update([("fov1_cell00001.tif", "single")])
    inference.update_with_confidence([("fov1_cell00002.tif", "budded", 0.9)])

    reopened = TileAnnotations(str(tmp_path / "fov1.tiles"))
    assert reopened.get("fov1_cell00001.tif") == "single"
    assert reopened.get("fov1_cell00002.tif") == "budded"


def test_prediction_never_replaces_a_human_tag_saved_meanwhile(tmp_path):
    viewer = TileAnnotations(str(tmp_path / "fov1.tiles"))
    inference = TileAnnotations(str(tmp_path / "fov1.tiles"))

    viewer.update([("fov1_cell00001.tif", "single")])
    inference.update_with_confidence([("fov1_cell00001.tif", "budded", 0.9)])

    reopened = TileAnnotations(str(tmp_path / "fov1.tiles"))
    assert reopened.get("fov1_cell00001.tif") == "single"
    assert reopened.confidence("fov1_cell00001.tif") is None
