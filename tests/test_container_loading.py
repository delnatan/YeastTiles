"""Confirms the two pixel-reading chokepoints (tileclass.load_thumbnail's
load_plane, tileclass.training.dataset's load_masked_crop) transparently
read through a packed .tiles container via its virtual
"<fov_id>.tiles/<cell_id>.tif" reference, same as they'd read a real file.
"""

import numpy as np

from tileclass.load_thumbnail import load_plane
from tileclass.scan import scan_container
from tileclass.tile_container import write_container
from tileclass.training.dataset import load_masked_crop


def _write_masked_crop_container(path, cell_id="fov_cell00001"):
    brightfield = np.full((8, 8), 100, dtype=np.uint8)
    fluorescence = np.full((8, 8), 200, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    crop = np.stack([brightfield, fluorescence, mask], axis=0)
    write_container(path, [(cell_id, 1, crop)])
    return crop


def test_load_plane_reads_through_container(tmp_path):
    container_path = tmp_path / "fov.tiles"
    crop = _write_masked_crop_container(container_path)

    plane = load_plane(f"{container_path}/fov_cell00001.tif")
    assert np.array_equal(plane, crop)


def test_load_masked_crop_reads_through_container(tmp_path):
    container_path = tmp_path / "fov.tiles"
    _write_masked_crop_container(container_path)

    result = load_masked_crop(f"{container_path}/fov_cell00001.tif")
    assert result.shape == (2, 8, 8)
    assert result.dtype == np.float32
    # Inside the mask (255): brightfield/fluorescence scaled to [0, 1].
    assert np.isclose(result[0, 4, 4], 100 / 255)
    assert np.isclose(result[1, 4, 4], 200 / 255)
    # Outside the mask: zeroed regardless of the underlying pixel value.
    assert result[0, 0, 0] == 0
    assert result[1, 0, 0] == 0


def test_scan_container_returns_natsorted_virtual_paths(tmp_path):
    container_path = tmp_path / "fov.tiles"
    cells = [
        (f"fov_cell{label:05d}", label, np.zeros((3, 4, 4), dtype=np.uint8))
        for label in (2, 10, 1)
    ]
    write_container(container_path, cells)

    paths = scan_container(container_path)
    assert paths == [
        f"{container_path}/fov_cell00001.tif",
        f"{container_path}/fov_cell00002.tif",
        f"{container_path}/fov_cell00010.tif",
    ]
