"""Tests for tileclass.tile_container -- the single-file-per-FOV tile
format that replaced one-tif-per-cell storage under 05_tiles/<fov_id>/.
"""

import numpy as np
import pytest

from tileclass.tile_container import (
    MAGIC,
    TileContainer,
    container_and_cell,
    get_container,
    is_container_ref,
    write_container,
)


def _make_cells(n=3, size=8):
    rng = np.random.default_rng(0)
    return [
        (f"fov_cell{label:05d}", label, rng.integers(0, 256, size=(3, size, size), dtype=np.uint8))
        for label in range(1, n + 1)
    ]


def test_round_trip_preserves_shape_dtype_and_values(tmp_path):
    cells = _make_cells()
    container_path = tmp_path / "fov.tiles"
    write_container(container_path, cells)

    container = TileContainer(container_path)
    assert len(container) == len(cells)
    for cell_id, _label, array in cells:
        assert cell_id in container
        restored = container.read(cell_id)
        assert restored.shape == array.shape
        assert restored.dtype == array.dtype
        assert np.array_equal(restored, array)


def test_cell_ids_match_input_order(tmp_path):
    cells = _make_cells()
    container_path = tmp_path / "fov.tiles"
    write_container(container_path, cells)

    assert TileContainer(container_path).cell_ids() == [c[0] for c in cells]


def test_bad_magic_is_rejected(tmp_path):
    bad_path = tmp_path / "not_a_container.tiles"
    bad_path.write_bytes(b"not a real container")

    with pytest.raises(ValueError):
        TileContainer(bad_path)


def test_is_container_ref_detects_virtual_paths(tmp_path):
    container_path = tmp_path / "fov.tiles"
    assert is_container_ref(f"{container_path}/fov_cell00001.tif")
    assert not is_container_ref(tmp_path / "fov" / "fov_cell00001.tif")


def test_container_and_cell_splits_virtual_path(tmp_path):
    container_path = tmp_path / "fov.tiles"
    ref = f"{container_path}/fov_cell00001.tif"
    parsed_container, cell_id = container_and_cell(ref)
    assert parsed_container == container_path
    assert cell_id == "fov_cell00001"


def test_get_container_caches_by_path(tmp_path):
    container_path = tmp_path / "fov.tiles"
    write_container(container_path, _make_cells())

    first = get_container(container_path)
    second = get_container(container_path)
    assert first is second


def test_zero_cells_still_produces_valid_container(tmp_path):
    container_path = tmp_path / "empty.tiles"
    write_container(container_path, [])

    container = TileContainer(container_path)
    assert len(container) == 0
    assert container.path.read_bytes().startswith(MAGIC)
