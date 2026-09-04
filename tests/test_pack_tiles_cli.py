"""Tests for the yeastprep-pack-tiles migration CLI (packs an
already-exported project's loose 05_tiles/<fov_id>/ tif crops into
<fov_id>.tiles containers)."""

import numpy as np
import polars as pl
import tifffile

from tileclass.tile_container import TileContainer
from yeastprep.core.pack_tiles_cli import main, pack_fov_dir


def _write_loose_fov(tiles_dir, fov_id, n_cells=3, size=4):
    fov_dir = tiles_dir / fov_id
    fov_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    arrays = {}
    for label in range(1, n_cells + 1):
        cell_id = f"{fov_id}_cell{label:05d}"
        array = rng.integers(0, 256, size=(3, size, size), dtype=np.uint8)
        tifffile.imwrite(fov_dir / f"{cell_id}.tif", array)
        arrays[cell_id] = array
    return fov_dir, arrays


def test_pack_fov_dir_round_trips_every_cell(tmp_path):
    tiles_dir = tmp_path / "05_tiles"
    fov_dir, arrays = _write_loose_fov(tiles_dir, "fov1")

    container_path, n_cells, error = pack_fov_dir(fov_dir)

    assert error is None
    assert n_cells == len(arrays)
    container = TileContainer(container_path)
    for cell_id, array in arrays.items():
        assert np.array_equal(container.read(cell_id), array)


def test_main_packs_project_and_leaves_originals_by_default(tmp_path):
    tiles_dir = tmp_path / "05_tiles"
    _write_loose_fov(tiles_dir, "fov1")
    _write_loose_fov(tiles_dir, "fov2", n_cells=1)

    exit_code = main([str(tmp_path)])

    assert exit_code == 0
    assert (tiles_dir / "fov1.tiles").exists()
    assert (tiles_dir / "fov2.tiles").exists()
    assert (tiles_dir / "fov1").is_dir()  # originals kept by default


def test_main_delete_originals_removes_packed_folders(tmp_path):
    tiles_dir = tmp_path / "05_tiles"
    _write_loose_fov(tiles_dir, "fov1")

    exit_code = main([str(tmp_path), "--delete-originals"])

    assert exit_code == 0
    assert (tiles_dir / "fov1.tiles").exists()
    assert not (tiles_dir / "fov1").exists()


def test_main_rewrites_tile_index_crop_paths_to_virtual_form(tmp_path):
    tiles_dir = tmp_path / "05_tiles"
    fov_dir, arrays = _write_loose_fov(tiles_dir, "fov1")
    cell_ids = sorted(arrays)
    pl.DataFrame(
        {
            "cell_id": cell_ids,
            "fov_id": ["fov1"] * len(cell_ids),
            "label": list(range(1, len(cell_ids) + 1)),
            "crop_path": [
                f"/home/original-machine/05_tiles/fov1/{cell_id}.tif" for cell_id in cell_ids
            ],
        }
    ).write_csv(tiles_dir / "tile_index.csv")

    assert main([str(tmp_path)]) == 0

    index = pl.read_csv(tiles_dir / "tile_index.csv")
    container_path = tiles_dir / "fov1.tiles"
    for row in index.iter_rows(named=True):
        assert row["crop_path"] == f"{container_path}/{row['cell_id']}.tif"


def test_main_no_tiles_reports_failure(tmp_path):
    assert main([str(tmp_path)]) == 1
