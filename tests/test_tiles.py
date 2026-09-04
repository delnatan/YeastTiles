"""Tests for yeastprep.core.tiles, using synthetic combined-channel tiffs
and hand-built masks -- no real acquisition data or cellpose inference
needed (masks are written directly as a `_seg.npy` sidecar, the same file
`segment_and_save` produces), matching test_segmentation.py's convention.
"""

import os

import numpy as np
import polars as pl
from pyvistra.io import save_tiff
from tileclass.tile_container import TileContainer, write_container

from yeastprep.core.combined_tiff import combine_channels
from yeastprep.core.segmentation import seg_npy_path
from yeastprep.core.tiles import (
    TileParams,
    append_tile_index,
    axis_slices,
    cell_geometry,
    export_tiles,
    fov_tile_status,
    tile_index_path,
    to_uint8,
)


def _write_synthetic_combined_tiff(path, Ny=128, Nx=128) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    brightfield = rng.integers(0, 4000, size=(Ny, Nx)).astype(np.uint16)
    target = rng.integers(0, 4000, size=(Ny, Nx)).astype(np.uint16)
    combined = combine_channels(brightfield, target)
    save_tiff(path, combined, input_axes="CYX")
    return brightfield, target


def _write_synthetic_masks(image_path, masks: np.ndarray):
    seg_path = seg_npy_path(image_path)
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(seg_path, {"masks": masks}, allow_pickle=True)


def _two_cell_mask(Ny=128, Nx=128) -> np.ndarray:
    """Two well-separated square blobs, labels 1 and 2, centered at known
    positions -- easy to check crop-window placement against."""
    mask = np.zeros((Ny, Nx), dtype=np.int32)
    mask[20:40, 20:40] = 1  # centroid ~ (29.5, 29.5)
    mask[80:100, 90:110] = 2  # centroid ~ (89.5, 99.5)
    return mask


def test_axis_slices_centers_window_when_fully_inside():
    src, dst = axis_slices(50.0, size=10, limit=128)
    assert (src.start, src.stop) == (45, 55)
    assert (dst.start, dst.stop) == (0, 10)


def test_axis_slices_clips_at_lower_edge():
    src, dst = axis_slices(2.0, size=10, limit=128)
    assert src.start == 0
    assert dst.start == 3  # window's left 3px fell off the edge


def test_cell_geometry_finds_both_labels_at_expected_centroids():
    mask = _two_cell_mask()
    geometries = cell_geometry(mask, size=32)

    assert [g.label for g in geometries] == [1, 2]
    g1 = geometries[0]
    row_src, col_src = g1.src
    # Cell 1 spans rows/cols 20:40 -> centroid 29.5 -> a 32px window
    # centered there starts at round(29.5) - 16 = 14.
    assert row_src.start == 14
    assert col_src.start == 14


def test_to_uint8_scales_full_range_to_0_255():
    image = np.array([[0, 50], [100, 200]], dtype=np.float32)
    out = to_uint8(image, lo_pct=0.0, hi_pct=100.0)
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255


def test_to_uint8_flat_image_returns_zeros():
    image = np.full((8, 8), 5.0)
    out = to_uint8(image, lo_pct=0.1, hi_pct=99.9)
    assert np.all(out == 0)


def test_export_tiles_writes_one_container_with_one_entry_per_cell(tmp_path):
    image_path = tmp_path / "fov1.tiff"
    _write_synthetic_combined_tiff(image_path)
    mask = _two_cell_mask()
    _write_synthetic_masks(image_path, mask)

    out_dir = tmp_path / "tiles"
    result = export_tiles(image_path, out_dir, TileParams(size=32))

    assert result.success, result.error
    assert result.n_cells == 2
    assert set(result.records["label"].to_list()) == {1, 2}

    container_path = out_dir / f"{image_path.stem}.tiles"
    assert container_path.exists()
    container = TileContainer(container_path)
    assert len(container) == 2

    for row in result.records.iter_rows(named=True):
        crop = container.read(row["cell_id"])
        assert crop.shape == (3, 32, 32)
        assert crop.dtype == np.uint8
        # Mask channel is binary (0 or 255) and non-empty for this cell.
        mask_channel = crop[2]
        assert set(np.unique(mask_channel).tolist()) <= {0, 255}
        assert mask_channel.max() == 255


def test_export_tiles_without_saved_mask_fails_without_raising(tmp_path):
    image_path = tmp_path / "unsegmented.tiff"
    _write_synthetic_combined_tiff(image_path)

    result = export_tiles(image_path, tmp_path / "tiles")

    assert not result.success
    assert "seg" in result.error.lower() or "segment" in result.error.lower()


def test_export_tiles_missing_image_fails_without_raising(tmp_path):
    result = export_tiles(tmp_path / "does_not_exist.tiff", tmp_path / "tiles")
    assert not result.success
    assert result.error is not None


def test_append_tile_index_dedupes_by_cell_id_keeping_latest(tmp_path):
    out_dir = tmp_path / "tiles"
    out_dir.mkdir()

    first = pl.DataFrame(
        {
            "cell_id": ["fov1_cell00001"],
            "fov_id": ["fov1"],
            "label": [1],
            "crop_path": [str(out_dir / "fov1" / "fov1_cell00001.tif")],
        }
    )
    append_tile_index(out_dir, first)

    second = pl.DataFrame(
        {
            "cell_id": ["fov1_cell00001", "fov1_cell00002"],
            "fov_id": ["fov1", "fov1"],
            "label": [1, 2],
            "crop_path": [
                str(out_dir / "fov1" / "fov1_cell00001.tif"),
                str(out_dir / "fov1" / "fov1_cell00002.tif"),
            ],
        }
    )
    append_tile_index(out_dir, second)

    index = pl.read_csv(tile_index_path(out_dir))
    assert sorted(index["cell_id"].to_list()) == ["fov1_cell00001", "fov1_cell00002"]


def _write_fov_container(out_dir, fov_id, n_cells=2):
    cells = [
        (f"{fov_id}_cell{label:05d}", label, np.zeros((3, 4, 4), dtype=np.uint8))
        for label in range(1, n_cells + 1)
    ]
    write_container(out_dir / f"{fov_id}.tiles", cells)


def test_fov_tile_status_reads_container_directly(tmp_path):
    out_dir = tmp_path / "tiles"
    out_dir.mkdir()
    _write_fov_container(out_dir, "fov1", n_cells=3)
    _write_fov_container(out_dir, "fov2", n_cells=1)

    statuses = fov_tile_status(out_dir)

    assert statuses["fov1"].n_cells == 3
    assert statuses["fov2"].n_cells == 1
    assert statuses["fov1"].up_to_date
    assert statuses["fov1"].mask_available


def test_fov_tile_status_survives_a_moved_project(tmp_path):
    """A container's freshness/count comes from its own file, not a
    recorded absolute path -- so this still works after `out_dir` itself
    is renamed/moved, unlike the old crop_path-in-tile_index.csv scheme
    this replaced."""
    original = tmp_path / "original_location"
    original.mkdir()
    _write_fov_container(original, "fov1", n_cells=5)

    moved = tmp_path / "moved_location"
    original.rename(moved)

    statuses = fov_tile_status(moved)
    assert statuses["fov1"].n_cells == 5


def test_fov_tile_status_stale_when_mask_newer_than_container(tmp_path):
    out_dir = tmp_path / "tiles"
    out_dir.mkdir()
    _write_fov_container(out_dir, "fov1")

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    seg_path = mask_dir / "fov1_seg.npy"
    seg_path.write_bytes(b"")
    # Make the mask newer than the container we just wrote.
    future = os.stat(out_dir / "fov1.tiles").st_mtime + 10
    os.utime(seg_path, (future, future))

    statuses = fov_tile_status(out_dir, mask_dir)
    assert statuses["fov1"].up_to_date is False


def test_fov_tile_status_missing_mask_dir_marks_unavailable(tmp_path):
    out_dir = tmp_path / "tiles"
    out_dir.mkdir()
    _write_fov_container(out_dir, "fov1")

    statuses = fov_tile_status(out_dir, mask_dir=tmp_path / "does_not_exist")
    assert statuses["fov1"].mask_available is False
    assert statuses["fov1"].up_to_date  # can't verify freshness -> assume up to date
