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


def _three_cell_mask(Ny=128, Nx=128) -> np.ndarray:
    mask = np.zeros((Ny, Nx), dtype=np.int32)
    mask[20:40, 20:40] = 1
    mask[20:40, 80:100] = 2
    mask[80:100, 50:70] = 3
    return mask


def _annotations_for(out_dir, fov_id):
    from tileclass.data.annotations import TileAnnotations

    return TileAnnotations(str(out_dir / f"{fov_id}.tiles"))


def test_reexport_after_mask_edit_moves_tags_to_renumbered_cells(tmp_path):
    image_path = tmp_path / "fov1.tiff"
    _write_synthetic_combined_tiff(image_path)
    out_dir = tmp_path / "tiles"
    mask = _three_cell_mask()
    _write_synthetic_masks(image_path, mask)
    assert export_tiles(image_path, out_dir, TileParams(size=32)).success

    annotations = _annotations_for(out_dir, "fov1")
    annotations.update({"fov1_cell00001": "dead", "fov1_cell00003": "budding"})
    annotations.update_with_confidence([("fov1_cell00002", "single", 0.8)])

    # Cell 1 deleted in the Cellpose GUI: the rest get renumbered 2->1, 3->2.
    edited = mask.copy()
    edited[edited == 1] = 0
    edited[edited == 2] = 1
    edited[edited == 3] = 2
    _write_synthetic_masks(image_path, edited)
    result = export_tiles(image_path, out_dir, TileParams(size=32))

    assert result.success, result.error
    assert (result.tags_moved, result.tags_dropped) == (2, 1)
    annotations = _annotations_for(out_dir, "fov1")
    assert dict(annotations.items()) == {"fov1_cell00001": "single", "fov1_cell00002": "budding"}
    assert annotations.confidence("fov1_cell00001") == 0.8  # still unreviewed
    assert annotations.confidence("fov1_cell00002") is None
    assert "fov1_cell00003\tbudding" in (out_dir / "fov1.txt.bak").read_text()


def test_reexport_with_unchanged_masks_leaves_tags_alone(tmp_path):
    image_path = tmp_path / "fov1.tiff"
    _write_synthetic_combined_tiff(image_path)
    out_dir = tmp_path / "tiles"
    _write_synthetic_masks(image_path, _three_cell_mask())
    export_tiles(image_path, out_dir, TileParams(size=32))
    _annotations_for(out_dir, "fov1").update({"fov1_cell00002": "dead"})

    result = export_tiles(image_path, out_dir, TileParams(size=48))

    assert (result.tags_moved, result.tags_dropped) == (0, 0)
    assert dict(_annotations_for(out_dir, "fov1").items()) == {"fov1_cell00002": "dead"}


def test_reexport_of_container_without_centroids_matches_by_content(tmp_path):
    image_path = tmp_path / "fov1.tiff"
    _write_synthetic_combined_tiff(image_path)
    out_dir = tmp_path / "tiles"
    mask = _three_cell_mask()
    _write_synthetic_masks(image_path, mask)
    export_tiles(image_path, out_dir, TileParams(size=32))

    # Rewrite the container the way older exports did: no centroid field.
    container_path = out_dir / "fov1.tiles"
    old = TileContainer(container_path)
    write_container(
        container_path,
        [(cell_id, entry["label"], old.read(cell_id)) for cell_id, entry in old.entries.items()],
    )
    _annotations_for(out_dir, "fov1").update({"fov1_cell00002": "dead", "fov1_cell00003": "budding"})

    # Cell 2 deleted: label 3 becomes 2 -- a different cell under an old label.
    edited = mask.copy()
    edited[edited == 2] = 0
    edited[edited == 3] = 2
    _write_synthetic_masks(image_path, edited)
    result = export_tiles(image_path, out_dir, TileParams(size=32))

    # Same-shaped cell 3 now sits under label 2 -- matched by its pixels,
    # not by the label or the (identical) mask shape.
    assert (result.tags_moved, result.tags_dropped) == (1, 1)
    assert dict(_annotations_for(out_dir, "fov1").items()) == {"fov1_cell00002": "budding"}


def test_append_tile_index_replaces_all_rows_of_a_reexported_fov(tmp_path):
    out_dir = tmp_path / "tiles"
    out_dir.mkdir()

    def rows(fov_id, labels):
        return pl.DataFrame(
            {
                "cell_id": [f"{fov_id}_cell{label:05d}" for label in labels],
                "fov_id": [fov_id] * len(labels),
                "label": labels,
                "crop_path": [f"{fov_id}.tiles/{fov_id}_cell{label:05d}" for label in labels],
            }
        )

    append_tile_index(out_dir, rows("fov1", [1, 2, 3, 4]))
    append_tile_index(out_dir, rows("fov2", [1]))
    append_tile_index(out_dir, rows("fov1", [1, 2]))
    index = pl.read_csv(tile_index_path(out_dir))
    assert index.filter(pl.col("fov_id") == "fov1")["label"].to_list() == [1, 2]
    assert index.filter(pl.col("fov_id") == "fov2")["label"].to_list() == [1]

    # A FOV re-exported with no cells left at all.
    append_tile_index(out_dir, rows("fov1", []).cast({"label": pl.Int64}), fov_ids=["fov1"])
    index = pl.read_csv(tile_index_path(out_dir))
    assert index["fov_id"].to_list() == ["fov2"]


def test_reexport_without_centroids_handles_a_changed_tile_size(tmp_path):
    image_path = tmp_path / "fov1.tiff"
    _write_synthetic_combined_tiff(image_path)
    out_dir = tmp_path / "tiles"
    _write_synthetic_masks(image_path, _three_cell_mask())
    export_tiles(image_path, out_dir, TileParams(size=32))
    container_path = out_dir / "fov1.tiles"
    old = TileContainer(container_path)
    write_container(
        container_path,
        [(cell_id, entry["label"], old.read(cell_id)) for cell_id, entry in old.entries.items()],
    )
    _annotations_for(out_dir, "fov1").update({"fov1_cell00001": "dead"})

    result = export_tiles(image_path, out_dir, TileParams(size=48))

    assert (result.tags_moved, result.tags_dropped) == (0, 0)
    assert dict(_annotations_for(out_dir, "fov1").items()) == {"fov1_cell00001": "dead"}


def test_edge_tile_contrast_ignores_padding_and_mask_keeps_only_its_cell():
    """A cell at the FOV corner: half its tile is padding. The contrast
    stretch must come from the image pixels alone (the padding isn't a
    dark part of the image), and the mask channel holds only this cell."""
    from yeastprep.core.tiles import crop_cell

    image = np.full((40, 40), 1000, np.uint16)
    image[:8, :8] = 1200  # the cell, brighter than its surroundings
    mask = np.zeros((40, 40), np.int32)
    mask[:8, :8] = 1
    mask[8:12, 8:12] = 2  # a neighbor inside the same crop window

    geom = next(g for g in cell_geometry(mask, 16) if g.label == 1)
    tile = crop_cell(mask, image, image, geom, TileParams(size=16, contrast_lo_pct=0, contrast_hi_pct=100))

    bf, _target, cell_mask = tile
    inside = bf[geom.dst]
    # Background maps to 0 and the cell to 255; counting the padding's
    # zeros as image would have put the 1000-valued background near 212.
    assert inside.min() == 0 and inside.max() == 255
    assert set(np.unique(cell_mask)) == {0, 255}
    assert (cell_mask[geom.dst] == 255).sum() == 64  # cell 1 only, not cell 2
