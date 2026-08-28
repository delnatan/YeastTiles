"""Tests for yeastprep.core.segmentation, using a synthetic focal-slice
image -- no real acquisition data needed. Runs the actual cellpose model
(no mocking), matching test_pipeline.py's convention of exercising the
real dependency rather than stubbing it.
"""

import numpy as np
from pyvistra.io import save_tiff

from yeastprep.core.combined_tiff import CELLPOSE_GUI_SUBDIR, combine_channels
from yeastprep.core.segmentation import (
    SegmentationParams,
    get_model,
    load_saved_masks,
    run_segmentation,
    seg_npy_path,
    segment_and_save,
)


def _synthetic_image(Ny=128, Nx=128, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(Ny, Nx)).astype(np.uint8)


def _write_synthetic_combined_tiff(path, Ny=128, Nx=128) -> np.ndarray:
    """Writes a combined-channel tiff matching core.pipeline.process_and_save's
    output format, and returns the brightfield channel (what segmentation
    should actually operate on)."""
    brightfield = _synthetic_image(Ny, Nx, seed=0)
    target = _synthetic_image(Ny, Nx, seed=1)
    combined = combine_channels(brightfield, target)
    save_tiff(path, combined, input_axes="CYX")
    return brightfield


def test_get_model_caches_by_path():
    model_a = get_model(None)
    model_b = get_model(None)
    assert model_a is model_b


def test_run_segmentation_shapes():
    image = _synthetic_image()
    model = get_model(None)
    params = SegmentationParams(diameter=30.0)

    result = run_segmentation(model, image, params)

    assert result.masks.shape == image.shape
    assert np.issubdtype(result.masks.dtype, np.integer)
    assert result.n_cells == int(result.masks.max())


def test_segment_and_save_writes_readable_seg_npy(tmp_path):
    path = tmp_path / "sample.tif"
    brightfield = _write_synthetic_combined_tiff(path)
    model = get_model(None)

    result = segment_and_save(path, model, SegmentationParams(diameter=30.0))

    assert result.success, result.error
    assert result.seg_path == tmp_path / CELLPOSE_GUI_SUBDIR / "sample_seg.npy"
    assert result.seg_path.exists()

    dat = np.load(result.seg_path, allow_pickle=True).item()
    # Segmentation only ever operates on the brightfield channel -- the
    # saved masks/image are 2D (Ny, Nx), not the combined (2, Ny, Nx) tiff.
    assert dat["masks"].shape == brightfield.shape
    assert int(dat["masks"].max()) == result.n_cells


def test_segment_and_save_reports_failure_without_raising(tmp_path):
    missing_path = tmp_path / "does_not_exist.tif"
    model = get_model(None)

    result = segment_and_save(missing_path, model)

    assert not result.success
    assert result.error is not None


def test_load_saved_masks_round_trips_through_segment_and_save(tmp_path):
    path = tmp_path / "sample.tif"
    brightfield = _write_synthetic_combined_tiff(path)
    model = get_model(None)

    result = segment_and_save(path, model, SegmentationParams(diameter=30.0))
    assert result.success

    masks = load_saved_masks(seg_npy_path(path))
    assert masks is not None
    assert masks.shape == brightfield.shape
    assert int(masks.max()) == result.n_cells


def test_load_saved_masks_missing_file_returns_none(tmp_path):
    assert load_saved_masks(tmp_path / "does_not_exist_seg.npy") is None
