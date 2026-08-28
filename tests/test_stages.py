"""Tests for yeastprep.core.stages, using synthetic project trees built
with tmp_path + pyvistra's save_tiff, matching test_segmentation.py's
convention of exercising real files rather than mocking the filesystem.
"""

import os
import time

import numpy as np
from pyvistra.io import save_tiff

from yeastprep.core import project as project_core
from yeastprep.core import stages
from yeastprep.core.combined_tiff import combine_channels
from yeastprep.core.segmentation import seg_npy_path


def _synthetic_image(Ny=32, Nx=32, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(Ny, Nx)).astype(np.uint8)


def _write_combined_tiff(path, seed=0):
    brightfield = _synthetic_image(seed=seed)
    target = _synthetic_image(seed=seed + 1)
    combined = combine_channels(brightfield, target)
    save_tiff(path, combined, input_axes="CYX")


def _write_fake_seg_npy(image_path):
    """A cellpose-shaped `_seg.npy` sidecar without running an actual
    model -- pipeline_status only checks for the sidecar's presence."""
    seg_path = seg_npy_path(image_path)
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(seg_path, {"masks": np.zeros((4, 4), dtype=np.int32)})


def test_pipeline_order_and_labels():
    keys = [spec.key for spec in stages.PIPELINE]
    assert keys == [
        stages.STAGE_RAW,
        project_core.STAGE_REDUCED,
        project_core.STAGE_DENOISED,
        project_core.STAGE_DECONVOLVED,
        stages.STAGE_SEGMENTATION,
        project_core.STAGE_TILES,
    ]
    optional_keys = {spec.key for spec in stages.PIPELINE if spec.optional}
    assert optional_keys == {project_core.STAGE_DENOISED, project_core.STAGE_DECONVOLVED}


def test_pipeline_status_empty_project(tmp_path):
    paths = project_core.ProjectPaths(tmp_path)
    states = stages.pipeline_status(paths)
    assert {s.key: s.status for s in states} == {
        stages.STAGE_RAW: "empty",
        project_core.STAGE_REDUCED: "empty",
        project_core.STAGE_DENOISED: "empty",
        project_core.STAGE_DECONVOLVED: "empty",
        stages.STAGE_SEGMENTATION: "empty",
        project_core.STAGE_TILES: "empty",
    }
    assert not any(s.active for s in states)


def test_pipeline_status_reduced_is_active_source(tmp_path):
    paths = project_core.ProjectPaths(tmp_path)
    (tmp_path / "sample.ims").write_bytes(b"")
    paths.reduced.mkdir()
    _write_combined_tiff(paths.reduced / "sample.tiff")

    states = {s.key: s for s in stages.pipeline_status(paths)}

    assert states[stages.STAGE_RAW].status == "done"
    assert states[project_core.STAGE_REDUCED].status == "done"
    assert states[project_core.STAGE_REDUCED].active
    assert states[project_core.STAGE_DENOISED].status == "empty"
    assert not states[project_core.STAGE_DENOISED].active


def test_pipeline_status_flags_stale_active_stage(tmp_path):
    paths = project_core.ProjectPaths(tmp_path)
    raw_path = tmp_path / "sample.ims"
    raw_path.write_bytes(b"")
    paths.reduced.mkdir()
    reduced_path = paths.reduced / "sample.tiff"
    _write_combined_tiff(reduced_path)

    # Make the raw source newer than its reduced output.
    now = time.time()
    os.utime(reduced_path, (now - 100, now - 100))
    os.utime(raw_path, (now, now))

    states = {s.key: s for s in stages.pipeline_status(paths)}
    reduced_state = states[project_core.STAGE_REDUCED]
    assert reduced_state.status == "stale"
    assert reduced_state.active  # still the active source despite being stale


def test_pipeline_status_segmentation_and_tiles(tmp_path):
    paths = project_core.ProjectPaths(tmp_path)
    paths.reduced.mkdir()
    reduced_path = paths.reduced / "sample.tiff"
    _write_combined_tiff(reduced_path)
    _write_fake_seg_npy(reduced_path)

    paths.tiles.mkdir()
    (paths.tiles / "sample").mkdir()
    (paths.tiles / "sample" / "sample_cell00001.tif").write_bytes(b"")
    (paths.tiles / "tile_index.csv").write_text(
        "cell_id,fov_id,label,crop_path\nsample_cell00001,sample,1,"
        f"{paths.tiles / 'sample' / 'sample_cell00001.tif'}\n"
    )

    states = {s.key: s for s in stages.pipeline_status(paths)}
    assert states[stages.STAGE_SEGMENTATION].status == "done"
    assert states[project_core.STAGE_TILES].status == "done"


def test_input_stage_for_denoise_always_reads_reduced(tmp_path):
    paths = project_core.ProjectPaths(tmp_path)
    assert stages.input_stage_for(project_core.STAGE_DENOISED, paths) == project_core.STAGE_REDUCED


def test_input_stage_for_deconvolve_prefers_denoised(tmp_path):
    paths = project_core.ProjectPaths(tmp_path)
    paths.reduced.mkdir()
    _write_combined_tiff(paths.reduced / "sample.tiff")
    assert (
        stages.input_stage_for(project_core.STAGE_DECONVOLVED, paths)
        == project_core.STAGE_REDUCED
    )

    paths.denoised.mkdir()
    _write_combined_tiff(paths.denoised / "sample.tiff")
    assert (
        stages.input_stage_for(project_core.STAGE_DECONVOLVED, paths)
        == project_core.STAGE_DENOISED
    )


def test_input_stage_for_segmentation_and_tiles_match_resolve_2d_source(tmp_path):
    paths = project_core.ProjectPaths(tmp_path)
    paths.reduced.mkdir()
    _write_combined_tiff(paths.reduced / "sample.tiff")
    paths.deconvolved.mkdir()
    _write_combined_tiff(paths.deconvolved / "sample.tiff")

    expected = project_core.resolve_2d_source(paths).name
    assert stages.input_stage_for(stages.STAGE_SEGMENTATION, paths) == expected
    assert stages.input_stage_for(project_core.STAGE_TILES, paths) == expected

    config = project_core.ProjectConfig(segmentation_source_stage=project_core.STAGE_REDUCED)
    assert (
        stages.input_stage_for(stages.STAGE_SEGMENTATION, paths, config)
        == project_core.STAGE_REDUCED
    )
