"""Pipeline tests using a synthetic multi-channel TIFF stack -- no real
acquisition file needed. Exercises the full load -> flatten -> save path
through pyvistra's actual I/O (not mocked), the same way process_and_save
will run against real data.
"""

import os
import time

import numpy as np
import tifffile
from pyvistra.io import save_tiff

from yeastprep.core.channels import ChannelSelection
from yeastprep.core.combined_tiff import (
    BRIGHTFIELD_CHANNEL,
    TARGET_CHANNEL,
    load_brightfield_channel,
)
from yeastprep.core.pipeline import (
    FlattenFieldParams,
    compute_focal_slice,
    compute_focus_diagnostics,
    load_volume,
    process_and_save,
    sum_project,
)
from yeastprep.core.project import stage_file_status

CHANNELS = ChannelSelection(brightfield=0, projection=1)


def _write_synthetic_stack(path, Nz=8, Ny=32, Nx=32, scale=(0.3, 0.1, 0.1)):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 1000, size=(2, Nz, Ny, Nx)).astype(np.uint16)
    save_tiff(path, data, scale=scale, input_axes="CZYX")
    return data


def test_load_volume_slices_correct_channels(tmp_path):
    path = tmp_path / "synthetic.tif"
    data = _write_synthetic_stack(path)

    volume = load_volume(path, CHANNELS)

    assert volume.img3d.shape == data.shape[1:]
    assert volume.projection3d.shape == data.shape[1:]
    assert np.array_equal(volume.img3d, data[0])
    assert np.array_equal(volume.projection3d, data[1])
    assert volume.scale == (0.3, 0.1, 0.1)


def test_compute_focus_diagnostics_and_focal_slice_shapes(tmp_path):
    path = tmp_path / "synthetic.tif"
    _write_synthetic_stack(path, Nz=8, Ny=32, Nx=32)
    volume = load_volume(path, CHANNELS)
    params = FlattenFieldParams(num_tiles_y=4, num_tiles_x=4)

    diagnostics = compute_focus_diagnostics(volume, params)
    assert diagnostics.coarse_focal_indices.shape == (4, 4)
    assert diagnostics.fine_focus_indices.shape == (32, 32)

    focal_slice = compute_focal_slice(volume, diagnostics, params)
    assert focal_slice.shape == (32, 32)
    assert focal_slice.dtype == np.uint16


def test_sum_project(tmp_path):
    path = tmp_path / "synthetic.tif"
    data = _write_synthetic_stack(path)
    volume = load_volume(path, CHANNELS)

    projected = sum_project(volume)
    assert projected.shape == data.shape[2:]
    assert np.array_equal(projected, data[1].sum(axis=0).astype(np.uint16))


def test_process_and_save_round_trips(tmp_path):
    path = tmp_path / "synthetic.tif"
    _write_synthetic_stack(path, Nz=8, Ny=32, Nx=32)

    outdir = tmp_path / "processed"
    params = FlattenFieldParams(num_tiles_y=4, num_tiles_x=4)

    result = process_and_save(path, outdir, params, CHANNELS)

    assert result.success, result.error
    assert result.output_path.exists()

    combined = tifffile.imread(result.output_path)
    assert combined.shape == (2, 32, 32)
    assert combined.dtype == np.uint16

    brightfield = load_brightfield_channel(result.output_path)
    assert np.array_equal(brightfield, combined[BRIGHTFIELD_CHANNEL])
    assert not np.array_equal(combined[BRIGHTFIELD_CHANNEL], combined[TARGET_CHANNEL])


def test_process_and_save_reports_failure_without_raising(tmp_path):
    missing_path = tmp_path / "does_not_exist.tif"
    outdir = tmp_path / "processed"

    result = process_and_save(missing_path, outdir)

    assert not result.success
    assert result.error is not None


def test_stage_file_status_before_and_after_processing(tmp_path):
    path = tmp_path / "synthetic.tif"
    _write_synthetic_stack(path, Nz=8, Ny=32, Nx=32)
    raw_folder = tmp_path / "raw"
    raw_folder.mkdir()
    raw_path = raw_folder / "synthetic.tiff"
    raw_path.write_bytes(b"")
    outdir = tmp_path / "reduced"

    assert stage_file_status(outdir, upstream_dir=raw_folder) == {}

    result = process_and_save(path, outdir)
    assert result.success

    statuses = stage_file_status(outdir, upstream_dir=raw_folder)
    assert statuses["synthetic"].exists
    assert statuses["synthetic"].up_to_date


def test_stage_file_status_stale_after_upstream_file_touched(tmp_path):
    path = tmp_path / "synthetic.tif"
    _write_synthetic_stack(path, Nz=8, Ny=32, Nx=32)
    raw_folder = tmp_path / "raw"
    raw_folder.mkdir()
    raw_path = raw_folder / "synthetic.tiff"
    raw_path.write_bytes(b"")
    outdir = tmp_path / "reduced"
    process_and_save(path, outdir)

    # Simulate the upstream file being re-produced after this stage ran --
    # the saved output is now stale and shouldn't be reported as current.
    future = time.time() + 10
    os.utime(raw_path, (future, future))

    statuses = stage_file_status(outdir, upstream_dir=raw_folder)
    assert not statuses["synthetic"].up_to_date
