"""Tests for yeastprep.core.deconvolve. Only the `enabled=False` passthrough
path is exercised without mocking -- a real PSF/deconvolution run needs a
PSF fixture this repo doesn't have, matching the existing
tmp_path/no-mocking convention where the code path allows it.
"""

import numpy as np
import pytest
from pyvistra.io import save_tiff

from yeastprep.core.combined_tiff import combine_channels, read_combined_tiff
from yeastprep.core.deconvolve import DeconvolveParams, deconvolve_and_save, run_deconvolve


def _synthetic_image(Ny=16, Nx=16, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(Ny, Nx)).astype(np.uint16)


def test_run_deconvolve_disabled_is_passthrough():
    target = _synthetic_image()
    result = run_deconvolve(target, DeconvolveParams(enabled=False))
    assert result is target


def test_run_deconvolve_enabled_requires_psf_path():
    target = _synthetic_image()
    with pytest.raises(ValueError, match="psf_path"):
        run_deconvolve(target, DeconvolveParams(enabled=True, psf_path=None))


def test_deconvolve_and_save_disabled_passthrough(tmp_path):
    path = tmp_path / "sample.tiff"
    brightfield = _synthetic_image(seed=0)
    target = _synthetic_image(seed=1)
    save_tiff(path, combine_channels(brightfield, target), input_axes="CYX")
    outdir = tmp_path / "03_deconvolved"

    result = deconvolve_and_save(path, outdir, DeconvolveParams(enabled=False))

    assert result.success, result.error
    assert result.output_path.exists()
    out_bf, out_target, _ = read_combined_tiff(result.output_path)
    assert np.array_equal(out_bf, brightfield)
    assert np.array_equal(out_target, target)


def test_deconvolve_and_save_reports_failure_without_raising(tmp_path):
    missing_path = tmp_path / "does_not_exist.tiff"
    result = deconvolve_and_save(missing_path, tmp_path / "out", DeconvolveParams(enabled=False))
    assert not result.success
    assert result.error is not None
