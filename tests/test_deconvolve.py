"""Tests for yeastprep.core.deconvolve, on small synthetic images and a
Gaussian stand-in PSF written to tmp_path (no mocking)."""

import numpy as np
import pytest
import tifffile
from pyvistra.io import save_tiff

from yeastprep.core.combined_tiff import combine_channels, read_combined_tiff
from yeastprep.core.deconvolve import (
    DeconvolveParams,
    deconvolve_and_save,
    psf_spacing_problem,
    read_pixel_size_um,
    run_deconvolve,
)


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


# ----------------------------------------------------------------------
# PSF pixel spacing must match the image's (divided by zoom)


def _gaussian_psf(n=9, sigma=1.2) -> np.ndarray:
    y, x = np.mgrid[:n, :n] - n // 2
    psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return (psf / psf.sum()).astype(np.float32)[None]


def _write_sample(path, pixel_um):
    image = combine_channels(_synthetic_image(seed=0), _synthetic_image(seed=1) + 100)
    save_tiff(path, image, scale=(1.0, pixel_um, pixel_um), input_axes="CYX")


def test_read_pixel_size_round_trips_and_ignores_uncalibrated_tiffs(tmp_path):
    from yeastprep.core.psf_calc import save_psf

    save_psf(tmp_path / "psf.tif", _gaussian_psf(), (0.1, 0.065, 0.065))
    assert read_pixel_size_um(tmp_path / "psf.tif") == pytest.approx(0.065, rel=1e-4)

    tifffile.imwrite(tmp_path / "plain.tif", np.zeros((4, 4), np.uint16))
    assert read_pixel_size_um(tmp_path / "plain.tif") is None


def test_psf_spacing_problem():
    assert psf_spacing_problem(0.065, 0.065) is None
    assert psf_spacing_problem(0.0325, 0.065, zoom=2) is None
    assert psf_spacing_problem(None, 0.065) is None  # can't check: allowed
    assert psf_spacing_problem(0.065, None) is None

    problem = psf_spacing_problem(0.094, 0.065)
    assert "0.0940 um" in problem and "0.0650 um" in problem
    assert "zoom" in psf_spacing_problem(0.065, 0.065, zoom=2)


def test_deconvolve_and_save_refuses_a_mismatched_psf(tmp_path):
    from yeastprep.core.psf_calc import save_psf

    sample = tmp_path / "sample.tiff"
    _write_sample(sample, pixel_um=0.065)
    nyquist_psf = tmp_path / "psf_nyquist.tif"
    save_psf(nyquist_psf, _gaussian_psf(), (0.1, 0.094, 0.094))

    result = deconvolve_and_save(
        sample, tmp_path / "out", DeconvolveParams(enabled=True, psf_path=str(nyquist_psf))
    )
    assert not result.success
    assert "Recompute the PSF" in result.error
    assert not (tmp_path / "out" / "sample.tiff").exists()


def test_deconvolve_and_save_runs_with_a_matching_psf(tmp_path):
    from yeastprep.core.psf_calc import save_psf

    sample = tmp_path / "sample.tiff"
    _write_sample(sample, pixel_um=0.065)
    psf = tmp_path / "psf.tif"
    save_psf(psf, _gaussian_psf(), (0.1, 0.065, 0.065))

    result = deconvolve_and_save(
        sample,
        tmp_path / "out",
        DeconvolveParams(enabled=True, psf_path=str(psf), num_iter=2),
    )
    assert result.success, result.error
    _bf, out_target, scale = read_combined_tiff(result.output_path)
    assert out_target.shape == (16, 16)
    assert scale[2] == pytest.approx(0.065, rel=1e-4)
