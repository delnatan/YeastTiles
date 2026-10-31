"""Tests for yeastprep.core.denoise: the per-image blind-spot denoiser on a
synthetic image with a known clean version, and `denoise_and_save`'s
per-channel merge with `run_denoise` monkeypatched to a cheap stand-in."""

import numpy as np
import pytest
from pyvistra.io import save_tiff

from yeastprep.core import denoise as denoise_module
from yeastprep.core.combined_tiff import (
    BRIGHTFIELD_CHANNEL,
    TARGET_CHANNEL,
    combine_channels,
    read_combined_tiff,
)
from yeastprep.core.denoise import DenoiseParams, denoise_and_save, denoise_image, run_denoise


def _synthetic_image(Ny=300, Nx=300, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(Ny, Nx)).astype(np.uint16)


def _blobs(n=300, seed=0) -> np.ndarray:
    """Smooth, cell-sized bright blobs on a flat background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n]
    clean = np.full((n, n), 100.0)
    for _ in range(25):
        cy, cx = rng.uniform(0, n, 2)
        clean += 200.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 8.0**2))
    return clean


def test_denoise_image_removes_noise_without_bias():
    clean = _blobs()
    noisy = (clean + np.random.default_rng(1).normal(0, 20, clean.shape)).astype(np.float32)

    denoised = denoise_image(noisy, steps=150)

    assert denoised.shape == noisy.shape and denoised.dtype == np.float32
    rmse_before = np.sqrt(np.mean((noisy - clean) ** 2))
    rmse_after = np.sqrt(np.mean((denoised - clean) ** 2))
    assert rmse_after < 0.5 * rmse_before
    assert abs(float(denoised.mean() - clean.mean())) < 2.0


def test_denoise_image_rejects_images_smaller_than_the_training_crop():
    with pytest.raises(ValueError, match="smaller than"):
        denoise_image(np.zeros((64, 64), np.float32), steps=1)


def test_run_denoise_keeps_the_input_dtype(monkeypatch):
    monkeypatch.setattr(
        denoise_module, "denoise_image", lambda image, steps: np.asarray(image, np.float32) - 300.4
    )
    brightfield = np.full((4, 4), 1000, np.uint16)
    target = np.full((4, 4), 100.0, np.float32)

    out_bf = run_denoise(brightfield, DenoiseParams())
    out_tg = run_denoise(target, DenoiseParams())

    assert out_bf.dtype == np.uint16 and np.all(out_bf == 700)
    assert out_tg.dtype == np.float32 and np.all(out_tg == 0)  # clipped at zero


def test_denoise_and_save_does_not_clobber_other_channel(tmp_path, monkeypatch):
    path = tmp_path / "sample.tiff"
    brightfield0 = _synthetic_image(seed=0)
    target0 = _synthetic_image(seed=1)
    save_tiff(path, combine_channels(brightfield0, target0), input_axes="CYX")
    outdir = tmp_path / "02_denoised"

    # Denoise channel 0 (brightfield) today.
    monkeypatch.setattr(
        denoise_module, "run_denoise", lambda image, params: np.full_like(image, 111)
    )
    result_bf = denoise_and_save(path, outdir, DenoiseParams(channels=(BRIGHTFIELD_CHANNEL,)))
    assert result_bf.success, result_bf.error
    bf1, tgt1, _ = read_combined_tiff(result_bf.output_path)
    assert np.all(bf1 == 111)
    assert np.array_equal(tgt1, target0)  # untouched channel carried over from source

    # Denoise channel 1 (target) tomorrow -- shouldn't lose today's result.
    monkeypatch.setattr(
        denoise_module, "run_denoise", lambda image, params: np.full_like(image, 222)
    )
    result_tgt = denoise_and_save(path, outdir, DenoiseParams(channels=(TARGET_CHANNEL,)))
    assert result_tgt.success, result_tgt.error
    bf2, tgt2, _ = read_combined_tiff(result_tgt.output_path)
    assert np.all(tgt2 == 222)
    assert np.all(bf2 == 111)  # yesterday's brightfield result preserved, not clobbered


def test_denoise_and_save_denoises_both_channels_by_default(tmp_path, monkeypatch):
    path = tmp_path / "sample.tiff"
    save_tiff(path, combine_channels(_synthetic_image(seed=0), _synthetic_image(seed=1)), input_axes="CYX")
    calls = []
    monkeypatch.setattr(
        denoise_module, "run_denoise", lambda image, params: calls.append(1) or np.zeros_like(image)
    )

    result = denoise_and_save(path, tmp_path / "02_denoised", DenoiseParams())

    assert result.success, result.error
    assert len(calls) == 2


def test_denoise_and_save_reports_failure_without_raising(tmp_path):
    result = denoise_and_save(tmp_path / "does_not_exist.tiff", tmp_path / "out", DenoiseParams())
    assert not result.success
    assert result.error is not None

    no_channels = denoise_and_save(tmp_path / "x.tiff", tmp_path / "out", DenoiseParams(channels=()))
    assert not no_channels.success
    assert "no channels" in no_channels.error
