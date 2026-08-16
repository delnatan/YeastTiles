"""Tests for yeastprep.core.denoise. `run_denoise` needs a real
jssl-denoise checkpoint, which isn't available as a test fixture, so
`denoise_and_save`'s merge-with-existing-output logic is tested with
`run_denoise` monkeypatched to a cheap stand-in (matching the pattern
`test_pipeline.py`/`test_segmentation.py` use of exercising real files,
just without a real model in the loop where one isn't available).
"""

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
from yeastprep.core.denoise import DenoiseParams, denoise_and_save, run_denoise


def _synthetic_image(Ny=16, Nx=16, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(Ny, Nx)).astype(np.uint16)


def test_run_denoise_requires_checkpoint_path():
    image = _synthetic_image()
    with pytest.raises(ValueError, match="checkpoint_path"):
        run_denoise(image, DenoiseParams(checkpoint_path=None))


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
    result_bf = denoise_and_save(
        path, outdir, DenoiseParams(channel=BRIGHTFIELD_CHANNEL, checkpoint_path="fake")
    )
    assert result_bf.success, result_bf.error
    bf1, tgt1, _ = read_combined_tiff(result_bf.output_path)
    assert np.all(bf1 == 111)
    assert np.array_equal(tgt1, target0)  # untouched channel carried over from source

    # Denoise channel 1 (target) tomorrow -- shouldn't lose today's result.
    monkeypatch.setattr(
        denoise_module, "run_denoise", lambda image, params: np.full_like(image, 222)
    )
    result_tgt = denoise_and_save(
        path, outdir, DenoiseParams(channel=TARGET_CHANNEL, checkpoint_path="fake")
    )
    assert result_tgt.success, result_tgt.error
    bf2, tgt2, _ = read_combined_tiff(result_tgt.output_path)
    assert np.all(tgt2 == 222)
    assert np.all(bf2 == 111)  # yesterday's brightfield result preserved, not clobbered


def test_denoise_and_save_reports_failure_without_raising(tmp_path):
    missing_path = tmp_path / "does_not_exist.tiff"
    result = denoise_and_save(
        missing_path, tmp_path / "out", DenoiseParams(checkpoint_path="fake")
    )
    assert not result.success
    assert result.error is not None
