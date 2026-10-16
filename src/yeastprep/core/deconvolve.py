"""Deconvolution driver: optional Poisson-ML deconvolution
(`core/deconvolution`) of a combined-channel tiff's target channel only,
writing into 03_deconvolved/.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from pyvistra.io import save_tiff

from .combined_tiff import combine_channels, read_combined_tiff
from .deconvolution import deconvolve


@dataclass
class DeconvolveParams:
    enabled: bool = False
    psf_path: str | None = None
    zoom: float = 1.0
    background: float = 0.0
    num_iter: int = 50


def load_psf(path) -> np.ndarray:
    """Read a PSF tiff back into a 2D array, squeezing any leading
    singleton z-axis left over from a 2D-sum-projected PSF saved by
    `core/psf_calc.py::save_psf`."""
    psf = np.asarray(tifffile.imread(path))
    while psf.ndim > 2 and psf.shape[0] == 1:
        psf = psf[0]
    return psf


def read_pixel_size_um(path) -> float | None:
    """A tiff's lateral pixel size in um, from its X resolution tag -- or
    None when it has no calibration (no tag, or tifffile's uncalibrated
    default of 1 pixel per unit)."""
    with tifffile.TiffFile(path) as tf:
        tag = tf.pages[0].tags.get("XResolution")
        if tag is None:
            return None
        numerator, denominator = tag.value
    if not numerator or numerator == denominator:
        return None
    return denominator / numerator


# How far a PSF's pixel spacing may stray from the image's (divided by
# zoom) before deconvolution refuses it. A Nyquist-sampled PSF is commonly
# 0.09 um/px against 0.065 um camera pixels -- a ~40% size error that
# deconvolves without complaint but sharpens to the wrong width.
_PSF_SPACING_TOLERANCE = 0.02


def psf_spacing_problem(
    psf_pixel_um: float | None, image_pixel_um: float | None, zoom: float = 1.0
) -> str | None:
    """Why a PSF can't be used on an image, or None if it can (or if either
    spacing is unknown, so there's nothing to check). The deconvolver
    models the PSF at the reconstruction grid's spacing, image pixel /
    zoom."""
    if psf_pixel_um is None or image_pixel_um is None:
        return None
    expected = image_pixel_um / zoom
    if abs(psf_pixel_um - expected) <= _PSF_SPACING_TOLERANCE * expected:
        return None
    zoom_note = f" / zoom {zoom:g}" if zoom != 1 else ""
    return (
        f"PSF pixel spacing is {psf_pixel_um:.4f} um but the image needs "
        f"{expected:.4f} um (pixel {image_pixel_um:.4f} um{zoom_note}). "
        "Recompute the PSF in the PSF Calculator with that lateral spacing."
    )


def run_deconvolve(
    target: np.ndarray, params: DeconvolveParams, pixel_size_um: float | None = None
) -> np.ndarray:
    """Deconvolve the target channel only -- brightfield has no PSF model
    here (see design.md). Casts the result back to uint16 so
    03_deconvolved/ stays a consistent 16-bit drop-in replacement for
    upstream stages, matching the project's "16-bit for 2D intermediates"
    convention (deconvolution runs internally in float32, per
    core/deconvolution/model.py).

    `pixel_size_um` is the image's lateral pixel size; when both it and the
    PSF file's spacing are known, a mismatch raises ValueError rather than
    deconvolving with a wrongly-sized PSF."""
    if not params.enabled:
        return target
    if not params.psf_path:
        raise ValueError("psf_path is required when deconvolution is enabled")
    problem = psf_spacing_problem(
        read_pixel_size_um(params.psf_path), pixel_size_um, params.zoom
    )
    if problem:
        raise ValueError(problem)
    psf = load_psf(params.psf_path)
    result = deconvolve(
        target,
        psf,
        zoom=params.zoom,
        background=params.background,
        num_iter=params.num_iter,
    )
    restored = result.restored.detach().cpu().numpy()
    return np.clip(restored, 0, 65535).astype(np.uint16)


@dataclass
class DeconvolveProcessResult:
    path: Path
    success: bool
    output_path: Path | None = None
    error: str | None = None


def deconvolve_and_save(
    path: Path, outdir: Path, params: DeconvolveParams = DeconvolveParams()
) -> DeconvolveProcessResult:
    """Deconvolve one combined-channel tiff's target channel and write the
    result (same 2-channel layout, brightfield passed through unchanged)
    into `outdir` (03_deconvolved/). `path` is typically a file already in
    02_denoised/ if that stage ran, else 01_reduced/ -- see
    core.project.resolve_2d_source. Catches any exception so a batch run
    doesn't abort on the first bad file."""
    path = Path(path)
    outdir = Path(outdir)
    try:
        outdir.mkdir(parents=True, exist_ok=True)

        brightfield, target, scale = read_combined_tiff(path)
        target = run_deconvolve(target, params, read_pixel_size_um(path))
        combined = combine_channels(brightfield, target)

        output_path = outdir / f"{path.stem}.tiff"
        save_tiff(output_path, combined, scale=scale, input_axes="CYX")

        return DeconvolveProcessResult(path=path, success=True, output_path=output_path)
    except Exception as exc:
        return DeconvolveProcessResult(path=path, success=False, error=str(exc))
