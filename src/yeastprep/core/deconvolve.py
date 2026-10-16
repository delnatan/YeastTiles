"""Deconvolution driver: optional Poisson-ML deconvolution
(`core/deconvolution`) of a combined-channel tiff's target channel only,
writing into 03_deconvolved/.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from .combined_tiff import read_combined_tiff, read_z_count, write_combined_tiff
from .deconvolution import deconvolve


@dataclass
class DeconvolveParams:
    enabled: bool = False
    psf_path: str | None = None
    zoom: float = 1.0
    # Raw counts per z-slice (the camera's dark offset). The target channel
    # is a raw z-sum, so the solver's background is this x the file's
    # slice count, plus `background` -- see `effective_background`.
    camera_offset: float = 100.0
    # Any further flat background, in summed counts.
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


def effective_background(params: DeconvolveParams, n_z: int | None) -> float:
    """The solver's background for a target channel summed over `n_z` raw
    z-slices: the camera offset contributes once per slice. Raises when
    there's an offset to account for but no slice count (a file reduced
    before the count was recorded)."""
    if not params.camera_offset:
        return params.background
    if n_z is None:
        raise ValueError(
            "This file doesn't record how many z-slices its target channel "
            "sums, so the camera offset can't be scaled to it. Re-run Data "
            "Reduction, or set the camera offset to 0."
        )
    return params.background + params.camera_offset * n_z


def run_deconvolve(
    target: np.ndarray,
    params: DeconvolveParams,
    pixel_size_um: float | None = None,
    n_z: int | None = None,
) -> np.ndarray:
    """Deconvolve the target channel only -- brightfield has no PSF model
    here (see design.md). The input is float32 summed counts (see
    core/pipeline.py's `sum_project`), deconvolved in float32 (per
    core/deconvolution/model.py), and only then min-max rescaled onto the
    full uint16 range (`rescale_to_uint16`) -- so sharpened peaks brighter
    than anything in the input aren't clipped, and dim structure isn't
    rounded away before deconvolution has seen it. Per FOV, so
    intensities aren't comparable across FOVs.

    `pixel_size_um` is the image's lateral pixel size; when both it and the
    PSF file's spacing are known, a mismatch raises ValueError rather than
    deconvolving with a wrongly-sized PSF. `n_z` is the number of raw
    z-slices summed into `target` (`read_z_count`), for
    `effective_background`."""
    if not params.enabled:
        return target
    if not params.psf_path:
        raise ValueError("psf_path is required when deconvolution is enabled")
    problem = psf_spacing_problem(
        read_pixel_size_um(params.psf_path), pixel_size_um, params.zoom
    )
    if problem:
        raise ValueError(problem)
    background = effective_background(params, n_z)
    psf = load_psf(params.psf_path)
    result = deconvolve(
        target,
        psf,
        zoom=params.zoom,
        background=background,
        num_iter=params.num_iter,
    )
    return rescale_to_uint16(result.restored.detach().cpu().numpy())


def rescale_to_uint16(image: np.ndarray) -> np.ndarray:
    """Min-max rescale onto the full uint16 range; all zeros for a flat
    image."""
    image = np.asarray(image, dtype=np.float64)
    lo, hi = image.min(), image.max()
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint16)
    scaled = (image - lo) * (np.iinfo(np.uint16).max / (hi - lo))
    return np.round(scaled).astype(np.uint16)


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
        n_z = read_z_count(path)
        target = run_deconvolve(target, params, read_pixel_size_um(path), n_z)

        output_path = outdir / f"{path.stem}.tiff"
        write_combined_tiff(output_path, brightfield, target, scale, n_z=n_z)

        return DeconvolveProcessResult(path=path, success=True, output_path=output_path)
    except Exception as exc:
        return DeconvolveProcessResult(path=path, success=False, error=str(exc))
