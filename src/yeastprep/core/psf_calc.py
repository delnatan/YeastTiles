"""2D sum-projected widefield PSF calculation for the deconvolution
sub-step of the enhancement stage (design.md 2b): an equivalent 2D PSF,
sum-projected from a 3D PSF computed at matching optical parameters (NA,
wavelength, lateral pixel spacing), oversampled axially to cover the
data's full defocus range.

`psfkit` already ships a full-featured PSF dialog
(`psfkit/pyvistra_gui/psf_dialog.py`) supporting widefield/confocal/
spinning-disk models and Zernike aberrations, but it routes its result
through pyvistra's `ImageBuffer`/output-selector machinery, which this app
doesn't embed. This module covers just the widefield + 2D-sum case
design.md actually calls for, reusing psfkit's own PSF math directly.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyvistra.io import save_tiff


@dataclass
class PSFCalcParams:
    wavelength_um: float = 0.525
    na: float = 1.4
    ni: float = 1.515
    ns: float | None = None
    dxy_um: float | None = None  # None -> psfkit's Nyquist-limited lateral spacing
    nxy: int = 128
    dz_um: float = 0.1
    axial_range_um: float = 10.0
    axial_oversample: int = 3
    pixel_integrated: bool = False


def compute_2d_psf(params: PSFCalcParams) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Compute a 3D widefield PSF then sum-project it over z into a
    single 2D plane -- the same `_2D_SUM` step psfkit's own PSF dialog
    uses (`psf_dialog.py`'s `_compute`: `np.sum(psf, axis=0,
    keepdims=True)`), just without the rest of that dialog's model/
    aberration options.

    Returns the PSF as (1, nxy, nxy) and its (dz, dy, dx) spacing in um.

    Imports psfkit lazily -- this is the only function in the module that
    needs it (see the `psf` extra in pyproject.toml), so the rest of the
    Deconvolve page keeps working with a user-supplied PSF tiff even if
    psfkit isn't installed.
    """
    from psfkit import Optics, compute_widefield_psf, nyquist_lateral

    optics = Optics(wavelength=params.wavelength_um, na=params.na, ni=params.ni, ns=params.ns)
    dxy = params.dxy_um or nyquist_lateral(optics)
    nz = max(1, round(params.axial_range_um / params.dz_um))
    spacing = (params.dz_um, dxy, dxy)

    psf3d = compute_widefield_psf(
        optics,
        shape=(nz, params.nxy, params.nxy),
        spacing=spacing,
        pixel_integrated=params.pixel_integrated,
        axial_oversample=params.axial_oversample,
    )
    psf2d = np.sum(psf3d, axis=0, keepdims=True).astype(np.float32)
    return psf2d, spacing


def save_psf(path, psf2d: np.ndarray, spacing: tuple[float, float, float]):
    save_tiff(Path(path), psf2d, scale=spacing, input_axes="ZYX")
