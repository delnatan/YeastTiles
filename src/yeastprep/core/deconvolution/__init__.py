"""A self-contained 2D Poisson-ML deconvolution submodule, known PSF, NLCG.

Vendored (trimmed to the 2D/NLCG/no-regularizer slice) from the
``resolvde`` project (github.com/delnatan/resolvde). Two things carried
over wholesale because they're the parts worth keeping:

- **The three-domain forward model** (:mod:`.model`, :mod:`.padding`):
  convolve on a PSF-padded grid, then crop/downsample down to the data
  grid, with an edge-tapered (not zero-padded) initial estimate. This is
  what keeps a deconvolved image free of bright/dark ringing at the field
  edges -- exactly the boundary-artifact problem a naive
  "pad with zeros" implementation runs into.
- **NLCG** (:mod:`.nlcg`): Schaefer, Schuster & Herz (2001)'s accelerated
  Poisson-ML restoration, with a closed-form exact step length (no
  backtracking line search) and Morozov's discrepancy principle for a
  parameter-light stopping rule.

Dropped relative to the source project: every other solver
(``erdecon_newton``, ``maxent_dual``, ...), all regularizers, tiling,
and joint background fitting. This package's use case is one 2D image
against one known PSF -- see :func:`deconvolve`.

:mod:`.photon_calibration` is independent of the rest (only depends on
``torch``) -- pull just that module if all you need is ADU-to-photon
conversion.
"""

from __future__ import annotations

from typing import Union

import torch
from torch import Tensor

from . import padding
from .model import ForwardModel, make_forward_model
from .nlcg import NLCGResult, nlcg_solver, nlcg_with_operator
from .operators import BlurFFT, Crop, Downsample
from .photon_calibration import (
    PhotonCalibration,
    SolveUnits,
    dark_frame_statistics,
    estimate_photon_calibration,
)
from .protocol import Compose, LinearOperator

__version__ = "0.1.0"

__all__ = [
    "BlurFFT",
    "Compose",
    "Crop",
    "Downsample",
    "ForwardModel",
    "LinearOperator",
    "NLCGResult",
    "PhotonCalibration",
    "SolveUnits",
    "dark_frame_statistics",
    "deconvolve",
    "estimate_photon_calibration",
    "make_forward_model",
    "nlcg_solver",
    "nlcg_with_operator",
]


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def deconvolve(
    data,
    psf,
    zoom: Union[float, tuple[float, float]] = 1.0,
    background: float = 0.0,
    device: Union[torch.device, str, None] = None,
    **nlcg_kwargs,
) -> NLCGResult:
    """Deconvolve one 2D image against one PSF -- the single entry point
    most callers need.

    Builds the :class:`.model.ForwardModel`, moves data to the target
    device, seeds the solve with an edge-tapered init (see
    :func:`.padding.default_init`), runs :func:`.nlcg.nlcg_with_operator`,
    and crops ``restored`` from the PSF-padded solve domain down to the
    visible (data-grid, or ``zoom``-refined) domain -- ``pred`` is already
    data-shaped, since ``model.op`` maps padded -> data.

    Args:
        data: Observed 2D detector image (numpy array or tensor). Already
            in whatever units the likelihood should be exact in -- convert
            ADU to photons first via :class:`.photon_calibration.SolveUnits`
            or :class:`.photon_calibration.PhotonCalibration` if needed.
        psf: Point spread function, a **centered** 2D array (peak at the
            array center), at visible-space pixel spacing (data spacing /
            zoom).
        zoom: Visible pixels per data pixel, >= 1. ``1.0`` (the default)
            reconstructs on the data's own grid.
        background: Constant background level, same units as ``data``.
        device: Target device. ``None`` auto-selects cuda > mps > cpu.
        **nlcg_kwargs: Forwarded to :func:`.nlcg.nlcg_with_operator` (e.g.
            ``num_iter``, ``slack``, ``tol``, ``verbose``).

    Returns:
        :class:`.nlcg.NLCGResult`, with ``restored`` cropped to
        ``model.visible_shape``.
    """
    data_t: Tensor = torch.as_tensor(data, dtype=torch.float32)
    if data_t.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {tuple(data_t.shape)}")

    resolved_device = torch.device(device) if device is not None else _auto_device()
    data_t = data_t.to(device=resolved_device)
    psf_t: Tensor = torch.as_tensor(psf, dtype=torch.float32, device=resolved_device)

    model = make_forward_model(psf_t, tuple(data_t.shape), zoom=zoom, device=resolved_device)
    init = padding.default_init(data_t, model, background=background)

    result = nlcg_with_operator(
        data_t, model.op, background=background, init=init, **nlcg_kwargs
    )
    result.restored = result.restored[model.valid_slices]
    return result
