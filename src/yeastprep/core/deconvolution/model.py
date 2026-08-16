"""ForwardModel dataclass + make_forward_model() factory.

Vendored (unchanged in substance) from resolvde/model.py.

Canonical imaging chain::

    x (padded visible) -> BlurFFT -> Crop(padded -> visible, offset-exact)
    -> Downsample -> data

Three-domain model:

- **data**: detector grid (what the camera recorded)
- **visible**: data grid refined by ``zoom >= 1`` (reconstruction resolution)
- **padded**: visible + PSF margins -- the solver's domain, so emitters just
  outside the FOV can explain edge pixels and convolution stays wrap-free
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from . import shapes
from .operators import BlurFFT, Crop, Downsample
from .protocol import Compose, LinearOperator

__all__ = ["ForwardModel", "make_forward_model"]


@dataclass(frozen=True, eq=False)
class ForwardModel:
    """Forward operator plus the shape bookkeeping needed to use it.

    ``op`` maps ``padded_shape -> data_shape`` (forward) and back
    (adjoint). Solvers iterate on arrays of ``padded_shape``;
    ``valid_slices`` crops a padded-domain reconstruction back to
    ``visible_shape``.
    """

    op: LinearOperator
    data_shape: tuple[int, ...]
    visible_shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    valid_slices: tuple[slice, ...]
    device: torch.device


def make_forward_model(
    psf: Tensor,
    data_shape: tuple[int, ...],
    zoom: float | tuple[float, ...] = 1.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> ForwardModel:
    """Build the canonical forward model: convolve -> crop -> downsample.

    With ``visible_shape == data_shape`` (the common ``zoom == 1`` case),
    the downsample stage is identity and is omitted.

    Args:
        psf: Point spread function sampled at *visible-space* pixel
            spacing (data spacing / zoom) -- a **centered** PSF; there is
            no corner-origin convention anywhere here.
        data_shape: Shape of the observed data (e.g. ``(Ny, Nx)`` for 2D).
        zoom: Visible pixels per data pixel, >= 1 (>1 reconstructs on a
            finer grid). Scalar or per-axis tuple.
        device: Device every stage is pinned to. Defaults to ``psf``'s own
            device.
        dtype: Real dtype for every stage -- float32 default and, on MPS,
            the only option.

    Returns:
        ``ForwardModel``. ``op.forward`` maps ``padded_shape ->
        data_shape``; ``op.adjoint`` maps back.
    """
    ndim = len(data_shape)
    if isinstance(zoom, (int, float)):
        zoom_t = (float(zoom),) * ndim
    else:
        zoom_t = tuple(float(z) for z in zoom)
    if len(zoom_t) != ndim:
        raise ValueError(f"zoom has {len(zoom_t)} entries, expected {ndim}")
    if any(z < 1.0 for z in zoom_t):
        raise ValueError(
            f"zoom must be >= 1 (visible grid at least as fine as data); got {zoom_t}"
        )

    psf_t = torch.as_tensor(psf)
    if psf_t.ndim != ndim:
        raise ValueError(f"psf is {psf_t.ndim}D but data_shape is {ndim}D")
    if device is None:
        device = psf_t.device
    device = torch.device(device)

    data_shape = tuple(int(d) for d in data_shape)
    visible_shape = tuple(max(1, round(d * z)) for d, z in zip(data_shape, zoom_t))
    kernel_shape = tuple(int(s) for s in psf_t.shape)
    padded_shape, padding = shapes.compute_padded_shape(visible_shape, kernel_shape)
    valid_slices = shapes.get_valid_slices(padded_shape, visible_shape, padding)

    blur = BlurFFT.build(psf_t, padded_shape, normalize=True, device=device, dtype=dtype)

    # Crop padded -> visible on the fine grid first, using the exact
    # (possibly asymmetric, for even-sized PSFs) pad_before offset. Doing
    # this before downsampling -- rather than downsampling the whole padded
    # domain and center-cropping the coarse result -- avoids a sub-pixel
    # registration shift: a naive center-crop of the downsampled *padded*
    # domain has no knowledge of padding's asymmetry and assumes the
    # visible content sits exactly centered, which only holds for
    # odd-sized PSFs.
    crop_start = tuple(pad_before for pad_before, _ in padding)
    visible_crop = Crop.build(padded_shape, visible_shape, start=crop_start)

    if visible_shape == data_shape:
        op = Compose(visible_crop, blur)
    else:
        # Effective ratio (not the nominal zoom) so downsampling
        # visible_shape lands on data_shape exactly, with no further crop
        # needed.
        eff_scale = tuple(v / d for v, d in zip(visible_shape, data_shape))
        downsampler = Downsample.build(eff_scale, visible_shape, device=device, dtype=dtype)
        op = Compose(downsampler, visible_crop, blur)

    return ForwardModel(
        op=op,
        data_shape=data_shape,
        visible_shape=visible_shape,
        padded_shape=padded_shape,
        valid_slices=valid_slices,
        device=device,
    )
