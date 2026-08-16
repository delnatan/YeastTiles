"""Edge-taper padding for the visible -> padded reconstruction domain.

Vendored (trimmed) from resolvde/padding.py: dropped ``flat_init``/
``flat_init_value`` (tiling-oriented constant-value inits; this package
only ever solves one whole image, not a grid of tiles).

Two different things are both called "padding" here, and only one of them
should ever be zero:

- ``operators.BlurFFT`` zero-pads the *kernel* out to ``fft_shape`` so a
  circular FFT computes a *linear* convolution. That zero-pad is a
  numerical requirement of the FFT trick, not a model of the data --
  changing it would compute the wrong convolution.
- ``shapes.compute_padded_shape`` extends ``visible_shape`` to
  ``padded_shape`` so emitters just outside the FOV can explain edge
  pixels (the three-domain model in ``model.py``). Feeding an implicit
  ``0`` into that margin -- what a naive default init does -- makes it a
  real, sharp content discontinuity: the PSF then smears that dark edge
  back into the visible field, especially along a widefield PSF's wide
  axial missing cone (relevant even in 2D, since a sum-projected 3D PSF
  inherits that same broad skirt).

``edge_taper_pad`` addresses the second case: it replicates the edge
pixels into the margin, then tapers them to ``0`` with a half-cosine
(Tukey) window, so the margin decays smoothly instead of stopping dead --
this is what keeps the deconvolution free of the bright/dark ringing a
zero-padded boundary would otherwise produce.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["edge_taper_pad", "default_init"]


def _taper_ramp(width: int, alpha: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Weights for one margin, ordered outer edge -> interior.

    ``0`` at the outermost sample, ramping (half-cosine) up to ``1``;
    ``alpha`` is the fraction of ``width`` (from the outer edge inward)
    that ramps -- the remainder, nearer the interior, stays flat at ``1``.
    """
    if width <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    taper_len = max(1, min(width, round(alpha * width)))
    flat_len = width - taper_len
    if taper_len == 1:
        t = torch.zeros(1, device=device, dtype=dtype)
    else:
        t = torch.linspace(0.0, 1.0, taper_len, device=device, dtype=dtype)
    ramp = 0.5 * (1 - torch.cos(math.pi * t))
    return torch.cat([ramp, torch.ones(flat_len, device=device, dtype=dtype)])


def _taper_window(
    padded_dim: int, before: int, after: int, alpha: float, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """1D weight vector of length ``padded_dim``: ``1`` across the interior,
    half-cosine ramps down to ``0`` at each outer edge."""
    interior = padded_dim - before - after
    before_ramp = _taper_ramp(before, alpha, device, dtype)
    after_ramp = torch.flip(_taper_ramp(after, alpha, device, dtype), dims=(0,))
    middle = torch.ones(interior, device=device, dtype=dtype)
    return torch.cat([before_ramp, middle, after_ramp])


def edge_taper_pad(
    x: Tensor,
    padded_shape: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
    alpha: float = 1.0,
) -> Tensor:
    """Extend ``x`` into ``padded_shape`` by replicating its edge pixels
    into each margin, then tapering them to ``0`` with a per-axis
    half-cosine (Tukey) window.

    ``x``'s trailing ``len(padding)`` axes must have size ``padded_shape[i]
    - padding[i][0] - padding[i][1]``; any leading axes broadcast through
    unchanged. ``alpha`` is the fraction of each margin (from its outer
    edge inward) that tapers -- ``1.0`` (the default) tapers the whole
    margin, matching how wide it already is: the width is exactly the
    PSF's own support (``kernel_dim - 1``), so there's no slack to spend on
    a flat run before the taper starts.
    """
    ndim = len(padding)
    expected = tuple(p - before - after for p, (before, after) in zip(padded_shape, padding))
    if tuple(x.shape[-ndim:]) != expected:
        raise ValueError(
            f"x's trailing shape {tuple(x.shape[-ndim:])} doesn't match "
            f"padded_shape {padded_shape} minus padding {padding} ({expected})"
        )

    lead_ndim = x.ndim - ndim
    lead_shape = x.shape[:lead_ndim]

    flat_pad: list[int] = []
    for before, after in reversed(padding):
        flat_pad.extend([before, after])

    # F.pad(mode="replicate") needs an (N, C, *spatial) tensor matching
    # ndim (3D/4D/5D input for 1D/2D/3D padding); flatten any leading axes
    # into one batch dim and restore them afterward.
    x_batched = x.reshape(-1, 1, *x.shape[lead_ndim:])
    x_pad = F.pad(x_batched, flat_pad, mode="replicate")
    x_pad = x_pad.reshape(*lead_shape, *padded_shape)

    window = None
    for axis, (p_dim, (before, after)) in enumerate(zip(padded_shape, padding)):
        w = _taper_window(p_dim, before, after, alpha, x.device, x.dtype)
        shape = [1] * ndim
        shape[axis] = p_dim
        w = w.reshape(shape)
        window = w if window is None else window * w

    return x_pad * window


def default_init(
    data: Tensor,
    model,  # ForwardModel: model.py, no import cycle (see nlcg.py's own note)
    background: float = 0.0,
    alpha: float = 1.0,
) -> Tensor:
    """Edge-tapered initial estimate for a solver's padded reconstruction
    domain, built directly from the observed ``data``.

    Upsamples ``data`` to ``model.visible_shape`` (nearest-neighbor, only
    when ``zoom > 1`` makes the two differ), subtracts ``background``, and
    extends the result into ``model.padded_shape`` with
    :func:`edge_taper_pad`. Returned in the same (raw, unnormalized) units
    as ``data``.
    """
    ndim = len(model.data_shape)
    visible = data
    if tuple(model.visible_shape) != tuple(model.data_shape):
        lead_shape = data.shape[:-ndim]
        batched = data.reshape(-1, 1, *data.shape[-ndim:])
        batched = F.interpolate(batched, size=tuple(model.visible_shape), mode="nearest")
        visible = batched.reshape(*lead_shape, *model.visible_shape)

    visible = torch.clamp(visible - background, min=1e-6)

    padding = tuple(
        (s.start, p - s.stop) for s, p in zip(model.valid_slices, model.padded_shape)
    )
    return edge_taper_pad(visible, model.padded_shape, padding, alpha=alpha)
