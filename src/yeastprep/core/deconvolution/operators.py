"""BlurFFT, Crop, Downsample linear operators.

Vendored (trimmed) from resolvde/operators.py: dropped ``Upsample``,
``NormalFFT``/``hessian_normal_symbol`` (regularizer machinery -- this
package's NLCG solver carries no regularizer), ``gaussian_kernel``, and
``BlurFFT.compose``/``fuse_kernel_into_operator`` (ICF fusion -- no ICF
here). What's left is exactly what :mod:`.model` needs to build the
wrap-free convolve -> crop -> downsample chain.

Each operator is an immutable bag of precomputed device tensors plus two
pure functions: a ``build()`` classmethod does the expensive one-time
planning, ``forward``/``adjoint`` touch only what construction prepared.
All operators act on the trailing ``len(in_shape)`` axes and broadcast
over any leading axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from . import shapes

__all__ = ["BlurFFT", "Crop", "Downsample"]


def _spatial_dims(ndim: int) -> tuple[int, ...]:
    return tuple(range(-ndim, 0))


def _kernel_otf(
    kernel: Tensor, fft_shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Embed a centered kernel into an ``fft_shape`` buffer and return its
    half-spectrum (rfftn) OTF."""
    kernel = kernel.to(device=device, dtype=dtype)
    kernel_shape = tuple(int(s) for s in kernel.shape)
    dims = _spatial_dims(len(fft_shape))
    kernel_pad = torch.zeros(fft_shape, dtype=dtype, device=device)
    # Align the kernel's center (floor convention, k // 2 -- matching
    # fftshift/ifftshift's own convention) with the buffer's center, so
    # ifftshift below moves the peak to exactly index 0 regardless of
    # fft_shape/kernel_shape parity.
    offset = tuple(f // 2 - k // 2 for f, k in zip(fft_shape, kernel_shape))
    embed = tuple(slice(o, o + k) for o, k in zip(offset, kernel_shape))
    kernel_pad[embed] = kernel
    kernel_pad = torch.fft.ifftshift(kernel_pad, dim=dims)
    return torch.fft.rfftn(kernel_pad, dim=dims)


# =============================================================================
# BlurFFT
# =============================================================================


@dataclass(frozen=True, eq=False)
class BlurFFT:
    """Wrap-free linear convolution: in_shape == out_shape == signal_shape.

    Accepts a **centered** PSF (peak at the array center) at construction;
    there is no corner-origin convention anywhere in this public API.

    ``_pad_cache`` holds the zero-padded ``fft_shape`` embed buffer, reused
    across calls instead of reallocated every ``forward``/``adjoint`` --
    solvers here call this many times per solve, and the margin outside the
    embedded signal is always zero and never written to, so only the first
    call per (batch-shape, dtype, device) combination pays for allocation +
    zeroing. Safe because the *returned* tensor always comes from
    ``irfftn``'s own fresh allocation (never aliases the reused buffer) and
    every embed write fully overwrites the signal region before use. Not
    safe for concurrent/re-entrant calls on the same instance -- fine here
    since the solver iterates sequentially.
    """

    otf: Tensor  # complex64 half-spectrum (rfftn layout) at fft_shape
    kernel: Tensor  # real-space, normalized, centered PSF
    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    fft_shape: tuple[int, ...]  # 5-smooth, >= signal + kernel - 1 per axis
    crop_slices: tuple[slice, ...]  # fft_shape -> signal_shape
    operator_norm_sq: float
    _pad_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def build(
        cls,
        psf: Tensor,
        signal_shape: tuple[int, ...],
        *,
        normalize: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "BlurFFT":
        signal_shape = tuple(int(s) for s in signal_shape)
        ndim = len(signal_shape)

        psf_t = torch.as_tensor(psf)
        kernel_shape = tuple(int(s) for s in psf_t.shape[-ndim:])
        if device is None:
            device = psf_t.device
        device = torch.device(device)
        psf_t = psf_t.to(device=device, dtype=dtype)
        if normalize:
            psf_t = psf_t / psf_t.sum()

        fft_shape = shapes.fast_padded_shape(signal_shape, kernel_shape)
        otf = _kernel_otf(psf_t, fft_shape, dtype=dtype, device=device)

        crop_slices = tuple(slice(0, s) for s in signal_shape)
        operator_norm_sq = float(otf.abs().max().item() ** 2)

        return cls(
            otf=otf,
            kernel=psf_t,
            in_shape=signal_shape,
            out_shape=signal_shape,
            fft_shape=fft_shape,
            crop_slices=crop_slices,
            operator_norm_sq=operator_norm_sq,
        )

    def _pad_buffer(self, x: Tensor, lead: tuple[int, ...]) -> Tensor:
        """Reused zero buffer for the embed step (see class docstring)."""
        key = (lead, x.dtype, x.device)
        buf = self._pad_cache.get(key)
        if buf is None:
            buf = x.new_zeros(*lead, *self.fft_shape)
            self._pad_cache[key] = buf
        return buf

    def _apply(self, x: Tensor, otf: Tensor) -> Tensor:
        ndim = len(self.in_shape)
        dims = _spatial_dims(ndim)
        lead = x.shape[:-ndim]
        x_pad = self._pad_buffer(x, lead)
        embed = (Ellipsis, *[slice(0, s) for s in self.in_shape])
        x_pad[embed] = x
        spec = torch.fft.rfftn(x_pad, dim=dims) * otf
        y_pad = torch.fft.irfftn(spec, s=self.fft_shape, dim=dims)
        return y_pad[(Ellipsis, *self.crop_slices)]

    def forward(self, x: Tensor) -> Tensor:
        return self._apply(x, self.otf)

    def adjoint(self, y: Tensor) -> Tensor:
        return self._apply(y, self.otf.conj())


# =============================================================================
# Crop
# =============================================================================


@dataclass(frozen=True, eq=False)
class Crop:
    """Center-cropping operator. Forward: slice. Adjoint: zero-pad."""

    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    slices: tuple[slice, ...]
    padding: tuple[tuple[int, int], ...]
    operator_norm_sq: float = 1.0

    @classmethod
    def build(
        cls,
        original_shape: tuple[int, ...],
        target_shape: tuple[int, ...],
        start: tuple[int, ...] | None = None,
    ) -> "Crop":
        original_shape = tuple(int(s) for s in original_shape)
        target_shape = tuple(int(s) for s in target_shape)
        if len(original_shape) != len(target_shape):
            raise ValueError(
                f"original_shape {original_shape} and target_shape "
                f"{target_shape} must have the same number of dimensions"
            )
        if start is not None and len(start) != len(original_shape):
            raise ValueError(
                f"start {start} must have the same number of dimensions as "
                f"original_shape {original_shape}"
            )

        slices = []
        padding = []
        for i, (orig, tgt) in enumerate(zip(original_shape, target_shape)):
            if tgt > orig:
                raise ValueError(f"target size {tgt} > original size {orig} on axis {i}")
            if start is None:
                axis_start = (orig - tgt) // 2
            else:
                axis_start = int(start[i])
                if axis_start < 0 or axis_start + tgt > orig:
                    raise ValueError(
                        f"start {axis_start} on axis {i} puts crop window "
                        f"[{axis_start}, {axis_start + tgt}) outside "
                        f"original size {orig}"
                    )
            stop = axis_start + tgt
            slices.append(slice(axis_start, stop))
            padding.append((axis_start, orig - stop))

        return cls(
            in_shape=original_shape,
            out_shape=target_shape,
            slices=tuple(slices),
            padding=tuple(padding),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x[(Ellipsis, *self.slices)]

    def adjoint(self, y: Tensor) -> Tensor:
        flat_pad: list[int] = []
        for before, after in reversed(self.padding):
            flat_pad.extend([before, after])
        return torch.nn.functional.pad(y, flat_pad, mode="constant", value=0.0)


# =============================================================================
# Fractional-area downsampling
# =============================================================================


def _dense_area_overlap_matrix(n_large: int, n_small: int) -> np.ndarray:
    """Dense ``(n_small, n_large)`` fractional-area overlap weights.

    ``W[i, j] = |[i*scale, (i+1)*scale) ∩ [j, j+1))|``, ``scale = n_large /
    n_small``. Built once in numpy float64, converted to a device tensor at
    construction.
    """
    scale = n_large / n_small
    i = np.arange(n_small, dtype=np.float64)[:, None]
    j = np.arange(n_large, dtype=np.float64)[None, :]
    i_start, i_end = i * scale, (i + 1) * scale
    j_start, j_end = j, j + 1.0
    overlap = np.minimum(i_end, j_end) - np.maximum(i_start, j_start)
    return np.maximum(overlap, 0.0)


class _no_tf32:
    """Force full float32 matmul precision for the wrapped block.

    On CUDA, matmul silently drops to TF32 if the process has enabled it
    globally. A no-op on CPU/MPS, where the flag doesn't affect matmul
    precision.
    """

    def __enter__(self):
        self._prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        return self

    def __exit__(self, *exc_info):
        torch.backends.cuda.matmul.allow_tf32 = self._prev
        return False


@dataclass(frozen=True, eq=False)
class _AxisPlan:
    n_large: int
    n_small: int
    fast: bool  # True: exact integer scale, reshape/sum path
    k: int | None  # integer scale factor, set iff fast
    weight: Tensor | None  # dense (n_small, n_large) overlap matrix, set iff not fast


def _build_axis_plans(
    in_axis_sizes: tuple[int, ...],
    scale: tuple[float, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[_AxisPlan, ...]:
    plans = []
    for n_in, s in zip(in_axis_sizes, scale):
        n_large, n_small = n_in, max(1, int(round(n_in / s)))

        k = round(s)
        fast = abs(s - k) < 1e-9 and k >= 1 and n_large == k * n_small

        if fast:
            plans.append(_AxisPlan(n_large, n_small, True, int(k), None))
        else:
            weight = torch.as_tensor(
                _dense_area_overlap_matrix(n_large, n_small), dtype=dtype, device=device
            )
            plans.append(_AxisPlan(n_large, n_small, False, None, weight))
    return tuple(plans)


def _resample_axis(x: Tensor, axis: int, plan: _AxisPlan, mode: str) -> Tensor:
    if plan.n_large == plan.n_small:
        return x

    if plan.fast:
        k = plan.k
        moved = torch.movedim(x, axis, -1)
        if mode == "forward":
            out = moved.reshape(*moved.shape[:-1], plan.n_small, k).sum(-1)
            return torch.movedim(out, -1, axis)
        if mode == "adjoint":
            return torch.repeat_interleave(x, k, dim=axis)
        raise ValueError(mode)  # pragma: no cover

    w = plan.weight
    moved = torch.movedim(x, axis, -1)
    batch_shape = moved.shape[:-1]
    flat = moved.reshape(-1, moved.shape[-1])

    with _no_tf32():
        if mode == "forward":
            out = flat @ w.T
        elif mode == "adjoint":
            out = flat @ w
        else:
            raise ValueError(mode)  # pragma: no cover

    out = out.reshape(*batch_shape, out.shape[-1])
    return torch.movedim(out, -1, axis)


def _resample_apply(x: Tensor, axis_plans: tuple[_AxisPlan, ...], mode: str) -> Tensor:
    ndim = len(axis_plans)
    result = x
    for i, plan in enumerate(axis_plans):
        axis = -(ndim - i)
        result = _resample_axis(result, axis, plan, mode)
    return result


def _normalize_scale(scale: float | tuple[float, ...], ndim: int) -> tuple[float, ...]:
    if isinstance(scale, (int, float)):
        return (float(scale),) * ndim
    scale_t = tuple(float(s) for s in scale)
    if len(scale_t) != ndim:
        raise ValueError(f"scale has {len(scale_t)} entries, expected {ndim}")
    return scale_t


@dataclass(frozen=True, eq=False)
class Downsample:
    """Fractional-area downsampling (fine -> coarse grid), scale >= 1 per axis.

    Forward preserves intensity (sums the fine voxels each coarse voxel
    covers); adjoint is the exact transpose (spreads each coarse value back
    over the fine pixels it covers).
    """

    in_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    axis_plans: tuple[_AxisPlan, ...]
    operator_norm_sq: float

    @classmethod
    def build(
        cls,
        scale: float | tuple[float, ...],
        in_shape: tuple[int, ...],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "Downsample":
        in_shape = tuple(int(s) for s in in_shape)
        scale_t = _normalize_scale(scale, len(in_shape))
        if any(s < 1.0 for s in scale_t):
            raise ValueError(f"Downsample requires scale >= 1 (fine -> coarse); got {scale_t}")
        plans = _build_axis_plans(
            in_shape, scale_t, device=torch.device(device), dtype=dtype
        )
        out_shape = tuple(p.n_small for p in plans)
        norm_sq = 1.0
        for p in plans:
            norm_sq *= p.n_large / p.n_small
        return cls(
            in_shape=in_shape, out_shape=out_shape, axis_plans=plans, operator_norm_sq=float(norm_sq)
        )

    def forward(self, x: Tensor) -> Tensor:
        return _resample_apply(x, self.axis_plans, "forward")

    def adjoint(self, y: Tensor) -> Tensor:
        return _resample_apply(y, self.axis_plans, "adjoint")
