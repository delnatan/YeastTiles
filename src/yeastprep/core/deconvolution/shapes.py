"""Shape and padding arithmetic for the two-domain forward model.

Pure Python on tuples -- no torch dependency. Vendored (trimmed) from
resolvde/shapes.py: dropped ``compute_visible_shape`` (model.py computes
``visible_shape`` inline; the helper had no other caller in this package).

The mental model:

- **data**: detector grid (what the camera recorded)
- **visible**: data grid refined by zoom >= 1 (reconstruction resolution)
- **padded**: visible + PSF margins (the solver's domain; wrap-free
  convolution, and emitters just outside the FOV can explain edge pixels)
"""

from __future__ import annotations

__all__ = [
    "compute_padded_shape",
    "get_valid_slices",
    "_next_smooth_number",
    "fast_padded_shape",
]


def compute_padded_shape(
    signal_shape: tuple[int, ...],
    kernel_shape: tuple[int, ...],
    *,
    min_pad: int | tuple[int | None, ...] | None = None,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Compute padded shape and per-axis (before, after) padding.

    Pads each axis by ``kernel_dim - 1``, the exact minimum for wrap-free
    linear convolution via circular FFT. ``min_pad`` overrides that on
    specific axes (e.g. ``0`` for an axis known to be confined within the
    signal already).
    """
    signal_shape = tuple(int(s) for s in signal_shape)
    kernel_shape = tuple(int(s) for s in kernel_shape)
    ndim = len(signal_shape)

    if len(kernel_shape) != ndim:
        raise ValueError(
            f"kernel_shape has {len(kernel_shape)} dimensions, "
            f"expected {ndim} to match signal_shape"
        )

    if min_pad is None:
        min_pad_tuple: tuple[int | None, ...] | None = None
    elif isinstance(min_pad, int):
        min_pad_tuple = (min_pad,) * ndim
    else:
        min_pad_tuple = tuple(min_pad)

    padding_list = []
    padded_shape_list = []

    for i in range(ndim):
        signal_dim = signal_shape[i]
        kernel_dim = kernel_shape[i]

        total_pad = kernel_dim - 1
        if min_pad_tuple is not None and min_pad_tuple[i] is not None:
            total_pad = min_pad_tuple[i]

        pad_before = total_pad // 2
        pad_after = total_pad - pad_before

        padding_list.append((pad_before, pad_after))
        padded_shape_list.append(signal_dim + total_pad)

    return tuple(padded_shape_list), tuple(padding_list)


def get_valid_slices(
    padded_shape: tuple[int, ...],
    signal_shape: tuple[int, ...],
    padding: tuple[tuple[int, int], ...] | None = None,
) -> tuple[slice, ...]:
    """Slices to extract the un-padded ``signal_shape`` region from a padded array.

    If ``padding`` is omitted, symmetric padding is assumed.
    """
    padded_shape = tuple(int(s) for s in padded_shape)
    signal_shape = tuple(int(s) for s in signal_shape)
    ndim = len(padded_shape)

    if len(signal_shape) != ndim:
        raise ValueError(
            f"padded_shape has {ndim} dimensions, "
            f"but signal_shape has {len(signal_shape)} dimensions. "
            f"Both must have the same ndim."
        )

    if padding is None:
        padding_list = []
        for p_dim, s_dim in zip(padded_shape, signal_shape):
            total_pad = p_dim - s_dim
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            padding_list.append((pad_before, pad_after))
        padding = tuple(padding_list)

    slices = []
    for i in range(ndim):
        pad_before, pad_after = padding[i]
        start = pad_before
        stop = padded_shape[i] - pad_after
        slices.append(slice(start, stop))

    return tuple(slices)


def _next_smooth_number(n: int) -> int:
    """Smallest integer >= n whose prime factors are only 2, 3, or 5.

    cuFFT and Apple Metal both reach peak throughput on 5-smooth sizes;
    ``scipy.fft.next_fast_len`` also accepts 7 and 11, which is fine for
    FFTW on CPU but can be much slower on GPU.
    """
    candidate = n
    while True:
        m = candidate
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return candidate
        candidate += 1


def fast_padded_shape(
    signal_shape: tuple[int, ...],
    kernel_shape: tuple[int, ...],
    min_pad: int | tuple[int | None, ...] | None = None,
) -> tuple[int, ...]:
    """Smallest 5-smooth padded shape >= signal + kernel - 1, per axis.

    ``min_pad`` relaxes the per-axis padding requirement (e.g. ``0`` for an
    axis known to be confined within the signal already); the returned
    shape is always floored at ``kernel_shape`` so the kernel still fits.
    """
    ndim = len(signal_shape)
    if min_pad is None:
        pads: tuple[int | None, ...] = (None,) * ndim
    elif isinstance(min_pad, int):
        pads = (min_pad,) * ndim
    else:
        pads = tuple(min_pad)

    result = []
    for n, m, p in zip(signal_shape, kernel_shape, pads):
        pad_needed = (m - 1) if p is None else int(p)
        target = max(n + pad_needed, m)
        result.append(_next_smooth_number(target))
    return tuple(result)
