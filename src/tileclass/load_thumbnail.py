"""Minimal single-plane image loader: tifffile + PIL + numpy only.

Unlike pyvistra's ``io.load_image`` (which returns a lazy 5D proxy for
arbitrary T/Z/C navigation), this returns a single already-materialized
``(C, H, W)`` plane -- all a classification tile ever needs. Multi-page
TIFFs / stacks are collapsed to T=0 and the middle Z slice, same as
``pyvistra.data.thumbnail_cache.DecodeCache.decode``'s ``proxy[0, z_mid]``.
"""

from pathlib import Path

import numpy as np
from PIL import Image

TARGET_ORDER = "tzcyx"

TIFF_SUFFIXES = (".tif", ".tiff")

# Anything PIL can open. Extend as needed.
PIL_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

SUPPORTED_SUFFIXES = TIFF_SUFFIXES + PIL_SUFFIXES


def _is_rgb_image(arr):
    """``(Y, X, 3|4)`` with spatial dims larger than the color axis."""
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        if arr.shape[0] > 4 and arr.shape[1] > 4:
            return True
    return False


def _to_5d(data, dims=None):
    """Reorder/reshape *data* to ``(T, Z, C, Y, X)``.

    ``dims``: axis-order string matching ``data.ndim`` (e.g. ``"zcyx"``),
    or ``None`` to fall back to ndim-based heuristics.
    """
    if dims:
        dims = dims.lower()
        if len(dims) != data.ndim:
            raise ValueError(
                f"dims string length ({len(dims)}) must match data ndim ({data.ndim})"
            )
        present = [d for d in TARGET_ORDER if d in dims]
        perm = [dims.index(d) for d in present]
        data = np.transpose(data, perm)
        target_shape = [
            data.shape[present.index(d)] if d in present else 1
            for d in TARGET_ORDER
        ]
        return data.reshape(target_shape)

    ndim = data.ndim
    if ndim == 2:  # (Y, X)
        return data[np.newaxis, np.newaxis, np.newaxis, :, :]
    if ndim == 3:
        return data[np.newaxis, np.newaxis, :, :, :]  # (C, Y, X)
    if ndim == 4:  # (Z, C, Y, X)
        return data[np.newaxis, :, :, :, :]
    if ndim == 5:  # (T, Z, C, Y, X)
        return data
    raise ValueError(f"Unsupported array ndim: {ndim}")


def load_plane(path, dims=None):
    """Load *path* and return a ``(C, H, W)`` plane (T=0, middle Z).

    Raises whatever the underlying decoder raises on a corrupt/unreadable
    file -- callers (``ThumbnailDecodeWorker``) already treat decode
    failure as skip-and-continue.
    """
    suffix = Path(path).suffix.lower()
    if suffix in TIFF_SUFFIXES:
        import tifffile

        arr = tifffile.imread(path)
    else:
        arr = np.asarray(Image.open(path))

    data = _to_5d(arr, dims=dims)
    _T, Z, _C, _H, _W = data.shape
    z_mid = Z // 2
    return np.array(data[0, z_mid, :, :, :], copy=True)
