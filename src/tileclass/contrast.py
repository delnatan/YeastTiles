"""Pure helpers for computing display contrast limits.

No Qt, no vispy. Used by renderers (one-shot auto-contrast on data load)
and by the channel panel (interactive percentile tighten/loosen).
"""

import numpy as np


def compute_percentile_clim(plane, pct_low=0.5, pct_high=99.98, ignore_zeros=True):
    """Return ``(vmin, vmax)`` from percentile bounds of *plane*.

    Signed data (with at least one finite value on each side of zero) is
    auto-detected and returned as a *symmetric* range ``(-r, r)`` where
    ``r`` is the percentile bracket of ``|plane|``. This keeps 0 anchored
    to the midpoint of the colormap — the correct convention for any
    diverging map (RdBu, coolwarm, …) and for any quantity with a
    natural zero (residuals, phase, differential images).

    Args:
        plane: 2D array (a single channel slice).
        pct_low: Lower percentile in [0, 100].
        pct_high: Upper percentile in [0, 100].
        ignore_zeros: If True, drop zero-valued pixels before computing
            percentiles on non-negative data. Zeros are typically
            background/padding and would squash the dynamic range
            otherwise. Falls back to the full plane if every pixel is
            zero. For signed data the parameter is ignored (zeros are
            kept; the symmetric bracket is data-driven).

    Returns:
        ``(float, float)``. The range is scale-invariant: when the
        percentile bracket collapses (e.g. very sparse data), falls back
        to the actual min/max of the valid pixels; if that is also
        degenerate, pads by a dtype-aware epsilon so the renderer never
        divides by zero. Never injects an absolute floor like ``+1``,
        which would destroy contrast on small-magnitude float data
        (e.g. unnormalized PSFs).
    """
    arr = np.asarray(plane)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0

    # Signed data: pick a symmetric bracket around 0 so diverging
    # colormaps align their neutral midpoint with zero.
    if finite.min() < 0 and finite.max() > 0:
        mag = np.abs(finite)
        # pct_high alone defines the symmetric envelope; the low cutoff
        # is implicit (anything below it is mapped to grey/centre).
        r = float(np.nanpercentile(mag, pct_high))
        if not np.isfinite(r) or r <= 0:
            r = float(mag.max())
        if r <= 0:
            r = _degenerate_pad(0.0, arr.dtype) / 2.0
        return -r, r

    valid = arr[arr > 0] if ignore_zeros else arr
    valid = valid[np.isfinite(valid)] if valid.size else valid
    if valid.size == 0:
        valid = finite

    vmin, vmax = np.nanpercentile(valid, (pct_low, pct_high))
    vmin, vmax = float(vmin), float(vmax)
    if vmax > vmin:
        return vmin, vmax

    # Percentile bracket collapsed on `valid` (e.g. a binary/label mask
    # whose only nonzero value is a constant like 1 -- ignore_zeros then
    # discards the very zeros needed to tell background from
    # foreground). Retry against the full finite array before giving up
    # on real contrast: if it has range, that range still separates
    # background from foreground even though `valid` alone didn't.
    dmin = float(np.nanmin(finite))
    dmax = float(np.nanmax(finite))
    if dmax > dmin:
        return dmin, dmax

    # Truly constant data — pad by a scale-aware epsilon.
    return dmin, dmin + _degenerate_pad(dmin, arr.dtype)


def _degenerate_pad(value, dtype):
    """Smallest meaningful range for a constant array of given dtype/value.

    Integers get ``1`` (the smallest representable step). Floats get an
    epsilon scaled to the value's own magnitude, with a unit floor so a
    constant-zero plane still yields a finite ``(0, eps)`` range.
    """
    if np.issubdtype(dtype, np.integer):
        return 1
    eps = np.finfo(dtype).eps if np.issubdtype(dtype, np.floating) else 1e-7
    return max(abs(value), 1.0) * float(eps) * 1024.0
