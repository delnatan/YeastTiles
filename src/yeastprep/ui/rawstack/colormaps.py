"""Trimmed colormap registry for the raw-stack canvas.

Vendored (trimmed) from pyvistra/colormaps.py: just the two-color
additive-blending set used for microscope channels, plus a handful of
vispy's built-in named colormaps and the emission-wavelength lookup used
for auto-assigning a channel's default color. No bundled-LUT / custom
registry machinery -- this viewer doesn't need it.
"""

from vispy.color import Colormap
from vispy.color import get_colormap as _vispy_get_colormap

_TWO_COLOR = {
    "Orange": ["black", "#ffb100"],
    "Green": ["black", "#49FF49"],
    "Cyan": ["black", "#5BD6FF"],
    "Magenta": ["black", "magenta"],
    "Yellow": ["black", "yellow"],
    "White": ["black", "white"],
}

_VISPY_NAMED = ["viridis", "gray", "hot", "turbo"]

_WAVELENGTH_COLORMAPS = [
    (500, "Cyan"),  # blue emission, e.g. DAPI ~460nm
    (560, "Green"),  # green emission, e.g. GFP ~509nm
    (600, "Orange"),  # yellow/orange emission
]
_WAVELENGTH_FALLBACK = "Magenta"  # red/far-red emission


def names() -> list:
    return list(_TWO_COLOR) + list(_VISPY_NAMED)


def colormap_for_wavelength(wavelength_nm) -> "str | None":
    if wavelength_nm is None:
        return None
    try:
        wl = float(wavelength_nm)
    except (TypeError, ValueError):
        return None
    if not wl > 0:
        return None
    for breakpoint, name in _WAVELENGTH_COLORMAPS:
        if wl < breakpoint:
            return name
    return _WAVELENGTH_FALLBACK


def get(name: str) -> tuple:
    """Returns (vispy.color.Colormap, display_color_hex_or_None)."""
    if name in _TWO_COLOR:
        stops = _TWO_COLOR[name]
        return Colormap(stops), stops[1]
    try:
        return _vispy_get_colormap(name), None
    except Exception:
        stops = _TWO_COLOR["White"]
        return Colormap(stops), stops[1]
