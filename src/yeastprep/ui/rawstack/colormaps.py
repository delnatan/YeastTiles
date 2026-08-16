"""Trimmed colormap registry for the raw-stack canvas.

Vendored (trimmed) from pyvistra/colormaps.py: just the two-color
additive-blending set used for microscope channels, plus a handful of
vispy's built-in named colormaps and the wavelength lookup used for
auto-assigning a channel's default color -- see `default_colormap_for_channel`,
called from `panel.py`/`canvas.py` when a new file loads. No bundled-LUT /
custom registry machinery -- this viewer doesn't need it.
"""

from vispy.color import Colormap
from vispy.color import get_colormap as _vispy_get_colormap

from yeastprep.core.channels import is_brightfield_name

_TWO_COLOR = {
    "Orange": ["black", "#ffb100"],
    "Green": ["black", "#49FF49"],
    "Cyan": ["black", "#5BD6FF"],
    "Magenta": ["black", "magenta"],
    "Yellow": ["black", "yellow"],
    "White": ["black", "white"],
}

_VISPY_NAMED = ["viridis", "gray", "hot", "turbo"]

_BRIGHTFIELD_COLORMAP = "gray"

# Anchored at four common laser lines -- 405/488/560/650nm -> cyan/green/
# orange/magenta -- with the breakpoint between each pair of anchors at
# their midpoint, so a wavelength close to any one of the four gets that
# color rather than landing in a neighboring band.
_WAVELENGTH_COLORMAPS = [
    (446.5, "Cyan"),  # < 446.5nm, e.g. 405nm (DAPI/Hoechst excitation)
    (524.0, "Green"),  # 446.5-524nm, e.g. 488nm (GFP/Alexa488 excitation)
    (605.0, "Orange"),  # 524-605nm, e.g. 560nm (mCherry/Alexa568 excitation)
]
_WAVELENGTH_FALLBACK = "Magenta"  # >= 605nm, e.g. 650nm (Alexa647 excitation)


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


def default_colormap_for_channel(meta: dict) -> str:
    """A channel's default display color: gray for anything that looks
    like transmitted light (by name, or by having no excitation/emission
    wavelength at all -- brightfield/DIC/phase channels usually record
    neither), else whichever of cyan/green/orange/magenta is closest to
    its excitation wavelength (preferred, since it's what vendors usually
    label channels by -- "the 488 channel") or, failing that, its emission
    wavelength."""
    name = str(meta.get("name") or "")
    excitation = meta.get("excitation_wavelength")
    emission = meta.get("emission_wavelength")
    if is_brightfield_name(name) or (excitation is None and emission is None):
        return _BRIGHTFIELD_COLORMAP
    return colormap_for_wavelength(excitation if excitation is not None else emission) or "White"


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
