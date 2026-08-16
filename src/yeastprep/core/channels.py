"""Brightfield / target-channel selection.

Replaces the hardcoded ch0=brightfield, ch1=DAPI assumption baked into
notebooks/01_flatten_field.ipynb with an explicit, best-effort-inferred,
user-overridable selection.
"""

from dataclasses import dataclass

_BRIGHTFIELD_NAME_HINTS = (
    "bf",
    "brightfield",
    "bright field",
    "dic",
    "transmitted",
    "trans",
    "white light",
)
_TARGET_NAME_HINTS = ("dapi", "hoechst")
_TARGET_WAVELENGTH_RANGE_NM = (400.0, 500.0)  # DAPI/Hoechst emission ~460nm


@dataclass(frozen=True)
class ChannelSelection:
    brightfield: int
    projection: int  # DAPI, or whatever design.md 2b's "target channel" is


def _channel_name(meta: dict) -> str:
    return str(meta.get("name") or "").strip().lower()


def _channel_emission_nm(meta: dict):
    value = meta.get("emission_wavelength")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_brightfield_name(name: str) -> bool:
    """Whether a channel's name looks like transmitted light rather than a
    fluorescence channel -- exposed for `ui/rawstack/colormaps.py`'s
    default-colormap-per-channel assignment, which needs the same
    brightfield/fluorescence distinction this module already makes for
    picking `ChannelSelection.brightfield`."""
    return any(hint in name.strip().lower() for hint in _BRIGHTFIELD_NAME_HINTS)


def _find_by_name(channels_meta: list[dict], hints: tuple[str, ...]):
    for idx, meta in enumerate(channels_meta):
        name = _channel_name(meta)
        if any(hint in name for hint in hints):
            return idx
    return None


def _find_by_wavelength(channels_meta: list[dict], lo: float, hi: float):
    for idx, meta in enumerate(channels_meta):
        nm = _channel_emission_nm(meta)
        if nm is not None and lo <= nm <= hi:
            return idx
    return None


def infer_channel_selection(channels_meta: list[dict] | None) -> ChannelSelection | None:
    """Best-effort guess of (brightfield, target) channel indices from
    ``meta['channels']`` (pyvistra's per-channel metadata list).

    Returns None when it can't confidently guess (missing/empty metadata,
    as for plain TIFFs, or fewer than 2 channels) -- callers should fall
    back to (0, 1) and let the user confirm/correct it.
    """
    if not channels_meta or len(channels_meta) < 2:
        return None

    brightfield = _find_by_name(channels_meta, _BRIGHTFIELD_NAME_HINTS)
    target = _find_by_name(channels_meta, _TARGET_NAME_HINTS)
    if target is None:
        target = _find_by_wavelength(channels_meta, *_TARGET_WAVELENGTH_RANGE_NM)

    if brightfield is None or target is None or brightfield == target:
        return None

    return ChannelSelection(brightfield=brightfield, projection=target)
