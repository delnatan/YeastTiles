"""Cross-session app state: recent folders, default parameters, window
geometry. Backed by QSettings -- no prior convention in this repo to
follow (tileclass persists folder-scoped state only, via sidecar files;
see data/annotations.py), but QSettings is the standard Qt mechanism for
this kind of app-global, not-folder-scoped state.
"""

from qtpy.QtCore import QByteArray, QSettings

from yeastprep.core.channels import ChannelSelection
from yeastprep.core.pipeline import FlattenFieldParams
from yeastprep.core.segmentation import SegmentationParams
from yeastprep.core.tiles import TileParams

_ORG = "BurgessLab"
_APP = "YeastTiles-yeastprep"
_MAX_RECENT = 10


def _settings() -> QSettings:
    return QSettings(_ORG, _APP)


def _recent(key: str) -> list[str]:
    value = _settings().value(key, [])
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _add_recent(key: str, path: str):
    entries = [path] + [p for p in _recent(key) if p != path]
    _settings().setValue(key, entries[:_MAX_RECENT])


def get_recent_input_dirs() -> list[str]:
    return _recent("recent_input_dirs")


def add_recent_input_dir(path: str):
    _add_recent("recent_input_dirs", path)


def get_recent_output_dirs() -> list[str]:
    return _recent("recent_output_dirs")


def add_recent_output_dir(path: str):
    _add_recent("recent_output_dirs", path)


def get_recent_segmentation_dirs() -> list[str]:
    return _recent("recent_segmentation_dirs")


def add_recent_segmentation_dir(path: str):
    _add_recent("recent_segmentation_dirs", path)


def get_recent_tile_input_dirs() -> list[str]:
    return _recent("recent_tile_input_dirs")


def add_recent_tile_input_dir(path: str):
    _add_recent("recent_tile_input_dirs", path)


def get_recent_tile_output_dirs() -> list[str]:
    return _recent("recent_tile_output_dirs")


def add_recent_tile_output_dir(path: str):
    _add_recent("recent_tile_output_dirs", path)


def get_default_params() -> FlattenFieldParams:
    s = _settings()
    s.beginGroup("default_params")
    params = FlattenFieldParams(
        num_tiles_y=int(s.value("num_tiles_y", FlattenFieldParams().num_tiles_y)),
        num_tiles_x=int(s.value("num_tiles_x", FlattenFieldParams().num_tiles_x)),
        inverted_variance_prominence=float(
            s.value(
                "inverted_variance_prominence",
                FlattenFieldParams().inverted_variance_prominence,
            )
        ),
        poly_degree=(
            int(s.value("poly_degree_x", 2)),
            int(s.value("poly_degree_y", 2)),
        ),
        offset_um=float(s.value("offset_um", FlattenFieldParams().offset_um)),
    )
    s.endGroup()
    return params


def set_default_params(params: FlattenFieldParams):
    s = _settings()
    s.beginGroup("default_params")
    s.setValue("num_tiles_y", params.num_tiles_y)
    s.setValue("num_tiles_x", params.num_tiles_x)
    s.setValue("inverted_variance_prominence", params.inverted_variance_prominence)
    s.setValue("poly_degree_x", params.poly_degree[0])
    s.setValue("poly_degree_y", params.poly_degree[1])
    s.setValue("offset_um", params.offset_um)
    s.endGroup()


def get_default_channels() -> ChannelSelection | None:
    s = _settings()
    bf = s.value("default_channels/brightfield")
    proj = s.value("default_channels/projection")
    if bf is None or proj is None:
        return None
    return ChannelSelection(brightfield=int(bf), projection=int(proj))


def set_default_channels(channels: ChannelSelection):
    s = _settings()
    s.setValue("default_channels/brightfield", channels.brightfield)
    s.setValue("default_channels/projection", channels.projection)


def get_default_segmentation_params() -> SegmentationParams:
    s = _settings()
    s.beginGroup("default_segmentation_params")
    defaults = SegmentationParams()
    model_path = s.value("model_path", defaults.model_path)
    params = SegmentationParams(
        model_path=model_path or None,
        diameter=float(s.value("diameter", defaults.diameter)),
        flow_threshold=float(s.value("flow_threshold", defaults.flow_threshold)),
        cellprob_threshold=float(
            s.value("cellprob_threshold", defaults.cellprob_threshold)
        ),
        remove_edge_masks=_as_bool(
            s.value("remove_edge_masks", defaults.remove_edge_masks)
        ),
    )
    s.endGroup()
    return params


def set_default_segmentation_params(params: SegmentationParams):
    s = _settings()
    s.beginGroup("default_segmentation_params")
    s.setValue("model_path", params.model_path or "")
    s.setValue("diameter", params.diameter)
    s.setValue("flow_threshold", params.flow_threshold)
    s.setValue("cellprob_threshold", params.cellprob_threshold)
    s.setValue("remove_edge_masks", params.remove_edge_masks)
    s.endGroup()


def get_default_tile_params() -> TileParams:
    s = _settings()
    s.beginGroup("default_tile_params")
    defaults = TileParams()
    params = TileParams(
        size=int(s.value("size", defaults.size)),
        contrast_lo_pct=float(s.value("contrast_lo_pct", defaults.contrast_lo_pct)),
        contrast_hi_pct=float(s.value("contrast_hi_pct", defaults.contrast_hi_pct)),
    )
    s.endGroup()
    return params


def set_default_tile_params(params: TileParams):
    s = _settings()
    s.beginGroup("default_tile_params")
    s.setValue("size", params.size)
    s.setValue("contrast_lo_pct", params.contrast_lo_pct)
    s.setValue("contrast_hi_pct", params.contrast_hi_pct)
    s.endGroup()


def _as_bool(value) -> bool:
    # QSettings round-trips bools as the string "true"/"false" on some
    # platforms (INI-backed backends) instead of native bool.
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def save_window_geometry(window):
    _settings().setValue("window_geometry", window.saveGeometry())


def restore_window_geometry(window):
    geometry = _settings().value("window_geometry")
    if isinstance(geometry, QByteArray):
        window.restoreGeometry(geometry)
