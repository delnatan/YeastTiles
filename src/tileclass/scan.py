"""List the cells of a packed tile container (see tile_container.py) as
the virtual path strings every other part of tileclass uses as a tile's
identity."""

import os
import re

from .tile_container import TileContainer

_DIGITS = re.compile(r"(\d+)")


def _natsort_key(path):
    """Split on digit runs so "tile2" sorts before "tile10"."""
    parts = _DIGITS.split(path)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def scan_container(path):
    """Return a virtual `"<container>.tiles/<cell_id>"` reference for
    every cell in the `.tiles` container at `path`, sorted naturally."""
    container = TileContainer(path)
    paths = [f"{path}/{cell_id}" for cell_id in container.cell_ids()]
    paths.sort(key=_natsort_key)
    return paths


def filter_by_fov(paths, fov_names):
    """Keep only paths whose filename is `{fov}_cell...` for one of
    fov_names -- matches the `{fov_id}_cell{label:05d}` naming yeastprep's
    tile export writes (core/tiles.py's `export_tiles`)."""
    prefixes = tuple(f"{name}_cell" for name in fov_names)
    return [p for p in paths if os.path.basename(p).startswith(prefixes)]
