"""Recursively collect supported image files under a folder, or every cell
in a packed tile container (see tile_container.py).

`scan_folder` predates the container format and is kept only for
NN_workflow/08_classify_tile_set.py (a standalone script superseded by
core/classify.py's `classify_pool`) and general ad hoc use -- every FOV
tile crop `tileclass`/`yeastprep` themselves produce or consume now comes
from `scan_container`, since `export_tiles` no longer writes loose files."""

import os
import re

from .load_thumbnail import SUPPORTED_SUFFIXES
from .tile_container import TileContainer

_DIGITS = re.compile(r"(\d+)")


def _natsort_key(path):
    """Split on digit runs so "tile2" sorts before "tile10"."""
    parts = _DIGITS.split(path)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def scan_folder(folder):
    """Return every supported image path under *folder*, recursively,
    sorted naturally. Skips hidden files/directories."""
    paths = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in names:
            if name.startswith("."):
                continue
            if name.lower().endswith(SUPPORTED_SUFFIXES):
                paths.append(os.path.join(root, name))
    paths.sort(key=_natsort_key)
    return paths


def scan_container(path):
    """Return a virtual `"<container>.tiles/<cell_id>.tif"` reference for
    every cell in the `.tiles` container at `path`, sorted naturally --
    the container-file counterpart to `scan_folder`, so `filter_by_fov`
    and every downstream identity-string consumer (annotations, thumbnail
    grid, ...) work the same regardless of which one produced the list."""
    container = TileContainer(path)
    paths = [f"{path}/{cell_id}.tif" for cell_id in container.cell_ids()]
    paths.sort(key=_natsort_key)
    return paths


def filter_by_fov(paths, fov_names):
    """Keep only paths whose filename is `{fov}_cell...` for one of
    fov_names -- matches the `{fov_id}_cell{label:05d}` naming yeastprep's
    tile export writes (core/tiles.py's `export_tiles`)."""
    prefixes = tuple(f"{name}_cell" for name in fov_names)
    return [p for p in paths if os.path.basename(p).startswith(prefixes)]
