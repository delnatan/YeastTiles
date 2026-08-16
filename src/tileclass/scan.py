"""Recursively collect supported image files under a folder."""

import os
import re

from .load_thumbnail import SUPPORTED_SUFFIXES

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


def filter_by_fov(paths, fov_names):
    """Keep only paths whose filename is `{fov}_cell...` for one of
    fov_names -- matches the `{fov_id}_cell{label:05d}` naming yeastprep's
    tile export writes (core/tiles.py's `export_tiles`)."""
    prefixes = tuple(f"{name}_cell" for name in fov_names)
    return [p for p in paths if os.path.basename(p).startswith(prefixes)]
