"""Tiny dependency-free filesystem-presence helper. Kept in its own module
rather than in `project.py` (which both `tiles.py` and this would
otherwise be the obvious home for) because `project.py` already imports
`TileParams` from `tiles.py` -- putting a `tiles.py`-needed helper in
`project.py` would create an import cycle.
"""

from pathlib import Path


def is_hidden(path: Path) -> bool:
    """Whether `path`'s name starts with `.` -- true both for ordinary
    dotfiles and for the macOS AppleDouble sidecars (`._<name>`) that
    external/non-HFS+ volumes (exFAT/FAT32 drives, network shares, ...)
    silently create next to every real file, same stem and extension.
    Neither should ever be read as project data."""
    return path.name.startswith(".")


def list_visible(path: Path | None, pattern: str = "*") -> list[Path]:
    """`path.glob(pattern)` with hidden entries (see `is_hidden`) filtered
    out. Needed because pathlib's glob, unlike a real shell, does *not*
    exclude a leading dot from a `*` match -- `*.tiff` happily matches
    `._foo.tiff` -- so every real directory scan should go through this
    rather than calling `.glob()` directly."""
    if path is None or not path.is_dir():
        return []
    return [p for p in path.glob(pattern) if not is_hidden(p)]


def dir_has_files(path: Path | None, pattern: str = "*") -> bool:
    """Whether `path` exists and has at least one *visible* entry matching
    `pattern` -- used to tell "an upstream/producer folder that was never
    populated" apart from "one that was emptied out after the fact" (most
    likely archived to external storage once its downstream products were
    made) from ones that still have something to compare against. `pattern`
    defaults to "any file at all", but callers whose folder can also hold
    unrelated files (config sidecars, `.DS_Store`, ...) should narrow it to
    whatever actually signals "this folder is in use for its real purpose"
    -- an unrelated file sitting alongside real data shouldn't count as
    "still populated"."""
    return bool(list_visible(path, pattern))
