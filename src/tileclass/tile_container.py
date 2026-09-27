"""Single-file-per-FOV tile container: replaces one-tif-per-cell storage
under `05_tiles/<fov_id>/` with one `05_tiles/<fov_id>.tiles` file holding
every cell crop, each compressed independently so a single cell can be
read back without touching any other.

File layout: `MAGIC`, an 8-byte little-endian index length, a JSON index
(`{"cells": [{cell_id, label, offset, nbytes, shape, dtype, codec}, ...]}`),
then the concatenated per-cell compressed payloads at the offsets the
index records (offsets are relative to the first payload byte, not the
file start).

Every existing consumer of a tile's "path" (the thumbnail grid,
`TileAnnotations`, `PooledAnnotations`, `classify_pool`) already treats it
as an opaque identity string built with plain `os.path` operations, never
opening it itself -- see their module docstrings. That lets a virtual path
`"<fov_id>.tiles/<cell_id>.tif"` (a string that *looks* like a file living
inside the container "directory") stand in for a real crop path everywhere
except the two functions that actually decode pixels
(`load_thumbnail.load_plane`, `training/dataset.load_masked_crop`) --
`is_container_ref`/`container_and_cell` are the parse those two use.
"""

import json
import re
import struct
import zlib
from pathlib import Path
from threading import Lock

import numpy as np

MAGIC = b"YTLC1\n"
_HEADER_LEN_FMT = "<Q"


def is_container_ref(path) -> bool:
    """True if `path` is a virtual `<container>.tiles/<cell_id>.<ext>`
    reference rather than a real file. Pure string check (parent's suffix),
    no disk I/O -- safe to call on every load without extra stat() cost."""
    return Path(path).parent.suffix == ".tiles"


def container_and_cell(path) -> tuple[Path, str]:
    """Split a virtual reference into (container_path, cell_id)."""
    p = Path(path)
    return p.parent, p.stem


# cell_id is `{fov_id}_cell{label:05d}` (core/tiles.py's export_tiles).
_CELL_ID = re.compile(r"^(?P<fov>.*)_cell(?P<index>\d+)$")


def tile_display_name(path, with_fov=True) -> str:
    """How the UI names a tile: "<fov_id> · cell 17", or just "cell 17".
    The `.tif` in a tile reference is only part of its identity string
    (and of the annotation sidecar's keys), never something to show.
    Anything not named like an exported cell falls back to its stem."""
    stem = Path(path).stem
    match = _CELL_ID.match(stem)
    if match is None:
        return stem
    cell = f"cell {int(match['index'])}"
    return f"{match['fov']} · {cell}" if with_fov else cell


def write_container(path, cells, codec="zlib", level=6) -> None:
    """Write `cells` (an iterable of `(cell_id, label, array)`) to `path`
    as one container file. Writes to a sibling `.tmp` file and
    `Path.replace()`s over `path` so a crash mid-write never leaves a
    truncated container where a good one used to be."""
    path = Path(path)
    entries = []
    payloads = []
    offset = 0
    for cell_id, label, array in cells:
        array = np.ascontiguousarray(array)
        raw = array.tobytes()
        payload = zlib.compress(raw, level) if codec == "zlib" else raw
        entries.append(
            {
                "cell_id": cell_id,
                "label": label,
                "offset": offset,
                "nbytes": len(payload),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "codec": codec,
            }
        )
        payloads.append(payload)
        offset += len(payload)

    index_bytes = json.dumps({"cells": entries}).encode("utf-8")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(_HEADER_LEN_FMT, len(index_bytes)))
        f.write(index_bytes)
        for payload in payloads:
            f.write(payload)
    tmp_path.replace(path)


class TileContainer:
    """One parsed container's index, plus on-demand per-cell reads. Opens
    the file fresh for each `.read()` (cheap: one seek, no directory walk)
    rather than holding a handle across calls, matching `load_plane`'s
    existing per-call-open behavior for real files -- the expensive part
    this whole format exists to avoid is per-file *open* overhead when
    there are thousands of *cells*, not one extra open per FOV."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            magic = f.read(len(MAGIC))
            if magic != MAGIC:
                raise ValueError(f"{self.path}: not a tile container (bad magic)")
            (index_len,) = struct.unpack(_HEADER_LEN_FMT, f.read(8))
            index = json.loads(f.read(index_len))
            self._data_start = f.tell()
        self.entries = {entry["cell_id"]: entry for entry in index["cells"]}

    def __len__(self):
        return len(self.entries)

    def __contains__(self, cell_id):
        return cell_id in self.entries

    def cell_ids(self):
        return list(self.entries)

    def read(self, cell_id) -> np.ndarray:
        entry = self.entries[cell_id]
        with open(self.path, "rb") as f:
            f.seek(self._data_start + entry["offset"])
            raw = f.read(entry["nbytes"])
        if entry["codec"] == "zlib":
            raw = zlib.decompress(raw)
        return np.frombuffer(raw, dtype=entry["dtype"]).reshape(entry["shape"])


# Process-lifetime cache of open containers, keyed by resolved path --
# reused across many `.read()` calls (a shuffled training epoch, a paged
# viewer session) so the JSON index is parsed once per container, not once
# per cell. Unbounded: a project has at most a few hundred FOVs, each
# index is tiny (cell_id/offset/shape per cell, no pixel data), nowhere
# near enough to matter memory-wise.
_container_cache: dict[str, TileContainer] = {}
_container_cache_lock = Lock()


def get_container(path) -> TileContainer:
    key = str(Path(path))
    with _container_cache_lock:
        container = _container_cache.get(key)
        if container is None:
            container = TileContainer(path)
            _container_cache[key] = container
        return container


def group_training_provenance(paths) -> dict[str, list[int]]:
    """Compact form of a list of container-ref crop paths, for recording
    training provenance in meta.json (see `training/supervised.py`'s
    `_save_weights` and `training/vicreg.py`'s `_save_backbone`): a
    container-ref path repeats its container's own path once per cell
    contributed, which is most of what made the old flat path list balloon
    to one line per cell -- grouping by container and keeping only each
    cell's already-recorded integer `label` (the container index's own
    per-cell field) says the same thing in a fraction of the text.

    Returns `{container_path: [label, ...]}`. Every `path` must be a
    container ref -- every crop tileclass/yeastprep produce or consume now
    comes from one (see scan.py's module docstring).
    """
    containers: dict[str, list[int]] = {}
    for path in paths:
        container_path, cell_id = container_and_cell(str(path))
        label = get_container(container_path).entries[cell_id]["label"]
        containers.setdefault(str(container_path), []).append(label)
    return {k: sorted(v) for k, v in sorted(containers.items())}


def training_provenance_contains(provenance: dict[str, list[int]], path) -> bool:
    """Membership test against `group_training_provenance`'s output --
    the read-side counterpart used by `training/vicreg.py`'s
    `warm_start_overlap` to check a newly pooled crop against a
    previously recorded one without re-expanding the grouping back into a
    flat set of path strings."""
    container_path, cell_id = container_and_cell(str(path))
    entry = get_container(container_path).entries.get(cell_id)
    if entry is None:
        return False
    return entry["label"] in provenance.get(str(container_path), ())
