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
