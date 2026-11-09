"""What the classification pool knows about its FOVs, loaded on demand.

Checking or unchecking a FOV only changes which FOVs a page *looks at*; it
must not read or drop any tiles. So per-FOV facts (tile count, tag tallies,
vocabulary) are loaded the first time something asks for them, kept in a
`PoolDataCache`, and revalidated by file stamps -- unchecking a FOV leaves its
entry alone, re-checking it is free, and a sidecar rewritten by a tile viewer
is noticed the next time it's asked about.

No Qt in here: the summaries are built on worker threads (see
`common/lazy_refresh.py`).
"""

import os
from dataclasses import dataclass, field
from threading import Lock

from tileclass.data.annotations import TileAnnotations, normalized_category_key
from tileclass.tile_container import get_container


@dataclass(frozen=True)
class FovTally:
    """One FOV's tiles and tags, as of `stamp`."""

    stamp: tuple
    n_tiles: int
    counts: dict  # category -> tiles tagged with it (human or AI)
    human_counts: dict  # ... of which a human set or confirmed
    untagged: int
    vocabulary: list


@dataclass
class PoolSummary:
    """The tallies of several FOVs, added up."""

    fov_dirs: list
    n_tiles: int = 0
    counts: dict = field(default_factory=dict)
    human_counts: dict = field(default_factory=dict)
    untagged: int = 0
    vocabulary: list = field(default_factory=list)


def _stamp(paths) -> tuple:
    stamp = []
    for path in paths:
        try:
            st = os.stat(path)
            stamp.append((st.st_mtime_ns, st.st_size))
        except OSError:
            stamp.append(None)  # no sidecar yet is a state like any other
    return tuple(stamp)


class PoolDataCache:
    def __init__(self):
        self._entries: dict[str, FovTally] = {}
        # Held across a load, so two pages asking for the same FOV at once
        # read it once.
        self._lock = Lock()

    def tally(self, fov_dir) -> FovTally:
        fov_dir = str(fov_dir)
        # Stamped before reading, so a write landing mid-read leaves a stamp
        # that no longer matches and is simply reloaded next time.
        stamp = _stamp([fov_dir, *TileAnnotations.source_paths(fov_dir)])
        with self._lock:
            entry = self._entries.get(fov_dir)
            if entry is None or entry.stamp != stamp:
                entry = self._load(fov_dir, stamp)
                self._entries[fov_dir] = entry
            return entry

    def forget_project(self, root):
        prefix = os.path.join(str(root), "")
        with self._lock:
            for key in [k for k in self._entries if k.startswith(prefix)]:
                del self._entries[key]

    @staticmethod
    def _load(fov_dir, stamp) -> FovTally:
        store = TileAnnotations(fov_dir)
        counts, human_counts, untagged = {}, {}, 0
        cell_ids = get_container(fov_dir).cell_ids()
        for cell_id in cell_ids:
            key = store.relpath(f"{fov_dir}/{cell_id}")
            category = store.get(key)
            if not category:
                untagged += 1
                continue
            counts[category] = counts.get(category, 0) + 1
            if store.confidence(key) is None:
                human_counts[category] = human_counts.get(category, 0) + 1
        return FovTally(
            stamp, len(cell_ids), counts, human_counts, untagged, store.categories()
        )

    def summarize(self, fov_dirs) -> PoolSummary:
        """`fov_dirs`' tallies added up; the vocabulary is the union of
        theirs, as `PooledAnnotations.categories()` would give."""
        summary = PoolSummary(list(fov_dirs))
        seen = set()
        for fov_dir in fov_dirs:
            t = self.tally(fov_dir)
            summary.n_tiles += t.n_tiles
            summary.untagged += t.untagged
            for name, n in t.counts.items():
                summary.counts[name] = summary.counts.get(name, 0) + n
            for name, n in t.human_counts.items():
                summary.human_counts[name] = summary.human_counts.get(name, 0) + n
            for name in t.vocabulary:
                key = normalized_category_key(name)
                if key not in seen:
                    seen.add(key)
                    summary.vocabulary.append(name)
        return summary
