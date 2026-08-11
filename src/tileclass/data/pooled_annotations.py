"""Multi-folder annotation pooling: browse/annotate several source folders
in one tileclass session while keeping each folder's own annotation
sidecar file completely independent (see annotations.py's module
docstring for that file format) -- no merged store, no format change.

`PooledAnnotations` owns one `TileAnnotations` per pooled folder and
dispatches every per-tile call to the right one by matching the tile's
absolute path against each folder's root. It duck-types just enough of
`TileAnnotations`'s surface for `main_window.py`, `ManageCategoriesDialog`,
and `AnnotationStatsPanel` to use it as a drop-in replacement -- with one
deliberate difference: per-tile methods (`get`, `confidence`, `update`,
`update_with_confidence`) are keyed by *absolute path* rather than a
folder-relative one, since a bare relpath isn't unique across a pool
(`relpath()` here is the identity function for exactly this reason).

Folder-vocabulary operations (`add_category`, `remove_category`,
`rename_category`) propagate to *every* pooled folder, so the picker's
choices stay in sync and a folder later reopened on its own still has the
full vocabulary. This assumes pooled folders are meant to share one
category vocabulary (design.md's stated use case: pooling experiments for
one consistent training run) -- if they don't, `categories()`'s union
still works, propagation on add/rename/remove is just slightly more
generous than "this one folder only".
"""

import os

from .annotations import TileAnnotations


class PooledAnnotations:
    def __init__(self, folders):
        self.folders = [os.path.normpath(os.path.abspath(f)) for f in folders]
        self._stores = {folder: TileAnnotations(folder) for folder in self.folders}

    # ------------------------------------------------------------------
    # Per-tile: keyed by absolute path (see module docstring)

    def relpath(self, abs_path):
        """Identity: the "key" this store uses for per-tile lookups is
        just the normalized absolute path -- see module docstring."""
        return os.path.normpath(os.path.abspath(abs_path))

    def _owner(self, key):
        best = None
        for folder in self.folders:
            if key == folder or key.startswith(folder + os.sep):
                if best is None or len(folder) > len(best):
                    best = folder
        if best is None:
            raise KeyError(f"{key!r} is not under any pooled folder: {self.folders}")
        return self._stores[best]

    def get(self, key, default=None):
        """`default` covers a known tile with no tag yet (ordinary
        dict.get semantics) -- a path outside every pooled folder is a
        caller bug (main_window.py only ever passes paths straight from
        the folders' own scan), so that still raises KeyError via
        `_owner` rather than silently returning `default` too."""
        owner = self._owner(key)
        return owner.get(owner.relpath(key), default)

    def confidence(self, key):
        owner = self._owner(key)
        return owner.confidence(owner.relpath(key))

    def update(self, items):
        """Batch-tag many absolute paths at once -- one save per pooled
        folder actually touched, not one save per tile (matches
        TileAnnotations.update's batching, just fanned out per owner)."""
        by_owner = {}
        for key, category in items:
            owner = self._owner(key)
            by_owner.setdefault(id(owner), (owner, []))[1].append(
                (owner.relpath(key), category)
            )
        for owner, pairs in by_owner.values():
            owner.update(pairs)

    def update_with_confidence(self, items):
        by_owner = {}
        for key, category, confidence in items:
            owner = self._owner(key)
            by_owner.setdefault(id(owner), (owner, []))[1].append(
                (owner.relpath(key), category, confidence)
            )
        for owner, triples in by_owner.values():
            owner.update_with_confidence(triples)

    def tagged_items(self):
        """Every (abs_path, category, confidence) currently tagged across
        the whole pool -- confidence is None for a human-set/confirmed
        tag, a [0, 1] float for an unreviewed AI prediction (see
        annotations.py). The one method with no `TileAnnotations`
        counterpart -- added for the training workflow, which needs
        resolvable absolute paths, not relpaths scoped to a folder it
        doesn't know about."""
        items = []
        for folder, store in self._stores.items():
            for relpath, category in store.items():
                items.append((os.path.join(folder, relpath), category, store.confidence(relpath)))
        return items

    def values(self):
        for store in self._stores.values():
            yield from store.values()

    # ------------------------------------------------------------------
    # Folder vocabulary -- propagated to every pooled folder

    def categories(self):
        seen = []
        for store in self._stores.values():
            for name in store.categories():
                if name not in seen:
                    seen.append(name)
        return seen

    def add_category(self, name):
        for store in self._stores.values():
            store.add_category(name)

    def remove_category(self, name):
        for store in self._stores.values():
            store.remove_category(name)

    def rename_category(self, old, new):
        for store in self._stores.values():
            store.rename_category(old, new)

    def usage_count(self, name):
        return sum(store.usage_count(name) for store in self._stores.values())

    # ------------------------------------------------------------------
    # Display settings -- also propagated (pooled tiles are assumed to
    # share one axes-order / channel-color convention; see module docstring)

    @property
    def dims(self):
        for store in self._stores.values():
            if store.dims:
                return store.dims
        return None

    def set_dims(self, dims):
        for store in self._stores.values():
            store.set_dims(dims)

    @property
    def channel_colors(self):
        merged = {}
        for folder in self.folders:
            merged.update(self._stores[folder].channel_colors)
        return merged

    def set_channel_colors(self, channel_colors):
        for store in self._stores.values():
            store.set_channel_colors(channel_colors)

    # ------------------------------------------------------------------

    @property
    def root_dir(self):
        """Single-folder case: that folder (matches TileAnnotations
        exactly). Pooled case: every pooled folder, newline-joined -- used
        only for the window's tooltip, so verbose is fine."""
        if len(self.folders) == 1:
            return self.folders[0]
        return "\n".join(self.folders)

    def label(self):
        """Short display string for the window title."""
        if len(self.folders) == 1:
            return os.path.basename(self.folders[0])
        return f"{len(self.folders)} pooled folders"
