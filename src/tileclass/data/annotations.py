"""Folder-level tile annotations: relative path -> category name.

Persisted as a tab-delimited text file named after the image folder and
placed as a *sibling* of that folder (not inside it), so annotating a
folder never mixes extra files in with the images themselves. Keys are
paths relative to the common image directory rather than bare filenames,
so a folder with nested subfolders still round-trips without collisions.

Header lines (prefixed with "#", written before the relpath/category
data) carry folder-level settings so they don't need to be re-entered
every time the same folder is reopened:

    #categories\tCategoryA\tCategoryB\tCategoryC
    #dims\ttzcyx
    #channel_colors\t0=#ff8800:additive:1.0\t1=#00ff00:overlay:0.8

``#channel_colors`` is the fast thumbnail grid's per-channel color/blend
-mode/opacity overlay settings (see ``data/overlay_state.py``); only
channels overridden from the default need an entry, so it's sparse and
absent entirely for folders that never touched the Colors... panel.

The category vocabulary is optional — if absent, `categories()` falls
back to whatever category names are actually in use (the original
behavior), so old annotation files without a header still round-trip.

A ``#categories`` line can be pre-written by hand for a folder that has
never been opened/annotated yet, to seed it with the same preset list
used elsewhere (e.g. copy another folder's category vocabulary into a
new sibling file before opening that folder in the tiled viewer).
`save()` always writes the canonical tab-delimited form above, but a
hand-typed header doesn't need literal tab characters — a comma- or
space-separated list works too, e.g. ``#categories: CategoryA,
CategoryB, CategoryC``. See `_parse_categories_header`.

`TileAnnotations` is a `MutableMapping` of relpath -> category, so
editing/replacing entries uses plain dict idiom: `annotations[relpath]`,
`annotations[relpath] = category`, `del annotations[relpath]`,
`annotations.update({...})` for a batch tag (one save instead of one
save per entry), plus the free `Mapping` extras (`.get()`, `in`, `len()`,
`.items()`, ...).

A tag line optionally carries a 3rd tab-delimited column: the confidence
of an AI-predicted category that hasn't been reviewed by a human yet --

    relpath/to/tile.tif\tsingle\t0.923

Absent (2-column line) means either a human set the tag directly, or a
human has since confirmed/overridden an AI prediction -- every path
through `__setitem__`/`update()`/`__delitem__` (the "someone is directly
tagging this" API) drops any existing confidence for the relpaths it
touches. Only `update_with_confidence()` (used by auto-annotation) sets
one. This is how the two are told apart on reload, without a separate
sidecar file or a fourth "reviewed" state to track.

`clear_unconfirmed()` is the inverse of `update_with_confidence()`: it
drops every tag that still carries a confidence, leaving human-set tags
alone -- for wiping stale predictions from an older model before
re-running a newer one over the same folder.
"""

import os
from collections.abc import MutableMapping


def normalized_category_key(name):
    """Case/whitespace-insensitive identity for a category name -- shared by
    `TileAnnotations.add_category`, `PooledAnnotations.categories`, and
    `ManageCategoriesDialog`'s duplicate-detection, so all three agree on
    what counts as "the same category" instead of drifting independently
    (the concrete failure this prevents: pooling one project's "two" with
    another's " two" or "Two" as three distinct vocabulary entries)."""
    return name.strip().casefold()


class TileAnnotations(MutableMapping):
    """Category tags for a folder of images, backed by a text file."""

    def __init__(self, root_dir):
        self.root_dir = os.path.normpath(root_dir)
        self.file_path = self._annotation_file_path(self.root_dir)
        self._categories = {}  # relpath -> category name
        self.confidences = {}  # relpath -> AI confidence, only if unreviewed
        self._category_vocab = []  # explicit predefined category list, if any
        self.dims = None  # persisted axes-order string (e.g. "tzcyx"), or None
        self.channel_colors = {}  # channel_idx -> (color_hex, blend_mode, opacity)
        self.load()

    @staticmethod
    def _annotation_file_path(root_dir):
        parent = os.path.dirname(root_dir)
        name = os.path.basename(root_dir)
        if not name:
            # root_dir was the filesystem root; nothing to nest alongside.
            return os.path.join(root_dir, "pyvistra_annotations.txt")
        return os.path.join(parent, f"{name}.txt")

    def load(self):
        """(Re)load categories and header settings from the annotation file,
        if it exists."""
        self._categories.clear()
        self.confidences.clear()
        self._category_vocab = []
        self.dims = None
        self.channel_colors = {}
        if not os.path.exists(self.file_path):
            return
        with open(self.file_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.rstrip("\n")
                if self._is_categories_header(line):
                    self._category_vocab = self._parse_categories_header(
                        line[len("#categories") :]
                    )
                    continue
                parts = line.split("\t")
                if parts[0] == "#dims":
                    if len(parts) == 2 and parts[1]:
                        self.dims = parts[1]
                    continue
                if parts[0] == "#channel_colors":
                    self.channel_colors = self._parse_channel_colors(parts[1:])
                    continue
                if len(parts) not in (2, 3) or not parts[0]:
                    continue
                relpath, category = parts[0], parts[1]
                self._categories[relpath] = category
                if len(parts) == 3:
                    try:
                        self.confidences[relpath] = float(parts[2])
                    except ValueError:
                        pass

    @staticmethod
    def _is_categories_header(line):
        """True if line is a ``#categories`` header, canonical or
        hand-typed (``#categories``, ``#categories\t...``,
        ``#categories: ...``, ``#categories ...``) — but not some
        unrelated ``#categoriesX`` line."""
        prefix = "#categories"
        if not line.startswith(prefix):
            return False
        return len(line) == len(prefix) or not line[len(prefix)].isalnum()

    @staticmethod
    def _parse_categories_header(rest):
        """Parse the remainder of a ``#categories`` header line (after
        the ``#categories`` prefix) into a list of category names.

        Accepts the canonical tab-delimited form written by `save()` as
        well as a comma- or space-separated form, so a preset list can
        be hand-typed into a sibling file for a folder that hasn't been
        opened/annotated yet without needing literal tab characters."""
        rest = rest.lstrip(":").strip()
        if not rest:
            return []
        if "\t" in rest:
            delimiter = "\t"
        elif "," in rest:
            delimiter = ","
        else:
            delimiter = " "
        return [name.strip() for name in rest.split(delimiter) if name.strip()]

    @staticmethod
    def _parse_channel_colors(entries):
        """Parse ``["0=#ff8800:additive:1.0", ...]`` into
        ``{0: ("#ff8800", "additive", 1.0)}``, skipping malformed entries
        so a hand-edited or truncated file still loads."""
        result = {}
        for entry in entries:
            if not entry:
                continue
            try:
                idx_str, rest = entry.split("=", 1)
                color_hex, blend_mode, opacity_str = rest.split(":")
                result[int(idx_str)] = (color_hex, blend_mode, float(opacity_str))
            except (ValueError, IndexError):
                continue
        return result

    @staticmethod
    def _format_channel_colors(channel_colors):
        return "\t".join(
            f"{idx}={color_hex}:{blend_mode}:{opacity}"
            for idx, (color_hex, blend_mode, opacity) in sorted(
                channel_colors.items()
            )
        )

    def save(self):
        """Write the current header settings and categories out, data
        sorted by relative path."""
        with open(self.file_path, "w", encoding="utf-8") as f:
            if self._category_vocab:
                f.write("#categories\t" + "\t".join(self._category_vocab) + "\n")
            if self.dims:
                f.write(f"#dims\t{self.dims}\n")
            if self.channel_colors:
                f.write(
                    "#channel_colors\t"
                    + self._format_channel_colors(self.channel_colors)
                    + "\n"
                )
            for relpath, category in sorted(self._categories.items()):
                confidence = self.confidences.get(relpath)
                if confidence is None:
                    f.write(f"{relpath}\t{category}\n")
                else:
                    f.write(f"{relpath}\t{category}\t{confidence:.4f}\n")

    # --- MutableMapping protocol: relpath -> category ---
    #
    # Gives editing/replacing entries plain dict idiom, plus the usual
    # Mapping extras (.get(), `in`, len(), .items(), ...) for free.

    def __getitem__(self, relpath):
        return self._categories[relpath]

    def __setitem__(self, relpath, category):
        """Tag relpath with category (empty/None clears the tag) and save."""
        self._apply(relpath, category)
        self.save()

    def __delitem__(self, relpath):
        del self._categories[relpath]
        self.confidences.pop(relpath, None)
        self.save()

    def __iter__(self):
        return iter(self._categories)

    def __len__(self):
        return len(self._categories)

    def update(self, other=(), **kwargs):
        """Batch-tag many relpaths at once, saving only once.

        Batch-annotating N tiles one `self[relpath] = category` at a time
        would rewrite the whole annotation file N times; this applies the
        same mutations but saves a single time at the end — the standard
        dict.update() contract, just persisted.
        """
        items = other.items() if hasattr(other, "items") else other
        for relpath, category in items:
            self._apply(relpath, category)
        for relpath, category in kwargs.items():
            self._apply(relpath, category)
        self.save()

    def _apply(self, relpath, category):
        """Set/clear relpath's category directly -- always drops any
        confidence, since this is the "a human is tagging this" path
        (see `update_with_confidence` for the AI-prediction path)."""
        if category:
            self._categories[relpath] = category
        else:
            self._categories.pop(relpath, None)
        self.confidences.pop(relpath, None)

    def get(self, relpath, default=None):
        """Return the category for relpath, or default if untagged."""
        return self._categories.get(relpath, default)

    def confidence(self, relpath):
        """AI confidence for relpath's current category, or None if it
        was set/confirmed by a human (see module docstring)."""
        return self.confidences.get(relpath)

    def update_with_confidence(self, items):
        """Batch-tag relpaths from an AI classifier: `(relpath, category,
        confidence)` triples, saving once. Unlike `update()`/
        `__setitem__`, this *keeps* the confidence -- it marks the tag as
        an unconfirmed prediction until a human overwrites it through the
        plain category-only API."""
        for relpath, category, confidence in items:
            if category:
                self._categories[relpath] = category
                self.confidences[relpath] = confidence
            else:
                self._categories.pop(relpath, None)
                self.confidences.pop(relpath, None)
        self.save()

    def clear_unconfirmed(self):
        """Remove every tag that's still an unreviewed AI prediction
        (confidence is not None), leaving human-set/confirmed tags
        untouched. Saves once (no-op if nothing to remove). Returns the
        number of tags removed -- for a classifier to be able to
        re-predict a folder after a new training run without disturbing
        anything a human has already confirmed."""
        stale = list(self.confidences)
        for relpath in stale:
            self._categories.pop(relpath, None)
            self.confidences.pop(relpath, None)
        if stale:
            self.save()
        return len(stale)

    def categories(self):
        """Predefined category vocabulary, if one has been set; otherwise
        the sorted list of distinct category names currently in use."""
        if self._category_vocab:
            return list(self._category_vocab)
        return sorted(set(self._categories.values()))

    def add_category(self, name):
        """Add name to the predefined vocabulary and persist.

        On first use (vocabulary still empty), seeds the vocabulary with
        whatever category names are already in use so existing tags don't
        silently drop out of the picker. No-ops if a case/whitespace variant
        of `name` is already present -- see `normalized_category_key`. This
        only ever appends to (or skips adding to) the `#categories` header
        line; it never touches existing per-tile tags.
        """
        name = name.strip()
        if not name:
            return
        if not self._category_vocab:
            self._category_vocab = self.categories()
        key = normalized_category_key(name)
        if any(normalized_category_key(existing) == key for existing in self._category_vocab):
            return
        self._category_vocab.append(name)
        self.save()

    def usage_count(self, name):
        """Number of tiles currently tagged with category `name`."""
        return sum(1 for category in self._categories.values() if category == name)

    def remove_category(self, name):
        """Remove name from the predefined vocabulary and persist.

        Existing tags using this category are left untouched (still shown
        on their tiles) — only removed from the picker's choices.
        """
        if name in self._category_vocab:
            self._category_vocab.remove(name)
            self.save()

    def rename_category(self, old, new):
        """Rename a vocabulary entry and update any tiles tagged with it."""
        new = new.strip()
        if not new or old == new:
            return
        if old in self._category_vocab:
            self._category_vocab[self._category_vocab.index(old)] = new
        for relpath, category in self._categories.items():
            if category == old:
                self._categories[relpath] = new
        self.save()

    def set_dims(self, dims):
        """Persist the axes-order string used for this folder."""
        self.dims = dims
        self.save()

    def set_channel_colors(self, channel_colors):
        """Persist the fast thumbnail grid's per-channel color/blend-mode
        /opacity overlay settings: ``{channel_idx: (color_hex, blend_mode,
        opacity)}``."""
        self.channel_colors = dict(channel_colors)
        self.save()

    def relpath(self, abs_path):
        """Convert an absolute image path to this store's relpath key."""
        return os.path.relpath(os.path.abspath(abs_path), self.root_dir)
