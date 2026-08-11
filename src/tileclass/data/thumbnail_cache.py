"""In-memory decode cache + CPU-side compositing for the thumbnail grid.

Byte-budgeted LRU + background-thread request/callback pattern, keyed by
image path. Also owns the CPU-side channel compositing math since this
render path has no GL context at all. Pure numpy: no Qt -- colors are
flat RGB (picked via a plain color dialog, see
``widgets/thumbnail_colors_panel.py``), not gradient colormaps, so
there's no LUT/colormap-registry lookup needed here.

Delivery contract: ``callback(path)`` fires on the worker thread once
``path`` has been decoded into the cache -- Qt consumers must marshal to
the GUI thread themselves.
"""

import threading
from collections import OrderedDict

import numpy as np

from ..contrast import compute_percentile_clim
from .overlay_state import OVERLAY
from ..load_thumbnail import load_plane


def composite_to_rgb(plane, channel_params):
    """Composite a ``(C, H, W)`` plane into an ``(H, W, 3)`` uint8 image.

    Each channel is normalized to ``[0, 1]`` by clim + gamma, then
    blended onto an accumulating canvas (starting at black) in
    channel-index order:

    - ``additive``: ``canvas += color * t * opacity`` (channels
      brighten on top of each other).
    - ``overlay``: ``alpha = t * opacity; canvas = canvas*(1-alpha) +
      color*alpha`` -- data-driven alpha compositing, so a segmentation
      mask channel tints over whatever's already composited instead of
      just brightening it. Useful for eyeballing masks.

    Canvas is clipped to ``[0, 1]`` after every channel so overlay math
    never reads an out-of-range "background".

    Args:
        plane: ``(C, H, W)`` array.
        channel_params: sequence of ``(clim, gamma, color_rgb, visible,
            blend_mode, opacity)``, one per channel. ``color_rgb`` is an
            ``(r, g, b)`` triple in ``[0, 1]``. Extra entries beyond
            ``plane.shape[0]`` are ignored.
    """
    C, H, W = plane.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(min(C, len(channel_params))):
        clim, gamma, color_rgb, visible, blend_mode, opacity = channel_params[c]
        if not visible:
            continue
        vmin, vmax = clim
        span = vmax - vmin
        if span <= 0:
            span = 1.0
        data = plane[c].astype(np.float32, copy=False)
        t = np.clip((data - vmin) / span, 0.0, 1.0)
        if gamma != 1.0:
            t = np.power(t, gamma, dtype=np.float32)
        color = np.asarray(color_rgb, dtype=np.float32)
        weighted = (t * opacity)[..., np.newaxis]
        if blend_mode == OVERLAY:
            rgb = rgb * (1.0 - weighted) + color * weighted
        else:
            rgb = rgb + color * weighted
        np.clip(rgb, 0.0, 1.0, out=rgb)
    return (rgb * 255.0).astype(np.uint8)


class DecodeCache:
    """Byte-budgeted LRU of decoded ``(C, H, W)`` planes, keyed by path.

    Each entry also carries a per-channel default clim -- one-shot
    auto-contrast computed at decode time (intensity ranges differ per
    image, so a single global clim would wash most tiles out unless the
    user explicitly overrides a channel via the global panel).
    """

    def __init__(self, max_bytes=256 * 1024 * 1024):
        self._max_bytes = int(max_bytes)
        self._cache = OrderedDict()  # path -> (plane, default_clims)
        self._nbytes = 0
        self._lock = threading.Lock()

    def get(self, path):
        """Return the cached ``(plane, default_clims)`` entry for
        ``path``, or ``None``."""
        with self._lock:
            entry = self._cache.get(path)
            if entry is not None:
                self._cache.move_to_end(path)
            return entry

    def decode(self, path, dims=None):
        """Decode ``path`` synchronously and store it. Safe to call from
        a background thread. Returns the cached entry."""
        plane = load_plane(path, dims=dims)
        default_clims = tuple(
            compute_percentile_clim(plane[c], 0.5, 99.5)
            for c in range(plane.shape[0])
        )
        entry = (plane, default_clims)
        self._store(path, entry)
        return entry

    def _store(self, path, entry):
        plane = entry[0]
        nbytes = plane.nbytes
        with self._lock:
            old = self._cache.pop(path, None)
            if old is not None:
                self._nbytes -= old[0].nbytes
            self._cache[path] = entry
            self._nbytes += nbytes
            while self._nbytes > self._max_bytes and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                self._nbytes -= evicted[0].nbytes

    def __contains__(self, path):
        with self._lock:
            return path in self._cache

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._nbytes = 0


class ThumbnailDecodeWorker:
    """Background decode of a priority-ordered path list.

    ``set_priority`` is latest-wins: it replaces whatever the loop
    hasn't reached yet, so switching pages/sorting reprioritizes instead
    of queuing up stale work.
    """

    def __init__(self, cache, callback, dims=None):
        self._cache = cache
        self._callback = callback
        self._dims = dims
        self._lock = threading.Condition()
        self._queue = []
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="tileclass-thumbnail-decode", daemon=True
        )
        self._thread.start()

    def set_priority(self, paths):
        """Replace the pending queue with ``paths`` (highest priority
        first), skipping ones already cached."""
        with self._lock:
            self._queue = [p for p in paths if p not in self._cache]
            self._lock.notify()

    def set_dims(self, dims):
        with self._lock:
            self._dims = dims

    def close(self):
        with self._lock:
            self._closed = True
            self._queue = []
            self._lock.notify()
        self._thread.join(timeout=1.0)

    def _run(self):
        while True:
            with self._lock:
                while not self._closed and not self._queue:
                    self._lock.wait()
                if self._closed:
                    return
                path = self._queue.pop(0)
                dims = self._dims
            if path in self._cache:
                continue
            try:
                self._cache.decode(path, dims=dims)
            except Exception:
                continue
            try:
                self._callback(path)
            except Exception:
                pass
