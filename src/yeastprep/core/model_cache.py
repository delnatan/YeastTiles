"""Loaded-model cache shared by the denoise and segmentation stages.

Loading a checkpoint pulls its weights onto the GPU, so live previews
reuse one loaded model across calls. Two things a plain path-keyed dict
got wrong:

- Retraining overwrites the same file (denoise training saves to
  `denoise_<channel>.pt` by default), and the old weights stayed in use
  until restart. Entries are keyed by the file's mtime and size too.
- Nothing was ever dropped, so every checkpoint tried in a session stayed
  on the GPU. Only the few most recently used are kept.
"""

import os
from collections import OrderedDict
from threading import Lock


def _file_identity(path) -> tuple:
    """`path` plus its mtime and size when it's a file -- or just `path`
    for a non-file model name (cellpose's built-in "cpsam", or None)."""
    if path is None:
        return (None,)
    try:
        stat = os.stat(path)
    except OSError:
        return (str(path),)
    return (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)


class ModelCache:
    def __init__(self, max_entries: int = 2):
        self._max_entries = max_entries
        self._entries: OrderedDict = OrderedDict()
        self._lock = Lock()

    def get(self, path, device, load):
        """The model for (`path`, `device`), calling `load()` if it isn't
        cached or the file changed since it was loaded. Preview and batch
        threads can ask at the same time; loading is serialized so a model
        is never loaded twice."""
        key = (_file_identity(path), str(device))
        with self._lock:
            model = self._entries.get(key)
            if model is not None:
                self._entries.move_to_end(key)
                return model
            model = load()
            self._entries[key] = model
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return model

    def clear(self):
        with self._lock:
            self._entries.clear()
