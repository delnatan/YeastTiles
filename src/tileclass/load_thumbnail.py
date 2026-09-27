"""Decode one tile from a packed `.tiles` container into the ``(C, H, W)``
plane the thumbnail grid composites."""

import numpy as np

from .tile_container import container_and_cell, get_container


def load_plane(path):
    """Return the ``(C, H, W)`` plane for a `<container>.tiles/<cell_id>.tif`
    reference (see tile_container.py). A 2D tile comes back as one
    channel. Raises on an unreadable container or unknown cell --
    `ThumbnailDecodeWorker` treats that as skip-and-continue."""
    container_path, cell_id = container_and_cell(path)
    arr = get_container(container_path).read(cell_id)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    # read() returns a read-only view onto the decompressed buffer.
    return np.array(arr, copy=True)
