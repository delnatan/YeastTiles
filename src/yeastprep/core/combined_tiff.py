"""Shared I/O for the combined-channel 2D tiff every 2D-image stage folder
(01_reduced/, 02_denoised/, 03_deconvolved/) holds one of per FOV: channel 0
= flattened brightfield, channel 1 = sum-projected target, per design.md's
"one file per FOV instead of two parallel folders" convention.

Pulled out of core/pipeline.py (which produces these files) since
core/denoise.py, core/deconvolve.py and core/tiles.py all need to read them
back too, and none of those are really "about" the flatten stage.
"""

import numpy as np
import tifffile

# Channel order within the combined 2D output tiff -- the one place this
# convention is defined. Segmentation (core/segmentation.py) only ever
# reads BRIGHTFIELD_CHANNEL; core/tiles.py and the denoise/deconvolve
# stages read both.
BRIGHTFIELD_CHANNEL = 0
TARGET_CHANNEL = 1


def combine_channels(brightfield: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Stack a flattened brightfield slice and a target-channel image into
    one (2, Ny, Nx) array -- BRIGHTFIELD_CHANNEL first, TARGET_CHANNEL
    second. One file per FOV instead of two parallel folders matched by
    filename stem: fewer places for the two halves of a FOV to drift out
    of sync, and the Cellpose GUI already supports picking which channel
    to segment on when opening a multi-channel tiff, so nothing is lost
    for manual correction either."""
    return np.stack([brightfield, target], axis=0)


def load_brightfield_channel(path) -> np.ndarray:
    """Read just the brightfield channel back out of a combined-channel
    tiff -- the common case: segmentation and every current display panel
    only ever need this one channel, never the target/DAPI one."""
    return tifffile.imread(path)[BRIGHTFIELD_CHANNEL]


def load_combined_channels(path) -> tuple[np.ndarray, np.ndarray]:
    """Read both channels back out of a combined-channel tiff -- needed by
    tile export and the denoise/deconvolve stages, unlike
    `load_brightfield_channel`."""
    arr = tifffile.imread(path)
    return arr[BRIGHTFIELD_CHANNEL], arr[TARGET_CHANNEL]


def read_combined_tiff(path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Read a combined-channel tiff back into brightfield/target arrays
    plus its pixel scale. Plain tifffile, not pyvistra's generic
    `load_image`, which is built for raw acquisition stacks and mis-infers
    axis order for these already-2D, non-acquisition tiffs."""
    with tifffile.TiffFile(path) as tf:
        arr = tf.asarray()
        page = tf.pages[0]
        dz = (tf.imagej_metadata or {}).get("spacing", 1.0)
        xres = page.tags["XResolution"].value
        yres = page.tags["YResolution"].value
        dx = xres[1] / xres[0] if xres[0] else 1.0
        dy = yres[1] / yres[0] if yres[0] else 1.0
    return arr[BRIGHTFIELD_CHANNEL], arr[TARGET_CHANNEL], (dz, dy, dx)
