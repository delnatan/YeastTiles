"""Field-flattening core: per-tile focus detection, polynomial surface fit,
and focal-plane resampling.

Moved from ``notebooks/focalstackutils.py`` (see that file's re-export shim).
``find_focus_index_map`` is split into ``compute_tile_variance_stack`` (the
expensive per-pixel variance pass) and ``peaks_from_variance_stack`` (the
cheap per-tile peak search) so a UI can cache the former and re-run only the
latter when just the prominence threshold changes.
"""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.polynomial import polyvander2d
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks


@dataclass(frozen=True)
class TileInfo:
    tile_height: int
    tile_width: int


@dataclass(frozen=True)
class TileVarianceStack:
    """Per-tile, per-z intensity variance, and the grid geometry it was
    computed on. ``variances`` has shape (num_tiles_y, num_tiles_x, Nz)."""

    variances: np.ndarray
    tile_info: TileInfo


def compute_tile_variance_stack(
    img3d: np.ndarray[tuple[int, int, int]],
    num_tiles_y: int = 8,
    num_tiles_x: int = 8,
) -> TileVarianceStack:
    """Divide the FOV into a coarse grid and compute intensity variance
    across each tile, for every z-plane.

    This is the expensive step (touches every pixel of the volume) --
    callers that only need to retune ``inverted_variance_prominence``
    should cache this result and only re-run `peaks_from_variance_stack`.
    """
    Nz, Ny, Nx = img3d.shape

    # use ceiling division to find tile sizes
    tile_y = int(np.ceil(Ny / num_tiles_y))
    tile_x = int(np.ceil(Nx / num_tiles_x))

    # calculate how many pixels are missing to fill the grid
    pad_y = (tile_y * num_tiles_y) - Ny
    pad_x = (tile_x * num_tiles_x) - Nx

    if pad_y == 0 and pad_x == 0:
        # no padding needed
        wrk = img3d
    else:
        # pad image to support even tiles and leverage broadcasting
        wrk = np.pad(img3d, ((0, 0), (0, pad_y), (0, pad_x)), mode="reflect")

    reshaped = wrk.reshape(Nz, num_tiles_y, tile_y, num_tiles_x, tile_x)

    # reorder axes so that lateral dimensions are the last two
    # we want (num_tiles_y, num_tiles_x, Nz, tile_y, tile_x)
    tiles = reshaped.transpose(1, 3, 0, 2, 4)

    # take the variance across each tile's pixels, per z-plane
    variances = np.var(tiles, axis=(-2, -1))

    return TileVarianceStack(variances=variances, tile_info=TileInfo(tile_y, tile_x))


def peaks_from_variance_stack(
    stack: TileVarianceStack,
    inverted_variance_prominence: float = 100,
) -> np.ndarray[tuple[int, int]]:
    """Find the focal z-index per tile from a precomputed variance stack.

    Cheap step (`scipy.signal.find_peaks` per tile) -- safe to re-run on
    every prominence change without recomputing `variances`.
    """
    num_tiles_y, num_tiles_x, _Nz = stack.variances.shape
    focal_plane_indices = np.zeros((num_tiles_y, num_tiles_x), dtype=np.uint8)

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            # invert variances to find 'peaks' instead of troughs
            minima_indices, props = find_peaks(
                -stack.variances[i, j], prominence=inverted_variance_prominence
            )
            if len(minima_indices) > 0:
                focal_plane_indices[i, j] = minima_indices[np.argmax(props["prominences"])]

    return focal_plane_indices


def find_focus_index_map(
    img3d: np.ndarray[tuple[int, int, int]],
    num_tiles_y: int = 8,
    num_tiles_x: int = 8,
    inverted_variance_prominence: int = 100,
) -> tuple[np.ndarray[tuple[int, int]], TileInfo]:
    """Divides image into square grids and finds the focal plane for each
    tile. Drop-in equivalent of the original (pre-split) function --
    composes `compute_tile_variance_stack` + `peaks_from_variance_stack`.
    """
    stack = compute_tile_variance_stack(img3d, num_tiles_y, num_tiles_x)
    focal_plane_indices = peaks_from_variance_stack(stack, inverted_variance_prominence)
    return focal_plane_indices, stack.tile_info


def fit_focal_indices_to_poly2d(
    focal_indices: np.ndarray[tuple[int, int]],
    Nx: int,
    Ny: int,
    tile_info: TileInfo,
    poly_degree: tuple[int, int] = (2, 2),
) -> np.ndarray[tuple[int, int]]:
    coarse_x_vec = np.arange(0, Nx, tile_info.tile_width)
    coarse_y_vec = np.arange(0, Ny, tile_info.tile_height)
    XC, YC = np.meshgrid(coarse_x_vec, coarse_y_vec)
    valid_mask = focal_indices != 0
    coarse_x_valid = XC[valid_mask]
    coarse_y_valid = YC[valid_mask]
    valid_z = focal_indices[valid_mask]
    A_valid = polyvander2d(coarse_x_valid, coarse_y_valid, deg=list(poly_degree))
    coefs, *_ = np.linalg.lstsq(A_valid, valid_z, rcond=None)

    fine_x = np.arange(0, Nx)
    fine_y = np.arange(0, Ny)
    XF, YF = np.meshgrid(fine_x, fine_y)
    A_fine = polyvander2d(XF, YF, deg=list(poly_degree))
    focus_indices = np.dot(A_fine, coefs).reshape((Ny, Nx))
    return focus_indices


def resample_focal_slice(
    img3d: np.ndarray[tuple[int, int, int]],
    focus_indices: np.ndarray[tuple[int, int]],
    dz_um: float = 0.293,
    offset_um: float = 1.2,
) -> np.ndarray[tuple[int, int]]:
    Nz, Ny, Nx = img3d.shape
    # express coordinates in pixel-space
    z_coords = focus_indices + (offset_um / dz_um)
    z_coords = np.clip(z_coords, 0, Nz - 1)
    y_coords, x_coords = np.mgrid[0:Ny, 0:Nx].astype(float)
    coords_resample = np.stack([z_coords, y_coords, x_coords], axis=0)
    focal_slice = map_coordinates(img3d, coords_resample, order=2)
    return focal_slice.astype(np.uint16)
