"""Synthetic-data regression tests for yeastprep.core.focus.

No real acquisition files needed -- these build small in-memory 3D arrays
with a known, deliberately-placed focal structure and check the pipeline
recovers it. Also serves as the numeric-regression proof that moving
notebooks/focalstackutils.py into core/focus.py (and splitting
find_focus_index_map) didn't change behavior: run these same assertions
against the pre-refactor notebooks/focalstackutils.py to confirm parity.
"""

import numpy as np

from yeastprep.core.focus import (
    compute_tile_variance_stack,
    find_focus_index_map,
    fit_focal_indices_to_poly2d,
    peaks_from_variance_stack,
    resample_focal_slice,
)


def _synthetic_stack(true_focus_z, Nz=20, tile=16, num_tiles=2, width=1.5):
    """A (Nz, tile*num_tiles, tile*num_tiles) stack where each coarse tile
    has *minimum* intensity variance at its own `true_focus_z[i, j]` --
    matching this codebase's established convention (find_peaks is run on
    *negated* variance, i.e. it locates variance troughs, not peaks; see
    focus.py's `peaks_from_variance_stack`). Modeled as a checkerboard
    pattern whose contrast dips to ~0 at z0 and grows away from it.
    """
    size = tile * num_tiles
    checkerboard = np.indices((size, size)).sum(axis=0) % 2
    stack = np.zeros((Nz, size, size), dtype=np.float64)
    for i in range(num_tiles):
        for j in range(num_tiles):
            z0 = true_focus_z[i, j]
            for z in range(Nz):
                amplitude = 1000 * (1 - np.exp(-((z - z0) ** 2) / (2 * width**2)))
                block = checkerboard[i * tile : (i + 1) * tile, j * tile : (j + 1) * tile]
                stack[z, i * tile : (i + 1) * tile, j * tile : (j + 1) * tile] = (
                    amplitude * block
                )
    return stack


def test_compute_tile_variance_stack_shape():
    true_focus = np.array([[5, 10], [15, 8]])
    stack = _synthetic_stack(true_focus, Nz=20, tile=16, num_tiles=2)
    result = compute_tile_variance_stack(stack, num_tiles_y=2, num_tiles_x=2)
    assert result.variances.shape == (2, 2, 20)
    assert result.tile_info.tile_height == 16
    assert result.tile_info.tile_width == 16


def test_peaks_from_variance_stack_recovers_known_focus():
    true_focus = np.array([[5, 10], [15, 8]])
    stack = _synthetic_stack(true_focus, Nz=20, tile=16, num_tiles=2)
    variance_stack = compute_tile_variance_stack(stack, num_tiles_y=2, num_tiles_x=2)
    peaks = peaks_from_variance_stack(variance_stack, inverted_variance_prominence=1.0)
    # Peak-finding on a smooth Gaussian-modulated variance curve should land
    # within a couple of z-slices of the true focus.
    assert np.all(np.abs(peaks.astype(int) - true_focus) <= 2)


def test_find_focus_index_map_matches_split_functions():
    """find_focus_index_map must remain a drop-in composition of the two
    split functions (no numeric change from the pre-split version)."""
    true_focus = np.array([[5, 10], [15, 8]])
    stack = _synthetic_stack(true_focus, Nz=20, tile=16, num_tiles=2)

    combined_peaks, combined_tile_info = find_focus_index_map(
        stack, num_tiles_y=2, num_tiles_x=2, inverted_variance_prominence=1.0
    )

    variance_stack = compute_tile_variance_stack(stack, num_tiles_y=2, num_tiles_x=2)
    separate_peaks = peaks_from_variance_stack(variance_stack, inverted_variance_prominence=1.0)

    assert np.array_equal(combined_peaks, separate_peaks)
    assert combined_tile_info == variance_stack.tile_info


def test_fit_focal_indices_to_poly2d_recovers_noiseless_quadratic():
    """A degree-2 fit to noiseless degree-2 data should have ~zero residual."""
    Ny, Nx = 64, 64
    tile = 16
    num_tiles_y, num_tiles_x = 4, 4

    y, x = np.mgrid[0:Ny, 0:Nx]
    true_surface = 0.001 * (x - 32) ** 2 + 0.002 * (y - 32) ** 2 + 5

    coarse_y = np.arange(0, Ny, tile)
    coarse_x = np.arange(0, Nx, tile)
    XC, YC = np.meshgrid(coarse_x, coarse_y)
    coarse_focus = (0.001 * (XC - 32) ** 2 + 0.002 * (YC - 32) ** 2 + 5).round().astype(np.uint8)

    from yeastprep.core.focus import TileInfo

    fitted = fit_focal_indices_to_poly2d(
        coarse_focus, Nx, Ny, TileInfo(tile, tile), poly_degree=(2, 2)
    )
    # Coarse focus values were rounded to uint8, so allow a small tolerance.
    assert np.allclose(fitted, true_surface, atol=1.0)


def test_resample_focal_slice_constant_focus_equals_fixed_z():
    """A constant focus-index map should just extract img3d[fixed_z]."""
    Nz, Ny, Nx = 10, 8, 8
    img3d = np.arange(Nz * Ny * Nx, dtype=np.uint16).reshape(Nz, Ny, Nx)
    fixed_z = 4
    focus_indices = np.full((Ny, Nx), float(fixed_z))

    result = resample_focal_slice(img3d, focus_indices, dz_um=1.0, offset_um=0.0)
    assert np.array_equal(result, img3d[fixed_z])


def test_resample_focal_slice_clips_out_of_range_offset():
    Nz, Ny, Nx = 5, 4, 4
    img3d = np.arange(Nz * Ny * Nx, dtype=np.uint16).reshape(Nz, Ny, Nx)
    focus_indices = np.zeros((Ny, Nx))
    # offset pushes far past the top of the stack -- should clip to Nz - 1.
    result = resample_focal_slice(img3d, focus_indices, dz_um=1.0, offset_um=100.0)
    assert np.array_equal(result, img3d[Nz - 1])
