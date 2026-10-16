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
    """A degree-2 fit to noiseless degree-2 data, sampled at each tile's
    center (where its focus measurement applies), has ~zero residual."""
    Ny, Nx = 64, 64
    tile = 16

    def surface(x, y):
        return 0.004 * (x - 20) ** 2 + 0.003 * (y - 40) ** 2 + 5

    y, x = np.mgrid[0:Ny, 0:Nx]
    centers = np.arange(0, Nx, tile) + (tile - 1) / 2
    XC, YC = np.meshgrid(centers, centers)

    from yeastprep.core.focus import TileInfo

    fitted = fit_focal_indices_to_poly2d(
        surface(XC, YC), Nx, Ny, TileInfo(tile, tile), poly_degree=(2, 2)
    )
    # Placing samples at tile corners instead shifts the surface by half a
    # tile, which is off by several z-slices at these slopes.
    assert np.allclose(fitted, surface(x, y), atol=1e-6)


def test_fit_focal_indices_to_poly2d_edge_tile_center_skips_padding():
    """A tile that's partly reflect-padding is centered on its real pixels,
    and a tile that's entirely padding is left out."""
    from yeastprep.core.focus import TileInfo

    # 10 px in 6 tiles -> 2 px tiles; tile 5 (pixels 10-11) is all padding.
    Nx = Ny = 10
    slope = 0.5
    centers = np.array([0.5, 2.5, 4.5, 6.5, 8.5, 99.0])
    focal = np.tile(slope * centers + 3, (6, 1))
    fitted = fit_focal_indices_to_poly2d(focal, Nx, Ny, TileInfo(2, 2), poly_degree=(1, 1))
    assert np.allclose(fitted, slope * np.arange(Nx) + 3)


def test_fit_focal_indices_to_poly2d_too_few_peaks_is_a_clear_error():
    import pytest

    from yeastprep.core.focus import TileInfo

    focal = np.zeros((4, 4), dtype=np.uint8)
    focal[0, :3] = 7  # 3 usable tiles, a (2, 2) surface has 9 coefficients
    with pytest.raises(ValueError, match="only 3 of 16 tiles"):
        fit_focal_indices_to_poly2d(focal, 64, 64, TileInfo(16, 16), poly_degree=(2, 2))


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
