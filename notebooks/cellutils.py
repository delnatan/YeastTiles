"""
Module containing utilities for compiling cells for cell classification

"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import scipy.ndimage as ndi
import tifffile

Slices = tuple[slice, slice]


class CellGeometry(NamedTuple):
    label: int
    src: Slices  # region to read from
    dst: Slices  # region to write into


def axis_slices(center: float, size: int, limit: int) -> tuple[slice, slice]:
    """For one axis: (source slice, dest slice) for a size-length window
    centered on `center`, clipped to the valid range [0, limit)."""
    half = size // 2
    start = int(round(center)) - half
    end = start + size
    src_start, src_end = max(start, 0), min(end, limit)
    dst_start, dst_end = src_start - start, src_end - start
    return slice(src_start, src_end), slice(dst_start, dst_end)


def cell_geometry(mask: np.ndarray, size: int) -> list[CellGeometry]:
    """Precompute read/write slices for every cell. Only needs each cell's
    centroid in pixel coordinates -- no intensity data involved."""
    labels = np.unique(mask)
    labels = labels[labels != 0]
    centroids = ndi.center_of_mass(mask > 0, labels=mask, index=labels)

    geometries = []
    for label, (cy, cx) in zip(labels, centroids):
        row_src, row_dst = axis_slices(cy, size, mask.shape[0])
        col_src, col_dst = axis_slices(cx, size, mask.shape[1])
        geometries.append(
            CellGeometry(int(label), (row_src, col_src), (row_dst, col_dst))
        )
    return geometries


def to_uint8(image: np.ndarray, lo_pct=0.1, hi_pct=99.9) -> np.ndarray:
    lo, hi = np.percentile(image, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = np.clip((image.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.uint8)


def apply_crop(
    image: np.ndarray, geom: CellGeometry, size: int, pad_value: float = 0
) -> np.ndarray:
    """Place image[geom.src] into a (size, size) tile at geom.dst. Identical
    for any channel or the mask -- the geometry already encodes everything
    about edge clipping."""
    tile = np.full((size, size), pad_value, dtype=image.dtype)
    tile[geom.dst] = image[geom.src]
    return tile


def crop_cell(
    mask: np.ndarray,
    image1: np.ndarray,
    image2: np.ndarray,
    geom: CellGeometry,
    size: int = 64,
) -> np.ndarray:
    """Crop one cell from two intensity channels + its mask, contrast-normalized.
    Returns a (3, size, size) stack: [image1, image2, cell_mask]."""
    cell_mask = (mask == geom.label).astype(np.uint8) * 255
    ch1 = to_uint8(apply_crop(image1, geom, size))
    ch2 = to_uint8(apply_crop(image2, geom, size))
    ch_mask = apply_crop(cell_mask, geom, size)
    return np.stack([ch1, ch2, ch_mask], axis=0)


def export_fov(
    mask: np.ndarray,
    image1: np.ndarray,
    image2: np.ndarray,
    fov_id: str,
    out_dir: Path,
    size: int = 64,
    channel_names: tuple[str, str] = ("brightfield", "fluorescence"),
) -> pl.DataFrame:
    """Crop every cell in one FOV, save as multi-channel TIFFs, return an index."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for geom in cell_geometry(mask, size):
        cell_id = f"{fov_id}_cell{geom.label:05d}"
        crop = crop_cell(mask, image1, image2, geom, size=size)

        crop_path = out_dir / f"{cell_id}.tif"
        tifffile.imwrite(
            crop_path,
            crop,
            metadata={"axes": "CYX", "channels": [*channel_names, "mask"]},
        )
        records.append(
            {
                "cell_id": cell_id,
                "fov_id": fov_id,
                "label": geom.label,
                "crop_path": str(crop_path),
            }
        )
    return pl.DataFrame(records)
