"""Cell-tile export: crops each segmented cell out of a combined-channel
tiff (core/combined_tiff.py's `combine_channels` convention: channel 0 =
flattened brightfield, channel 1 = sum-projected target) into a
homogeneously sized ``(3, size, size)`` tile -- [brightfield, target,
binary cell mask] -- per design.md stage 3.

Restructures notebooks/cellutils.py's prototype (`cell_geometry` /
`crop_cell` / `export_fov`) into the same load -> process ->
save-with-per-file-error-catch shape as core/segmentation.py's
`segment_and_save`: reads the mask back from cellpose's own `_seg.npy`
sidecar (written by `segment_and_save`, possibly hand-corrected in the
real Cellpose GUI afterward) rather than re-running segmentation here.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import scipy.ndimage as ndi
import tifffile

from .combined_tiff import load_combined_channels
from .fs_status import dir_has_files
from .segmentation import load_saved_masks, seg_npy_path

Slices = tuple[slice, slice]

CHANNEL_NAMES = ("brightfield", "target", "mask")

TILE_INDEX_NAME = "tile_index.csv"


@dataclass
class TileParams:
    size: int = 64
    contrast_lo_pct: float = 0.1
    contrast_hi_pct: float = 99.9


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
    centroid in pixel coordinates -- no intensity data involved, so this is
    cheap enough to call synchronously from the UI thread for a live crop-
    window preview (unlike segmentation, there's no model forward pass)."""
    labels = np.unique(mask)
    labels = labels[labels != 0]
    if len(labels) == 0:
        return []
    centroids = ndi.center_of_mass(mask > 0, labels=mask, index=labels)

    geometries = []
    for label, (cy, cx) in zip(labels, centroids):
        row_src, row_dst = axis_slices(cy, size, mask.shape[0])
        col_src, col_dst = axis_slices(cx, size, mask.shape[1])
        geometries.append(
            CellGeometry(int(label), (row_src, col_src), (row_dst, col_dst))
        )
    return geometries


def to_uint8(image: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
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
    brightfield: np.ndarray,
    target: np.ndarray,
    geom: CellGeometry,
    params: TileParams,
) -> np.ndarray:
    """Crop one cell from both intensity channels + its mask, contrast-
    normalized. Returns a (3, size, size) stack: [brightfield, target,
    cell_mask]. Only this cell's own label is kept in the mask channel --
    neighboring cells that happen to fall inside the same crop window are
    background there, matching design.md's "images are not masked, but the
    per-cell binary mask is a separate channel" spec."""
    size = params.size
    cell_mask = (mask == geom.label).astype(np.uint8) * 255
    ch_bf = to_uint8(
        apply_crop(brightfield, geom, size), params.contrast_lo_pct, params.contrast_hi_pct
    )
    ch_target = to_uint8(
        apply_crop(target, geom, size), params.contrast_lo_pct, params.contrast_hi_pct
    )
    ch_mask = apply_crop(cell_mask, geom, size)
    return np.stack([ch_bf, ch_target, ch_mask], axis=0)


@dataclass
class TileExportResult:
    path: Path
    success: bool
    n_cells: int | None = None
    records: pl.DataFrame | None = None
    error: str | None = None


def _empty_index() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"cell_id": pl.Utf8, "fov_id": pl.Utf8, "label": pl.Int64, "crop_path": pl.Utf8}
    )


def export_tiles(
    path, out_dir, params: TileParams = TileParams()
) -> TileExportResult:
    """Crop every cell in one combined-channel tiff's saved segmentation and
    write each as its own (3, size, size) tiff under `out_dir/<fov_id>/` --
    one subfolder per FOV, so each FOV's annotation sidecar (a `tileclass`
    concept: one text file per folder) stays separate and combining several
    FOVs' annotations later is just pooling their subfolders. Catches any
    exception so a batch run over many files doesn't abort on the first bad
    one (matches core/segmentation.py's `segment_and_save`). Requires a
    saved `_seg.npy` sidecar to already exist next to `path` (from the
    Segmentation stage's batch run, or corrected in the real Cellpose GUI);
    tile export never runs segmentation itself."""
    path = Path(path)
    try:
        masks = load_saved_masks(seg_npy_path(path))
        if masks is None:
            return TileExportResult(
                path=path,
                success=False,
                error="No saved segmentation (_seg.npy) found -- run Segmentation first.",
            )

        brightfield, target = load_combined_channels(path)
        out_dir = Path(out_dir)

        fov_id = path.stem
        fov_dir = out_dir / fov_id
        fov_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for geom in cell_geometry(masks, params.size):
            cell_id = f"{fov_id}_cell{geom.label:05d}"
            crop = crop_cell(masks, brightfield, target, geom, params)

            crop_path = fov_dir / f"{cell_id}.tif"
            tifffile.imwrite(
                crop_path,
                crop,
                photometric="minisblack",
                metadata={"axes": "CYX", "channels": list(CHANNEL_NAMES)},
            )
            records.append(
                {
                    "cell_id": cell_id,
                    "fov_id": fov_id,
                    "label": geom.label,
                    "crop_path": str(crop_path),
                }
            )

        df = pl.DataFrame(records) if records else _empty_index()
        return TileExportResult(path=path, success=True, n_cells=len(records), records=df)
    except Exception as exc:
        return TileExportResult(path=path, success=False, error=str(exc))


def tile_index_path(out_dir) -> Path:
    return Path(out_dir) / TILE_INDEX_NAME


def append_tile_index(out_dir, records: pl.DataFrame):
    """Merge `records` into `out_dir`'s running tile index, the per-corpus
    cell_id/fov_id/crop_path table that a later training run reads. Keyed
    on `cell_id` so re-exporting a FOV (new params, corrected mask) updates
    its rows in place rather than duplicating them -- 'last write wins'."""
    if records.is_empty():
        return
    index_path = tile_index_path(out_dir)
    if index_path.exists():
        existing = pl.read_csv(index_path)
        combined = pl.concat([existing, records], how="vertical_relaxed").unique(
            subset=["cell_id"], keep="last"
        )
    else:
        combined = records
    combined = combined.sort(["fov_id", "label"])
    combined.write_csv(index_path)


@dataclass
class FovTileStatus:
    n_cells: int
    up_to_date: bool  # newest surviving tile file's mtime vs. this FOV's _seg.npy sidecar
    # False only when a mask_dir was given but is empty/missing -- see
    # project.FileStatus.upstream_available for why that's kept distinct
    # from a genuine staleness signal.
    mask_available: bool = True


def fov_tile_status(out_dir, mask_dir=None) -> dict[str, FovTileStatus]:
    """Per-source-FOV tile summary, read from `out_dir`'s tile_index.csv --
    the one place that already tracks which cell_id/crop_path came from
    which fov_id (see append_tile_index). A FOV counts as up to date if a
    surviving tile file is no older than its `_seg.npy` sidecar in
    `mask_dir` (tiles are cropped straight from that saved mask), so a mask
    re-run since the last export shows up as stale without re-reading any
    image data.

    Deliberately touches disk at most once per FOV (one `stat()` on the
    first crop we find still present), not once per crop: a FOV's cells
    all export together in one `export_tiles()` batch, so any surviving
    crop's mtime is as good a freshness signal as the oldest code's
    max-over-every-crop, and a project can have orders of magnitude more
    individual cell crops than FOVs -- stat-ing every single one just to
    render this summary was the actual reason project-open scans could
    still take a long time even after that scan was moved off the GUI
    thread (see core/project_scan.py)."""
    index_path = tile_index_path(out_dir)
    if not index_path.exists():
        return {}
    try:
        df = pl.read_csv(index_path)
    except Exception:
        return {}

    by_fov: dict[str, list[Path]] = {}
    for row in df.iter_rows(named=True):
        by_fov.setdefault(row["fov_id"], []).append(Path(row["crop_path"]))

    mask_dir = Path(mask_dir) if mask_dir else None
    # Narrowed to `*_seg.npy` sidecars specifically (not "any file") --
    # mask_dir is a 2D-stage folder that can also hold incidental files
    # (`.DS_Store`, ...) which shouldn't count as "still populated".
    mask_available = mask_dir is None or dir_has_files(mask_dir, "*_seg.npy")

    statuses: dict[str, FovTileStatus] = {}
    for fov_id, crop_paths in by_fov.items():
        newest_tile_mtime = None
        for crop_path in crop_paths:
            try:
                newest_tile_mtime = crop_path.stat().st_mtime
                break
            except OSError:
                continue
        if newest_tile_mtime is None:
            continue  # none of this FOV's recorded crops still exist on disk

        up_to_date = True
        if mask_dir is not None and mask_available:
            seg_path = mask_dir / f"{fov_id}_seg.npy"
            try:
                up_to_date = newest_tile_mtime >= seg_path.stat().st_mtime
            except OSError:
                up_to_date = False
        statuses[fov_id] = FovTileStatus(
            n_cells=len(crop_paths), up_to_date=up_to_date, mask_available=mask_available
        )
    return statuses
