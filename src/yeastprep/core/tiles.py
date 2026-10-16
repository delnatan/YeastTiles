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

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import scipy.ndimage as ndi
from tileclass.data.annotations import TileAnnotations
from tileclass.tile_container import TileContainer, write_container

from .combined_tiff import load_combined_channels
from .fs_status import dir_has_files
from .segmentation import load_saved_masks, seg_npy_path

Slices = tuple[slice, slice]

TILE_INDEX_NAME = "tile_index.csv"

# Minimum mask overlap (IoU, compared in the old tile's own crop window)
# for a re-exported cell to count as "the same cell" an existing tag was
# made on -- see `_match_new_cell`.
MIN_CARRY_OVER_IOU = 0.5
# For containers exported before centroids were recorded, mask shape alone
# can't tell two similar-sized cells apart, so the brightfield pixels must
# match too (Pearson r, same crop window) -- see `_match_by_content`.
MIN_CARRY_OVER_CORR = 0.9


@dataclass
class TileParams:
    size: int = 64
    contrast_lo_pct: float = 0.1
    contrast_hi_pct: float = 99.9


class CellGeometry(NamedTuple):
    label: int
    src: Slices  # region to read from
    dst: Slices  # region to write into
    centroid: tuple[float, float] | None = None  # (row, col), pixel coords


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

    return [
        _geometry_at(int(label), (cy, cx), size, mask.shape)
        for label, (cy, cx) in zip(labels, centroids)
    ]


def _geometry_at(label: int, centroid, size: int, shape) -> CellGeometry:
    cy, cx = centroid
    row_src, row_dst = axis_slices(cy, size, shape[0])
    col_src, col_dst = axis_slices(cx, size, shape[1])
    return CellGeometry(
        label, (row_src, col_src), (row_dst, col_dst), (float(cy), float(cx))
    )


def cell_id_for(fov_id: str, label: int) -> str:
    return f"{fov_id}_cell{label:05d}"


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


def crop_to_uint8(
    image: np.ndarray, geom: CellGeometry, size: int, lo_pct: float, hi_pct: float
) -> np.ndarray:
    """Crop one channel and contrast-stretch it to 8 bits, taking the
    percentiles over the image pixels only. For a cell near the FOV edge,
    part of the tile is padding, and counting that as zeros would drag the
    low percentile to 0 and wash the cell out."""
    tile = np.zeros((size, size), dtype=np.uint8)
    tile[geom.dst] = to_uint8(image[geom.src], lo_pct, hi_pct)
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
    lo, hi = params.contrast_lo_pct, params.contrast_hi_pct
    ch_bf = crop_to_uint8(brightfield, geom, size, lo, hi)
    ch_target = crop_to_uint8(target, geom, size, lo, hi)
    # Compared after cropping: padding is 0, never a cell label.
    ch_mask = (apply_crop(mask, geom, size) == geom.label).astype(np.uint8) * 255
    return np.stack([ch_bf, ch_target, ch_mask], axis=0)


@dataclass
class TileExportResult:
    path: Path
    success: bool
    n_cells: int | None = None
    records: pl.DataFrame | None = None
    error: str | None = None
    # Existing tags (`<fov_id>.txt`) re-keyed to a re-exported cell's new
    # cell_id, and dropped because their cell couldn't be found again --
    # see `_carry_over_tags`.
    tags_moved: int = 0
    tags_dropped: int = 0


def _empty_index() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"cell_id": pl.Utf8, "fov_id": pl.Utf8, "label": pl.Int64, "crop_path": pl.Utf8}
    )


def export_tiles(
    path, out_dir, params: TileParams = TileParams()
) -> TileExportResult:
    """Crop every cell in one combined-channel tiff's saved segmentation and
    pack them into a single `out_dir/<fov_id>.tiles` container (see
    `tileclass.tile_container`) -- one container per FOV, so each FOV's
    annotation sidecar (a `tileclass` concept: one text file per container)
    stays separate and combining several FOVs' annotations later is just
    pooling their containers. Catches any exception so a batch run over
    many files doesn't abort on the first bad one (matches
    core/segmentation.py's `segment_and_save`). Requires a saved `_seg.npy`
    sidecar to already exist next to `path` (from the Segmentation stage's
    batch run, or corrected in the real Cellpose GUI); tile export never
    runs segmentation itself."""
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
        out_dir.mkdir(parents=True, exist_ok=True)

        fov_id = path.stem
        container_path = out_dir / f"{fov_id}.tiles"

        records = []
        cells = []
        for geom in cell_geometry(masks, params.size):
            cell_id = cell_id_for(fov_id, geom.label)
            crop = crop_cell(masks, brightfield, target, geom, params)
            # The centroid lets a later re-export find this same cell again
            # after its mask labels get renumbered -- see _carry_over_tags.
            cells.append((cell_id, geom.label, crop, {"centroid": list(geom.centroid)}))
            records.append(
                {
                    "cell_id": cell_id,
                    "fov_id": fov_id,
                    "label": geom.label,
                    # Virtual reference, not a real path on disk -- see
                    # tile_container.py's module docstring. Every consumer
                    # of this column already treats it as an opaque string.
                    "crop_path": f"{container_path}/{cell_id}",
                }
            )

        tag_mapping = _carry_over_tags(container_path, masks, cells, fov_id)
        write_container(container_path, cells)
        tags_moved = tags_dropped = 0
        if tag_mapping:
            annotations = TileAnnotations(str(container_path))
            # Keep the pre-remap tags recoverable -- dropped ones are
            # human work that can't be redone from the new tiles alone.
            shutil.copy2(annotations.file_path, annotations.file_path + ".bak")
            annotations.remap(tag_mapping)
            tags_moved = sum(new is not None for new in tag_mapping.values())
            tags_dropped = len(tag_mapping) - tags_moved

        df = pl.DataFrame(records) if records else _empty_index()
        return TileExportResult(
            path=path,
            success=True,
            n_cells=len(records),
            records=df,
            tags_moved=tags_moved,
            tags_dropped=tags_dropped,
        )
    except Exception as exc:
        return TileExportResult(path=path, success=False, error=str(exc))


def _carry_over_tags(container_path, masks, new_cells, fov_id) -> dict[str, str | None]:
    """Re-keying for `container_path`'s existing tags before it's
    overwritten with `new_cells`: tagged old cell_id -> the cell_id that
    same cell has now, or None if it can't be found again. Only entries
    that change are returned (empty: nothing to do).

    Tags are keyed by cell_id, i.e. by mask label, and labels don't
    survive mask edits -- deleting or adding a cell in the Cellpose GUI
    renumbers the rest. Without this, re-exporting after a correction
    would silently move every human tag onto whichever cell inherited
    that label number."""
    if not Path(container_path).exists():
        return {}
    annotations = TileAnnotations(str(container_path))
    if not len(annotations):
        return {}
    old = TileContainer(container_path)
    new_crops = {cell_id: crop for cell_id, _label, crop, _extra in new_cells}

    matches = {}
    for old_id, entry in old.entries.items():
        if old_id not in annotations:
            continue
        old_crop = old.read(old_id)
        centroid = entry.get("centroid")
        if centroid is not None:
            label = _match_new_cell(old_crop[2] > 0, centroid, masks)
            matches[old_id] = cell_id_for(fov_id, label) if label is not None else None
        else:
            # Exported before centroids were recorded: no telling where
            # the old cell was, so look for a new tile that's a near copy.
            matches[old_id] = _match_by_content(old_crop, new_crops)

    # Two old cells landing on one new cell (masks merged) is ambiguous.
    claimed = Counter(new_id for new_id in matches.values() if new_id is not None)
    matches = {
        old_id: new_id if new_id is not None and claimed[new_id] == 1 else None
        for old_id, new_id in matches.items()
    }
    return {old_id: new_id for old_id, new_id in matches.items() if new_id != old_id}


def _match_new_cell(old_mask, centroid, masks) -> int | None:
    """The new mask label covering most of an old cell's footprint, in the
    old tile's own crop window -- or None if nothing overlaps it enough to
    be the same cell."""
    size = old_mask.shape[-1]  # the old export's tile size, not this one's
    window = apply_crop(masks, _geometry_at(0, centroid, size, masks.shape), size)
    labels, counts = np.unique(window[old_mask], return_counts=True)
    counts = counts[labels != 0]
    labels = labels[labels != 0]
    if len(labels) == 0:
        return None
    label = int(labels[np.argmax(counts)])
    return label if _iou(old_mask, window == label) >= MIN_CARRY_OVER_IOU else None


def _match_by_content(old_crop, new_crops) -> str | None:
    """The new cell whose tile is (nearly) the same picture as `old_crop`:
    overlapping mask footprint and correlated brightfield pixels, both
    compared in the crop window (same cell -> same centroid -> same
    window). Best correlation wins; None if no tile is close enough --
    conservative on purpose, since a dropped tag beats a misplaced one."""
    old_mask = old_crop[2] > 0
    best, best_corr = None, MIN_CARRY_OVER_CORR
    for cell_id, crop in new_crops.items():
        if _iou(*_common_window(old_mask, crop[2] > 0)) < MIN_CARRY_OVER_IOU:
            continue
        a, b = _common_window(old_crop[0], crop[0])
        corr = _pearson(a, b)
        if corr >= best_corr:
            best, best_corr = cell_id, corr
    return best


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel() - a.mean()
    b = b.astype(np.float64).ravel() - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / denom) if denom else 0.0


def _common_window(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two crops centered on the same point but possibly of different tile
    sizes, both cut down to the smaller one's window (see `axis_slices`:
    a window starts `size // 2` before its center)."""
    n = min(a.shape[-1], b.shape[-1])

    def center(x):
        start = x.shape[-1] // 2 - n // 2
        return x[start : start + n, start : start + n]

    return center(a), center(b)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 0.0


def tile_index_path(out_dir) -> Path:
    return Path(out_dir) / TILE_INDEX_NAME


def append_tile_index(out_dir, records: pl.DataFrame, fov_ids=None):
    """Merge `records` into `out_dir`'s running tile index, the per-corpus
    cell_id/fov_id/crop_path table that a later training run reads.

    Re-exporting a FOV replaces *all* of its rows, not just the cell_ids
    that come back: a corrected mask can have fewer cells than before, and
    their old rows would otherwise point at cells the container no longer
    has. `fov_ids` names the FOVs being replaced -- needed when a FOV now
    has no cells at all, so `records` can't say which FOV it was -- and
    defaults to whichever FOVs `records` covers."""
    fov_ids = set(fov_ids) if fov_ids is not None else set(records["fov_id"].to_list())
    index_path = tile_index_path(out_dir)
    if index_path.exists():
        # Rows written before tile refs dropped their `.tif` suffix.
        existing = pl.read_csv(index_path, schema_overrides={"fov_id": pl.Utf8}).with_columns(
            pl.col("crop_path").str.strip_suffix(".tif")
        )
        existing = existing.filter(~pl.col("fov_id").is_in(list(fov_ids)))
        combined = pl.concat([existing, records], how="vertical_relaxed").unique(
            subset=["cell_id"], keep="last"
        )
    elif records.is_empty():
        return
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
    """Per-source-FOV tile summary, read straight from each
    `out_dir/<fov_id>.tiles` container's own index. A FOV counts as up to
    date if its container is no older than its `_seg.npy` sidecar in
    `mask_dir` (tiles are cropped straight from that saved mask), so a mask
    re-run since the last export shows up as stale without re-reading any
    image data.

    One container = one file, so this is one `stat()` (freshness) plus one
    small index parse (cell count) per FOV, not per cell."""
    out_dir = Path(out_dir)
    mask_dir = Path(mask_dir) if mask_dir else None
    # Narrowed to `*_seg.npy` sidecars specifically (not "any file") --
    # mask_dir is a 2D-stage folder that can also hold incidental files
    # (`.DS_Store`, ...) which shouldn't count as "still populated".
    mask_available = mask_dir is None or dir_has_files(mask_dir, "*_seg.npy")

    statuses: dict[str, FovTileStatus] = {}
    for container_path in sorted(out_dir.glob("*.tiles")):
        fov_id = container_path.stem
        try:
            n_cells = len(TileContainer(container_path))
            container_mtime = container_path.stat().st_mtime
        except OSError:
            continue

        up_to_date = True
        if mask_dir is not None and mask_available:
            seg_path = mask_dir / f"{fov_id}_seg.npy"
            try:
                up_to_date = container_mtime >= seg_path.stat().st_mtime
            except OSError:
                up_to_date = False
        statuses[fov_id] = FovTileStatus(
            n_cells=n_cells, up_to_date=up_to_date, mask_available=mask_available
        )
    return statuses
