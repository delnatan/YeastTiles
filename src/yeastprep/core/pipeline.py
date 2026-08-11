"""Reusable field-flattening driver.

Generalizes notebooks/01_flatten_field.ipynb's notebook-only `process_file`
into functions callable from the yeastprep UI (which needs the
intermediate diagnostics, not just the final files) and from the batch CLI
(`core/cli.py`, which only needs `process_and_save`).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
from pyvistra.io import load_image, save_tiff

from .channels import ChannelSelection
from .focus import (
    TileVarianceStack,
    compute_tile_variance_stack,
    fit_focal_indices_to_poly2d,
    peaks_from_variance_stack,
    resample_focal_slice,
)

DEFAULT_CHANNELS = ChannelSelection(brightfield=0, projection=1)


@dataclass
class FlattenFieldParams:
    num_tiles_y: int = 8
    num_tiles_x: int = 8
    inverted_variance_prominence: float = 100
    poly_degree: tuple[int, int] = (2, 2)
    offset_um: float = 0.7


@dataclass
class LoadedVolume:
    path: Path
    img3d: np.ndarray  # brightfield channel, (Nz, Ny, Nx)
    projection3d: np.ndarray  # target/DAPI channel, (Nz, Ny, Nx)
    scale: tuple[float, float, float]  # (dz, dy, dx) in um
    channels_meta: list[dict] = field(default_factory=list)


def load_volume(path: Path, channels: ChannelSelection = DEFAULT_CHANNELS) -> LoadedVolume:
    """Load a raw multi-channel stack and slice out the brightfield +
    target channels as plain (Nz, Ny, Nx) arrays."""
    path = Path(path)
    img, meta = load_image(str(path))
    img3d = np.asarray(img[0, :, channels.brightfield, :, :])
    projection3d = np.asarray(img[0, :, channels.projection, :, :])
    dz, dy, dx = meta["scale"]
    return LoadedVolume(
        path=path,
        img3d=img3d,
        projection3d=projection3d,
        scale=(dz, dy, dx),
        channels_meta=list(meta.get("channels") or []),
    )


@dataclass
class FocusDiagnostics:
    variance_stack: TileVarianceStack
    coarse_focal_indices: np.ndarray  # (num_tiles_y, num_tiles_x)
    fine_focus_indices: np.ndarray  # (Ny, Nx)


def compute_focus_diagnostics(
    volume: LoadedVolume, params: FlattenFieldParams
) -> FocusDiagnostics:
    """Stages A-C: variance stack -> per-tile peaks -> fitted surface."""
    _Nz, Ny, Nx = volume.img3d.shape
    variance_stack = compute_tile_variance_stack(
        volume.img3d, params.num_tiles_y, params.num_tiles_x
    )
    coarse_focal_indices = peaks_from_variance_stack(
        variance_stack, params.inverted_variance_prominence
    )
    fine_focus_indices = fit_focal_indices_to_poly2d(
        coarse_focal_indices,
        Nx,
        Ny,
        variance_stack.tile_info,
        poly_degree=params.poly_degree,
    )
    return FocusDiagnostics(
        variance_stack=variance_stack,
        coarse_focal_indices=coarse_focal_indices,
        fine_focus_indices=fine_focus_indices,
    )


def compute_focal_slice(
    volume: LoadedVolume, diagnostics: FocusDiagnostics, params: FlattenFieldParams
) -> np.ndarray:
    """Stage D: resample the flattened focal-plane image."""
    dz_um = volume.scale[0]
    return resample_focal_slice(
        volume.img3d,
        diagnostics.fine_focus_indices,
        dz_um=dz_um,
        offset_um=params.offset_um,
    )


def sum_project(volume: LoadedVolume) -> np.ndarray:
    return np.sum(volume.projection3d, axis=0).astype(np.uint16)


# Channel order within the combined 2D output tiff -- the one place this
# convention is defined. Segmentation (core/segmentation.py) only ever
# reads BRIGHTFIELD_CHANNEL; a later tile-cropping stage would read both.
BRIGHTFIELD_CHANNEL = 0
TARGET_CHANNEL = 1


def combine_channels(focal_slice: np.ndarray, projected: np.ndarray) -> np.ndarray:
    """Stack the flattened brightfield slice and the sum-projected target
    channel into one (2, Ny, Nx) array -- BRIGHTFIELD_CHANNEL first,
    TARGET_CHANNEL second. One file per FOV instead of two parallel
    folders matched by filename stem: fewer places for the two halves of
    a FOV to drift out of sync, and the Cellpose GUI already supports
    picking which channel to segment on when opening a multi-channel
    tiff, so nothing is lost for manual correction either."""
    return np.stack([focal_slice, projected], axis=0)


@dataclass
class ProcessResult:
    path: Path
    success: bool
    output_path: Path | None = None
    error: str | None = None


def process_and_save(
    path: Path,
    outdir: Path,
    params: FlattenFieldParams = FlattenFieldParams(),
    channels: ChannelSelection = DEFAULT_CHANNELS,
) -> ProcessResult:
    """Full pipeline for one file: load, flatten, sum-project, save both
    channels combined into a single tiff. Catches any exception so a
    batch run over many files doesn't abort on the first bad one (unlike
    the original notebook loop)."""
    path = Path(path)
    outdir = Path(outdir)
    try:
        outdir.mkdir(parents=True, exist_ok=True)

        volume = load_volume(path, channels)
        diagnostics = compute_focus_diagnostics(volume, params)
        focal_slice = compute_focal_slice(volume, diagnostics, params)
        projected = sum_project(volume)
        combined = combine_channels(focal_slice, projected)

        output_path = outdir / f"{path.stem}.tiff"
        save_tiff(output_path, combined, scale=volume.scale, input_axes="CYX")

        return ProcessResult(path=path, success=True, output_path=output_path)
    except Exception as exc:
        return ProcessResult(path=path, success=False, error=str(exc))


def load_brightfield_channel(path) -> np.ndarray:
    """Read just the brightfield channel back out of a combined-channel
    tiff written by `process_and_save` -- the common case: segmentation
    (core/segmentation.py) and every current display panel only ever
    need this one channel, never the target/DAPI one."""
    return tifffile.imread(path)[BRIGHTFIELD_CHANNEL]


def find_processed_output(raw_path, output_folder) -> Path | None:
    """Pure filesystem check: has `raw_path` already been flattened into
    `output_folder` (in the "processed" subfolder), with output still
    newer than the source file?

    Returns the combined-channel tiff's path if so, else None. Shared by
    the UI layer's "is this file processed" status dot (file_list_panel)
    and its "load the saved output instead of recomputing" fast-inspect
    path (ui/pages/data_reduction_page.py) -- one rule, defined once,
    with no Qt dependency.
    """
    raw_path = Path(raw_path)
    output_path = Path(output_folder) / "processed" / f"{raw_path.stem}.tiff"
    if not (raw_path.exists() and output_path.exists()):
        return None
    if output_path.stat().st_mtime >= raw_path.stat().st_mtime:
        return output_path
    return None
