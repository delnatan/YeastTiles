"""Reusable field-flattening driver.

Generalizes notebooks/01_flatten_field.ipynb's notebook-only `process_file`
into functions callable from the yeastprep UI (which needs the
intermediate diagnostics, not just the final files) and from the batch CLI
(`core/cli.py`, which only needs `process_and_save`).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pyvistra.io import load_image, save_tiff

from .channels import ChannelSelection
from .combined_tiff import combine_channels
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
    # Subtracted from every z-slice of the target channel before summing,
    # each slice clipped at zero -- see `sum_project`.
    projection_baseline: float = 0.0


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


def sum_project(volume: LoadedVolume, baseline: float = 0.0) -> np.ndarray:
    """Sum the target channel over z, then min-max rescale it onto the full
    uint16 range. A full-stack sum easily exceeds 65535 (a plain `astype`
    wrapped bright nuclei around to dim ones), and absolute intensity isn't
    meaningful downstream anyway -- the rescale just keeps the sum's
    relative intensities, at full 16-bit precision, so deconvolution
    output looks natural. Per FOV, so intensities aren't comparable across
    FOVs.

    `baseline` (e.g. the camera's dark offset, in raw counts) is
    subtracted from each z-slice with the result clipped at zero, so
    out-of-focus slices that sit at the offset add nothing to the sum
    instead of a pedestal under the signal. The clip is what makes it do
    anything: an unclipped constant would come straight back out in the
    min-max rescale. It also discards the below-baseline half of the noise
    wherever a slice sits right at the baseline, so set it at (or a little
    below) the offset, not up into real background."""
    summed = np.zeros(volume.projection3d.shape[1:], dtype=np.float64)
    for plane in volume.projection3d:  # one slice at a time: no float copy of the stack
        summed += np.maximum(plane.astype(np.float64) - baseline, 0.0)
    lo, hi = summed.min(), summed.max()
    if hi <= lo:
        return np.zeros(summed.shape, dtype=np.uint16)
    scaled = (summed - lo) * (np.iinfo(np.uint16).max / (hi - lo))
    return np.round(scaled).astype(np.uint16)


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
        projected = sum_project(volume, params.projection_baseline)
        combined = combine_channels(focal_slice, projected)

        output_path = outdir / f"{path.stem}.tiff"
        save_tiff(output_path, combined, scale=volume.scale, input_axes="CYX")

        return ProcessResult(path=path, success=True, output_path=output_path)
    except Exception as exc:
        return ProcessResult(path=path, success=False, error=str(exc))


