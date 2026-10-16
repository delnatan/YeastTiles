"""Reusable field-flattening driver.

Generalizes notebooks/01_flatten_field.ipynb's notebook-only `process_file`
into functions callable from the yeastprep UI (which needs the
intermediate diagnostics, not just the final files) and from the batch CLI
(`core/cli.py`, which only needs `process_and_save`).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pyvistra.io import load_image

from .channels import ChannelSelection
from .combined_tiff import write_combined_tiff
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
    offset_um: float = 1.0


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
    """Sum the target channel over z, kept as float32 raw summed counts --
    no offset subtraction, no rescale. A full-stack sum easily exceeds
    65535 (a plain uint16 `astype` wrapped bright nuclei around to dim
    ones), and squeezing it into uint16 here would cost deconvolution
    twice: its Poisson model wants counts, not an arbitrary per-FOV
    stretch, and sharpened peaks above an input already pinned at 65535
    had nowhere to go but the clip. The camera offset is left in too: it's
    a flat camera_offset x Nz pedestal, which deconvolution models as its
    background (core/deconvolve.py's `effective_background`) rather than
    having it clipped away slice by slice, which biased the noise. It
    stays float32 counts through deconvolution too: no stage ever min-max
    rescales it (one zinger would crush every dim cell), and contrast is
    set by percentile at tile export. A 2D float32 plane is still a
    fraction of the raw stack's size."""
    summed = np.zeros(volume.projection3d.shape[1:], dtype=np.float64)
    for plane in volume.projection3d:  # one slice at a time: no float copy of the stack
        summed += plane
    return summed.astype(np.float32)


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

        output_path = outdir / f"{path.stem}.tiff"
        write_combined_tiff(
            output_path,
            focal_slice,
            projected,
            volume.scale,
            n_z=volume.projection3d.shape[0],
        )

        return ProcessResult(path=path, success=True, output_path=output_path)
    except Exception as exc:
        return ProcessResult(path=path, success=False, error=str(exc))


