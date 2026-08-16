"""Denoise driver: an optional jssl-denoise pass over a combined-channel
tiff, writing into 02_denoised/. Split out of the old core/enhancement.py
so denoising and deconvolution are independently toggleable stages (see
core/deconvolve.py for the other half).

Brightfield and the target/fluorescence channel are different imaging
modalities, so each needs its own trained checkpoint -- denoising is done
one channel per run (`DenoiseParams.channel`), not both channels sharing
one checkpoint. `denoise_and_save` merges its result into whatever's
already in 02_denoised/ for that file rather than overwriting it wholesale,
so denoising channel 0 today and channel 1 tomorrow doesn't clobber
yesterday's result.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jssl_denoise import Denoiser
from pyvistra.io import save_tiff
from tileclass.classifiers.device import select_device

from .combined_tiff import BRIGHTFIELD_CHANNEL, TARGET_CHANNEL, combine_channels, read_combined_tiff


@dataclass
class DenoiseParams:
    channel: int = BRIGHTFIELD_CHANNEL
    checkpoint_path: str | None = None
    tta: bool = True


# The filename convention `ui/pages/train_denoise_page.py` defaults a freshly
# trained checkpoint's save path to (project_root/denoise_<slug>.pt) --
# defined here, not there, so `find_project_checkpoint` below (used by
# DenoisePage to auto-fill checkpoint fields for a project that already has
# checkpoints sitting in it) reads the exact same convention rather than a
# second hand-copied one that could drift out of sync.
_CHECKPOINT_CHANNEL_SLUGS = {BRIGHTFIELD_CHANNEL: "brightfield", TARGET_CHANNEL: "target"}


def checkpoint_filename_for_channel(channel: int) -> str:
    slug = _CHECKPOINT_CHANNEL_SLUGS.get(channel, "channel")
    return f"denoise_{slug}.pt"


def find_project_checkpoint(project_root, channel: int) -> str | None:
    """A `checkpoint_filename_for_channel(channel)` file sitting directly in
    `project_root`, if one exists -- lets DenoisePage offer a whole
    project's worth of files for batch denoising without the user having to
    manually Browse to (or previously have saved) a checkpoint path first,
    as long as the checkpoint follows the naming convention a Training-tab
    run already saves to by default."""
    path = Path(project_root) / checkpoint_filename_for_channel(channel)
    return str(path) if path.is_file() else None


def _default_device() -> torch.device:
    return select_device()


_denoiser_cache: dict[tuple[str, str], Denoiser] = {}


def get_denoiser(checkpoint_path: str, device: torch.device | None = None) -> Denoiser:
    """Load (or reuse a cached) Denoiser. Loading pulls weights onto the
    device, so callers that recompute repeatedly (live preview) should
    keep reusing the same checkpoint_path rather than reloading per call."""
    device = device or _default_device()
    key = (str(checkpoint_path), str(device))
    denoiser = _denoiser_cache.get(key)
    if denoiser is None:
        denoiser = Denoiser.load(checkpoint_path, device=str(device))
        _denoiser_cache[key] = denoiser
    return denoiser


def run_denoise(image: np.ndarray, params: DenoiseParams) -> np.ndarray:
    """Denoise one channel's image with `params.checkpoint_path`. Casts back
    to uint16 so 02_denoised/ stays a consistent 16-bit drop-in replacement
    for 01_reduced/, matching the project's "16-bit for 2D intermediates"
    convention (jssl-denoise returns float)."""
    if not params.checkpoint_path:
        raise ValueError("checkpoint_path is required to denoise")
    denoiser = get_denoiser(params.checkpoint_path)
    denoised, _ = denoiser.denoise(image, tta=params.tta)
    return np.clip(denoised, 0, 65535).astype(np.uint16)


@dataclass
class DenoiseProcessResult:
    path: Path
    success: bool
    output_path: Path | None = None
    error: str | None = None


def denoise_and_save(
    path: Path, outdir: Path, params: DenoiseParams = DenoiseParams()
) -> DenoiseProcessResult:
    """Denoise the selected channel of one combined-channel tiff and write
    the result (same 2-channel layout) into `outdir` (02_denoised/). If
    that file already has a saved output there -- e.g. the other channel
    was denoised in an earlier run -- the untouched channel is carried over
    from that existing output rather than from the raw `path` source, so
    running channel 0 today and channel 1 tomorrow doesn't lose either
    result. Catches any exception so a batch run doesn't abort on the first
    bad file (matches core/pipeline.py's process_and_save and
    core/segmentation.py's segment_and_save)."""
    path = Path(path)
    outdir = Path(outdir)
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        output_path = outdir / f"{path.stem}.tiff"

        brightfield, target, scale = read_combined_tiff(path)
        if output_path.exists():
            existing_brightfield, existing_target, _ = read_combined_tiff(output_path)
        else:
            existing_brightfield, existing_target = brightfield, target

        if params.channel == BRIGHTFIELD_CHANNEL:
            brightfield = run_denoise(brightfield, params)
            target = existing_target
        else:
            target = run_denoise(target, params)
            brightfield = existing_brightfield

        combined = combine_channels(brightfield, target)
        save_tiff(output_path, combined, scale=scale, input_axes="CYX")

        return DenoiseProcessResult(path=path, success=True, output_path=output_path)
    except Exception as exc:
        return DenoiseProcessResult(path=path, success=False, error=str(exc))
