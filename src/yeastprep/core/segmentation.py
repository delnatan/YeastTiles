"""Cellpose driver for the segmentation stage: takes the brightfield
channel of the combined-channel tiff produced by `core/pipeline.py` and
finds yeast cells. The target/DAPI channel in that same file is never
touched here -- segmentation is brightfield-only.

Restructures notebooks/02_cookie_cut.ipynb's model-load + `.eval(...)` +
`remove_edge_masks` cells into functions callable from the yeastprep UI
(live preview as `offset_um` etc. are tuned -- the focal plane choice
directly affects segmentation quality) and from a batch runner.

Mask correction itself is *not* reimplemented here: `segment_and_save`
writes cellpose's own `_seg.npy` sidecar next to each FOV's brightfield-only
companion tiff (`combined_tiff.write_brightfield_tiff`; via
`cellpose.io.masks_flows_to_seg`, the same helper the Cellpose GUI's "save"
action uses), so the real Cellpose GUI can open that companion file
directly for manual correction, per design.md. Cellpose 4's SAM model
dropped channel selection, so the GUI needs a real single-channel file --
opening the 2-channel combined tiff directly would segment/train on the
target channel too.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from cellpose import io as cp_io
from cellpose import utils as cp_utils
from cellpose.models import CellposeModel

from tileclass.classifiers.device import select_device

from .combined_tiff import brightfield_tiff_path, load_brightfield_channel, write_brightfield_tiff

DEFAULT_MODEL_PATH: str | None = None  # None -> cellpose's built-in default (cpsam)


@dataclass
class SegmentationParams:
    model_path: str | None = DEFAULT_MODEL_PATH
    diameter: float = 30.0
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    remove_edge_masks: bool = True


def _default_device() -> torch.device:
    return select_device()


_model_cache: dict[tuple[str | None, str], CellposeModel] = {}


def get_model(model_path: str | None, device: torch.device | None = None) -> CellposeModel:
    """Load (or reuse a cached) CellposeModel. Loading pulls weights onto
    the GPU, so callers that recompute repeatedly (live preview) should
    keep reusing the same model_path rather than reloading per call."""
    device = device or _default_device()
    key = (model_path, str(device))
    model = _model_cache.get(key)
    if model is None:
        model = CellposeModel(pretrained_model=model_path or "cpsam", device=device)
        _model_cache[key] = model
    return model


@dataclass
class SegmentationResult:
    masks: np.ndarray  # (Ny, Nx) int label image, 0 = background
    flows: list
    styles: np.ndarray
    n_cells: int


def run_segmentation(
    model: CellposeModel, image: np.ndarray, params: SegmentationParams
) -> SegmentationResult:
    masks_list, flows_list, styles = model.eval(
        [image],
        diameter=params.diameter,
        flow_threshold=params.flow_threshold,
        cellprob_threshold=params.cellprob_threshold,
    )
    masks = masks_list[0]
    flows = flows_list[0]
    if params.remove_edge_masks:
        masks = cp_utils.remove_edge_masks(masks)
    n_cells = int(masks.max())
    return SegmentationResult(masks=masks, flows=flows, styles=styles, n_cells=n_cells)


@dataclass
class SegmentProcessResult:
    path: Path
    success: bool
    seg_path: Path | None = None
    n_cells: int | None = None
    error: str | None = None


def seg_npy_path(image_path) -> Path:
    """Where cellpose's `_seg.npy` sidecar lives for a given focal-slice
    tiff -- the one place this naming convention is defined, shared by
    `segment_and_save` (which writes it) and the UI's fast-inspect path
    (which reads it back via `load_saved_masks`).

    Sidecars now live next to the brightfield-only companion tiff (see
    `combined_tiff.write_brightfield_tiff`), not the 2-channel combined
    tiff itself, so the real Cellpose GUI's own Save/Train read and write
    the same file our batch path does. Falls back to the pre-existing
    (combined-tiff-adjacent) location if only that one exists, so masks
    saved before this convention changed aren't orphaned."""
    image_path = Path(image_path)
    bf_path = brightfield_tiff_path(image_path)
    bf_seg_path = bf_path.with_name(f"{bf_path.stem}_seg.npy")
    if bf_seg_path.exists():
        return bf_seg_path
    legacy_seg_path = image_path.with_name(f"{image_path.stem}_seg.npy")
    if legacy_seg_path.exists():
        return legacy_seg_path
    return bf_seg_path


def segment_and_save(
    path: Path, model: CellposeModel, params: SegmentationParams = SegmentationParams()
) -> SegmentProcessResult:
    """Segment one combined-channel tiff (brightfield channel only) and
    write cellpose's `<stem>_seg.npy` next to the brightfield-only
    companion tiff (generating it first if missing/stale) -- so the real
    Cellpose GUI, opened on that same companion file, sees these masks for
    correction. Catches any exception so a batch run doesn't abort on the
    first bad file (matches core/pipeline.py's process_and_save)."""
    path = Path(path)
    try:
        image = load_brightfield_channel(path)
        result = run_segmentation(model, image, params)
        bf_path = write_brightfield_tiff(path)
        cp_io.masks_flows_to_seg(
            [image], [result.masks], [result.flows], [str(bf_path)], channels=[0, 0]
        )
        return SegmentProcessResult(
            path=path, success=True, seg_path=seg_npy_path(path), n_cells=result.n_cells
        )
    except Exception as exc:
        return SegmentProcessResult(path=path, success=False, error=str(exc))


def load_saved_masks(seg_path) -> np.ndarray | None:
    """Read back a cellpose-native `_seg.npy` sidecar's masks array, or
    None if it doesn't exist / isn't readable. Centralizes knowledge of
    the seg.npy schema next to the code that writes it, rather than
    leaving `np.load(...).item()["masks"]` inlined in the UI layer."""
    seg_path = Path(seg_path)
    if not seg_path.exists():
        return None
    try:
        return np.load(seg_path, allow_pickle=True).item()["masks"]
    except Exception:
        return None
