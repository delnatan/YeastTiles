"""Per-tiles-output-folder JSON sidecar: `<output_folder>/.yeastprep_tile_params.json`.

Scoped to the *output* folder rather than the 2D-images-plus-masks input
folder: unlike segmentation_folder_config.py's params (which only affect
that one folder's masks), tile size/contrast here define the shape of the
corpus accumulating in the output folder -- re-running export against a
different input folder into the same output folder should still reuse
these params, so every tile written there stays a consistent (size, size)
and comparable contrast-wise (design.md's "pooling" requirement).
"""

import json
from pathlib import Path

from yeastprep.core.tiles import TileParams

_SIDECAR_NAME = ".yeastprep_tile_params.json"


def sidecar_path(output_folder) -> Path:
    return Path(output_folder) / _SIDECAR_NAME


def load_tile_folder_config(output_folder) -> TileParams | None:
    path = sidecar_path(output_folder)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    defaults = TileParams()
    return TileParams(
        size=int(data.get("size", defaults.size)),
        contrast_lo_pct=float(data.get("contrast_lo_pct", defaults.contrast_lo_pct)),
        contrast_hi_pct=float(data.get("contrast_hi_pct", defaults.contrast_hi_pct)),
    )


def save_tile_folder_config(output_folder, params: TileParams):
    path = sidecar_path(output_folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "size": params.size,
        "contrast_lo_pct": params.contrast_lo_pct,
        "contrast_hi_pct": params.contrast_hi_pct,
    }
    path.write_text(json.dumps(data, indent=2))
