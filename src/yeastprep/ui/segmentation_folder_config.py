"""Per-2D-images-folder JSON sidecar: `<folder>/.yeastprep_segmentation_params.json`.

Separately scoped from folder_config.py's flatten-stage sidecar: the
Segmentation tab's 2D images folder (SegmentationFolderPanel) is
insulated from the flatten stage's output folder (design.md's stage
boundary between 3D->2D data reduction and 2D processing), so the tuned
segmentation params for that folder shouldn't live in -- or require --
the flatten stage's sidecar.
"""

import json
from pathlib import Path

from yeastprep.core.segmentation import SegmentationParams

_SIDECAR_NAME = ".yeastprep_segmentation_params.json"


def sidecar_path(folder) -> Path:
    return Path(folder) / _SIDECAR_NAME


def load_segmentation_folder_config(folder) -> SegmentationParams | None:
    path = sidecar_path(folder)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    defaults = SegmentationParams()
    return SegmentationParams(
        model_path=data.get("model_path", defaults.model_path),
        diameter=float(data.get("diameter", defaults.diameter)),
        flow_threshold=float(data.get("flow_threshold", defaults.flow_threshold)),
        cellprob_threshold=float(
            data.get("cellprob_threshold", defaults.cellprob_threshold)
        ),
        remove_edge_masks=bool(
            data.get("remove_edge_masks", defaults.remove_edge_masks)
        ),
    )


def save_segmentation_folder_config(folder, params: SegmentationParams):
    path = sidecar_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "model_path": params.model_path,
        "diameter": params.diameter,
        "flow_threshold": params.flow_threshold,
        "cellprob_threshold": params.cellprob_threshold,
        "remove_edge_masks": params.remove_edge_masks,
    }
    path.write_text(json.dumps(data, indent=2))
