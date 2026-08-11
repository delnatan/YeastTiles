"""Per-output-folder JSON sidecar: `<output_folder>/.yeastprep_params.json`.

Mirrors tileclass/data/annotations.py's convention of a folder-scoped
sidecar for reproducibility -- tuned params (prominence, poly degree,
offset) and channel order are legitimately per-acquisition-session, not
just per-app-global (see ui/settings.py for the global fallback).

Scoped to the 3D->2D flatten stage only -- see segmentation_folder_config.py
for the Segmentation stage's own, separately-scoped sidecar. The two
stages read/write different folders (design.md's stage boundary), so their
persisted config is deliberately not shared.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from yeastprep.core.channels import ChannelSelection
from yeastprep.core.pipeline import FlattenFieldParams

_SIDECAR_NAME = ".yeastprep_params.json"


def sidecar_path(output_folder) -> Path:
    return Path(output_folder) / _SIDECAR_NAME


def load_folder_config(output_folder) -> dict | None:
    """Returns {'params': FlattenFieldParams, 'channels': ChannelSelection,
    'last_run': str | None, 'files_processed': list[str]}, or None if no
    sidecar exists / it's unreadable."""
    path = sidecar_path(output_folder)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    raw_params = data.get("params", {})
    defaults = FlattenFieldParams()
    params = FlattenFieldParams(
        num_tiles_y=int(raw_params.get("num_tiles_y", defaults.num_tiles_y)),
        num_tiles_x=int(raw_params.get("num_tiles_x", defaults.num_tiles_x)),
        inverted_variance_prominence=float(
            raw_params.get(
                "inverted_variance_prominence", defaults.inverted_variance_prominence
            )
        ),
        poly_degree=tuple(raw_params.get("poly_degree", list(defaults.poly_degree))),
        offset_um=float(raw_params.get("offset_um", defaults.offset_um)),
    )

    raw_channels = data.get("channels")
    channels = (
        ChannelSelection(
            brightfield=int(raw_channels["brightfield"]),
            projection=int(raw_channels["projection"]),
        )
        if raw_channels
        else None
    )

    return {
        "params": params,
        "channels": channels,
        "last_run": data.get("last_run"),
        "files_processed": list(data.get("files_processed", [])),
    }


def save_folder_config(
    output_folder,
    params: FlattenFieldParams,
    channels: ChannelSelection,
    files_processed: list[str] | None = None,
):
    path = sidecar_path(output_folder)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_folder_config(output_folder)
    all_processed = set(existing["files_processed"]) if existing else set()
    if files_processed:
        all_processed.update(files_processed)

    data = {
        "params": {
            "num_tiles_y": params.num_tiles_y,
            "num_tiles_x": params.num_tiles_x,
            "inverted_variance_prominence": params.inverted_variance_prominence,
            "poly_degree": list(params.poly_degree),
            "offset_um": params.offset_um,
        },
        "channels": {
            "brightfield": channels.brightfield,
            "projection": channels.projection,
        },
        "last_run": datetime.now(timezone.utc).isoformat(),
        "files_processed": sorted(all_processed),
    }
    path.write_text(json.dumps(data, indent=2))
