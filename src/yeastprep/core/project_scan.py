"""Pure (Qt-free) filesystem scan behind `ui/project_tree_panel.py`'s tree
and `ui/common/stage_breadcrumb.py`'s pipeline chips.

Globbing every stage folder and `stat()`-ing every file in it (to compare
mtimes against each stage's producer folder) is the expensive part of
opening a project -- on a slow external/network drive, each of those is a
real disk round trip, and a project with hundreds of raw stacks can turn
into thousands of them. Kept here as one plain function with no Qt
dependency so `ui/worker.py`'s `SimplePipelineWorker` can run it on a
background `QThread` (see `ProjectScanController`) instead of blocking the
GUI thread -- previously `ProjectTreePanel.refresh()` and
`PipelineBreadcrumb.refresh()` each did this same walk synchronously,
independently, on every project open.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import project as project_core
from . import stages as stages_core
from . import tiles as tiles_core
from .combined_tiff import CELLPOSE_GUI_SUBDIR
from .fs_status import list_visible

STAGE_RAW = stages_core.STAGE_RAW
_2D_STAGES = (
    project_core.STAGE_REDUCED,
    project_core.STAGE_DENOISED,
    project_core.STAGE_DECONVOLVED,
)

_DIGITS = re.compile(r"(\d+)")


def natsort_key(path: Path):
    parts = _DIGITS.split(path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


@dataclass
class LeafInfo:
    kind: str  # "file" | "tile_fov"
    stage: str
    path: str  # str(file path) for "file"; fov_id for "tile_fov"
    display_name: str
    icon_state: str
    tooltip: str = ""
    seg_icon_state: str | None = None
    seg_tooltip: str = ""


@dataclass
class StageScanResult:
    stage: str
    exists: bool
    is_active_2d: bool = False
    empty_msg: str = ""
    count: int = 0
    extra_count_text: str = ""
    leaves: list[LeafInfo] = field(default_factory=list)


@dataclass
class ProjectScanSnapshot:
    root: str
    active_2d_stage: str | None
    stages: dict[str, StageScanResult]
    pipeline_states: list  # list[stages_core.StageState]


def _stage_dir(root: Path, stage: str) -> Path | None:
    if stage == STAGE_RAW:
        return root
    return root / stage


def _producer_dir(root: Path, stage: str, config: project_core.ProjectConfig | None) -> Path | None:
    """Mirrors `ui/project_tree_panel.py`'s old `_producer_dir` /
    `core/stages.py`'s `_producer_dir` -- whichever folder feeds *into*
    `stage`, for staleness comparison."""
    if stage == project_core.STAGE_REDUCED:
        return root
    if stage == project_core.STAGE_DENOISED:
        return root / project_core.STAGE_REDUCED
    if stage == project_core.STAGE_DECONVOLVED:
        paths = project_core.ProjectPaths(root)
        override = config.deconvolve_source_stage if config else None
        return project_core.resolve_deconvolve_source(paths, override)
    return None


def scan_project(
    root: str, raw_pattern: str, segmentation_override: str | None, stage_keys
) -> ProjectScanSnapshot:
    root_path = Path(root)
    paths = project_core.ProjectPaths(root_path)
    config = project_core.load_project_config(root_path)

    active_source = project_core.resolve_2d_source(paths, segmentation_override)
    active_2d_stage = active_source.name if active_source is not None else None

    stages = {
        stage: _scan_stage(root_path, stage, raw_pattern, active_2d_stage, config)
        for stage in stage_keys
    }
    pipeline_states = stages_core.pipeline_status(paths, config)

    return ProjectScanSnapshot(
        root=str(root_path),
        active_2d_stage=active_2d_stage,
        stages=stages,
        pipeline_states=pipeline_states,
    )


def _scan_stage(
    root: Path,
    stage: str,
    raw_pattern: str,
    active_2d_stage: str | None,
    config: project_core.ProjectConfig | None,
) -> StageScanResult:
    is_active = stage == active_2d_stage
    folder = _stage_dir(root, stage)
    if folder is None or not folder.is_dir():
        empty_msg = "no project open" if stage == STAGE_RAW else "not created yet"
        return StageScanResult(stage=stage, exists=False, is_active_2d=is_active, empty_msg=empty_msg)

    if stage == project_core.STAGE_TILES:
        return _scan_tiles(root, folder, active_2d_stage)

    pattern = (raw_pattern or "*.ims") if stage == STAGE_RAW else "*.tiff"
    file_paths = sorted(list_visible(folder, pattern), key=natsort_key)

    producer_dir = _producer_dir(root, stage, config) if stage in _2D_STAGES else None
    statuses = project_core.stage_file_status(folder, upstream_dir=producer_dir)
    reduced_dir = _stage_dir(root, project_core.STAGE_REDUCED) if stage == STAGE_RAW else None

    denoise_channels_done: dict[str, list[int]] = {}
    if stage == project_core.STAGE_DENOISED and config:
        denoise_channels_done = config.denoise_channels_done

    seg_statuses = {}
    if stage in _2D_STAGES and is_active:
        # _seg.npy sidecars live in the brightfield-companion subfolder,
        # not flat in `folder` -- see combined_tiff.CELLPOSE_GUI_SUBDIR.
        seg_statuses = project_core.segmentation_file_status(folder / CELLPOSE_GUI_SUBDIR)

    leaves = []
    for path in file_paths:
        tooltip = ""
        if stage in _2D_STAGES:
            status = statuses.get(path.stem)
            up_to_date = bool(status and status.up_to_date)
            upstream_available = bool(status and status.upstream_available)
            if not upstream_available:
                icon_state = "archived"
                tooltip = (
                    "Source folder for this stage isn't present on this computer "
                    "(likely moved to storage) -- can't verify freshness, showing "
                    "as up to date."
                )
            elif stage == project_core.STAGE_DENOISED and up_to_date:
                n_channels = len(denoise_channels_done.get(path.stem, ()))
                icon_state = "partial" if n_channels == 1 else "done"
            else:
                icon_state = "done" if up_to_date else "stale"
        elif stage == STAGE_RAW:
            produced = reduced_dir is not None and (reduced_dir / f"{path.stem}.tiff").exists()
            icon_state = "done" if produced else "unprocessed"
        else:
            icon_state = "done"

        seg_icon_state = None
        seg_tooltip = ""
        seg_status = seg_statuses.get(path.stem)
        if seg_status is not None:
            seg_icon_state = "done" if seg_status.up_to_date else "stale"
            seg_tooltip = (
                "Segmented, up to date"
                if seg_status.up_to_date
                else "Segmented, but the source file changed since"
            )

        leaves.append(
            LeafInfo(
                kind="file",
                stage=stage,
                path=str(path),
                display_name=path.name,
                icon_state=icon_state,
                tooltip=tooltip,
                seg_icon_state=seg_icon_state,
                seg_tooltip=seg_tooltip,
            )
        )

    return StageScanResult(stage=stage, exists=True, is_active_2d=is_active, count=len(leaves), leaves=leaves)


def _scan_tiles(root: Path, folder: Path, active_2d_stage: str | None) -> StageScanResult:
    stage_dir = _stage_dir(root, active_2d_stage) if active_2d_stage else None
    # _seg.npy sidecars live in the brightfield-companion subfolder, not
    # flat in stage_dir -- see combined_tiff.CELLPOSE_GUI_SUBDIR.
    mask_dir = stage_dir / CELLPOSE_GUI_SUBDIR if stage_dir is not None else None
    fov_statuses = tiles_core.fov_tile_status(folder, mask_dir)
    n_cells = sum(status.n_cells for status in fov_statuses.values())

    leaves = []
    for fov_id in sorted(fov_statuses, key=lambda stem: natsort_key(Path(stem))):
        status = fov_statuses[fov_id]
        n = status.n_cells
        tooltip = ""
        if not status.mask_available:
            icon_state = "archived"
            tooltip = (
                "Source/mask folder for this FOV isn't present on this computer "
                "(likely moved to storage) -- can't verify freshness, showing as "
                "up to date."
            )
        elif status.up_to_date:
            icon_state = "done"
        else:
            icon_state = "stale"
            tooltip = "Segmentation for this FOV changed since these tiles were exported."

        leaves.append(
            LeafInfo(
                kind="tile_fov",
                stage=project_core.STAGE_TILES,
                path=fov_id,
                display_name=f"{fov_id}  ({n} cell{'s' if n != 1 else ''})",
                icon_state=icon_state,
                tooltip=tooltip,
            )
        )

    return StageScanResult(
        stage=project_core.STAGE_TILES,
        exists=True,
        is_active_2d=False,
        count=len(leaves),
        extra_count_text=f", {n_cells} cell(s)",
        leaves=leaves,
    )
