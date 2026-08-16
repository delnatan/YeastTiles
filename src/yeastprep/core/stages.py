"""The linear-but-branching pipeline sequence (raw -> reduced -> [denoise]
-> [deconvolve] -> segment -> tile) as data, so the tree panel and the
breadcrumb widget both read stage order/labels/status from one place
instead of each hand-rolling it (as `ui/project_tree_panel.py`'s
`_STAGE_ORDER`/`_STAGE_LABELS` and each page's own input-stage assumptions
used to).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import project

STAGE_RAW = "raw"
# Virtual stage: segmentation writes `_seg.npy` sidecars in place next to
# whichever 2D stage is its source (see project.segmentation_mask_dir), it
# has no numbered folder of its own.
STAGE_SEGMENTATION = "segmentation"


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    optional: bool  # True only for the denoise/deconvolve stages


PIPELINE: tuple[StageSpec, ...] = (
    StageSpec(STAGE_RAW, "Raw", optional=False),
    StageSpec(project.STAGE_REDUCED, "Reduced", optional=False),
    StageSpec(project.STAGE_DENOISED, "Denoise", optional=True),
    StageSpec(project.STAGE_DECONVOLVED, "Deconvolve", optional=True),
    StageSpec(STAGE_SEGMENTATION, "Segment", optional=False),
    StageSpec(project.STAGE_TILES, "Tile", optional=False),
)


@dataclass(frozen=True)
class StageState:
    key: str
    label: str
    optional: bool
    status: str  # "empty" | "stale" | "done"
    # Whether this is the stage Segmentation/Tile Generation currently read
    # from (project.resolve_2d_source) -- kept separate from `status` so a
    # stale-but-active stage doesn't lose its "stale" warning.
    active: bool


def _has_any(folder: Path, pattern: str) -> bool:
    return folder.is_dir() and any(folder.glob(pattern))


def _producer_dir(
    paths: project.ProjectPaths, stage_key: str, config: project.ProjectConfig
) -> Path | None:
    """Whichever folder feeds *into* `stage_key`, for staleness comparison.
    Mirrors `ui/project_tree_panel.py`'s `_producer_dir` -- kept here too
    since `pipeline_status` needs the same rule and shouldn't import a Qt
    widget module to get it."""
    if stage_key == project.STAGE_REDUCED:
        return paths.root
    if stage_key == project.STAGE_DENOISED:
        return paths.reduced
    if stage_key == project.STAGE_DECONVOLVED:
        return project.resolve_deconvolve_source(paths, config.deconvolve_source_stage)
    return None


def _folder_stage_status(
    paths: project.ProjectPaths, stage_key: str, config: project.ProjectConfig
) -> str:
    stage_dir = paths.stage_dir(stage_key)
    if not _has_any(stage_dir, "*.tiff"):
        return "empty"
    producer = _producer_dir(paths, stage_key, config)
    statuses = project.stage_file_status(stage_dir, upstream_dir=producer)
    if statuses and all(s.up_to_date for s in statuses.values()):
        return "done"
    return "stale"


def pipeline_status(
    paths: project.ProjectPaths, config: project.ProjectConfig | None = None
) -> list[StageState]:
    """Per-stage status for the whole pipeline, composed entirely from
    existing primitives (`stage_file_status`, `resolve_2d_source`, plain
    globs) -- no new file-scanning logic."""
    config = config or project.ProjectConfig()
    active_source = project.resolve_2d_source(paths, config.segmentation_source_stage)
    active_key = active_source.name if active_source is not None else None

    states = []
    for spec in PIPELINE:
        if spec.key == STAGE_RAW:
            status = "done" if _has_any(paths.root, config.raw_pattern or "*.ims") else "empty"
        elif spec.key == STAGE_SEGMENTATION:
            mask_dir = project.segmentation_mask_dir(paths, config.segmentation_source_stage)
            has_masks = mask_dir is not None and _has_any(mask_dir, "*_seg.npy")
            status = "done" if has_masks else "empty"
        elif spec.key == project.STAGE_TILES:
            status = "done" if _has_any(paths.tiles, "*.tif") else "empty"
        else:
            status = _folder_stage_status(paths, spec.key, config)

        states.append(
            StageState(
                key=spec.key,
                label=spec.label,
                optional=spec.optional,
                status=status,
                active=(spec.key == active_key),
            )
        )
    return states


def input_stage_for(
    stage_key: str,
    paths: project.ProjectPaths,
    config: project.ProjectConfig | None = None,
) -> str | None:
    """The stage key that `stage_key` should read its 2D input from, given
    the current project state -- centralizes the input-resolution rule each
    page previously reimplemented ad hoc (Denoise always reads
    `STAGE_REDUCED`; Deconvolve reads denoised-else-reduced; Segmentation
    and Tile Generation both read whichever stage `resolve_2d_source`
    picks)."""
    config = config or project.ProjectConfig()
    if stage_key == project.STAGE_DENOISED:
        return project.STAGE_REDUCED
    if stage_key == project.STAGE_DECONVOLVED:
        upstream = project.resolve_deconvolve_source(paths, config.deconvolve_source_stage)
        return upstream.name if upstream is not None else None
    if stage_key in (STAGE_SEGMENTATION, project.STAGE_TILES):
        source = project.resolve_2d_source(paths, config.segmentation_source_stage)
        return source.name if source is not None else None
    return None
