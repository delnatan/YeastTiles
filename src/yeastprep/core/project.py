"""Single-project folder layout: the project root IS the folder holding the
raw 3D stacks (`raw_pattern` glob, e.g. `*.ims`) -- there is no separate
"raw input folder" picker. Numbered stage subfolders live right alongside
those raw files in the same root, replacing the old scheme of four
independently picked stage folders connected only by hand-wired "use X
output" buttons in ui/main_window.py.

    <project_root>/
      <stem>.ims / .czi / .nd2   raw multi-channel Z-stacks (untouched)
      01_reduced/        <stem>.tiff   16-bit, 2ch (CYX): [brightfield, target]
      02_denoised/       <stem>.tiff   16-bit, 2ch          -- optional stage
      03_deconvolved/    <stem>.tiff   16-bit, 2ch          -- optional stage
      05_tiles/          <fov>/<fov>_cell#####.tif  8-bit, 3ch + tile_index.csv

Segmentation has no numbered folder of its own: cellpose's own GUI needs
`_seg.npy` sitting directly next to the image it corrects
(core/segmentation.py's `seg_npy_path`), so masks are always written in
the `.cellpose_bf/` subfolder of whichever stage folder is currently the
segmentation source, alongside the brightfield-only companion tiffs
cellpose's GUI actually opens (core/combined_tiff.py's
`write_brightfield_tiff` -- cellpose 4's SAM model dropped channel
selection, so it can't be pointed at the 2-channel tiffs directly) -- see
`resolve_2d_source`/`segmentation_mask_dir`. The numbering deliberately
skips "04" so the visible numbering still communicates that there's a 4th
conceptual stage even though it produces no folder of its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .channels import ChannelSelection
from .combined_tiff import CELLPOSE_GUI_SUBDIR
from .deconvolve import DeconvolveParams
from .denoise import DenoiseParams
from .fs_status import is_hidden, list_visible
from .pipeline import DEFAULT_CHANNELS, FlattenFieldParams
from .segmentation import SegmentationParams
from .tiles import TileParams

STAGE_REDUCED = "01_reduced"
STAGE_DENOISED = "02_denoised"
STAGE_DECONVOLVED = "03_deconvolved"
STAGE_TILES = "05_tiles"

# Downstream-to-upstream priority for "the" 2D source feeding Segmentation /
# Tile Generation -- whichever of these is non-empty and most downstream wins.
STAGE_PRIORITY = (STAGE_DECONVOLVED, STAGE_DENOISED, STAGE_REDUCED)

_CONFIG_NAME = ".yeastprep_project.json"


@dataclass
class ProjectPaths:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)

    @property
    def reduced(self) -> Path:
        return self.root / STAGE_REDUCED

    @property
    def denoised(self) -> Path:
        return self.root / STAGE_DENOISED

    @property
    def deconvolved(self) -> Path:
        return self.root / STAGE_DECONVOLVED

    @property
    def tiles(self) -> Path:
        return self.root / STAGE_TILES

    @property
    def config_path(self) -> Path:
        return self.root / _CONFIG_NAME

    def stage_dir(self, stage: str) -> Path:
        return self.root / stage


def _has_tiffs(folder: Path) -> bool:
    return bool(list_visible(folder, "*.tiff"))


def upstream_2d_stage(paths: ProjectPaths, stages: tuple[str, ...]) -> Path | None:
    """The most-downstream non-empty stage folder among `stages`, in the
    order given. Used both by `resolve_2d_source` (Segmentation/Tile
    Generation's input, any of the three 2D stages) and by the Deconvolve
    stage (its own input is 02_denoised/ if that ran, else 01_reduced/ --
    never 03_deconvolved/, that's its own output)."""
    for stage in stages:
        candidate = paths.stage_dir(stage)
        if _has_tiffs(candidate):
            return candidate
    return None


def resolve_2d_source(paths: ProjectPaths, override: str | None = None) -> Path | None:
    """The 2D-image folder Segmentation/Tile Generation should read from:
    `override` (a STAGE_* name) if given, else the most downstream
    non-empty stage folder (deconvolved > denoised > reduced), else None
    if nothing has been produced yet."""
    if override is not None:
        candidate = paths.stage_dir(override)
        return candidate if _has_tiffs(candidate) else None
    return upstream_2d_stage(paths, STAGE_PRIORITY)


_DECONVOLVE_STAGE_PRIORITY = (STAGE_DENOISED, STAGE_REDUCED)


def resolve_deconvolve_source(paths: ProjectPaths, override: str | None = None) -> Path | None:
    """The 2D-image folder Deconvolve should read from: `override` (REDUCED
    or DENOISED) if given and non-empty, else denoised-if-it-ran else
    reduced. Mirrors `resolve_2d_source` but excludes DECONVOLVED from the
    candidates -- that's this stage's own output, never a valid input."""
    if override is not None:
        candidate = paths.stage_dir(override)
        return candidate if _has_tiffs(candidate) else None
    return upstream_2d_stage(paths, _DECONVOLVE_STAGE_PRIORITY)


def segmentation_mask_dir(
    paths: ProjectPaths, segmentation_source_stage: str | None
) -> Path | None:
    """Whichever `.cellpose_bf/` subfolder currently holds (or should hold)
    `_seg.npy` sidecars: inside the project config's recorded
    `segmentation_source_stage` if set, else inside whichever folder
    `resolve_2d_source` would currently pick."""
    if segmentation_source_stage is not None:
        stage_dir = paths.stage_dir(segmentation_source_stage)
    else:
        stage_dir = resolve_2d_source(paths)
    return stage_dir / CELLPOSE_GUI_SUBDIR if stage_dir is not None else None


@dataclass
class FileStatus:
    exists: bool
    up_to_date: bool  # only meaningful when exists and an upstream dir was given
    # False only when an upstream_dir was given to stage_file_status but it
    # was empty/missing -- distinguishes "can't verify, producer's likely
    # archived elsewhere" from a genuine per-file staleness signal. Always
    # True when no upstream_dir was given at all.
    upstream_available: bool = True


def find_stage_output(stage_dir: Path, source_path) -> Path | None:
    """Single-file convenience over `stage_file_status`: has `source_path`
    already been produced into `stage_dir` (matched by stem), with the
    output still newer than the source? Returns the output path if so,
    else None. Handy in a UI click handler where re-globbing the whole
    folder just to check one file would be wasteful."""
    source_path = Path(source_path)
    output_path = stage_dir / f"{source_path.stem}.tiff"
    if not (source_path.exists() and output_path.exists()):
        return None
    if output_path.stat().st_mtime >= source_path.stat().st_mtime:
        return output_path
    return None


def stage_file_status(
    stage_dir: Path, upstream_dir: Path | None = None
) -> dict[str, FileStatus]:
    """FOV-stem -> FileStatus for every `<stem>.tiff` in `stage_dir`,
    optionally comparing mtimes against a same-stem file in `upstream_dir`
    (matched by stem rather than full name, since an upstream folder can
    hold a different extension -- e.g. 01_reduced/'s raw upstream is
    `*.ims`/`*.czi`/`*.nd2`, not `*.tiff`). Generalizes the near-duplicate
    mtime checks that used to live as `pipeline.find_processed_output` /
    `enhancement.find_enhanced_output` into the one place the tree widget
    needs."""
    statuses: dict[str, FileStatus] = {}
    if not stage_dir.is_dir():
        return statuses

    own_paths = list_visible(stage_dir, "*.tiff")

    upstream_by_stem: dict[str, Path] = {}
    if upstream_dir is not None and upstream_dir.is_dir():
        for upstream_path in upstream_dir.iterdir():
            if upstream_path.is_file() and not is_hidden(upstream_path):
                upstream_by_stem[upstream_path.stem] = upstream_path

    # Whether the upstream dir has *anything relevant* -- deliberately not
    # "has any file at all": `upstream_dir` can be the project root itself
    # (01_reduced/'s upstream), which also holds the config sidecar,
    # denoise checkpoints, `.DS_Store`, other stage folders, etc., none of
    # which will ever share a stem with a real FOV file. Intersecting on
    # stem is what tells "raw stacks archived off this computer" (no
    # overlap at all) apart from "raw stacks still here" (full or partial
    # overlap), regardless of incidental files sitting alongside them.
    own_stems = {p.stem for p in own_paths}
    upstream_available = upstream_dir is None or bool(set(upstream_by_stem) & own_stems)

    for path in own_paths:
        if upstream_dir is None or not upstream_available:
            up_to_date = True
        else:
            upstream_path = upstream_by_stem.get(path.stem)
            up_to_date = (
                upstream_path is not None
                and path.stat().st_mtime >= upstream_path.stat().st_mtime
            )
        statuses[path.stem] = FileStatus(
            exists=True, up_to_date=up_to_date, upstream_available=upstream_available
        )
    return statuses


def segmentation_file_status(mask_dir: Path) -> dict[str, FileStatus]:
    """FOV-stem -> FileStatus for every `<stem>_seg.npy` sidecar in
    `mask_dir`, comparing mtime against its sibling `<stem>.tiff` in the
    same folder -- segmentation masks live next to whichever 2D stage
    produced them (see segmentation_mask_dir), not in a folder of their
    own, so this mirrors stage_file_status's up-to-date check but against
    a sidecar instead of a same-named file in a separate upstream folder."""
    statuses: dict[str, FileStatus] = {}
    if not mask_dir.is_dir():
        return statuses
    for seg_path in list_visible(mask_dir, "*_seg.npy"):
        stem = seg_path.name[: -len("_seg.npy")]
        source_path = mask_dir / f"{stem}.tiff"
        up_to_date = (
            source_path.exists() and seg_path.stat().st_mtime >= source_path.stat().st_mtime
        )
        statuses[stem] = FileStatus(exists=True, up_to_date=up_to_date)
    return statuses


# ----------------------------------------------------------------------
# Project-level config sidecar: .yeastprep_project.json
#
# Replaces the four separate per-folder sidecars (folder_config.py,
# segmentation_folder_config.py, enhance_folder_config.py,
# tile_folder_config.py) with one file at the project root -- there's only
# ever one folder tree to scope config to now.
# ----------------------------------------------------------------------


@dataclass
class ProjectConfig:
    raw_pattern: str = "*.ims"
    channels: ChannelSelection = field(default_factory=lambda: DEFAULT_CHANNELS)
    flatten_params: FlattenFieldParams = field(default_factory=FlattenFieldParams)
    denoise_params: DenoiseParams = field(default_factory=DenoiseParams)
    # channel (0=brightfield, 1=target) -> checkpoint path -- brightfield and
    # the target/fluorescence channel are different imaging modalities, so
    # each needs its own trained network; this is what lets the Denoising
    # tab restore the right checkpoint when you switch which channel you're
    # working on, instead of remembering only whichever channel you saved
    # defaults for last.
    denoise_checkpoints: dict[int, str] = field(default_factory=dict)
    deconvolve_params: DeconvolveParams = field(default_factory=DeconvolveParams)
    segmentation_params: SegmentationParams = field(default_factory=SegmentationParams)
    tile_params: TileParams = field(default_factory=TileParams)
    segmentation_source_stage: str | None = None
    # Override for Deconvolve's own input stage (REDUCED or DENOISED) --
    # None means "auto: denoised if it ran, else reduced" (see
    # `resolve_deconvolve_source`). Separate from `segmentation_source_stage`
    # since Deconvolve's valid choices exclude DECONVOLVED itself (that's
    # its own output), unlike Segmentation/Tile Generation's.
    deconvolve_source_stage: str | None = None
    # stage -> {"last_run": iso timestamp, "files": [stems processed]}
    stage_runs: dict[str, dict] = field(default_factory=dict)
    # 02_denoised file stem -> channels (0=brightfield, 1=target) actually
    # run through jssl-denoise for it, as of the most recent successful
    # batch that touched it -- denoise_and_save merges per-channel
    # (core/denoise.py), so a file existing in 02_denoised/ doesn't by
    # itself say which channel(s) inside it were ever denoised versus just
    # carried over raw from the untouched channel. Lets the tree panel show
    # a half-circle (one channel done) apart from a full one (both).
    denoise_channels_done: dict[str, list[int]] = field(default_factory=dict)


def _flatten_params_from_dict(data: dict, defaults: FlattenFieldParams) -> FlattenFieldParams:
    return FlattenFieldParams(
        num_tiles_y=int(data.get("num_tiles_y", defaults.num_tiles_y)),
        num_tiles_x=int(data.get("num_tiles_x", defaults.num_tiles_x)),
        inverted_variance_prominence=float(
            data.get("inverted_variance_prominence", defaults.inverted_variance_prominence)
        ),
        poly_degree=tuple(data.get("poly_degree", list(defaults.poly_degree))),
        offset_um=float(data.get("offset_um", defaults.offset_um)),
    )


def _denoise_params_from_dict(data: dict, defaults: DenoiseParams) -> DenoiseParams:
    return DenoiseParams(
        channel=int(data.get("channel", defaults.channel)),
        checkpoint_path=data.get("checkpoint_path", defaults.checkpoint_path) or None,
        tta=bool(data.get("tta", defaults.tta)),
    )


def _deconvolve_params_from_dict(data: dict, defaults: DeconvolveParams) -> DeconvolveParams:
    return DeconvolveParams(
        enabled=bool(data.get("enabled", defaults.enabled)),
        psf_path=data.get("psf_path", defaults.psf_path) or None,
        zoom=float(data.get("zoom", defaults.zoom)),
        background=float(data.get("background", defaults.background)),
        num_iter=int(data.get("num_iter", defaults.num_iter)),
    )


def _segmentation_params_from_dict(
    data: dict, defaults: SegmentationParams
) -> SegmentationParams:
    return SegmentationParams(
        model_path=data.get("model_path", defaults.model_path) or None,
        diameter=float(data.get("diameter", defaults.diameter)),
        flow_threshold=float(data.get("flow_threshold", defaults.flow_threshold)),
        cellprob_threshold=float(data.get("cellprob_threshold", defaults.cellprob_threshold)),
        remove_edge_masks=bool(data.get("remove_edge_masks", defaults.remove_edge_masks)),
    )


def _tile_params_from_dict(data: dict, defaults: TileParams) -> TileParams:
    return TileParams(
        size=int(data.get("size", defaults.size)),
        contrast_lo_pct=float(data.get("contrast_lo_pct", defaults.contrast_lo_pct)),
        contrast_hi_pct=float(data.get("contrast_hi_pct", defaults.contrast_hi_pct)),
    )


def load_project_config(root) -> ProjectConfig | None:
    path = ProjectPaths(root).config_path
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    defaults = ProjectConfig()

    channels_raw = data.get("channels")
    channels = (
        ChannelSelection(
            brightfield=int(channels_raw["brightfield"]),
            projection=int(channels_raw["projection"]),
        )
        if channels_raw
        else defaults.channels
    )

    return ProjectConfig(
        raw_pattern=data.get("raw_pattern", defaults.raw_pattern),
        channels=channels,
        flatten_params=_flatten_params_from_dict(
            data.get("flatten_params", {}), defaults.flatten_params
        ),
        denoise_params=_denoise_params_from_dict(
            data.get("denoise_params", {}), defaults.denoise_params
        ),
        denoise_checkpoints={
            int(channel): path for channel, path in data.get("denoise_checkpoints", {}).items()
        },
        deconvolve_params=_deconvolve_params_from_dict(
            data.get("deconvolve_params", {}), defaults.deconvolve_params
        ),
        segmentation_params=_segmentation_params_from_dict(
            data.get("segmentation_params", {}), defaults.segmentation_params
        ),
        tile_params=_tile_params_from_dict(data.get("tile_params", {}), defaults.tile_params),
        segmentation_source_stage=data.get("segmentation_source_stage"),
        deconvolve_source_stage=data.get("deconvolve_source_stage"),
        stage_runs=data.get("stage_runs", {}),
        denoise_channels_done={
            stem: [int(c) for c in channels]
            for stem, channels in data.get("denoise_channels_done", {}).items()
        },
    )


def save_project_config(root, config: ProjectConfig):
    path = ProjectPaths(root).config_path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "raw_pattern": config.raw_pattern,
        "channels": {
            "brightfield": config.channels.brightfield,
            "projection": config.channels.projection,
        },
        "flatten_params": {
            "num_tiles_y": config.flatten_params.num_tiles_y,
            "num_tiles_x": config.flatten_params.num_tiles_x,
            "inverted_variance_prominence": config.flatten_params.inverted_variance_prominence,
            "poly_degree": list(config.flatten_params.poly_degree),
            "offset_um": config.flatten_params.offset_um,
        },
        "denoise_params": {
            "channel": config.denoise_params.channel,
            "checkpoint_path": config.denoise_params.checkpoint_path or "",
            "tta": config.denoise_params.tta,
        },
        "denoise_checkpoints": {
            str(channel): path for channel, path in config.denoise_checkpoints.items()
        },
        "deconvolve_params": {
            "enabled": config.deconvolve_params.enabled,
            "psf_path": config.deconvolve_params.psf_path or "",
            "zoom": config.deconvolve_params.zoom,
            "background": config.deconvolve_params.background,
            "num_iter": config.deconvolve_params.num_iter,
        },
        "segmentation_params": {
            "model_path": config.segmentation_params.model_path or "",
            "diameter": config.segmentation_params.diameter,
            "flow_threshold": config.segmentation_params.flow_threshold,
            "cellprob_threshold": config.segmentation_params.cellprob_threshold,
            "remove_edge_masks": config.segmentation_params.remove_edge_masks,
        },
        "tile_params": {
            "size": config.tile_params.size,
            "contrast_lo_pct": config.tile_params.contrast_lo_pct,
            "contrast_hi_pct": config.tile_params.contrast_hi_pct,
        },
        "segmentation_source_stage": config.segmentation_source_stage,
        "deconvolve_source_stage": config.deconvolve_source_stage,
        "stage_runs": config.stage_runs,
        "denoise_channels_done": {
            stem: sorted(channels) for stem, channels in config.denoise_channels_done.items()
        },
    }
    path.write_text(json.dumps(data, indent=2))


def mark_stage_run(root, stage: str, files_processed: list[str]):
    """Record a batch run's completion for `stage` -- last_run timestamp +
    accumulated set of processed file stems, merged into whatever
    `stage_runs` already exists rather than overwritten (matches the old
    per-folder sidecars' `files_processed` accumulation)."""
    config = load_project_config(root) or ProjectConfig()
    existing = config.stage_runs.get(stage, {})
    all_processed = set(existing.get("files", []))
    all_processed.update(files_processed)
    config.stage_runs[stage] = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "files": sorted(all_processed),
    }
    save_project_config(root, config)


def mark_denoise_channels(root, channel: int, stems: list[str]):
    """Record that `channel` was actually run through jssl-denoise (not just
    carried over) for each of `stems`, merged into whatever
    `denoise_channels_done` already has for that stem -- so denoising
    channel 0 today and channel 1 tomorrow accumulates to "both done"
    rather than the later run clobbering the earlier one's record."""
    config = load_project_config(root) or ProjectConfig()
    done = config.denoise_channels_done
    for stem in stems:
        channels = set(done.get(stem, []))
        channels.add(channel)
        done[stem] = sorted(channels)
    save_project_config(root, config)
