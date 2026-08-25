"""Turns a tree selection (stage, path) into the specific list of tasks
valid for it right now, given the project's current stage-resolution state
(`core/stages.consumers_for`) -- the single place that answers "what can I
do with this file," consumed by `SelectionActionsPanel`. Kept in the UI
layer (not `core/stages.py`) because it names *pages*, a UI concept; the
actual eligibility logic it wraps stays in core.
"""

from dataclasses import dataclass

from yeastprep.core import project as project_core
from yeastprep.core import stages as stages_core

from .project_tree_panel import ProjectTreePanel


@dataclass(frozen=True)
class Action:
    label: str
    page_key: str
    mode: str  # "live" | "saved" | "open_viewer_fov"


# Which page owns each stage as a *consumer* -- i.e. the page a selection
# should route to if that stage currently reads its input from the
# selected file's stage. Order matches core.stages._CONSUMER_CANDIDATES so
# actions list in a stable, pipeline-ordered sequence.
_CONSUMER_ACTION = {
    project_core.STAGE_DENOISED: ("Denoise this file", "denoise"),
    project_core.STAGE_DECONVOLVED: ("Deconvolve this file", "deconvolve"),
    stages_core.STAGE_SEGMENTATION: ("Segment this file", "segmentation"),
    project_core.STAGE_TILES: ("Use for Tile Generation", "tile_generation"),
}

# Stage a saved-output file was itself produced by -- offers a "view/
# re-tune" action back into the page that produced it, in addition to
# whatever downstream tasks it can feed.
_PRODUCER_ACTION = {
    project_core.STAGE_DENOISED: ("Open in Denoise (view/re-tune)", "denoise"),
    project_core.STAGE_DECONVOLVED: ("Open in Deconvolve (view/re-tune)", "deconvolve"),
}

_PREVIEWABLE_STAGES = (
    project_core.STAGE_REDUCED,
    project_core.STAGE_DENOISED,
    project_core.STAGE_DECONVOLVED,
)


def _consumer_actions(consumers: list[str]) -> list[Action]:
    actions = []
    for consumer in consumers:
        entry = _CONSUMER_ACTION.get(consumer)
        if entry is None:
            continue
        label, page_key = entry
        actions.append(Action(label, page_key, "live"))
    return actions


def actions_for_selection(stage: str, path: str, tree_panel: ProjectTreePanel) -> list[Action]:
    if stage == project_core.STAGE_TILES:
        # `path` here is a FOV id, not a file path (see
        # ProjectTreePanel._refresh_tiles_children / _on_current_changed).
        return [Action("Open in Tile Viewer (this FOV)", "tile_generation", "open_viewer_fov")]

    if stage == stages_core.STAGE_RAW:
        return [Action("Reduce this file", "data_reduction", "live")]

    paths = tree_panel.project_paths()
    config = project_core.load_project_config(tree_panel.project_root()) or project_core.ProjectConfig()
    consumers = stages_core.consumers_for(stage, paths, config)

    actions: list[Action] = []
    producer = _PRODUCER_ACTION.get(stage)
    if producer is not None:
        label, page_key = producer
        actions.append(Action(label, page_key, "saved"))
    actions.extend(_consumer_actions(consumers))
    if stage in _PREVIEWABLE_STAGES:
        actions.append(Action("Preview", "preview", "live"))
    return actions
