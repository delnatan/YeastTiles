"""Project config sidecar: round-trip, and never silently discarding a
config file that exists but can't be parsed."""

from yeastprep.core.project import (
    ProjectConfig,
    ProjectPaths,
    load_project_config,
    mark_stage_run,
    save_project_config,
)


def test_round_trip(tmp_path):
    config = ProjectConfig(raw_pattern="*.czi")
    config.segmentation_params.diameter = 42.0
    save_project_config(tmp_path, config)

    loaded = load_project_config(tmp_path)
    assert loaded.raw_pattern == "*.czi"
    assert loaded.segmentation_params.diameter == 42.0
    assert not list(tmp_path.glob(".yeastprep_project.json.tmp"))


def test_missing_config_loads_as_none(tmp_path):
    assert load_project_config(tmp_path) is None


def test_unreadable_config_is_backed_up_not_overwritten(tmp_path):
    config_path = ProjectPaths(tmp_path).config_path
    broken = '{"raw_pattern": "*.nd2", "flatten_params": {'  # truncated
    config_path.write_text(broken)

    assert load_project_config(tmp_path) is None
    # The usual caller pattern: fall back to defaults, record a run, save.
    mark_stage_run(tmp_path, "01_reduced", ["fov1"])

    backups = list(tmp_path.glob(".yeastprep_project.json.unreadable-*"))
    assert len(backups) == 1
    assert backups[0].read_text() == broken
    assert load_project_config(tmp_path).stage_runs["01_reduced"]["files"] == ["fov1"]


def test_wrongly_typed_config_counts_as_unreadable(tmp_path):
    ProjectPaths(tmp_path).config_path.write_text('{"channels": {"brightfield": 0}}')
    assert load_project_config(tmp_path) is None

    save_project_config(tmp_path, ProjectConfig())
    assert len(list(tmp_path.glob(".yeastprep_project.json.unreadable-*"))) == 1
