"""The classification pool's per-page FOV scopes: shared projects, separate
FOV checks, and carrying over exclusions saved before scopes existed."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QSettings  # noqa: E402

from yeastprep.ui import settings  # noqa: E402
from yeastprep.ui.classifier_pool import ClassifierPool  # noqa: E402


@pytest.fixture
def settings_path(tmp_path, monkeypatch):
    path = str(tmp_path / "settings.ini")
    monkeypatch.setattr(settings, "_settings", lambda: QSettings(path, QSettings.IniFormat))
    return path


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    tiles = root / "05_tiles"
    tiles.mkdir(parents=True)
    for fov in ("a", "b", "c"):
        (tiles / f"{fov}.tiles").write_bytes(b"")
    return root


def _fov(project, name):
    return str(project / "05_tiles" / f"{name}.tiles")


def test_scopes_check_fovs_independently(settings_path, project):
    pool = ClassifierPool()
    pool.add_project(str(project))
    training, classify = pool.scope("training"), pool.scope("classify")

    classify.set_fovs_included([_fov(project, "a"), _fov(project, "b")], False)

    assert classify.checked_fov_dirs() == [_fov(project, "c")]
    assert training.checked_fov_dirs() == [_fov(project, n) for n in "abc"]
    assert training.roots() == classify.roots() == [str(project)]


def test_scopes_persist_and_project_removal_clears_them(settings_path, project):
    pool = ClassifierPool()
    pool.add_project(str(project))
    pool.scope("training").set_fov_included(_fov(project, "b"), False)

    reloaded = ClassifierPool()
    assert not reloaded.scope("training").is_fov_included(_fov(project, "b"))
    assert reloaded.scope("classify").is_fov_included(_fov(project, "b"))

    reloaded.remove_project(str(project))
    reloaded.add_project(str(project))
    assert reloaded.scope("training").is_fov_included(_fov(project, "b"))


def test_legacy_exclusions_carry_over_to_every_scope(settings_path, project):
    s = QSettings(settings_path, QSettings.IniFormat)
    s.setValue("classifier_pool/roots", [str(project)])
    s.setValue("classifier_pool/excluded_fovs", [_fov(project, "a")])
    s.sync()

    pool = ClassifierPool()
    for name in ("training", "classify"):
        assert not pool.scope(name).is_fov_included(_fov(project, "a"))

    # Re-checking everything in one scope must stick, not fall back to the
    # legacy list on the next load.
    pool.scope("classify").set_fov_included(_fov(project, "a"), True)
    reloaded = ClassifierPool()
    assert reloaded.scope("classify").is_fov_included(_fov(project, "a"))
    assert not reloaded.scope("training").is_fov_included(_fov(project, "a"))
