"""Checking/unchecking pool FOVs loads nothing: per-FOV data is read on
demand into a cache, and views refresh lazily, off the GUI thread."""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QSettings, Qt  # noqa: E402
from qtpy.QtWidgets import QApplication, QLabel  # noqa: E402
from tileclass.data.annotations import TileAnnotations  # noqa: E402
from tileclass.data.pooled_annotations import PooledAnnotations  # noqa: E402
from tileclass.tile_container import write_container  # noqa: E402

from yeastprep.ui import classifier_pool, settings  # noqa: E402
from yeastprep.ui.classifier_pool import ClassifierPool, ClassifierPoolWidget  # noqa: E402
from yeastprep.ui.common.lazy_refresh import LazyRefresh  # noqa: E402
from yeastprep.ui.pool_data import PoolDataCache  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings_path(tmp_path, monkeypatch):
    path = str(tmp_path / "settings.ini")
    monkeypatch.setattr(settings, "_settings", lambda: QSettings(path, QSettings.IniFormat))
    return path


def _crop():
    return np.zeros((3, 8, 8), dtype=np.uint8)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    tiles = root / "05_tiles"
    tiles.mkdir(parents=True)
    for fov, n in (("a", 4), ("b", 3), ("c", 2)):
        write_container(
            tiles / f"{fov}.tiles",
            [(f"{fov}_cell{i:05d}", i, _crop()) for i in range(n)],
        )
    return root


def _fov(project, name):
    return str(project / "05_tiles" / f"{name}.tiles")


def _tag(project, name, tags):
    store = TileAnnotations(_fov(project, name))
    store.update(tags)
    return store


def _wait(qapp, cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        qapp.processEvents()
        time.sleep(0.005)
    return cond()


# ----------------------------------------------------------------------
# Cache


def test_summary_matches_pooled_annotations(project):
    fovs = [_fov(project, n) for n in "abc"]
    _tag(project, "a", [("a_cell00000", "alive"), ("a_cell00001", "dead")])
    _tag(project, "b", [("b_cell00002", "alive")])
    store = TileAnnotations(_fov(project, "b"))
    store.update_with_confidence([("b_cell00000", "dead", 0.9)])

    summary = PoolDataCache().summarize(fovs)

    pooled = PooledAnnotations(fovs)
    assert summary.n_tiles == 9
    assert summary.counts == {"alive": 2, "dead": 2}
    assert summary.human_counts == {"alive": 2, "dead": 1}  # one AI guess
    assert summary.untagged == 5
    assert summary.vocabulary == pooled.categories()


def test_cache_reloads_only_what_changed(project, monkeypatch):
    cache = PoolDataCache()
    a, b = _fov(project, "a"), _fov(project, "b")
    loads = []
    real = PoolDataCache._load
    monkeypatch.setattr(
        PoolDataCache, "_load", staticmethod(lambda f, s: (loads.append(f), real(f, s))[1])
    )

    cache.summarize([a, b])
    cache.summarize([a])  # b unchecked: nothing to do
    cache.summarize([a, b])  # b checked again: still cached
    assert loads == [a, b]

    _tag(project, "b", [("b_cell00000", "alive")])  # e.g. a viewer saved tags
    summary = cache.summarize([a, b])
    assert loads == [a, b, b]
    assert summary.counts == {"alive": 1}


# ----------------------------------------------------------------------
# Signals and the pool tree


def test_selection_signal_is_per_scope_and_skips_noops(settings_path, project):
    pool = ClassifierPool()
    pool.add_project(str(project))
    training, classify = pool.scope("training"), pool.scope("classify")
    seen = {"training": 0, "classify": 0, "projects": 0}
    training.selection_changed.connect(lambda: seen.__setitem__("training", seen["training"] + 1))
    classify.selection_changed.connect(lambda: seen.__setitem__("classify", seen["classify"] + 1))
    classify.projects_changed.connect(lambda: seen.__setitem__("projects", seen["projects"] + 1))

    classify.set_fov_included(_fov(project, "a"), False)
    classify.set_fov_included(_fov(project, "a"), False)  # already unchecked

    assert seen == {"training": 0, "classify": 1, "projects": 0}
    pool.remove_project(str(project))
    assert seen["projects"] == 1


def test_toggling_a_fov_does_not_rebuild_or_rescan(qapp, settings_path, project, monkeypatch):
    pool = ClassifierPool()
    pool.add_project(str(project))
    widget = ClassifierPoolWidget(pool.scope("classify"), hint="")
    other = ClassifierPoolWidget(pool.scope("training"), hint="")

    calls = []
    real = classifier_pool.discover_fov_dirs
    monkeypatch.setattr(
        classifier_pool, "discover_fov_dirs", lambda root: (calls.append(root), real(root))[1]
    )

    leaf = widget.pool_tree.topLevelItem(0).child(1)
    leaf.setCheckState(0, Qt.Unchecked)
    # A bulk change from the context menu moves the checkmarks in place.
    fovs = [_fov(project, n) for n in "abc"]
    widget.pool.set_fovs_included(fovs, False)

    assert calls == []
    states = [widget.pool_tree.topLevelItem(0).child(i).checkState(0) for i in range(3)]
    assert states == [Qt.Unchecked] * 3
    # The other page's checks were never touched.
    assert other.pool.is_fov_included(fovs[0])


# ----------------------------------------------------------------------
# LazyRefresh


def _lazy(qapp, **kwargs):
    label = QLabel()
    log = {"capture": 0, "compute": 0, "applied": []}

    def capture():
        log["capture"] += 1
        return log["capture"]

    def compute(arg):
        log["compute"] += 1
        return arg

    lazy = LazyRefresh(
        label, capture, compute, log["applied"].append, delay_ms=20, **kwargs
    )
    return label, lazy, log


def test_lazy_refresh_waits_until_visible_and_coalesces(qapp):
    label, lazy, log = _lazy(qapp)
    for _ in range(5):
        lazy.invalidate()
    _wait(qapp, lambda: False, timeout=0.2)
    assert log["compute"] == 0  # hidden: nothing happens

    label.show()
    assert _wait(qapp, lambda: log["applied"])
    assert log["compute"] == 1

    for _ in range(5):
        lazy.invalidate()
    assert _wait(qapp, lambda: len(log["applied"]) == 2)
    _wait(qapp, lambda: False, timeout=0.2)
    assert log["compute"] == 2  # five invalidations, one run
    lazy.shutdown()
    label.close()


def test_lazy_refresh_drops_a_run_overtaken_by_a_newer_one(qapp):
    release = []
    label = QLabel()
    applied = []

    def compute(arg):
        while not release:
            time.sleep(0.005)
        return arg

    counter = iter(range(100))
    lazy = LazyRefresh(label, lambda: next(counter), compute, applied.append, delay_ms=10)
    label.show()
    assert _wait(qapp, lambda: lazy._task.is_running())
    lazy.invalidate()  # while run 0 is still going
    _wait(qapp, lambda: False, timeout=0.1)
    release.append(1)
    assert _wait(qapp, lambda: applied)
    assert applied == [1]  # the stale result 0 was never shown
    lazy.shutdown()
    label.close()
