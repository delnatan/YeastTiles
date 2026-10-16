"""GUI-side batch plumbing, run headless: the shared `BatchRunner`, the
project tree's signals and bulk checking, and dropping stale previews."""

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QCoreApplication, QSettings  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

from yeastprep.ui import settings  # noqa: E402
from yeastprep.ui.common.batch_runner import BatchRunner  # noqa: E402
from yeastprep.ui.worker import BatchWorker, DebouncedController, SimplePipelineWorker  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Keep the user's real recent-projects list etc. out of the tests."""
    path = str(tmp_path / "settings.ini")
    monkeypatch.setattr(settings, "_settings", lambda: QSettings(path, QSettings.IniFormat))


def _wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError("condition not met")
        QCoreApplication.processEvents()
        time.sleep(0.005)


class _Worker(BatchWorker):
    """Processes names; blocks on `gate` (if given) inside each file."""

    def __init__(self, paths, gate=None):
        super().__init__(paths)
        self.gate = gate
        self.entered = threading.Event()  # set once the first file is underway

    def _process(self, path):
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(10)
        return str(path)


def _collect(runner):
    events = {"results": [], "outcomes": []}
    runner.file_result.connect(events["results"].append)
    runner.finished.connect(events["outcomes"].append)
    return events


# ----------------------------------------------------------------------
# BatchRunner


def test_runner_reports_results_then_one_outcome(qapp):
    runner = BatchRunner("Batch test")
    events = _collect(runner)
    runner.start(_Worker(["a", "b"]))
    _wait_until(lambda: events["outcomes"])

    assert events["results"] == ["a", "b"]
    (outcome,) = events["outcomes"]
    assert (outcome.done, outcome.total, outcome.cancelled, outcome.error) == (2, 2, False, None)
    assert outcome.summary() == "Batch test complete (2 file(s))."
    assert not runner.is_running()
    assert runner.describe_running() is None


def test_runner_refuses_a_second_run_and_cancels_between_files(qapp):
    gate = threading.Event()
    runner = BatchRunner("Batch test")
    events = _collect(runner)
    worker = _Worker(["a", "b", "c"], gate)
    runner.start(worker)

    assert runner.is_running()
    assert runner.describe_running() == "Batch test (0/3)"
    with pytest.raises(RuntimeError):
        runner.start(_Worker(["x"]))

    assert worker.entered.wait(10)
    runner.cancel()
    gate.set()  # let file "a" finish; the run stops before "b"
    _wait_until(lambda: events["outcomes"])

    assert events["results"] == ["a"]
    (outcome,) = events["outcomes"]
    assert outcome.cancelled and outcome.done == 1
    assert outcome.summary() == "Batch test cancelled after 1 of 3 file(s)."


def test_runner_shutdown_still_delivers_the_outcome(qapp):
    """On app exit the page's finished handler must still run (it records
    which files the run completed) even though the event loop is ending."""
    gate = threading.Event()
    runner = BatchRunner("Batch test")
    events = _collect(runner)
    worker = _Worker(["a", "b"], gate)
    runner.start(worker)
    assert worker.entered.wait(10)
    threading.Timer(0.1, gate.set).start()

    runner.shutdown()  # no processEvents() after this

    assert events["results"] == ["a"]
    assert len(events["outcomes"]) == 1 and events["outcomes"][0].cancelled
    assert not runner.is_running()


# ----------------------------------------------------------------------
# Stale preview results


@dataclass
class _Result:
    request_id: int
    source_id: int
    payload: object
    computed: object


def test_invalidate_drops_pending_and_in_flight_requests(qapp):
    controller = DebouncedController(
        SimplePipelineWorker(lambda payload, params: payload, _Result),
        debounce_ms=10,
        dedupe_key_fn=lambda payload, source_id, params: (source_id, params),
    )
    results = []
    controller.result_ready.connect(results.append)
    try:
        controller.recompute_now("old file", 1, "params")
        controller.schedule("old file again", 1, "other params")
        controller.invalidate()
        _wait_until(lambda: results)
        time.sleep(0.05)  # past the debounce interval
        QCoreApplication.processEvents()

        # The in-flight result still arrives, but no longer counts as
        # latest; the debounced one was never sent.
        assert [r.payload for r in results] == ["old file"]
        assert results[0].request_id != controller.latest_request_id()

        # Re-requesting the same input afterwards isn't deduped away.
        controller.schedule("old file", 1, "params")
        _wait_until(lambda: len(results) == 2)
        assert results[1].request_id == controller.latest_request_id()
    finally:
        controller.shutdown()


# ----------------------------------------------------------------------
# Project tree


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    for name in ("a", "b", "c"):
        (root / f"{name}.ims").write_bytes(b"")
    reduced = root / "01_reduced"
    reduced.mkdir()
    (reduced / "a.tiff").write_bytes(b"")  # "a" is already reduced
    return root


@pytest.fixture
def tree(qapp, project):
    from yeastprep.ui.project_tree_panel import ProjectTreePanel

    panel = ProjectTreePanel()
    refreshed = []
    panel.refreshed.connect(lambda: refreshed.append(True))
    assert panel.set_project_root(str(project))
    _wait_until(lambda: refreshed and panel.all_paths_for_stage("raw"))
    yield panel
    panel.shutdown()


def test_marking_a_result_is_not_a_checkbox_change(tree, project):
    """Every page re-evaluates its batch button on `checked_changed`; a
    status-icon update fired it too, re-enabling a running batch's button."""
    changes = []
    tree.checked_changed.connect(changes.append)
    tree.mark_result("raw", project / "a.ims", True)
    tree.mark_result("raw", project / "b.ims", False, "boom")
    assert changes == []


def test_check_only_unreduced_files(tree, project):
    changes = []
    tree.checked_changed.connect(changes.append)
    raw = tree._stage_items["raw"]
    leaves = [raw.child(i) for i in range(raw.childCount())]

    tree._set_checked("raw", leaves, lambda state: state[0] == "unprocessed")

    assert sorted(Path(p).name for p in tree.checked_paths_for_stage("raw")) == ["b.ims", "c.ims"]
    assert changes == ["raw"]  # one notification for the whole bulk change

    tree._set_checked("raw", leaves, lambda _state: False)
    assert tree.checked_paths_for_stage("raw") == []


def test_opening_a_missing_project_folder_warns_instead_of_creating_it(qapp, tmp_path, monkeypatch):
    from yeastprep.ui import project_tree_panel
    from yeastprep.ui.project_tree_panel import ProjectTreePanel

    warnings = []
    monkeypatch.setattr(
        project_tree_panel.QMessageBox, "warning", lambda *args: warnings.append(args[2])
    )
    panel = ProjectTreePanel()
    try:
        missing = tmp_path / "unplugged_drive" / "project"
        assert panel.set_project_root(str(missing)) is False
        assert not missing.exists()
        assert panel.project_root() == ""
        assert "not found" in warnings[0]
    finally:
        panel.shutdown()


# ----------------------------------------------------------------------
# One-shot background jobs


def test_background_task_delivers_result_or_error(qapp):
    from yeastprep.ui.common.background_task import BackgroundTask

    task = BackgroundTask()
    got = []
    task.succeeded.connect(lambda r: got.append(("ok", r)))
    task.failed.connect(lambda m: got.append(("failed", m)))

    gate = threading.Event()
    task.start(lambda: gate.wait(10) and threading.current_thread().name)
    assert task.is_running()
    with pytest.raises(RuntimeError):
        task.start(lambda: None)
    gate.set()
    _wait_until(lambda: got)
    assert got[0][0] == "ok" and got[0][1] != threading.current_thread().name
    assert not task.is_running()

    task.start(lambda: 1 / 0)
    _wait_until(lambda: len(got) == 2)
    assert got[1] == ("failed", "division by zero")


def test_background_task_shutdown_waits_and_drops_the_result(qapp):
    from yeastprep.ui.common.background_task import BackgroundTask

    task = BackgroundTask()
    got = []
    task.succeeded.connect(got.append)
    finished = threading.Event()

    def slow():
        time.sleep(0.1)
        finished.set()
        return "late"

    task.start(slow)
    task.shutdown()
    assert finished.is_set() and not task.is_running()
    QCoreApplication.processEvents()
    assert got == []


def test_denoise_training_worker_reads_frames_itself(qapp, tmp_path):
    """Frames are read on the worker's thread; a cancel while reading ends
    the run with no checkpoint, and an unreadable frame is an error."""
    import numpy as np
    import tifffile

    from yeastprep.ui.worker import TrainingWorker

    frame = tmp_path / "f.tiff"
    tifffile.imwrite(frame, np.zeros((2, 8, 8), np.uint16))

    worker = TrainingWorker([frame, frame], channel=1, config=None)
    assert [f.shape for f in worker._load_frames()] == [(8, 8), (8, 8)]

    cancelled = []
    worker.cancelled.connect(cancelled.append)
    worker.cancel()
    worker.run()
    assert cancelled == [None]

    errors = []
    broken = TrainingWorker([tmp_path / "missing.tiff"], channel=0, config=None)
    broken.error.connect(errors.append)
    broken.run()
    assert errors and errors[0].startswith("Couldn't read training frames")


def test_errors_from_superseded_preview_requests_are_dropped(qapp):
    def compute(payload, params):
        raise ValueError(f"bad {payload}")

    controller = DebouncedController(SimplePipelineWorker(compute, _Result), debounce_ms=10)
    errors = []
    controller.error.connect(errors.append)
    try:
        controller.recompute_now("old file", 1, None)
        controller.invalidate()  # e.g. another file was opened meanwhile
        controller.recompute_now("new file", 2, None)
        _wait_until(lambda: errors)
        time.sleep(0.05)
        QCoreApplication.processEvents()
        assert errors == ["bad new file"]
    finally:
        controller.shutdown()
