"""ui.worker.BatchWorker: a run that dies outside any per-file catch must
still emit `finished`, or the page that started it stays stuck "running"."""

from pathlib import Path

from yeastprep.core.segmentation import SegmentationParams
from yeastprep.ui.worker import BatchWorker, SegmentationBatchWorker


class _RecordingWorker(BatchWorker):
    def _process(self, path):
        if Path(path).name == "bad":
            raise RuntimeError("disk on fire")
        return Path(path).name


def _run(worker):
    events = []
    worker.file_result.connect(lambda result: events.append(("result", result)))
    worker.error.connect(lambda message: events.append(("error", message)))
    worker.finished.connect(lambda: events.append(("finished",)))
    worker.run()
    return events


def test_successful_run_reports_each_file_then_finishes():
    events = _run(_RecordingWorker(["a", "b"]))
    assert events == [("result", "a"), ("result", "b"), ("finished",)]


def test_unexpected_failure_reports_error_then_still_finishes():
    events = _run(_RecordingWorker(["a", "bad", "c"]))
    assert events == [
        ("result", "a"),
        ("error", "RuntimeError: disk on fire"),
        ("finished",),
    ]


def test_setup_failure_still_finishes(monkeypatch):
    import yeastprep.ui.worker as worker_module

    def broken_get_model(_path):
        raise FileNotFoundError("no such model")

    monkeypatch.setattr(worker_module, "get_model", broken_get_model)
    events = _run(SegmentationBatchWorker([Path("fov.tiff")], SegmentationParams()))
    assert events == [("error", "FileNotFoundError: no such model"), ("finished",)]
