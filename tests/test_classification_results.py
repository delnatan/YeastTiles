"""The Classify Tiles page's Results tab: per-FOV category counts, the two
CSV files it saves, and the inference run that feeds them."""

import csv
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QSettings  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402
from tileclass.data.annotations import TileAnnotations  # noqa: E402
from tileclass.tile_container import write_container  # noqa: E402

from yeastprep.ui import settings  # noqa: E402
from yeastprep.ui.classifier_pool import ClassifierPool  # noqa: E402
from yeastprep.ui.diagnostics.classification_results_panel import (  # noqa: E402
    results_table,
    write_summary_csv,
    write_tile_csv,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def scope(qapp, tmp_path, monkeypatch):
    path = str(tmp_path / "settings.ini")
    monkeypatch.setattr(settings, "_settings", lambda: QSettings(path, QSettings.IniFormat))
    tiles = tmp_path / "project" / "05_tiles"
    tiles.mkdir(parents=True)
    crop = np.zeros((3, 8, 8), dtype=np.uint8)
    for fov, n in (("a", 4), ("b", 2)):
        write_container(tiles / f"{fov}.tiles", [(f"{fov}_cell{i:05d}", i, crop) for i in range(n)])
    store = TileAnnotations(str(tiles / "a.tiles"))
    store.add_category("single")
    store.add_category("tetrad")
    store.update([("a_cell00000", "single")])
    store.update_with_confidence([("a_cell00001", "tetrad", 0.8), ("a_cell00002", "single", 0.6)])

    pool = ClassifierPool()
    pool.add_project(str(tmp_path / "project"))
    return pool.scope("classify")


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def test_results_table_counts_tags_per_fov(scope, tmp_path):
    table = results_table(scope.snapshot())

    assert table.header() == ["project", "fov", "tiles", "single", "tetrad", "unclassified", "human_set"]
    assert table.rows == [["project", "a", 4, 2, 1, 1, 1], ["project", "b", 2, 0, 0, 2, 0]]
    assert table.total() == ["", "All 2 FOV(s)", 6, 2, 1, 3, 1]

    write_summary_csv(tmp_path / "summary.csv", table)
    assert _read(tmp_path / "summary.csv") == [
        table.header(),
        ["project", "a", "4", "2", "1", "1", "1"],
        ["project", "b", "2", "0", "0", "2", "0"],
    ]


def test_tile_csv_has_one_row_per_cell(scope, tmp_path):
    n = write_tile_csv(tmp_path / "tiles.csv", scope.checked_fov_dirs())

    rows = _read(tmp_path / "tiles.csv")
    assert n == 6 and len(rows) == 7
    assert rows[0] == ["project", "fov", "cell", "category", "source", "confidence"]
    assert rows[1] == ["project", "a", "0", "single", "human", ""]
    assert rows[2] == ["project", "a", "1", "tetrad", "ai", "0.8000"]
    assert rows[4] == ["project", "a", "3", "", "", ""]


def _wait(qapp, cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        qapp.processEvents()
        time.sleep(0.005)
    return cond()


def test_page_runs_inference_and_fills_the_results_tab(qapp, scope, tmp_path, monkeypatch):
    from yeastprep.ui.pages import classify_tiles_page
    from yeastprep.ui.pages.classify_tiles_page import ClassifyTilesPage

    class FakeClassifier:
        def __init__(self, weights_path=None, meta_path=None):
            pass

        def predict(self, paths):
            return [("tetrad", 0.9) for _ in paths]

    monkeypatch.setattr(classify_tiles_page, "YeastEfficientNetClassifier", FakeClassifier)
    page = ClassifyTilesPage(scope)
    monkeypatch.setattr(page.infer_picker, "weights_path", lambda: tmp_path / "weights.pth")
    page.show()
    page.right_tabs.setCurrentIndex(page._results_tab_index)
    try:
        page._run_inference()
        assert _wait(qapp, lambda: page._inference_thread is None)
        # 3 blank tiles filled, the 2 earlier predictions replaced.
        assert "3 newly tagged, 2 earlier prediction(s) replaced" in page.inference_log.toPlainText()
        assert not page.inference_progress.isVisible()

        panel = page.results_panel
        assert _wait(qapp, lambda: panel._data.rows and panel._data.total()[4] == 5)
        assert panel.table.rowCount() == 3  # two FOVs + the total
        assert panel.table.item(2, 4).text() == "5"  # tetrad, all FOVs
    finally:
        page.shutdown()
        page.close()
