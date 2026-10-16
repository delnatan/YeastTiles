"""The Classify Tiles page's embedding explorer, run headless: lasso ->
grid -> tag -> saved to the sidecar and recolored on the plot."""

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtCore import QPoint  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402
from tileclass.data.pooled_annotations import PooledAnnotations  # noqa: E402
from tileclass.tile_container import write_container  # noqa: E402

from yeastprep.ui.diagnostics.embedding_explorer import EmbeddingExplorerWidget  # noqa: E402
from yeastprep.ui.diagnostics.embedding_scatter_widget import UNLABELED_LABEL  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _crop(seed, size=32):
    rng = np.random.default_rng(seed)
    bf = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    target = rng.integers(0, 255, size=(size, size), dtype=np.uint8)
    mask = np.full((size, size), 255, dtype=np.uint8)
    return np.stack([bf, target, mask], axis=0)


@pytest.fixture
def explorer_setup(qapp, tmp_path):
    container = tmp_path / "fov.tiles"
    cells = [(f"fov_cell{i:05d}", i, _crop(i)) for i in range(6)]
    write_container(container, cells)
    paths = [os.path.normpath(str(container / cell_id)) for cell_id, _, _ in cells]

    pooled = PooledAnnotations([container])
    pooled.add_category("single")
    pooled.add_category("budding")
    pooled.update([(paths[5], "single")])

    labels = [UNLABELED_LABEL] * 5 + ["single"]
    xy = np.random.default_rng(0).normal(size=(6, 2))

    explorer = EmbeddingExplorerWidget()
    explorer.show_embedding(xy, labels, paths, pooled)
    yield explorer, paths, container
    explorer.shutdown()


def test_lasso_fills_grid_with_unlabeled_first_all_selected(explorer_setup):
    explorer, paths, _ = explorer_setup
    explorer._on_lasso([5, 0, 1])
    assert explorer.grid.paths == [paths[0], paths[1], paths[5]]
    assert explorer.grid.selected_paths == explorer.grid.paths


def test_number_key_tags_selection_saves_and_recolors(explorer_setup):
    explorer, paths, container = explorer_setup
    explorer._on_lasso([0, 1, 2])
    explorer.assign_number(2)  # "budding"

    fresh = PooledAnnotations([container])  # re-read from disk
    assert [fresh.get(p) for p in paths[:3]] == ["budding"] * 3
    assert all(fresh.confidence(p) is None for p in paths[:3])  # human-set
    assert explorer.labels_for(paths) == ["budding"] * 3 + [UNLABELED_LABEL] * 2 + ["single"]

    # Labeled points are drawn last (on top): 2 unlabeled, then 4 labeled.
    sizes = explorer.scatter._points.get_sizes()
    assert list(sizes[:2]) == [5, 5] and all(sizes[2:] == 16)


def test_deselected_tiles_are_left_alone(explorer_setup):
    explorer, paths, container = explorer_setup
    explorer._on_lasso([0, 1, 2])
    explorer.grid._select(paths[1], ctrl_or_cmd=True, shift=False)  # Ctrl-click off
    explorer.assign_category("single")

    fresh = PooledAnnotations([container])
    assert [fresh.get(p) for p in paths[:3]] == ["single", None, "single"]
    # The plot rings exactly the grid selection.
    assert len(explorer.scatter._highlight.get_offsets()) == 2


def test_clear_tag(explorer_setup):
    explorer, paths, container = explorer_setup
    explorer._on_lasso([5])
    explorer.assign_category("")
    assert PooledAnnotations([container]).get(paths[5]) is None
    assert explorer.labels_for([paths[5]]) == [UNLABELED_LABEL]


def test_hover_preview_shows_and_hides(explorer_setup):
    explorer, _, _ = explorer_setup
    explorer._on_point_hovered(3, QPoint(10, 10))
    assert explorer.preview.isVisible()
    assert "fov" in explorer.preview.caption.text()
    explorer._on_point_hovered(-1, QPoint(0, 0))
    assert not explorer.preview.isVisible()


def test_scatter_lasso_reports_indices_inside(explorer_setup):
    explorer, _, _ = explorer_setup
    got = []
    explorer.scatter.lassoSelected.connect(got.append)
    xy = explorer.scatter._xy
    lo, hi = xy.min(axis=0) - 1, xy.max(axis=0) + 1
    explorer.scatter._on_lasso([(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])])
    assert got == [list(range(6))]
