"""A `classification`-only install (no `prep` extra) must still launch
yeastprep and its batch CLIs. Rather than needing a second environment,
each check runs in a subprocess that hides the `prep`-only packages from
the import system, so a stray module-level `import cellpose` fails here
instead of on the machine that doesn't have it."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

PREP_ONLY_PACKAGES = ("cellpose", "jssl_denoise", "psfkit")

_HIDE_PACKAGES = f"""
import importlib.abc, importlib.util, sys

class _HidePrepPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in {PREP_ONLY_PACKAGES!r}:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)

sys.meta_path.insert(0, _HidePrepPackages())
_real_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **k: (
    None if name.split(".")[0] in {PREP_ONLY_PACKAGES!r} else _real_find_spec(name, *a, **k)
)
"""


def _run_without_prep(code: str):
    result = subprocess.run(
        [sys.executable, "-c", _HIDE_PACKAGES + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "QT_QPA_PLATFORM": "offscreen"},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr


def test_batch_clis_import_without_prep_extra():
    _run_without_prep(
        """
        import yeastprep.core.cli
        import yeastprep.core.project
        """
    )


def test_yeastprep_window_builds_without_prep_extra():
    _run_without_prep(
        """
        from qtpy.QtWidgets import QApplication
        app = QApplication([])

        from yeastprep.ui.main_window import YeastPrepWindow
        from yeastprep.ui.pages.missing_extra_page import MissingExtraPage

        window = YeastPrepWindow()
        assert isinstance(window.denoise_page, MissingExtraPage)
        assert isinstance(window.segmentation_page, MissingExtraPage)
        assert not isinstance(window.classify_tiles_page, MissingExtraPage)
        assert not isinstance(window.classifier_training_page, MissingExtraPage)
        window.close()
        """
    )


def test_fallback_mask_outlines_match_cellpose():
    masks_to_outlines = pytest.importorskip("cellpose.utils").masks_to_outlines
    from yeastprep.ui.common.mask_outlines import _label_boundaries

    yy, xx = np.mgrid[:64, :64]
    masks = np.zeros((64, 64), dtype=np.int32)
    masks[(yy - 20) ** 2 + (xx - 20) ** 2 < 12**2] = 1
    masks[(yy - 44) ** 2 + (xx - 40) ** 2 < 10**2] = 2
    masks[0:8, 50:64] = 3  # touches the image edge

    np.testing.assert_array_equal(_label_boundaries(masks), masks_to_outlines(masks))
