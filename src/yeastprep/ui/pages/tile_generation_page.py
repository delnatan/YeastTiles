"""Tile Generation page: crops every segmented cell out of the
combined-channel tiffs (design.md stage 3) using whichever mask each file
already has saved (from the Segmentation stage's batch run, or corrected
in the real Cellpose GUI). Self-contained -- owns its own input (2D images
+ masks) and output (tiles) folder selection, batch worker, and status
feedback; works standalone off whatever's already on disk, with no
dependency on the Segmentation page being open (or ever having run in this
session). The one convenience that legitimately needs the *other* page's
state -- "use the Segmentation folder" -- is deliberately not wired here;
the shell (which is the only thing that knows about both pages) connects
that button's click.
"""

import re
import subprocess
import sys
from pathlib import Path

from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core.pipeline import load_brightfield_channel
from yeastprep.core.segmentation import load_saved_masks, seg_npy_path
from yeastprep.core.tiles import TileParams

from .. import settings, tile_folder_config
from ..diagnostics.tile_generation_preview_panel import TileGenerationPreviewPanel
from ..file_list_panel import FileListPanel
from ..tile_folder_panel import TileFolderPanel
from ..tile_params_panel import TileParamsPanel
from ..worker import TileBatchWorker
from .page_progress import PageProgress

_DIGITS = re.compile(r"(\d+)")


def _natsort_key(path: Path):
    parts = _DIGITS.split(path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


class TileGenerationPage(QWidget):
    progress_changed = Signal(object)  # PageProgress

    def __init__(self, parent=None):
        super().__init__(parent)

        self._last_focal_slice = None
        self._last_masks = None
        self._batch_thread = None
        self._batch_worker = None

        self._build_ui()
        self._wire_up()

        self.tile_params_panel.set_params(settings.get_default_tile_params())

    # ------------------------------------------------------------------
    # UI construction

    def _build_ui(self):
        outer = QVBoxLayout(self)

        row = QHBoxLayout()
        outer.addLayout(row, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.tile_folder_panel = TileFolderPanel()
        left_layout.addWidget(self.tile_folder_panel)

        # Click handler wired by the shell, not here -- see module docstring.
        self.use_segmentation_folder_btn = QPushButton("Use Segmentation folder")
        left_layout.addWidget(self.use_segmentation_folder_btn)

        self.file_list_panel = FileListPanel()
        left_layout.addWidget(self.file_list_panel, 1)

        # No cross-page auto-refresh (pages are insulated -- see module
        # docstring), so this is the way to pick up files that appeared
        # after the folder was first pointed here: a batch run finishing on
        # the Segmentation page, new corrected masks written by the real
        # Cellpose GUI, files dropped in from elsewhere, etc.
        self.refresh_btn = QPushButton("Refresh file list")
        left_layout.addWidget(self.refresh_btn)

        self.tile_params_panel = TileParamsPanel()
        left_layout.addWidget(self.tile_params_panel)

        left.setMinimumWidth(320)
        left.setMaximumWidth(420)
        row.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_panel = TileGenerationPreviewPanel()
        right_layout.addWidget(self.preview_panel, 1)

        batch_row = QHBoxLayout()
        self.export_btn = QPushButton("Export Tiles for Selected")
        batch_row.addWidget(self.export_btn)
        self.open_tile_viewer_btn = QPushButton("Open in Tile Viewer")
        batch_row.addWidget(self.open_tile_viewer_btn)
        right_layout.addLayout(batch_row)

        row.addWidget(right, 1)

        self.status_label = QLabel("")
        outer.addWidget(self.status_label)

    # ------------------------------------------------------------------
    # Wiring

    def _wire_up(self):
        self.tile_folder_panel.input_folder_changed.connect(self._on_input_folder_changed)
        self.tile_folder_panel.output_folder_changed.connect(self._on_output_folder_changed)
        self.file_list_panel.file_selected.connect(self._on_file_selected)

        self.tile_params_panel.params_changed.connect(self._on_params_changed)
        self.tile_params_panel.recompute_requested.connect(self._recompute_now)
        self.tile_params_panel.save_defaults_requested.connect(self._save_folder_defaults)
        self.tile_params_panel.reset_defaults_requested.connect(self._reset_folder_defaults)

        self.export_btn.clicked.connect(self._start_batch)
        self.open_tile_viewer_btn.clicked.connect(self._open_tile_viewer)
        self.refresh_btn.clicked.connect(self._rescan_folder)

    # ------------------------------------------------------------------
    # Folder handling

    def _on_input_folder_changed(self, _folder: str):
        self._rescan_folder()

    def _rescan_folder(self):
        folder = self.tile_folder_panel.input_folder()
        if not folder or not Path(folder).is_dir():
            self.file_list_panel.set_files([])
            return
        paths = sorted(Path(folder).glob("*.tiff"), key=_natsort_key)
        self.file_list_panel.set_files([str(p) for p in paths])

    def _on_output_folder_changed(self, folder: str):
        params = tile_folder_config.load_tile_folder_config(folder)
        if params:
            self.tile_params_panel.set_params(params)
            self.status_label.setText(
                "Loaded tile parameters from a previous run in this folder."
            )

    # ------------------------------------------------------------------
    # File selection -> live crop-window preview (requires a saved mask;
    # tile export never runs segmentation itself)

    def _on_file_selected(self, path: str):
        masks = load_saved_masks(seg_npy_path(path))
        if masks is None:
            self._last_focal_slice = None
            self._last_masks = None
            self.preview_panel.clear()
            self.status_label.setText(
                f"{Path(path).name}: no saved segmentation found -- run Segmentation first."
            )
            return

        try:
            focal_slice = load_brightfield_channel(path)
        except Exception as exc:
            self.status_label.setText(f"Failed to load {path}: {exc}")
            return

        self._last_focal_slice = focal_slice
        self._last_masks = masks
        self.preview_panel.set_data(focal_slice, masks, self.tile_params_panel.params().size)
        n_cells = int(masks.max())
        self.status_label.setText(
            f"{Path(path).name}: {n_cells} cell{'s' if n_cells != 1 else ''} with saved segmentation"
        )

    def _on_params_changed(self, _params: TileParams):
        if not self.tile_params_panel.is_auto_recompute():
            return
        self._redraw_preview()

    def _recompute_now(self):
        self._redraw_preview()

    def _redraw_preview(self):
        if self._last_masks is None:
            return
        self.preview_panel.set_data(
            self._last_focal_slice, self._last_masks, self.tile_params_panel.params().size
        )

    # ------------------------------------------------------------------
    # Folder-scoped defaults

    def _save_folder_defaults(self):
        folder = self.tile_folder_panel.output_folder()
        if not folder:
            QMessageBox.warning(self, "yeastprep", "Select a tiles output folder first.")
            return
        tile_folder_config.save_tile_folder_config(folder, self.tile_params_panel.params())
        self.status_label.setText("Saved as tile-output folder defaults.")

    def _reset_folder_defaults(self):
        folder = self.tile_folder_panel.output_folder()
        params = tile_folder_config.load_tile_folder_config(folder) if folder else None
        self.tile_params_panel.set_params(params or settings.get_default_tile_params())

    # ------------------------------------------------------------------
    # Batch export

    def _start_batch(self):
        input_folder = self.tile_folder_panel.input_folder()
        output_folder = self.tile_folder_panel.output_folder()
        if not input_folder or not output_folder:
            QMessageBox.warning(
                self,
                "yeastprep",
                "Select both a 2D images folder and a tiles output folder first.",
            )
            return

        paths = [Path(p) for p in self.file_list_panel.checked_paths()]
        if not paths:
            QMessageBox.warning(
                self,
                "yeastprep",
                f"No tiffs found in {input_folder}.\n"
                "Segment some files first (Segmentation page), or point this folder "
                "at an existing set of segmented tiffs.",
            )
            return

        self._batch_thread = QThread()
        self._batch_worker = TileBatchWorker(
            paths, Path(output_folder), self.tile_params_panel.params()
        )
        self._batch_worker.moveToThread(self._batch_thread)
        self._thread_started_connection = self._batch_thread.started.connect(
            self._batch_worker.run
        )
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.file_result.connect(self._on_batch_file_result)
        self._batch_worker.finished.connect(self._on_batch_finished)

        self.export_btn.setEnabled(False)
        self.progress_changed.emit(PageProgress(active=True, done=0, total=len(paths)))

        self._batch_thread.start()

    def _on_batch_progress(self, done: int, total: int, name: str):
        self.progress_changed.emit(PageProgress(active=True, done=done, total=total, message=name))
        self.status_label.setText(f"Exporting tiles from {name} ({done}/{total})")

    def _on_batch_file_result(self, result):
        self.file_list_panel.mark_result(str(result.path), result.success, result.error)
        if result.success:
            self.status_label.setText(f"{result.path.name}: {result.n_cells} tile(s) exported")
        else:
            self.status_label.setText(f"{result.path.name}: {result.error}")

    def _on_batch_finished(self):
        self.progress_changed.emit(PageProgress(active=False))
        self.export_btn.setEnabled(True)
        self.status_label.setText("Batch tile export complete.")

        self._batch_thread.quit()
        self._batch_thread.wait()
        self._batch_thread = None
        self._batch_worker = None

    def _open_tile_viewer(self):
        folder = self.tile_folder_panel.output_folder()
        if not folder or not Path(folder).is_dir():
            QMessageBox.warning(self, "yeastprep", "Select a tiles output folder first.")
            return
        subprocess.Popen(
            [sys.executable, "-m", "tileclass", folder],
            start_new_session=True,
        )

    # ------------------------------------------------------------------

    def shutdown(self):
        settings.set_default_tile_params(self.tile_params_panel.params())
        if self._batch_thread is not None:
            self._batch_thread.quit()
            self._batch_thread.wait()
