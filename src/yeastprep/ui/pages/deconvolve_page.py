"""Deconvolve page: optional Poisson-ML deconvolution
(`core/deconvolution`) of the target channel only, reading from
02_denoised/ if that stage ran, else 01_reduced/, and writing into
03_deconvolved/. Split out of the old enhance_page.py so deconvolution is
independently toggleable from denoising (see denoise_page.py). Reads its
input/batch-selection through the shared ProjectTreePanel (see
main_window.py) rather than owning its own folder panel.

A second tab hosts the PSF Calculator (`psfkit`-backed), for producing the
PSF file this stage consumes.
"""

from pathlib import Path

from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from yeastprep.core import project as project_core
from yeastprep.core.combined_tiff import load_combined_channels
from yeastprep.core.deconvolve import DeconvolveParams

from .. import settings
from ..batch_progress_bar import BatchProgressBar
from ..common.preview_source_label import PreviewSourceLabel
from ..deconvolve_params_panel import DeconvolveParamsPanel
from ..diagnostics.deconvolve_preview_panel import DeconvolvePreviewPanel
from ..project_tree_panel import ProjectTreePanel
from ..psf_calculator_panel import PSFCalculatorPanel
from ..worker import DeconvolveBatchWorker, DeconvolveController
from .page_progress import PageProgress

# Deconvolution only ever applies to the target/fluorescence channel --
# brightfield has no PSF model and is passed through unchanged (see
# core/deconvolve.py) -- so the preview is labeled explicitly rather than
# leaving it implicit the way an unlabeled "Before"/"After" would.
_CHANNEL_LABEL = "Target/fluorescence channel"

# Deconvolve's own valid input choices -- excludes STAGE_DECONVOLVED, that's
# this stage's own output (see project_core.resolve_deconvolve_source).
_SOURCE_STAGES = (project_core.STAGE_REDUCED, project_core.STAGE_DENOISED)
_SOURCE_STAGE_LABELS = {
    project_core.STAGE_REDUCED: "01 · Reduced (2D)",
    project_core.STAGE_DENOISED: "02 · Denoised",
}


class DeconvolvePage(QWidget):
    progress_changed = Signal(object)  # PageProgress

    def __init__(self, tree_panel: ProjectTreePanel, parent=None):
        super().__init__(parent)
        self.tree_panel = tree_panel

        self._last_target = None
        self._source_stage = None
        self._source_generation = 0
        self._batch_thread = None
        self._batch_worker = None

        self._build_ui()
        self._wire_up()

        self.params_panel.set_params(settings.get_default_deconvolve_params())
        self._refresh_source_status()

    # ------------------------------------------------------------------
    # UI construction

    def _build_ui(self):
        outer = QVBoxLayout(self)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.tabs.addTab(self._build_main_tab(), "Deconvolve")

        self.psf_calculator_panel = PSFCalculatorPanel()
        self.tabs.addTab(self.psf_calculator_panel, "PSF Calculator")

        self.status_label = QLabel(
            "Deconvolution applies to the target/fluorescence channel only -- "
            "the brightfield channel has no PSF model and is saved through unchanged."
        )
        outer.addWidget(self.status_label)

    def _build_main_tab(self) -> QWidget:
        tab = QWidget()
        row = QHBoxLayout(tab)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._build_source_group())
        self.params_panel = DeconvolveParamsPanel()
        left_layout.addWidget(self.params_panel)
        left_layout.addStretch(1)
        left.setMinimumWidth(340)
        left.setMaximumWidth(440)
        row.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_source_label = PreviewSourceLabel()
        right_layout.addWidget(self.preview_source_label)
        self.preview_panel = DeconvolvePreviewPanel()
        right_layout.addWidget(self.preview_panel, 1)

        action_row = QHBoxLayout()
        self.deconvolve_btn = QPushButton("Deconvolve && Save Selected")
        action_row.addWidget(self.deconvolve_btn)
        self.batch_progress = BatchProgressBar()
        action_row.addWidget(self.batch_progress, 1)
        right_layout.addLayout(action_row)

        row.addWidget(right, 1)

        return tab

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Input")
        v = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Source stage:"))
        self.source_combo = QComboBox()
        self.source_combo.addItem("Auto (Denoised if available, else Reduced)", None)
        for stage in _SOURCE_STAGES:
            self.source_combo.addItem(_SOURCE_STAGE_LABELS[stage], stage)
        row.addWidget(self.source_combo, 1)
        v.addLayout(row)

        self.source_status_label = QLabel("")
        self.source_status_label.setWordWrap(True)
        v.addWidget(self.source_status_label)

        return group

    # ------------------------------------------------------------------
    # Wiring

    def _wire_up(self):
        self.tree_panel.file_preview_requested.connect(self._on_file_selected)
        self.tree_panel.project_root_changed.connect(self._on_project_root_changed)
        self.tree_panel.refreshed.connect(self._refresh_source_status)
        self.source_combo.currentIndexChanged.connect(self._on_source_combo_changed)

        self.controller = DeconvolveController()
        self.controller.result_ready.connect(self._on_result_ready)
        self.controller.error.connect(self._on_error)

        self.params_panel.params_changed.connect(self._on_params_changed)
        self.params_panel.recompute_requested.connect(self._recompute_now)
        self.params_panel.save_defaults_requested.connect(self._save_project_defaults)
        self.params_panel.reset_defaults_requested.connect(self._reset_project_defaults)
        self.deconvolve_btn.clicked.connect(self._start_batch)

        self.psf_calculator_panel.psf_saved.connect(self.params_panel.set_psf_path)

    # ------------------------------------------------------------------
    # Input stage: `source_combo`'s override (REDUCED/DENOISED) if one is
    # picked, else auto (denoised if it ran, else reduced) -- never
    # 03_deconvolved/, that's this stage's own output. The override is
    # persisted per-project (`deconvolve_source_stage`) so it's remembered
    # across launches, same as segmentation_source_stage.

    def _input_stage(self) -> str | None:
        paths = self.tree_panel.project_paths()
        if paths is None:
            return None
        source = project_core.resolve_deconvolve_source(paths, self.source_combo.currentData())
        return source.name if source is not None else None

    def _on_project_root_changed(self, root: str):
        config = project_core.load_project_config(root)
        override = config.deconvolve_source_stage if config else None
        self.source_combo.blockSignals(True)
        idx = self.source_combo.findData(override)
        self.source_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.source_combo.blockSignals(False)
        self._refresh_source_status()

    def _on_source_combo_changed(self, _index):
        root = self.tree_panel.project_root()
        if root:
            config = project_core.load_project_config(root) or project_core.ProjectConfig()
            config.deconvolve_source_stage = self.source_combo.currentData()
            project_core.save_project_config(root, config)
        # The old preview/target belonged to whichever stage was active
        # before -- leaving it up would look like it's still the current
        # input, so it's cleared rather than left stale.
        self._last_target = None
        self.preview_panel.clear()
        self.preview_source_label.clear_path()
        self._refresh_source_status()
        self.tree_panel.refresh()

    def _refresh_source_status(self):
        paths = self.tree_panel.project_paths()
        if paths is None:
            self.source_status_label.setText("Open a project to choose an input stage.")
            return
        stage = self._input_stage()
        if stage is None:
            override = self.source_combo.currentData()
            if override is not None:
                self.source_status_label.setText(
                    f"No files in {_SOURCE_STAGE_LABELS[override]} yet."
                )
            else:
                self.source_status_label.setText(
                    "No reduced or denoised files yet -- run Data Reduction first."
                )
            return
        n = len(self.tree_panel.all_paths_for_stage(stage))
        self.source_status_label.setText(
            f"Reading from: {_SOURCE_STAGE_LABELS[stage]}  ({n} file{'s' if n != 1 else ''})"
        )

    # ------------------------------------------------------------------
    # File selection -> live preview

    def _on_file_selected(self, stage: str, path: str):
        if stage != self._input_stage():
            return
        self.preview_source_label.set_path(path)
        try:
            _brightfield, target = load_combined_channels(path)
        except Exception as exc:
            self.status_label.setText(f"Failed to load {path}: {exc}")
            return
        self._last_target = target
        self._source_generation += 1

        self.preview_panel.set_data(target, None, _CHANNEL_LABEL)

        if self.params_panel.is_auto_recompute():
            self.controller.schedule(
                target, self._source_generation, self.params_panel.params()
            )

    def _on_params_changed(self, params: DeconvolveParams):
        if self._last_target is None:
            return
        if not self.params_panel.is_auto_recompute():
            return
        self.controller.schedule(self._last_target, self._source_generation, params)

    def _recompute_now(self):
        if self._last_target is None:
            return
        self.controller.recompute_now(
            self._last_target, self._source_generation, self.params_panel.params()
        )

    def _on_result_ready(self, result):
        if result.request_id != self.controller.latest_request_id():
            return  # superseded by a newer request already queued
        self.preview_panel.set_data(result.target_before, result.target_after, _CHANNEL_LABEL)

    def _on_error(self, message: str):
        self.status_label.setText(f"Deconvolve error: {message}")

    # ------------------------------------------------------------------
    # Project-scoped defaults

    def _save_project_defaults(self):
        root = self.tree_panel.project_root()
        if not root:
            QMessageBox.warning(self, "yeastprep", "Open a project first.")
            return
        config = project_core.load_project_config(root) or project_core.ProjectConfig()
        config.deconvolve_params = self.params_panel.params()
        project_core.save_project_config(root, config)
        self.status_label.setText("Saved as project defaults.")

    def _reset_project_defaults(self):
        root = self.tree_panel.project_root()
        config = project_core.load_project_config(root) if root else None
        if config:
            self.params_panel.set_params(config.deconvolve_params)
        else:
            self.params_panel.set_params(settings.get_default_deconvolve_params())

    # ------------------------------------------------------------------
    # Batch deconvolve

    def _start_batch(self):
        paths_root = self.tree_panel.project_paths()
        input_stage = self._input_stage()
        if paths_root is None or input_stage is None:
            QMessageBox.warning(
                self,
                "yeastprep",
                "No reduced (2D) tiffs found. Run Data Reduction (and optionally "
                "Denoise) first.",
            )
            return

        paths = [Path(p) for p in self.tree_panel.checked_paths_for_stage(input_stage)]
        if not paths:
            QMessageBox.warning(self, "yeastprep", "No files checked to deconvolve.")
            return

        outdir = paths_root.deconvolved

        self._batch_thread = QThread()
        self._batch_worker = DeconvolveBatchWorker(paths, outdir, self.params_panel.params())
        self._batch_worker.moveToThread(self._batch_thread)
        self._thread_started_connection = self._batch_thread.started.connect(
            self._batch_worker.run
        )
        self._batch_worker.progress.connect(self._on_batch_progress)
        self._batch_worker.file_result.connect(self._on_batch_file_result)
        self._batch_worker.finished.connect(self._on_batch_finished)

        self._source_stage = input_stage
        self.deconvolve_btn.setEnabled(False)
        self._emit_progress(PageProgress(active=True, done=0, total=len(paths)))

        self._batch_thread.start()

    def _emit_progress(self, progress: PageProgress):
        self.progress_changed.emit(progress)
        self.batch_progress.apply(progress)

    def _on_batch_progress(self, done: int, total: int, name: str):
        self._emit_progress(PageProgress(active=True, done=done, total=total, message=name))
        self.status_label.setText(f"Deconvolving {name} ({done}/{total})")

    def _on_batch_file_result(self, result):
        if self._source_stage:
            self.tree_panel.mark_result(
                self._source_stage, result.path, result.success, result.error
            )
        if result.success:
            self.status_label.setText(f"{result.path.name}: saved to {result.output_path}")
        else:
            self.status_label.setText(f"{result.path.name}: {result.error}")

    def _on_batch_finished(self):
        self._emit_progress(PageProgress(active=False))
        self.deconvolve_btn.setEnabled(True)
        self.status_label.setText("Batch deconvolve complete.")

        root = self.tree_panel.project_root()
        if root and self._source_stage:
            processed = [
                Path(p).stem
                for p in self.tree_panel.checked_paths_for_stage(self._source_stage)
            ]
            project_core.mark_stage_run(root, project_core.STAGE_DECONVOLVED, processed)
        self.tree_panel.refresh()

        self._batch_thread.quit()
        self._batch_thread.wait()
        self._batch_thread = None
        self._batch_worker = None

    # ------------------------------------------------------------------

    def shutdown(self):
        settings.set_default_deconvolve_params(self.params_panel.params())
        self.controller.shutdown()
        if self._batch_thread is not None:
            self._batch_thread.quit()
            self._batch_thread.wait()
