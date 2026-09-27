"""Classifier Training page: trains the yeast-tile classifier on the
human-annotated tiles of the shared classification pool
(`ui/classifier_pool.py`). Two tabs, each with the same shape:

- Supervised Training fits the classifier (`tileclass.training.supervised`);
- VICReg Pretraining fits a headless backbone (`tileclass.training.vicreg`)
  that supervised runs (and later VICReg runs) can start from.

Settings (pool, parameters, starting point) sit in the left column; the
right side shows progress and the Deploy group. A run always saves to the
first pooled project's session folder (`core.classify.supervised_output_dir`
/ `vicreg_output_dir`), never straight to the live slot the tile viewer and
Classify Tiles use -- "Deploy Latest" copies it there (backing up whatever
was deployed) once you've looked at the run.
"""

from pathlib import Path

from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tileclass.checkpoint_import import import_checkpoint
from tileclass.training.vicreg import warm_start_overlap

from yeastprep.core.classify import find_project_checkpoint, supervised_output_dir, vicreg_output_dir

from ..classifier_pool import ClassifierPool, ClassifierPoolWidget
from ..classify_params_panel import SupervisedTrainParamsPanel, VicregTrainParamsPanel
from ..common.checkpoint_choice import (
    DEPLOYED_PATHS,
    CheckpointChoice,
    describe_checkpoint,
    pick_checkpoint_file,
)
from ..diagnostics.classifier_training_monitor_panel import ClassifierTrainingMonitorPanel
from ..worker import ClassifierTrainingWorker, ClassifierVicregWorker
from .page_progress import PageProgress


class _TrainingTab(QWidget):
    """One training mode. Never shown itself: `left_panel` (params +
    starting point) and `right_panel` (progress, Start/Cancel, Deploy) are
    placed by `ClassifierTrainingPage`. Subclasses fill in the class
    attributes and the worker/result hooks."""

    progress_changed = Signal(object)  # PageProgress

    kind = ""  # "classifier" | "backbone", see checkpoint_choice.py
    session_kind = ""  # core.classify.find_project_checkpoint's `kind`
    secondary_label = None  # second plotted metric, if any

    def __init__(self, pool: ClassifierPool, parent=None):
        super().__init__(parent)
        self.pool = pool
        self._thread = None
        self._worker = None
        self._latest_path: Path | None = None  # this session's last run

        self._build_ui()
        self.start_btn.clicked.connect(self._start_training)
        self.cancel_btn.clicked.connect(self._cancel_training)
        self.deploy_latest_btn.clicked.connect(self._deploy_latest)
        self.deploy_other_btn.clicked.connect(self._deploy_other)
        self.monitor_panel.datasetTabActivated.connect(self.refresh_dataset_summary)
        pool.changed.connect(self.refresh_dataset_summary)
        pool.changed.connect(self._refresh_deploy_labels)
        self._refresh_deploy_labels()

    # ------------------------------------------------------------------
    # Subclass hooks

    def _make_params_panel(self) -> QWidget:
        raise NotImplementedError

    def _output_dir(self, project_root) -> Path:
        raise NotImplementedError

    def _build_worker(self, records, output_dir, backbone_path):
        raise NotImplementedError

    def _on_epoch_progress(self, progress):
        raise NotImplementedError

    def _result_summary(self, result) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # UI

    def _build_ui(self):
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.params_panel = self._make_params_panel()
        left_layout.addWidget(self.params_panel)

        start_group = QGroupBox("Starting point (backbone)")
        start_layout = QVBoxLayout(start_group)
        self.starting_point = CheckpointChoice("backbone", allow_imagenet=True)
        self.starting_point.setToolTip(
            "Backbone weights this run starts from. A VICReg-pretrained "
            "backbone (trained on your own tiles) is recommended over "
            "ImageNet weights whenever one is available."
        )
        start_layout.addWidget(self.starting_point)
        left_layout.addWidget(start_group)
        left_layout.addStretch(1)

        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.monitor_panel = ClassifierTrainingMonitorPanel()
        right_layout.addWidget(self.monitor_panel, 1)

        self.progress_label = QLabel("Ready")
        right_layout.addWidget(self.progress_label)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        buttons.addWidget(self.start_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        buttons.addWidget(self.cancel_btn)
        right_layout.addLayout(buttons)

        deploy_group = QGroupBox("Deploy")
        deploy_layout = QVBoxLayout(deploy_group)
        self.latest_label = QLabel()
        self.latest_label.setWordWrap(True)
        deploy_layout.addWidget(self.latest_label)
        deploy_buttons = QHBoxLayout()
        self.deploy_latest_btn = QPushButton("Deploy Latest")
        self.deploy_latest_btn.setToolTip(
            "Copy the latest checkpoint into the live slot, backing up "
            "whatever is deployed now."
        )
        deploy_buttons.addWidget(self.deploy_latest_btn)
        self.deploy_other_btn = QPushButton("Deploy Other...")
        self.deploy_other_btn.setToolTip("Deploy a checkpoint file from anywhere.")
        deploy_buttons.addWidget(self.deploy_other_btn)
        deploy_buttons.addStretch(1)
        deploy_layout.addLayout(deploy_buttons)
        self.deployed_label = QLabel()
        self.deployed_label.setWordWrap(True)
        self.deployed_label.setStyleSheet("color: #a0a0a0;")
        deploy_layout.addWidget(self.deployed_label)
        right_layout.addWidget(deploy_group)

    # ------------------------------------------------------------------

    def refresh_dataset_summary(self):
        pooled = self.pool.pooled_annotations()
        if pooled is not None:
            self.monitor_panel.set_dataset_summary(pooled, self.pool.tile_paths())

    def _latest_checkpoint(self) -> Path | None:
        """This session's last run, else a previous session's output in the
        first pooled project's session folder."""
        if self._latest_path is not None and self._latest_path.exists():
            return self._latest_path
        roots = self.pool.roots()
        found = find_project_checkpoint(roots[0], self.session_kind) if roots else None
        return found[0] if found else None

    def _refresh_deploy_labels(self):
        latest = self._latest_checkpoint()
        self.deploy_latest_btn.setEnabled(latest is not None)
        if latest is None:
            self.latest_label.setText("Latest: nothing trained yet.")
        else:
            self.latest_label.setText(f"Latest: {latest}\n{describe_checkpoint(latest)}")
        deployed = DEPLOYED_PATHS[self.kind]
        self.deployed_label.setText(
            f"Deployed: {describe_checkpoint(deployed)}"
            if deployed.exists()
            else "Deployed: nothing yet."
        )

    # ------------------------------------------------------------------
    # Training

    def _start_training(self):
        if self._thread is not None:
            return
        records = self.pool.gather_confirmed_records()
        if not records:
            QMessageBox.warning(
                self,
                "yeastprep",
                "No human-annotated tiles to train on -- add a project with "
                "annotated FOVs to the classification pool.",
            )
            return

        output_dir = self._output_dir(self.pool.roots()[0])
        backbone_path = self.starting_point.weights_path()

        self.monitor_panel.clear(self.secondary_label)
        self.monitor_panel.log(f"records={len(records)}  saving to {output_dir}")
        start_line = f"starting point: {self.starting_point.describe()}"
        if backbone_path is not None:
            overlap = warm_start_overlap(
                [p for p, _ in records], meta_path=backbone_path.with_name("meta.json")
            )
            if overlap is not None:
                start_line += f"; {overlap[0]}/{overlap[1]} pooled crops already seen by it"
        self.monitor_panel.log(start_line)
        self.refresh_dataset_summary()

        self.progress_label.setText("Training...")
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_changed.emit(PageProgress(active=True))

        self._worker = self._build_worker(records, output_dir, backbone_path)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_epoch_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _finish(self, status_text: str):
        self.progress_label.setText(status_text)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_changed.emit(PageProgress(active=False))
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None

    def _apply_result(self, result):
        self._latest_path = Path(result.weights_path)
        self.monitor_panel.log(self._result_summary(result))
        self._refresh_deploy_labels()

    def _on_finished(self, result):
        self._apply_result(result)
        self._finish("Training completed.")

    def _on_cancelled(self, result):
        # `result` is the best epoch reached before cancelling (already
        # saved), or None if no epoch finished.
        if result is None:
            self._finish("Training cancelled -- no epoch finished yet, nothing saved.")
            return
        self._apply_result(result)
        self._finish("Training cancelled -- best epoch saved.")

    def _on_error(self, message: str):
        self.monitor_panel.log(f"ERROR: {message}")
        self._finish(f"Training error: {message.splitlines()[0] if message else ''}")

    def _cancel_training(self):
        if self._worker is not None:
            self._worker.cancel()
            self.progress_label.setText("Cancelling after current epoch...")
            self.cancel_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Deploy

    def _deploy_latest(self):
        latest = self._latest_checkpoint()
        if latest is not None:
            self._deploy(latest)

    def _deploy_other(self):
        chosen = pick_checkpoint_file(self, self.kind)
        if chosen is not None:
            self._deploy(chosen)

    def _deploy(self, weights_path: Path):
        live_weights = DEPLOYED_PATHS[self.kind]
        try:
            import_checkpoint(
                weights_path,
                weights_path.with_name("meta.json"),
                live_weights,
                live_weights.with_name("meta.json"),
            )
        except Exception as exc:
            QMessageBox.critical(self, "yeastprep", f"Deploy failed: {exc}")
            return
        self.monitor_panel.log(f"deployed {weights_path} -> {live_weights}")
        self._refresh_deploy_labels()

    def shutdown(self):
        if self._thread is not None:
            self._cancel_training()
            self._thread.quit()
            self._thread.wait()


class _SupervisedTrainingTab(_TrainingTab):
    kind = "classifier"
    session_kind = "supervised"
    secondary_label = "val accuracy"

    def _make_params_panel(self):
        return SupervisedTrainParamsPanel()

    def _output_dir(self, project_root):
        return supervised_output_dir(project_root)

    def _build_worker(self, records, output_dir, backbone_path):
        return ClassifierTrainingWorker(
            records,
            params=self.params_panel.params(),
            output_dir=output_dir,
            backbone_weights_path=backbone_path,
        )

    def _on_epoch_progress(self, progress):
        self.monitor_panel.append_supervised_epoch(progress)
        self.progress_label.setText(
            f"[{progress.stage}] epoch {progress.epoch}/{progress.total_epochs}  "
            f"loss={progress.avg_loss:.4f}"
        )

    def _result_summary(self, result):
        return (
            f"val_accuracy={result.val_accuracy:.3f}  train={result.train_count}  "
            f"val={result.val_count}  categories={result.categories}"
        )


class _VicregTrainingTab(_TrainingTab):
    kind = "backbone"
    session_kind = "vicreg"
    secondary_label = "std"

    def _make_params_panel(self):
        return VicregTrainParamsPanel()

    def _output_dir(self, project_root):
        return vicreg_output_dir(project_root)

    def _build_worker(self, records, output_dir, backbone_path):
        return ClassifierVicregWorker(
            records,
            params=self.params_panel.params(),
            output_dir=output_dir,
            backbone_weights_path=backbone_path,
        )

    def _on_epoch_progress(self, progress):
        self.monitor_panel.append_vicreg_epoch(progress)
        self.progress_label.setText(
            f"epoch {progress.epoch}/{progress.total_epochs}  loss={progress.avg_loss:.4f}"
        )

    def _result_summary(self, result):
        return (
            f"final_loss={result.final_loss:.4f}  categories={result.categories}  "
            f"singleton_categories={result.singleton_categories}"
        )


class ClassifierTrainingPage(QWidget):
    progress_changed = Signal(object)  # PageProgress

    def __init__(self, pool: ClassifierPool, parent=None):
        super().__init__(parent)
        self.pool = pool

        outer = QVBoxLayout(self)
        splitter = QSplitter()
        outer.addWidget(splitter, 1)

        # Left: the pool, then whichever tab's own settings match the tab
        # showing on the right.
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(ClassifierPoolWidget(pool))
        self.settings_stack = QStackedWidget()
        left_layout.addWidget(self.settings_stack, 1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(340)
        left_scroll.setMaximumWidth(440)
        splitter.addWidget(left_scroll)

        self.tabs = QTabWidget()
        self.supervised_tab = _SupervisedTrainingTab(pool)
        self.vicreg_tab = _VicregTrainingTab(pool)
        for tab, title in (
            (self.supervised_tab, "Supervised Training"),
            (self.vicreg_tab, "VICReg Pretraining"),
        ):
            self.tabs.addTab(tab.right_panel, title)
            self.settings_stack.addWidget(tab.left_panel)
            tab.progress_changed.connect(self.progress_changed)
        self.tabs.currentChanged.connect(self.settings_stack.setCurrentIndex)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def shutdown(self):
        self.supervised_tab.shutdown()
        self.vicreg_tab.shutdown()
