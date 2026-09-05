"""Classifier Training page: fits the yeast-tile classifier (supervised
fine-tuning, or VICReg self-supervised backbone pretraining) on annotated
tile crops pooled from one or more projects' `05_tiles/` FOV folders -- the
"cumulative dataset ... to make the network more general" requirement,
applied to classifier training the way `train_denoise_page.TrainDenoisePage`
applies it to denoise training (this page's pooling UX is a direct port of
that one, see `_classifier_pool_widget.ClassifierPoolWidget`).

Training and inference are deliberately split across two *pages* now (not
just two apps): this page always writes a training run's resulting
checkpoint to a project-local session folder
(`core.classify.supervised_output_dir`/`vicreg_output_dir`), never to
tileclass's live inference slot directly -- promoting a checkpoint there is
a separate, explicit "Deploy to Tile Classifier" action per tab, so a pooled
multi-project training run can never silently overwrite what tileclass's
auto-annotate is currently using without the user reviewing this page's
diagnostics first (see `_save_weights`'s `output_dir` docstring in
`tileclass.training.supervised`/`vicreg` for the underlying contract this
relies on). *Using* a trained/deployed checkpoint to classify tiles, or to
explore its embeddings, lives on the separate Classify Tiles page
(`classify_tiles_page.ClassifyTilesPage`) instead -- this page only ever
trains and (optionally) deploys.

One pool is shared by both tabs (QTabWidget below), each tab wraps its own
worker/params/diagnostics -- see `_SupervisedTrainingTab`/`_VicregTrainingTab`.
"""

import json
from pathlib import Path

from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tileclass.checkpoint_import import import_checkpoint
from tileclass.classifiers.yeast_efficientnet import META_PATH as LIVE_CLASSIFIER_META_PATH
from tileclass.classifiers.yeast_efficientnet import WEIGHTS_PATH as LIVE_CLASSIFIER_WEIGHTS_PATH
from tileclass.training.vicreg import VICREG_META_PATH as LIVE_VICREG_META_PATH
from tileclass.training.vicreg import VICREG_WEIGHTS_PATH as LIVE_VICREG_WEIGHTS_PATH
from tileclass.training.vicreg import warm_start_overlap

from yeastprep.core.classify import default_supervised_checkpoint_paths, default_vicreg_checkpoint_paths

from ..classify_params_panel import SupervisedTrainParamsPanel, VicregTrainParamsPanel
from ..common.checkpoint_file_picker import CheckpointFilePicker as _CheckpointFilePicker
from ..diagnostics.classifier_training_monitor_panel import ClassifierTrainingMonitorPanel
from ..project_tree_panel import ProjectTreePanel
from ..worker import ClassifierTrainingWorker, ClassifierVicregWorker
from ._classifier_pool_widget import ClassifierPoolWidget
from .page_progress import PageProgress


class _BaseTrainingTab(QWidget):
    """Shared shell for the two tabs below: params panel + checkpoint-output
    path field + Start/Cancel on the left, diagnostics panel + a Deploy
    group (checkpoint picker + Deploy button) on the right. Subclasses
    supply the params panel, the worker class, and the deploy destination;
    `_start_training`/`_finish` etc. are generic over that.

    Deploy is deliberately independent of whether a training run happened
    in this session at all: its `_CheckpointFilePicker` defaults to
    whatever this session just trained (once it has), but is just as
    happily pointed -- via Browse, or by typing a path -- at a checkpoint
    saved anywhere else entirely, e.g. from a previous session, a shared
    folder, or another user's run. Any `weights.pth` with a sibling
    `meta.json` of the right kind can be deployed."""

    progress_changed = Signal(object)  # PageProgress
    # Emitted (weights_path, is_vicreg) right after a run finishes -- lets
    # the Classify Tiles page default its own pickers to this session's
    # fresh checkpoint, the way `deploy_picker` already does here, without
    # duplicating any inference/embedding code in this page (see
    # `main_window.py`'s wiring of `ClassifierTrainingPage.checkpointTrained`
    # to `ClassifyTilesPage.set_default_checkpoint`).
    checkpointTrained = Signal(Path, bool)

    _live_weights_filename = "weights.pth"  # overridden per subclass
    _live_meta_filename = "meta.json"
    _is_vicreg = False  # overridden by _VicregTrainingTab

    def __init__(self, pool_widget: ClassifierPoolWidget, parent=None):
        super().__init__(parent)
        self.pool_widget = pool_widget
        self._thread = None
        self._worker = None
        self._checkpoint_path_is_default = True
        self._last_checkpoint_dir: Path | None = None

        self._build_ui()
        self._wire_up()
        self._refresh_default_checkpoint_path()

    # ------------------------------------------------------------------
    # UI construction -- subclasses provide the params panel via hooks.

    def _make_params_panel(self) -> QWidget:
        raise NotImplementedError

    def _default_checkpoint_paths(self, project_root) -> tuple[Path, Path]:
        raise NotImplementedError

    def _live_slot_paths(self) -> tuple[Path, Path]:
        raise NotImplementedError

    def _build_extra_left_widgets(self, left_layout: QVBoxLayout):
        """Hook for a subclass-specific widget below the checkpoint-output
        group, on the left (inputs) side -- used for a "warm-starting
        from..." indicator (see `_describe_warm_start`/`_refresh_warm_start_label`
        below). No-op by default."""

    def _pre_start_check(self) -> bool:
        """Hook called right before a training run actually starts (after
        the records/output-dir checks, before the worker is built) --
        return `False` to abort (a `QMessageBox` should already explain
        why, by that point). Overridden by `_SupervisedTrainingTab` to
        resolve its "Starting point" backbone picker into a
        `backbone_weights_path` for `_build_worker`/`_describe_warm_start`
        to use. No-op (always proceeds) by default."""
        return True

    def _describe_warm_start(self, records) -> str | None:
        """One-line description of what a training run starting right now
        would warm-start from, appended to `_start_training`'s log line so
        it's never a mystery after the fact -- overridden per subclass
        since the two kinds of training resolve this differently
        (VICReg: a fixed live-slot check; supervised: whatever
        `_pre_start_check` just resolved from the "Starting point" picker,
        see `_SupervisedTrainingTab`'s override). `None` (the default) logs
        nothing extra."""
        return None

    def _on_deployed(self):
        """Hook called after a successful `_deploy_to_tile_classifier` --
        overridden by `_VicregTrainingTab` to refresh its warm-start label,
        since deploying is exactly the action that changes what a future
        VICReg run would warm-start from."""

    def _deploy_expects_vicreg(self) -> bool:
        """Whether `deploy_picker`'s checkpoint should be a headless VICReg
        backbone (`True`, `_VicregTrainingTab`) or a classifier with a head
        (`False`, `_SupervisedTrainingTab`'s default) -- passed to
        `_CheckpointFilePicker.resolve`'s `expect_vicreg`."""
        return False

    def _deploy_wrong_kind_message(self) -> str:
        return (
            "{meta_path} doesn't look like the right kind of checkpoint for "
            "this tab's live slot -- pick a checkpoint from the matching "
            "training tab instead."
        )

    def _deploy_button_text(self) -> str:
        return "Deploy to Tile Classifier"

    def _deploy_button_tooltip(self) -> str:
        return (
            "Copy the checkpoint chosen above onto tileclass's live "
            "inference slot, backing up whatever's currently deployed "
            "first. Works for any checkpoint with this kind's sibling "
            "meta.json -- this session's just-trained one (filled in "
            "automatically once training finishes), an older session's, or "
            "one Browsed from anywhere."
        )

    def _build_ui(self):
        outer = QVBoxLayout(self)
        row = QHBoxLayout()
        outer.addLayout(row, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.params_panel = self._make_params_panel()
        left_layout.addWidget(self.params_panel)
        left_layout.addWidget(self._build_checkpoint_group())
        self._build_extra_left_widgets(left_layout)
        left_layout.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(340)
        left_scroll.setMaximumWidth(440)
        row.addWidget(left_scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.monitor_panel = ClassifierTrainingMonitorPanel()
        right_layout.addWidget(self.monitor_panel, 1)

        self.progress_bar_label = QLabel("Ready")
        right_layout.addWidget(self.progress_bar_label)

        buttons = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        buttons.addWidget(self.start_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        buttons.addWidget(self.cancel_btn)
        right_layout.addLayout(buttons)

        deploy_group = QGroupBox("Deploy")
        deploy_layout = QVBoxLayout(deploy_group)
        self.deploy_picker = _CheckpointFilePicker(
            placeholder=self._live_weights_filename,
            tooltip=(
                "Checkpoint to deploy -- defaults to this session's "
                "just-trained checkpoint once training finishes. Browse to "
                "pick any other one instead (an older session's, a backup, "
                "a colleague's), or use 'Deployed'/'View Metadata...' to "
                "inspect what's currently live before replacing it."
            ),
            deployed_path=self._live_slot_paths()[0],
            deployed_tooltip="Load the path of tileclass's currently deployed checkpoint.",
        )
        deploy_layout.addWidget(self.deploy_picker)
        self.deploy_btn = QPushButton(self._deploy_button_text())
        self.deploy_btn.setToolTip(self._deploy_button_tooltip())
        deploy_layout.addWidget(self.deploy_btn)
        right_layout.addWidget(deploy_group)

        row.addWidget(right, 1)

        self.status_label = QLabel("")
        outer.addWidget(self.status_label)

    def _build_checkpoint_group(self) -> QGroupBox:
        group = QGroupBox("Save Trained Weights To")
        v = QVBoxLayout(group)
        row = QHBoxLayout()
        self.checkpoint_path_edit = QLineEdit()
        self.checkpoint_path_edit.setToolTip(
            "Where this run's weights.pth/meta.json get written -- defaults "
            "to a session folder under the first pooled project's root, but "
            "can be any folder you choose (edit the path or Browse). Never "
            "written automatically to tileclass's live classifier -- use "
            "the Deploy group on the right for that, pointed at this folder "
            "(or anywhere else a checkpoint of this kind lives)."
        )
        row.addWidget(self.checkpoint_path_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_checkpoint_dir)
        row.addWidget(browse_btn)
        v.addLayout(row)
        return group

    # ------------------------------------------------------------------

    def _wire_up(self):
        self.pool_widget.pool_changed.connect(self._refresh_default_checkpoint_path)
        self.pool_widget.pool_changed.connect(self.refresh_dataset_summary)
        self.checkpoint_path_edit.textEdited.connect(self._on_checkpoint_path_edited)
        self.start_btn.clicked.connect(self._start_training)
        self.cancel_btn.clicked.connect(self._cancel_training)
        self.deploy_btn.clicked.connect(self._deploy_to_tile_classifier)
        self.monitor_panel.datasetTabActivated.connect(self.refresh_dataset_summary)

    def refresh_dataset_summary(self):
        pooled = self.pool_widget.pooled_annotations()
        if pooled is not None:
            self.monitor_panel.set_dataset_summary(pooled)

    # ------------------------------------------------------------------
    # Checkpoint output path -- defaults into the first pooled project's
    # session folder, same "obvious non-blank default unless you
    # deliberately Browse/edit" convention as TrainDenoisePage's checkpoint
    # field.

    def _default_checkpoint_dir(self) -> Path | None:
        fov_dirs = self.pool_widget.checked_fov_dirs()
        if not fov_dirs:
            return None
        # A pooled FOV dir is <project_root>/05_tiles/<fov_id>; walk up two
        # levels to recover the project root.
        project_root = Path(fov_dirs[0]).parent.parent
        weights_path, _meta_path = self._default_checkpoint_paths(project_root)
        return weights_path.parent

    def _refresh_default_checkpoint_path(self):
        if not self._checkpoint_path_is_default:
            return
        default_dir = self._default_checkpoint_dir()
        self.checkpoint_path_edit.setText(str(default_dir) if default_dir else "")

    def _on_checkpoint_path_edited(self, _text):
        self._checkpoint_path_is_default = False

    def _browse_checkpoint_dir(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Checkpoint Output Folder", self.checkpoint_path_edit.text()
        )
        if chosen:
            self.checkpoint_path_edit.setText(chosen)
            self._checkpoint_path_is_default = False

    def _output_dir(self) -> Path | None:
        text = self.checkpoint_path_edit.text().strip()
        return Path(text) if text else None

    # ------------------------------------------------------------------
    # Training -- subclasses implement `_build_worker`/`_on_epoch_progress`/
    # `_on_result` for their result-shape-specific handling.

    def _build_worker(self, records, output_dir):
        raise NotImplementedError

    def _on_epoch_progress(self, progress):
        raise NotImplementedError

    def _on_result(self, result):
        raise NotImplementedError

    def _start_training(self):
        if self._thread is not None:
            return

        records = self.pool_widget.gather_confirmed_records()
        if not records:
            QMessageBox.warning(
                self,
                "yeastprep",
                "No confirmed annotated tiles to train on -- add a project with "
                "annotated FOVs to the pool, and make sure at least one FOV is checked.",
            )
            return

        output_dir = self._output_dir()
        if output_dir is None:
            QMessageBox.warning(
                self, "yeastprep", "Choose a checkpoint output folder first."
            )
            return

        if not self._pre_start_check():
            return

        self.monitor_panel.clear(self._secondary_label())
        self.monitor_panel.log(f"records={len(records)}  output_dir={output_dir}")
        warm_start = self._describe_warm_start(records)
        if warm_start is not None:
            self.monitor_panel.log(warm_start)
        self.refresh_dataset_summary()
        self.status_label.setText("Training...")
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self._emit_progress(PageProgress(active=True, done=0, total=0))

        self._worker = self._build_worker(records, output_dir)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_epoch_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.error.connect(self._on_error)
        self._thread.start()

    def _secondary_label(self) -> str:
        return "val accuracy"

    def _emit_progress(self, progress: PageProgress):
        self.progress_changed.emit(progress)

    def _teardown_thread(self):
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._worker = None

    def _finish(self, status_text: str):
        self.status_label.setText(status_text)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self._emit_progress(PageProgress(active=False))
        self._teardown_thread()

    def _on_finished(self, result):
        self._last_checkpoint_dir = Path(result.weights_path).parent
        self.deploy_picker.set_default_path(
            self._last_checkpoint_dir / self._live_weights_filename
        )
        self.checkpointTrained.emit(Path(result.weights_path), self._is_vicreg)
        self._on_result(result)
        self._finish("Training completed.")

    def _on_cancelled(self):
        self._finish("Training cancelled.")

    def _on_error(self, message: str):
        self.monitor_panel.log(f"ERROR: {message}")
        self._finish(f"Training error: {message.splitlines()[0] if message else ''}")

    def _cancel_training(self):
        if self._worker is not None:
            self._worker.cancel()
            self.status_label.setText("Cancelling after current epoch...")
            self.cancel_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Deploy: promote whatever checkpoint `deploy_picker` points at into
    # tileclass's live slot -- this session's just-trained one by default,
    # but just as validly anything else typed/Browsed into the field (see
    # class docstring).

    def _deploy_to_tile_classifier(self):
        resolved = self.deploy_picker.resolve(
            expect_vicreg=self._deploy_expects_vicreg(),
            wrong_kind_message=self._deploy_wrong_kind_message(),
        )
        if resolved is None:
            return
        weights_src, meta_src = resolved
        weights_dest, meta_dest = self._live_slot_paths()
        try:
            import_checkpoint(weights_src, meta_src, weights_dest, meta_dest)
        except Exception as exc:
            QMessageBox.critical(self, "yeastprep", f"Deploy failed: {exc}")
            return
        self.monitor_panel.log(f"deployed {weights_src} -> {weights_dest}")
        self.status_label.setText("Deployed to tile classifier.")
        self._on_deployed()

    # ------------------------------------------------------------------

    def shutdown(self):
        if self._thread is not None:
            self._cancel_training()
            self._thread.quit()
            self._thread.wait()


class _SupervisedTrainingTab(_BaseTrainingTab):
    _live_weights_filename = "weights.pth"
    _live_meta_filename = "meta.json"

    def _make_params_panel(self) -> QWidget:
        return SupervisedTrainParamsPanel()

    def _default_checkpoint_paths(self, project_root):
        return default_supervised_checkpoint_paths(project_root)

    def _live_slot_paths(self):
        return LIVE_CLASSIFIER_WEIGHTS_PATH, LIVE_CLASSIFIER_META_PATH

    # ------------------------------------------------------------------
    # Starting point: `tileclass.training.supervised._init_model` never
    # warm-starts the *classifier head* from a deployed classifier (see
    # module docstring there -- every run's train/val split is recomputed
    # fresh, so warm-starting the head would leak). Its *backbone*,
    # however, can start from either a generic ImageNet-pretrained stem or
    # a VICReg-pretrained one (`backbone_weights_path`) -- and the latter
    # is the recommended default whenever a VICReg backbone has been
    # pretrained on this project's own tile crops (see the NN_workflow
    # scripts this GUI's training was ported from), since it starts
    # finetuning from features already tuned to this data instead of
    # generic photographs. This panel makes that choice explicit and
    # resolves it just before a run starts (`_pre_start_check`), rather
    # than the page silently always cold-starting from ImageNet the way it
    # used to.

    def _build_extra_left_widgets(self, left_layout):
        self._resolved_backbone_path: Path | None = None

        group = QGroupBox("Starting Point (Backbone)")
        v = QVBoxLayout(group)

        self.backbone_warm_start_cb = QCheckBox(
            "Start from VICReg-pretrained backbone (recommended)"
        )
        self.backbone_warm_start_cb.setChecked(True)
        self.backbone_warm_start_cb.setToolTip(
            "Checked (recommended): initialize the backbone below from a "
            "VICReg checkpoint -- pretrained on this project's own tile "
            "crops -- instead of generic ImageNet weights, then attach a "
            "freshly initialized classifier head on top. Unchecked: cold-"
            "start the backbone from ImageNet instead. Either way the "
            "classifier head itself is always freshly initialized (see "
            "tooltip above) -- this only changes the backbone's starting "
            "weights. Falls back to ImageNet automatically if no VICReg "
            "backbone is available yet."
        )
        v.addWidget(self.backbone_warm_start_cb)

        self.backbone_picker = _CheckpointFilePicker(
            placeholder="backbone.pth",
            tooltip=(
                "VICReg backbone checkpoint to start from -- defaults to "
                "the currently deployed backbone. Browse to pick a "
                "different one (an older VICReg session's backbone.pth, a "
                "backup, ...), or use 'View Metadata...' to check what "
                "it was pretrained on before committing to it."
            ),
            deployed_path=LIVE_VICREG_WEIGHTS_PATH,
            deployed_tooltip="Use tileclass's currently deployed VICReg backbone.",
        )
        v.addWidget(self.backbone_picker)

        self.backbone_status_label = QLabel()
        self.backbone_status_label.setWordWrap(True)
        v.addWidget(self.backbone_status_label)

        left_layout.addWidget(group)

        self.backbone_picker.set_default_path(LIVE_VICREG_WEIGHTS_PATH)
        self.backbone_warm_start_cb.stateChanged.connect(self._refresh_backbone_status)
        self.backbone_picker.pathChanged.connect(self._refresh_backbone_status)
        self._refresh_backbone_status()

    def _refresh_backbone_status(self, *_args):
        enabled = self.backbone_warm_start_cb.isChecked()
        self.backbone_picker.setEnabled(enabled)
        if not enabled:
            self.backbone_status_label.setText("Cold start: ImageNet-pretrained stem.")
            return

        text = self.backbone_picker.edit.text().strip()
        if not text or not Path(text).is_file():
            self.backbone_status_label.setText(
                "No VICReg backbone available yet -- will fall back to an "
                "ImageNet-pretrained stem. Pretrain one in the VICReg "
                "Pretraining tab first to enable this."
            )
            return

        meta_path = Path(text).with_name("meta.json")
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            meta = {}
        last_trained = meta.get("last_trained", "unknown")
        categories = meta.get("categories") or []
        counts = meta.get("category_counts") or {}
        categories_str = (
            ", ".join(f"{c} ({counts[c]})" if c in counts else c for c in categories)
            if categories
            else "n/a"
        )
        self.backbone_status_label.setText(
            f"{text}\ntrained: {last_trained}\n"
            f"categories seen: {categories_str}"
        )

    def _pre_start_check(self) -> bool:
        if not self.backbone_warm_start_cb.isChecked():
            self._resolved_backbone_path = None
            return True

        text = self.backbone_picker.edit.text().strip()
        if not text or not Path(text).is_file():
            # Nothing deployed/chosen yet -- fall back to ImageNet silently
            # rather than blocking on a checkpoint that was never trained,
            # same as `VICRegParams.warm_start`'s existence check.
            self._resolved_backbone_path = None
            return True

        resolved = self.backbone_picker.resolve(
            expect_vicreg=True,
            wrong_kind_message=(
                "{meta_path} doesn't look like a VICReg backbone checkpoint "
                "-- pick a backbone.pth from the VICReg Pretraining tab "
                "instead, or uncheck 'Start from VICReg-pretrained backbone' "
                "above to cold-start from ImageNet."
            ),
        )
        if resolved is None:
            return False
        self._resolved_backbone_path = resolved[0]
        return True

    def _describe_warm_start(self, records) -> str:
        if self._resolved_backbone_path is not None:
            return f"backbone: VICReg-pretrained ({self._resolved_backbone_path}); classifier head freshly initialized"
        return (
            "backbone: ImageNet-pretrained stem (no VICReg backbone "
            "selected/available); classifier head freshly initialized"
        )

    def _build_worker(self, records, output_dir):
        return ClassifierTrainingWorker(
            records,
            params=self.params_panel.params(),
            output_dir=output_dir,
            backbone_weights_path=self._resolved_backbone_path,
        )

    def _on_epoch_progress(self, progress):
        self.monitor_panel.append_supervised_epoch(progress)
        self.progress_bar_label.setText(
            f"[{progress.stage}] epoch {progress.epoch}/{progress.total_epochs}  "
            f"loss={progress.avg_loss:.4f}"
        )

    def _on_result(self, result):
        self.monitor_panel.log(
            f"val_accuracy={result.val_accuracy:.3f}  "
            f"train={result.train_count}  val={result.val_count}  "
            f"categories={result.categories}"
        )


class _VicregTrainingTab(_BaseTrainingTab):
    _live_weights_filename = "backbone.pth"
    _live_meta_filename = "meta.json"
    _is_vicreg = True

    def _make_params_panel(self) -> QWidget:
        return VicregTrainParamsPanel()

    def _default_checkpoint_paths(self, project_root):
        return default_vicreg_checkpoint_paths(project_root)

    def _live_slot_paths(self):
        return LIVE_VICREG_WEIGHTS_PATH, LIVE_VICREG_META_PATH

    def _secondary_label(self) -> str:
        return "std"

    def _deploy_expects_vicreg(self) -> bool:
        return True

    def _deploy_wrong_kind_message(self) -> str:
        return (
            "{meta_path} doesn't look like a VICReg backbone checkpoint -- "
            "pick a backbone.pth from this tab instead (this session's own, "
            "the Deployed backbone, or another VICReg session's backbone.pth)."
        )

    def _deploy_button_text(self) -> str:
        return "Deploy as VICReg Backbone"

    def _deploy_button_tooltip(self) -> str:
        return (
            "Copy the backbone chosen above onto the live VICReg warm-start "
            "slot, backing up whatever's currently deployed first. This "
            "backbone has no classification head -- it feeds future VICReg "
            "pretraining runs (see 'Warm start' above) and Supervised "
            "Training's 'Starting Point' backbone picker, not a "
            "classification head by itself."
        )

    def _build_worker(self, records, output_dir):
        return ClassifierVicregWorker(
            records, params=self.params_panel.params(), output_dir=output_dir
        )

    # ------------------------------------------------------------------
    # Warm-start visibility: `pretrain_vicreg` checks `params.warm_start`
    # (the "Warm start from deployed backbone" checkbox in
    # `VicregTrainParamsPanel`) and `LIVE_VICREG_WEIGHTS_PATH` (see
    # `training/vicreg.py`) to decide whether to warm-start or start cold
    # from ImageNet -- entirely independent of the "Checkpoint output"
    # field above (that only controls where the *result* of this run gets
    # saved). This label always reflects the checkbox's current state, so
    # it's kept live via `params_changed`/`pool_changed`, not just
    # refreshed at Start-time.

    def _build_extra_left_widgets(self, left_layout):
        group = QGroupBox("Warm start")
        v = QVBoxLayout(group)
        self.warm_start_label = QLabel()
        self.warm_start_label.setWordWrap(True)
        v.addWidget(self.warm_start_label)
        left_layout.addWidget(group)
        self.params_panel.params_changed.connect(self._refresh_warm_start_label)
        self.pool_widget.pool_changed.connect(self._refresh_warm_start_label)
        self._refresh_warm_start_label()

    def _warm_start_overlap(self):
        """(already_seen, total) of the currently pooled records against
        whatever `LIVE_VICREG_WEIGHTS_PATH` was last trained on, or `None`
        if there's no pool, no deployed backbone, or no recorded
        provenance to compare against (see `training.vicreg.warm_start_overlap`)."""
        records = self.pool_widget.gather_confirmed_records()
        if not records:
            return None
        return warm_start_overlap([p for p, _ in records])

    def _refresh_warm_start_label(self, *_args):
        if not (LIVE_VICREG_WEIGHTS_PATH.exists() and LIVE_VICREG_META_PATH.exists()):
            self.warm_start_label.setText(
                "No VICReg backbone deployed yet -- will start cold from ImageNet."
            )
            return

        last_trained = None
        try:
            last_trained = json.loads(LIVE_VICREG_META_PATH.read_text()).get("last_trained")
        except (OSError, ValueError):
            pass
        suffix = f" (trained: {last_trained})" if last_trained else ""

        if not self.params_panel.params().warm_start:
            self.warm_start_label.setText(
                f"From scratch: ImageNet-pretrained stem -- warm start disabled.\n"
                f"Deployed backbone{suffix} left untouched as a starting point:\n"
                f"{LIVE_VICREG_WEIGHTS_PATH}"
            )
            return

        text = f"Warm start: deployed backbone{suffix}\n{LIVE_VICREG_WEIGHTS_PATH}"
        overlap = self._warm_start_overlap()
        if overlap is not None:
            already_seen, total = overlap
            text += (
                f"\n{already_seen}/{total} pooled crop(s) already seen by this "
                f"backbone ({total - already_seen} new)."
            )
        self.warm_start_label.setText(text)

    def _describe_warm_start(self, records) -> str:
        if not LIVE_VICREG_WEIGHTS_PATH.exists():
            return "from scratch: ImageNet-pretrained stem (no backbone deployed yet)"
        if not self.params_panel.params().warm_start:
            return "from scratch: ImageNet-pretrained stem (warm start disabled)"
        suffix = ""
        overlap = warm_start_overlap([p for p, _ in records])
        if overlap is not None:
            already_seen, total = overlap
            suffix = f"; {already_seen}/{total} pooled crop(s) already seen by this backbone"
        return f"warm start: deployed backbone ({LIVE_VICREG_WEIGHTS_PATH}){suffix}"

    def _on_deployed(self):
        self._refresh_warm_start_label()

    def _on_epoch_progress(self, progress):
        self.monitor_panel.append_vicreg_epoch(progress)
        self.progress_bar_label.setText(
            f"epoch {progress.epoch}/{progress.total_epochs}  loss={progress.avg_loss:.4f}"
        )

    def _on_result(self, result):
        self.monitor_panel.log(
            f"final_loss={result.final_loss:.4f}  categories={result.categories}  "
            f"singleton_categories={result.singleton_categories}"
        )
        self.monitor_panel.log(
            "Open the Classify Tiles page's 'Explore Embeddings' group to "
            "plot this backbone's embeddings (its picker now defaults to "
            "this session's checkpoint)."
        )


class ClassifierTrainingPage(QWidget):
    """Top-level page: one shared pool feeding two independent training
    tasks (see module docstring)."""

    progress_changed = Signal(object)  # PageProgress
    # Bubbles up whichever tab's run just finished -- see
    # `_BaseTrainingTab.checkpointTrained`; `main_window.py` wires this to
    # `ClassifyTilesPage.set_default_checkpoint`.
    checkpointTrained = Signal(Path, bool)

    def __init__(self, tree_panel: ProjectTreePanel, parent=None):
        super().__init__(parent)
        self.tree_panel = tree_panel

        self._build_ui()
        self._wire_up()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        self.pool_widget = ClassifierPoolWidget()
        outer.addWidget(self.pool_widget)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.supervised_tab = _SupervisedTrainingTab(self.pool_widget)
        self.tabs.addTab(self.supervised_tab, "Supervised Training")
        self.vicreg_tab = _VicregTrainingTab(self.pool_widget)
        self.tabs.addTab(self.vicreg_tab, "VICReg Pretraining")

    def _wire_up(self):
        self.supervised_tab.progress_changed.connect(self.progress_changed)
        self.vicreg_tab.progress_changed.connect(self.progress_changed)
        self.supervised_tab.checkpointTrained.connect(self.checkpointTrained)
        self.vicreg_tab.checkpointTrained.connect(self.checkpointTrained)

    # ------------------------------------------------------------------

    def load_selection(self, stage: str, path: str, mode: str):
        """Only `mode == "open_viewer_fov"` applies here (see
        `selection_actions.actions_for_selection`'s STAGE_TILES branch) --
        `path` is a FOV id, this page pool-adds its parent project rather
        than scoping to that one FOV, since training wants the project's
        whole annotated pool, not a single FOV (unlike the tile viewer's
        "open_viewer_fov" mode, which does scope to just that FOV)."""
        if mode != "open_viewer_fov":
            return
        root = self.tree_panel.project_root()
        if root:
            self.pool_widget.add_project(root)

    def shutdown(self):
        self.supervised_tab.shutdown()
        self.vicreg_tab.shutdown()
