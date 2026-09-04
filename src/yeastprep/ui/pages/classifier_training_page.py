"""Classifier Training page: fits the yeast-tile classifier (supervised
fine-tuning, or VICReg self-supervised backbone pretraining) on annotated
tile crops pooled from one or more projects' `05_tiles/` FOV folders -- the
"cumulative dataset ... to make the network more general" requirement,
applied to classifier training the way `train_denoise_page.TrainDenoisePage`
applies it to denoise training (this page's pooling UX is a direct port of
that one, see `_classifier_pool_widget.ClassifierPoolWidget`).

Training and inference are deliberately split across two apps now: this page
always writes a training run's resulting checkpoint to a project-local
session folder (`core.classify.supervised_output_dir`/`vicreg_output_dir`),
never to tileclass's live inference slot directly -- promoting a checkpoint
there is a separate, explicit "Deploy to Tile Classifier" action per tab, so
a pooled multi-project training run can never silently overwrite what
tileclass's auto-annotate is currently using without the user reviewing this
page's diagnostics first (see `_save_weights`'s `output_dir` docstring in
`tileclass.training.supervised`/`vicreg` for the underlying contract this
relies on).

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
from tileclass.classifiers.yeast_efficientnet import YeastEfficientNetClassifier
from tileclass.main_window import MainWindow
from tileclass.training.linear_probe import extract_embeddings, knn_accuracy, tsne_2d
from tileclass.training.vicreg import VICREG_META_PATH as LIVE_VICREG_META_PATH
from tileclass.training.vicreg import VICREG_WEIGHTS_PATH as LIVE_VICREG_WEIGHTS_PATH
from tileclass.training.vicreg import load_backbone, warm_start_overlap

from yeastprep.core.classify import default_supervised_checkpoint_paths, default_vicreg_checkpoint_paths

from ..classify_params_panel import SupervisedTrainParamsPanel, VicregTrainParamsPanel
from ..common.json_tree_dialog import JsonTreeDialog
from ..diagnostics.classifier_training_monitor_panel import ClassifierTrainingMonitorPanel
from ..project_tree_panel import ProjectTreePanel
from ..worker import ClassifierInferenceWorker, ClassifierTrainingWorker, ClassifierVicregWorker
from ._classifier_pool_widget import ClassifierPoolWidget
from .page_progress import PageProgress


def _looks_like_vicreg_backbone(meta_path: Path) -> bool:
    """Whether `meta_path` was written by `tileclass.training.vicreg`'s
    `_save_backbone` rather than `tileclass.training.supervised`'s
    `_save_weights` -- both save a `meta.json` with a `categories` key, so
    that alone can't tell them apart, but only the VICReg one ever writes
    a `pairing` key (see `training/vicreg.py`'s `_save_backbone`). Used to
    reject a VICReg backbone picked (by mistake, via Browse) as an
    inference checkpoint before it fails deep inside a background thread
    with a raw PyTorch state_dict-mismatch error -- a headless backbone
    has no classification head to load into `YeastEfficientNetClassifier`."""
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return False
    return "pairing" in meta


class _CheckpointFilePicker(QWidget):
    """A checkpoint-weights-file field + Browse + Deployed + View Metadata
    row, with "still showing its default until the user edits/Browses/picks
    Deployed" tracking -- the shape shared by `_SupervisedTrainingTab`'s
    "Run Inference on Pool" weights field, `_VicregTrainingTab`'s "Evaluate
    Embeddings" weights field, the "Starting point" backbone field, and
    both tabs' Deploy field -- previously hand-duplicated per tab (its own
    `_..._is_default` flag, Browse dialog, Deployed-button wiring, and
    existence/sibling-meta.json/checkpoint-kind validation). `resolve()`
    centralizes that validation so callers just get back a ready-to-use
    (weights_path, meta_path) pair or `None` (a QMessageBox has already
    explained why, in that case).

    "View Metadata..." opens the sibling meta.json (whatever's currently
    typed/Browsed/Deployed into the field, not necessarily a validated
    pair) in a `JsonTreeDialog` -- so a user can inspect what a checkpoint
    was trained on and with what parameters before committing to using it
    for inference, further training, or deploying it live."""

    pathChanged = Signal(str)

    def __init__(
        self,
        placeholder: str,
        tooltip: str,
        deployed_path: Path,
        deployed_tooltip: str,
        parent=None,
    ):
        super().__init__(parent)
        self._deployed_path = deployed_path
        self._is_default = True

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.setToolTip(tooltip)
        self.edit.textEdited.connect(self._on_edited)
        layout.addWidget(self.edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        layout.addWidget(browse_btn)
        deployed_btn = QPushButton("Deployed")
        deployed_btn.setToolTip(deployed_tooltip)
        deployed_btn.clicked.connect(self._use_deployed)
        layout.addWidget(deployed_btn)
        metadata_btn = QPushButton("View Metadata...")
        metadata_btn.setToolTip(
            "Open this checkpoint's sibling meta.json in a browsable tree -- "
            "what it was trained on, with what parameters, and when."
        )
        metadata_btn.clicked.connect(self._view_metadata)
        layout.addWidget(metadata_btn)

    def _on_edited(self, text):
        self._is_default = False
        self.pathChanged.emit(text)

    def _browse(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, "Checkpoint", self.edit.text(), "PyTorch weights (*.pth)"
        )
        if path:
            self.edit.setText(path)
            self._is_default = False
            self.pathChanged.emit(path)

    def _use_deployed(self):
        path = str(self._deployed_path)
        self.edit.setText(path)
        self._is_default = False
        self.pathChanged.emit(path)

    def _view_metadata(self):
        text = self.edit.text().strip()
        if not text:
            QMessageBox.information(self, "yeastprep", "Choose a checkpoint first.")
            return
        meta_path = Path(text).with_name("meta.json")
        if not meta_path.is_file():
            QMessageBox.warning(
                self, "yeastprep", f"No meta.json found next to {text}."
            )
            return
        try:
            data = json.loads(meta_path.read_text())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "yeastprep", f"Could not read {meta_path}: {exc}")
            return
        dialog = JsonTreeDialog(f"Metadata -- {meta_path}", data, parent=self)
        dialog.exec_()

    def set_default_path(self, path: Path):
        """Called once a training run in this session produces a fresh
        checkpoint -- only takes effect while the user hasn't Browsed/typed/
        picked Deployed themselves, same "obvious default until you
        deliberately override it" convention as the checkpoint-output field."""
        if self._is_default:
            self.edit.setText(str(path))
            self.pathChanged.emit(str(path))

    def resolve(self, *, expect_vicreg: bool, wrong_kind_message: str) -> tuple[Path, Path] | None:
        """Validates the current text as a (weights.pth, sibling meta.json)
        pair of the expected kind -- `expect_vicreg=False` for a
        classifier checkpoint, `True` for a headless VICReg backbone (see
        `_looks_like_vicreg_backbone`). Shows a `QMessageBox.warning` and
        returns `None` for anything wrong; `wrong_kind_message` (a
        `str.format`-style template taking `meta_path`) supplies the
        tab-specific "pick a checkpoint from the other tab instead" hint."""
        text = self.edit.text().strip()
        if not text:
            QMessageBox.warning(self, "yeastprep", "Choose a checkpoint first.")
            return None
        weights_path = Path(text)
        meta_path = weights_path.with_name("meta.json")
        if not weights_path.is_file() or not meta_path.is_file():
            QMessageBox.warning(
                self,
                "yeastprep",
                f"Expected both {weights_path.name} and a sibling meta.json at "
                f"{weights_path.parent} -- one or both are missing.",
            )
            return None
        if _looks_like_vicreg_backbone(meta_path) != expect_vicreg:
            QMessageBox.warning(self, "yeastprep", wrong_kind_message.format(meta_path=meta_path))
            return None
        return weights_path, meta_path


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

    _live_weights_filename = "weights.pth"  # overridden per subclass
    _live_meta_filename = "meta.json"

    def __init__(self, pool_widget: ClassifierPoolWidget, parent=None):
        super().__init__(parent)
        self.pool_widget = pool_widget
        self._thread = None
        self._worker = None
        self._checkpoint_path_is_default = True
        self._last_checkpoint_dir: Path | None = None
        self._embedding_viewer_windows: list[MainWindow] = []

        self._build_ui()
        self._wire_up()
        self._refresh_default_checkpoint_path()

    # ------------------------------------------------------------------
    # UI construction -- subclasses provide the params panel and any
    # extra controls (e.g. VICReg's "Evaluate embeddings" button) via hooks.

    def _make_params_panel(self) -> QWidget:
        raise NotImplementedError

    def _default_checkpoint_paths(self, project_root) -> tuple[Path, Path]:
        raise NotImplementedError

    def _live_slot_paths(self) -> tuple[Path, Path]:
        raise NotImplementedError

    def _build_extra_actions(self, right_layout: QVBoxLayout):
        """Hook for a subclass-specific action row below Deploy -- only
        `_SupervisedTrainingTab` uses this (Run Inference on Pool); VICReg
        produces a headless backbone with no classification head, so
        inference doesn't apply there. No-op by default."""

    def _build_extra_left_widgets(self, left_layout: QVBoxLayout):
        """Hook for a subclass-specific widget below the checkpoint-output
        group, on the left (inputs) side -- used for a "warm-starting
        from..." indicator (see `_describe_warm_start`/`_refresh_warm_start_label`
        below). No-op by default."""

    def _on_checkpoint_ready(self):
        """Hook called once a training run finishes and `_last_checkpoint_dir`
        is set, right after `deploy_picker` has had its default path set to
        that fresh checkpoint -- overridden by `_SupervisedTrainingTab` to
        also default its inference-weights picker."""

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

        self._build_extra_actions(right_layout)

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
        self.monitor_panel.pointsSelected.connect(self._open_viewer_for_selection)

    def refresh_dataset_summary(self):
        pooled = self.pool_widget.pooled_annotations()
        if pooled is not None:
            self.monitor_panel.set_dataset_summary(pooled)

    def _open_viewer_for_selection(self, paths: list[str]) -> None:
        """Opens tiles lasso-selected on the embedding scatter
        (`ClassifierTrainingMonitorPanel.pointsSelected`) in an in-process
        tileclass viewer window, scoped to the currently checked FOV
        folders -- the same folders the embeddings themselves were pulled
        from -- so an unexpected cluster or an outlier sitting with the
        wrong label can be checked against its actual image and annotation.
        Each selection opens its own window (rather than reusing one), kept
        alive here since a parentless QMainWindow with no other reference
        would otherwise be garbage-collected out from under Qt."""
        folders = self.pool_widget.checked_fov_dirs()
        if not folders or not paths:
            return
        window = MainWindow(folders, paths, tiles_per_page=len(paths))
        window.show()
        self._embedding_viewer_windows.append(window)
        window.destroyed.connect(
            lambda: self._embedding_viewer_windows.remove(window)
        )

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
        self._on_checkpoint_ready()
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
        self.backbone_status_label.setText(
            f"{text}\ntrained: {last_trained}\n"
            f"categories seen: {', '.join(categories) if categories else 'n/a'}"
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

    # ------------------------------------------------------------------
    # Run Inference on Pool: batch-classify every tile crop across the
    # currently checked FOV folders with a chosen checkpoint, tagging
    # results as unreviewed AI predictions. Never overwrites an existing
    # tag (human-confirmed or a prior AI prediction) -- see
    # `core.classify.classify_pool`. The weights field defaults to this
    # session's just-trained checkpoint once one exists (so "train, then
    # immediately see how it does on the pool" needs no extra picking),
    # but is independently editable/Browse-able -- e.g. to evaluate an
    # older session's checkpoint, or the currently deployed classifier,
    # without having trained anything in this session at all.

    def _build_extra_actions(self, right_layout):
        self._inference_thread = None
        self._inference_worker = None

        group = QGroupBox("Run Inference")
        v = QVBoxLayout(group)

        self.infer_picker = _CheckpointFilePicker(
            placeholder="weights.pth",
            tooltip=(
                "Checkpoint to run inference with -- defaults to this session's "
                "just-trained weights once training finishes. Browse to pick a "
                "different one (an older session's checkpoint, a backup, ...), "
                "or use 'Deployed' for tileclass's current live classifier. "
                "Expects a sibling meta.json next to whatever weights.pth is chosen."
            ),
            deployed_path=LIVE_CLASSIFIER_WEIGHTS_PATH,
            deployed_tooltip="Use tileclass's currently deployed classifier weights.",
        )
        v.addWidget(self.infer_picker)

        self.infer_btn = QPushButton("Run Inference on Pool")
        self.infer_btn.setToolTip(
            "Classify every currently-untagged tile across the checked FOVs "
            "with the checkpoint above, tagging results as unreviewed AI "
            "predictions. Tiles that already have a tag are left untouched; "
            "human-confirmed tiles are used as a free accuracy check, logged below."
        )
        self.infer_btn.clicked.connect(self._run_inference)
        v.addWidget(self.infer_btn)

        right_layout.addWidget(group)

    def _on_checkpoint_ready(self):
        self.infer_picker.set_default_path(self._last_checkpoint_dir / self._live_weights_filename)

    def _run_inference(self):
        if self._inference_thread is not None:
            return

        resolved = self.infer_picker.resolve(
            expect_vicreg=False,
            wrong_kind_message=(
                "{meta_path} looks like a VICReg backbone checkpoint, not a "
                "classifier -- it has no classification head to run inference "
                "with. Pick a checkpoint from the Supervised Training tab "
                "instead (this session's own, the Deployed classifier, or "
                "another supervised session's weights.pth)."
            ),
        )
        if resolved is None:
            return
        weights_path, meta_path = resolved

        pooled = self.pool_widget.pooled_annotations()
        if pooled is None:
            QMessageBox.warning(
                self, "yeastprep", "No FOVs checked in the pool to run inference on."
            )
            return

        try:
            classifier = YeastEfficientNetClassifier(weights_path=weights_path, meta_path=meta_path)
        except Exception as exc:
            QMessageBox.critical(self, "yeastprep", f"Could not load checkpoint: {exc}")
            return

        self.status_label.setText("Running inference on pool...")
        self.infer_btn.setEnabled(False)
        self.start_btn.setEnabled(False)

        self._inference_worker = ClassifierInferenceWorker(pooled, classifier)
        self._inference_thread = QThread()
        self._inference_worker.moveToThread(self._inference_thread)
        self._inference_thread.started.connect(self._inference_worker.run)
        self._inference_worker.finished.connect(self._on_inference_finished)
        self._inference_worker.error.connect(self._on_inference_error)
        self._inference_thread.start()

    def _teardown_inference_thread(self):
        self._inference_thread.quit()
        self._inference_thread.wait()
        self._inference_thread = None
        self._inference_worker = None

    def _on_inference_finished(self, result):
        summary = (
            f"inference: {result.n_total} tile(s) scored, "
            f"{result.n_newly_tagged} newly tagged, categories={result.category_counts}"
        )
        if result.mean_confidence is not None:
            summary += f", mean_confidence={result.mean_confidence:.3f}"
        self.monitor_panel.log(summary)
        if result.n_human_confirmed:
            self.monitor_panel.log(
                f"  agreement with {result.n_human_confirmed} human-confirmed tile(s): "
                f"{result.accuracy_vs_human:.3f} ({result.n_agree_with_human}/{result.n_human_confirmed})"
            )
        self.refresh_dataset_summary()
        self.status_label.setText("Inference complete.")
        self.infer_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self._teardown_inference_thread()

    def _on_inference_error(self, message: str):
        self.monitor_panel.log(f"inference ERROR: {message}")
        self.status_label.setText(f"Inference error: {message.splitlines()[0] if message else ''}")
        self.infer_btn.setEnabled(True)
        self.start_btn.setEnabled(True)
        self._teardown_inference_thread()

    def shutdown(self):
        if self._inference_thread is not None:
            self._inference_thread.quit()
            self._inference_thread.wait()
        super().shutdown()


class _VicregTrainingTab(_BaseTrainingTab):
    _live_weights_filename = "backbone.pth"
    _live_meta_filename = "meta.json"

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
        self._evaluate_embeddings(result.weights_path)

    def _evaluate_embeddings(self, weights_path):
        """Post-hoc embedding-separability check (see
        `tileclass.training.linear_probe`): loads the backbone at
        `weights_path` back off disk and scatters a t-SNE projection of
        its embeddings over every currently pooled confirmed tile --
        t-SNE rather than a linear PCA projection because the accompanying
        `knn_accuracy` number is itself a local-neighborhood notion of
        separability, and a linear projection can visually smear apart
        clusters a neighborhood-preserving one would show cleanly
        separated (see `linear_probe.tsne_2d`'s docstring). Runs
        synchronously on the GUI thread -- called both automatically right
        after a training run finishes (`_on_result`, `weights_path` is
        that run's own result) and on demand via the "Evaluate Embeddings"
        button below (`weights_path` is whatever's in `evaluate_picker`,
        independent of any training run). Not wired to progress signals --
        there's nothing per-epoch to show here, unlike the loss plot.
        Runs on the GUI thread with no per-item progress to report, so
        this at least sets a wait cursor and disables the button for its
        duration -- otherwise a slow t-SNE pass over a large pool looks
        indistinguishable from a hang."""
        import torch
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QApplication

        from tileclass.classifiers.device import select_device

        records = self.pool_widget.gather_confirmed_records()
        if len(records) < 2:
            return
        self.status_label.setText("Evaluating embeddings...")
        self.evaluate_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            device = select_device()
            backbone = load_backbone(weights_path=weights_path, device=device)
            backbone.eval()
            paths = [p for p, _ in records]
            labels = [label for _, label in records]
            with torch.no_grad():
                embeddings = extract_embeddings(paths, backbone, device)
            xy = tsne_2d(embeddings)
            acc = knn_accuracy(embeddings, labels) if len(set(labels)) > 1 else None
            self.monitor_panel.show_embedding_scatter(xy, labels, paths=paths, knn_acc=acc)
            self.status_label.setText("Embeddings evaluated.")
        except Exception as exc:
            self.monitor_panel.log(f"embedding evaluation failed: {exc}")
            self.status_label.setText("Embedding evaluation failed.")
        finally:
            QApplication.restoreOverrideCursor()
            self.evaluate_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Evaluate Embeddings: plot the t-SNE diagnostic above on demand, for
    # any backbone checkpoint, without running any training -- same
    # weights-field + Browse/Deployed pattern as
    # `_SupervisedTrainingTab`'s "Run Inference on Pool". Runs synchronously
    # (see `_evaluate_embeddings`'s docstring); fine for the "confirmed
    # tiles" pool this always draws from, which is the same, typically
    # small, annotated set a VICReg run itself trains on -- not the
    # much larger full-pool `classify_pool` sweep that
    # `ClassifierInferenceWorker` backgrounds.

    def _build_extra_actions(self, right_layout):
        group = QGroupBox("Evaluate Embeddings")
        v = QVBoxLayout(group)

        self.evaluate_picker = _CheckpointFilePicker(
            placeholder="backbone.pth",
            tooltip=(
                "Backbone checkpoint to plot embeddings for -- defaults to this "
                "session's just-trained backbone once training finishes. Browse "
                "to pick a different one (an older session's checkpoint, a "
                "backup, ...), or use 'Deployed' for tileclass's current live "
                "VICReg backbone. Expects a sibling meta.json next to whatever "
                "backbone.pth is chosen."
            ),
            deployed_path=LIVE_VICREG_WEIGHTS_PATH,
            deployed_tooltip="Use tileclass's currently deployed VICReg backbone.",
        )
        v.addWidget(self.evaluate_picker)

        self.evaluate_btn = QPushButton("Evaluate Embeddings")
        self.evaluate_btn.setToolTip(
            "Plot a t-SNE projection of the checkpoint above's embeddings "
            "over every currently pooled confirmed tile, without running "
            "any training -- the same diagnostic that runs automatically "
            "after a VICReg training run finishes."
        )
        self.evaluate_btn.clicked.connect(self._evaluate_embeddings_clicked)
        v.addWidget(self.evaluate_btn)

        right_layout.addWidget(group)

    def _on_checkpoint_ready(self):
        self.evaluate_picker.set_default_path(self._last_checkpoint_dir / self._live_weights_filename)

    def _evaluate_embeddings_clicked(self):
        resolved = self.evaluate_picker.resolve(
            expect_vicreg=True,
            wrong_kind_message=(
                "{meta_path} doesn't look like a VICReg backbone checkpoint "
                "-- pick a backbone.pth from the VICReg Pretraining tab "
                "instead (this session's own, the Deployed backbone, or "
                "another session's backbone.pth)."
            ),
        )
        if resolved is None:
            return
        weights_path, _meta_path = resolved

        records = self.pool_widget.gather_confirmed_records()
        if len(records) < 2:
            QMessageBox.warning(
                self,
                "yeastprep",
                "Need at least 2 confirmed annotated tiles in the pool to "
                "evaluate embeddings.",
            )
            return

        self.monitor_panel.log(f"evaluating embeddings: {weights_path}")
        self._evaluate_embeddings(weights_path)


class ClassifierTrainingPage(QWidget):
    """Top-level page: one shared pool feeding two independent training
    tasks (see module docstring)."""

    progress_changed = Signal(object)  # PageProgress

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
