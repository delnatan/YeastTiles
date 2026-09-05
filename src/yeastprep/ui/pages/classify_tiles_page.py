"""Classify Tiles page: everything about *using* a trained checkpoint --
bulk-classifying a pool of tiles, and interactively exploring a backbone's
embeddings -- separated out from the Classifier Training page (see that
page's module docstring), which now only trains and deploys. Getting these
off the training tabs' narrow 340-440px column and onto their own page gives
both real room to work, especially the embedding scatter, which is meant for
actual visual inspection (lasso-select a cluster, open it in a tile viewer),
not just a post-training sanity check.

Deliberately owns its own `ClassifierPoolWidget` rather than sharing the
training page's: a classify run may target a different (often larger, less
curated) set of projects than the training pool, e.g. bulk-classifying a
whole new dataset that's never been used for training at all.
"""

from pathlib import Path

from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tileclass.classifiers.device import select_device
from tileclass.classifiers.yeast_efficientnet import WEIGHTS_PATH as LIVE_CLASSIFIER_WEIGHTS_PATH
from tileclass.classifiers.yeast_efficientnet import YeastEfficientNetClassifier
from tileclass.main_window import MainWindow
from tileclass.training.linear_probe import extract_embeddings, knn_accuracy, tsne_2d
from tileclass.training.vicreg import VICREG_WEIGHTS_PATH as LIVE_VICREG_WEIGHTS_PATH
from tileclass.training.vicreg import load_backbone

from yeastprep.core.classify import sample_unlabeled

from ..common.checkpoint_file_picker import CheckpointFilePicker
from ..diagnostics.embedding_scatter_widget import UNLABELED_LABEL, EmbeddingScatterWidget
from ..project_tree_panel import ProjectTreePanel
from ..worker import ClassifierInferenceWorker
from ._classifier_pool_widget import ClassifierPoolWidget
from .page_progress import PageProgress


class ClassifyTilesPage(QWidget):
    progress_changed = Signal(object)  # PageProgress -- unused (nothing here is a batch/stage job)

    def __init__(self, tree_panel: ProjectTreePanel, parent=None):
        super().__init__(parent)
        self.tree_panel = tree_panel
        self._inference_thread = None
        self._inference_worker = None
        self._embedding_viewer_windows: list[MainWindow] = []

        self._build_ui()
        self._wire_up()

    # ------------------------------------------------------------------
    # UI construction

    def _build_ui(self):
        outer = QVBoxLayout(self)
        self.pool_widget = ClassifierPoolWidget()
        outer.addWidget(self.pool_widget)

        splitter = QSplitter()
        outer.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._build_inference_group())
        left_layout.addWidget(self._build_embeddings_controls_group())
        left_layout.addStretch(1)
        left.setMinimumWidth(360)
        left.setMaximumWidth(460)
        splitter.addWidget(left)

        self.embedding_scatter = EmbeddingScatterWidget()
        splitter.addWidget(self.embedding_scatter)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.status_label = QLabel("")
        outer.addWidget(self.status_label)

    def _build_inference_group(self) -> QGroupBox:
        group = QGroupBox("Run Inference on Pool")
        v = QVBoxLayout(group)

        self.infer_picker = CheckpointFilePicker(
            placeholder="weights.pth",
            tooltip=(
                "Checkpoint to run inference with -- Browse to pick one (a "
                "training session's checkpoint, a backup, ...), or use "
                "'Deployed' for tileclass's current live classifier. Expects "
                "a sibling meta.json next to whatever weights.pth is chosen. "
                "Auto-fills with a just-trained checkpoint from the "
                "Classifier Training page, if you haven't picked one yet."
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

        self.predicted_table = QTableWidget(0, 2)
        self.predicted_table.setHorizontalHeaderLabels(["Category", "Predicted"])
        self.predicted_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.predicted_table.verticalHeader().setVisible(False)
        self.predicted_table.setMaximumHeight(160)
        v.addWidget(self.predicted_table)

        self.inference_log = QPlainTextEdit()
        self.inference_log.setReadOnly(True)
        self.inference_log.setMaximumBlockCount(2000)
        self.inference_log.setMaximumHeight(120)
        v.addWidget(self.inference_log)

        return group

    def _build_embeddings_controls_group(self) -> QGroupBox:
        group = QGroupBox("Explore Embeddings")
        v = QVBoxLayout(group)

        self.embed_picker = CheckpointFilePicker(
            placeholder="backbone.pth",
            tooltip=(
                "Backbone checkpoint to plot embeddings for -- Browse to "
                "pick one (a VICReg pretraining session's backbone.pth, a "
                "backup, ...), or use 'Deployed' for tileclass's current "
                "live VICReg backbone. Expects a sibling meta.json next to "
                "whatever backbone.pth is chosen. Auto-fills with a "
                "just-pretrained backbone from the Classifier Training "
                "page, if you haven't picked one yet."
            ),
            deployed_path=LIVE_VICREG_WEIGHTS_PATH,
            deployed_tooltip="Use tileclass's currently deployed VICReg backbone.",
        )
        v.addWidget(self.embed_picker)

        sample_row = QHBoxLayout()
        self.include_unlabeled_cb = QCheckBox("Include random sample of unlabeled tiles:")
        self.include_unlabeled_cb.setToolTip(
            "Also plot a random sample of currently-unlabeled tiles from the "
            "checked pool, in gray, so you can see where unlabeled data "
            "falls relative to labeled clusters -- without embedding the "
            "entire (possibly huge) pool."
        )
        sample_row.addWidget(self.include_unlabeled_cb)
        self.unlabeled_sample_size = QSpinBox()
        self.unlabeled_sample_size.setRange(1, 5000)
        self.unlabeled_sample_size.setValue(200)
        self.unlabeled_sample_size.setEnabled(False)
        sample_row.addWidget(self.unlabeled_sample_size)
        sample_row.addStretch(1)
        v.addLayout(sample_row)
        self.include_unlabeled_cb.toggled.connect(self.unlabeled_sample_size.setEnabled)

        self.evaluate_btn = QPushButton("Evaluate Embeddings")
        self.evaluate_btn.setToolTip(
            "Plot a t-SNE projection of the checkpoint above's embeddings "
            "over every currently pooled confirmed tile (and, if checked, a "
            "random sample of unlabeled ones). Lasso-select points on the "
            "plot to open them in a tile viewer."
        )
        self.evaluate_btn.clicked.connect(self._evaluate_embeddings)
        v.addWidget(self.evaluate_btn)

        return group

    # ------------------------------------------------------------------

    def _wire_up(self):
        self.embedding_scatter.pointsSelected.connect(self._open_viewer_for_selection)

    def load_selection(self, stage: str, path: str, mode: str):
        """Only `mode == "open_viewer_fov"` applies here (see
        `selection_actions.actions_for_selection`'s STAGE_TILES branch) --
        `path` is a FOV id, this page pool-adds its parent project (same
        convention as `ClassifierTrainingPage.load_selection`)."""
        if mode != "open_viewer_fov":
            return
        root = self.tree_panel.project_root()
        if root:
            self.pool_widget.add_project(root)

    def set_default_checkpoint(self, weights_path: Path, is_vicreg: bool) -> None:
        """Connected to `ClassifierTrainingPage.checkpointTrained` -- routes
        a just-finished training run's checkpoint to whichever picker it's
        relevant for, only taking effect while that picker is still showing
        its own default (see `CheckpointFilePicker.set_default_path`)."""
        picker = self.embed_picker if is_vicreg else self.infer_picker
        picker.set_default_path(weights_path)

    # ------------------------------------------------------------------
    # Run Inference on Pool -- batch-classify every tile crop across the
    # currently checked FOV folders with a chosen checkpoint, tagging
    # results as unreviewed AI predictions. Never overwrites an existing
    # tag (human-confirmed or a prior AI prediction) -- see
    # `core.classify.classify_pool`.

    def _run_inference(self):
        if self._inference_thread is not None:
            return

        resolved = self.infer_picker.resolve(
            expect_vicreg=False,
            wrong_kind_message=(
                "{meta_path} looks like a VICReg backbone checkpoint, not a "
                "classifier -- it has no classification head to run "
                "inference with. Pick a classifier weights.pth instead (a "
                "supervised training session's own, or the Deployed classifier)."
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
            f"{result.n_newly_tagged} newly tagged"
        )
        if result.mean_confidence is not None:
            summary += f", mean_confidence={result.mean_confidence:.3f}"
        self.inference_log.appendPlainText(summary)
        if result.n_human_confirmed:
            self.inference_log.appendPlainText(
                f"  agreement with {result.n_human_confirmed} human-confirmed tile(s): "
                f"{result.accuracy_vs_human:.3f} ({result.n_agree_with_human}/{result.n_human_confirmed})"
            )
        categories = sorted(result.category_counts)
        self.predicted_table.setRowCount(len(categories))
        for row, category in enumerate(categories):
            self.predicted_table.setItem(row, 0, QTableWidgetItem(category))
            self.predicted_table.setItem(
                row, 1, QTableWidgetItem(str(result.category_counts[category]))
            )
        self.status_label.setText("Inference complete.")
        self.infer_btn.setEnabled(True)
        self._teardown_inference_thread()

    def _on_inference_error(self, message: str):
        self.inference_log.appendPlainText(f"inference ERROR: {message}")
        self.status_label.setText(f"Inference error: {message.splitlines()[0] if message else ''}")
        self.infer_btn.setEnabled(True)
        self._teardown_inference_thread()

    # ------------------------------------------------------------------
    # Explore Embeddings -- t-SNE projection of a backbone's embeddings
    # over the pool's confirmed tiles, optionally plus a random sample of
    # unlabeled ones (see `core.classify.sample_unlabeled`). Runs
    # synchronously on the GUI thread, like the training page's old
    # "Evaluate Embeddings" did -- fine for the "confirmed tiles (+ a
    # bounded unlabeled sample)" pool this draws from, not a full-pool sweep.

    def _evaluate_embeddings(self):
        import torch
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QApplication

        resolved = self.embed_picker.resolve(
            expect_vicreg=True,
            wrong_kind_message=(
                "{meta_path} doesn't look like a VICReg backbone checkpoint "
                "-- pick a backbone.pth instead (a VICReg pretraining "
                "session's own, or the Deployed backbone)."
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

        paths = [p for p, _ in records]
        labels = [label for _, label in records]

        if self.include_unlabeled_cb.isChecked():
            pooled = self.pool_widget.pooled_annotations()
            n = self.unlabeled_sample_size.value()
            unlabeled_paths = sample_unlabeled(pooled, n) if pooled is not None else []
            paths = paths + unlabeled_paths
            labels = labels + [UNLABELED_LABEL] * len(unlabeled_paths)

        self.status_label.setText("Evaluating embeddings...")
        self.evaluate_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            device = select_device()
            backbone = load_backbone(weights_path=weights_path, device=device)
            backbone.eval()
            with torch.no_grad():
                embeddings = extract_embeddings(paths, backbone, device)
            xy = tsne_2d(embeddings)
            n_confirmed = len(records)
            confirmed_labels = labels[:n_confirmed]
            acc = (
                knn_accuracy(embeddings[:n_confirmed], confirmed_labels)
                if len(set(confirmed_labels)) > 1
                else None
            )
            self.embedding_scatter.show_embedding_scatter(xy, labels, paths=paths, knn_acc=acc)
            self.status_label.setText("Embeddings evaluated.")
        except Exception as exc:
            self.inference_log.appendPlainText(f"embedding evaluation failed: {exc}")
            self.status_label.setText("Embedding evaluation failed.")
        finally:
            QApplication.restoreOverrideCursor()
            self.evaluate_btn.setEnabled(True)

    def _open_viewer_for_selection(self, paths: list[str]) -> None:
        """Opens tiles lasso-selected on the embedding scatter in an
        in-process tileclass viewer window, scoped to the currently checked
        FOV folders -- the same folders the embeddings themselves were
        pulled from -- so a suspicious cluster, an outlier, or an unlabeled
        point sitting inside a labeled cluster can be checked against its
        actual image and annotation. Each selection opens its own window
        (rather than reusing one), kept alive here since a parentless
        QMainWindow with no other reference would otherwise be
        garbage-collected out from under Qt."""
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

    def shutdown(self):
        if self._inference_thread is not None:
            self._inference_thread.quit()
            self._inference_thread.wait()
