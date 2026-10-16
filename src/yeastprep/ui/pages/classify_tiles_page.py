"""Classify Tiles page: using a trained checkpoint on the shared
classification pool (`ui/classifier_pool.py`) -- bulk-classifying its
tiles, clearing stale AI predictions, exploring a backbone's embeddings
(lasso-select a cluster to open it in a tile viewer), and plotting
annotation counts against experimental variables.
"""

from pathlib import Path

from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tileclass.classifiers.device import select_device
from tileclass.classifiers.yeast_efficientnet import YeastEfficientNetClassifier
from tileclass.main_window import MainWindow
from tileclass.training.embeddings import (
    UMAP_METRICS,
    extract_embeddings,
    knn_accuracy,
    umap_2d,
)
from tileclass.training.vicreg import load_backbone

from yeastprep.core.classify import sample_unlabeled

from ..classifier_pool import ClassifierPool, ClassifierPoolWidget
from ..common.background_task import BackgroundTask
from ..common.checkpoint_choice import CheckpointChoice
from ..diagnostics.annotation_analytics_panel import AnnotationAnalyticsPanel
from ..diagnostics.embedding_scatter_widget import UNLABELED_LABEL, EmbeddingScatterWidget
from ..worker import ClassifierInferenceWorker
from .page_progress import PageProgress  # noqa: F401 -- shape of progress_changed, never emitted here


class ClassifyTilesPage(QWidget):
    progress_changed = Signal(object)  # PageProgress -- unused (nothing here is a batch/stage job)

    def __init__(self, pool: ClassifierPool, parent=None):
        super().__init__(parent)
        self.pool = pool
        self._inference_thread = None
        self._inference_worker = None
        self._embedding_viewer_windows: list[MainWindow] = []
        # (embeddings, labels, paths, knn_acc) from the last Evaluate, so
        # Re-plot can re-run just UMAP when only its settings changed.
        self._last_embeddings = None
        # Forward pass + UMAP (Evaluate), or UMAP alone (Re-plot) -- one at
        # a time, off the GUI thread.
        self._embedding_task = BackgroundTask(self)
        self._embedding_done_text = ""

        self._build_ui()
        self._wire_up()

    # ------------------------------------------------------------------
    # UI construction

    def _build_ui(self):
        outer = QVBoxLayout(self)

        splitter = QSplitter()
        outer.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(ClassifierPoolWidget(self.pool))
        left_layout.addWidget(self._build_inference_group())
        left_layout.addWidget(self._build_embeddings_controls_group())
        left_layout.addStretch(1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(360)
        left_scroll.setMaximumWidth(460)
        splitter.addWidget(left_scroll)

        self.embedding_scatter = EmbeddingScatterWidget()
        self.annotation_analytics = AnnotationAnalyticsPanel(self.pool)

        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self.embedding_scatter, "Embeddings")
        self._analytics_tab_index = self.right_tabs.addTab(
            self.annotation_analytics, "Annotation Analytics"
        )
        self.right_tabs.currentChanged.connect(self._on_right_tab_changed)
        splitter.addWidget(self.right_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.status_label = QLabel("")
        outer.addWidget(self.status_label)

    def _build_inference_group(self) -> QGroupBox:
        group = QGroupBox("Run Inference on Pool")
        v = QVBoxLayout(group)

        self.infer_picker = CheckpointChoice("classifier")
        self.infer_picker.setToolTip("Classifier checkpoint to run: the deployed one, or Browse.")
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

        self.clear_predictions_btn = QPushButton("Clear AI Predictions in Pool...")
        self.clear_predictions_btn.setToolTip(
            "Remove every unconfirmed AI prediction across the checked FOVs, "
            "keeping human-set and accepted annotations -- so a newer "
            "checkpoint can re-predict those tiles."
        )
        self.clear_predictions_btn.clicked.connect(self._clear_predictions)
        v.addWidget(self.clear_predictions_btn)

        self.inference_log = QPlainTextEdit()
        self.inference_log.setReadOnly(True)
        self.inference_log.setMaximumBlockCount(2000)
        self.inference_log.setMaximumHeight(120)
        v.addWidget(self.inference_log)

        return group

    def _build_embeddings_controls_group(self) -> QGroupBox:
        group = QGroupBox("Explore Embeddings")
        v = QVBoxLayout(group)

        self.embed_picker = CheckpointChoice("backbone")
        self.embed_picker.setToolTip("VICReg backbone to embed with: the deployed one, or Browse.")
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

        v.addLayout(self._build_umap_form())

        buttons = QHBoxLayout()
        self.evaluate_btn = QPushButton("Evaluate Embeddings")
        self.evaluate_btn.setToolTip(
            "Plot a UMAP projection of the checkpoint above's embeddings "
            "over every currently pooled confirmed tile (and, if checked, a "
            "random sample of unlabeled ones). Lasso-select points on the "
            "plot to open them in a tile viewer."
        )
        self.evaluate_btn.clicked.connect(self._evaluate_embeddings)
        buttons.addWidget(self.evaluate_btn)
        self.replot_btn = QPushButton("Re-plot")
        self.replot_btn.setToolTip(
            "Re-run only the UMAP layout on the last evaluated embeddings, "
            "with the settings above -- no new forward pass."
        )
        self.replot_btn.setEnabled(False)
        self.replot_btn.clicked.connect(self._replot_embeddings)
        buttons.addWidget(self.replot_btn)
        v.addLayout(buttons)

        return group

    def _build_umap_form(self) -> QFormLayout:
        form = QFormLayout()

        self.umap_neighbors = QSpinBox()
        self.umap_neighbors.setRange(2, 200)
        self.umap_neighbors.setValue(15)
        self.umap_neighbors.setToolTip(
            "Neighborhood size: small values emphasize fine local structure, "
            "large values the overall arrangement of clusters."
        )
        form.addRow("UMAP neighbors", self.umap_neighbors)

        self.umap_min_dist = QDoubleSpinBox()
        self.umap_min_dist.setRange(0.0, 0.99)
        self.umap_min_dist.setSingleStep(0.05)
        self.umap_min_dist.setDecimals(2)
        self.umap_min_dist.setValue(0.1)
        self.umap_min_dist.setToolTip(
            "How tightly points may pack: low values give dense, well-"
            "separated clumps; high values spread clusters out."
        )
        form.addRow("Min. distance", self.umap_min_dist)

        self.umap_metric = QComboBox()
        self.umap_metric.addItems(UMAP_METRICS)
        self.umap_metric.setToolTip(
            "Distance between embeddings. Cosine compares directions only, "
            "often a better fit for self-supervised embeddings."
        )
        form.addRow("Metric", self.umap_metric)

        self.umap_seed = QSpinBox()
        self.umap_seed.setRange(-1, 99999)
        self.umap_seed.setValue(0)
        self.umap_seed.setSpecialValueText("random")
        self.umap_seed.setToolTip(
            "Fixed seed: the same layout every time (single-threaded). "
            "'random': faster parallel run, a different layout each time."
        )
        form.addRow("Seed", self.umap_seed)

        return form

    def _umap_kwargs(self) -> dict:
        seed = self.umap_seed.value()
        return dict(
            n_neighbors=self.umap_neighbors.value(),
            min_dist=self.umap_min_dist.value(),
            metric=self.umap_metric.currentText(),
            seed=None if seed < 0 else seed,
        )

    # ------------------------------------------------------------------

    def _wire_up(self):
        self.embedding_scatter.pointsSelected.connect(self._open_viewer_for_selection)
        self._embedding_task.succeeded.connect(self._on_embeddings_ready)
        self._embedding_task.failed.connect(self._on_embeddings_failed)
        self.pool.changed.connect(self.annotation_analytics.refresh_from_pool)

    def _on_right_tab_changed(self, index: int):
        # Annotation files can change on disk from outside this page
        # entirely (a separately opened tile viewer, another yeastprep
        # window), so pull a fresh summary each time the tab becomes
        # visible rather than relying only on pool_changed.
        if index == self._analytics_tab_index:
            self.annotation_analytics.refresh_from_pool()

    # ------------------------------------------------------------------
    # Run Inference on Pool -- batch-classify every tile crop across the
    # currently checked FOV folders with a chosen checkpoint, tagging
    # results as unreviewed AI predictions. Never overwrites an existing
    # tag (human-confirmed or a prior AI prediction) -- see
    # `core.classify.classify_pool`.

    def _run_inference(self):
        if self._inference_thread is not None:
            return

        weights_path = self.infer_picker.weights_path()
        if weights_path is None:
            QMessageBox.warning(
                self, "yeastprep", "No classifier deployed yet -- choose one with Browse..."
            )
            return

        pooled = self.pool.pooled_annotations()
        if pooled is None:
            QMessageBox.warning(
                self, "yeastprep", "No FOVs checked in the pool to run inference on."
            )
            return

        try:
            classifier = YeastEfficientNetClassifier(
                weights_path=weights_path, meta_path=weights_path.with_name("meta.json")
            )
        except Exception as exc:
            QMessageBox.critical(self, "yeastprep", f"Could not load checkpoint: {exc}")
            return

        self.status_label.setText("Running inference on pool...")
        self._set_inference_running(True)

        self._inference_worker = ClassifierInferenceWorker(pooled, classifier)
        self._inference_thread = QThread()
        self._inference_worker.moveToThread(self._inference_thread)
        self._inference_thread.started.connect(self._inference_worker.run)
        self._inference_worker.finished.connect(self._on_inference_finished)
        self._inference_worker.error.connect(self._on_inference_error)
        self._inference_thread.start()

    def _clear_predictions(self):
        if self._inference_thread is not None:
            return
        pooled = self.pool.pooled_annotations()
        if pooled is None:
            QMessageBox.warning(self, "yeastprep", "No FOVs checked in the pool.")
            return
        n_predictions = sum(
            1 for _path, _category, confidence in pooled.tagged_items() if confidence is not None
        )
        if not n_predictions:
            QMessageBox.information(
                self, "yeastprep", "There are no unconfirmed AI predictions in the pool."
            )
            return
        answer = QMessageBox.question(
            self,
            "yeastprep",
            f"Remove {n_predictions} unconfirmed AI prediction(s) from the checked FOVs? "
            "Human-set and accepted annotations are kept.",
        )
        if answer != QMessageBox.Yes:
            return
        removed = pooled.clear_unconfirmed()
        self.inference_log.appendPlainText(f"cleared {removed} unconfirmed AI prediction(s)")
        self.annotation_analytics.refresh_from_pool()

    def _set_inference_running(self, running: bool):
        self.infer_btn.setEnabled(not running)
        self.infer_btn.setText("Running inference..." if running else "Run Inference on Pool")
        # Clearing predictions while inference is writing new ones would race it.
        self.clear_predictions_btn.setEnabled(not running)

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
        self.status_label.setText("Inference complete.")
        self._set_inference_running(False)
        self._teardown_inference_thread()
        # Newly-tagged predictions just landed on disk -- refresh the
        # Annotation Analytics tab's pool-wide summary rather than leaving
        # it showing pre-inference counts until the user happens to click
        # into that tab.
        self.annotation_analytics.refresh_from_pool()

    def _on_inference_error(self, message: str):
        self.inference_log.appendPlainText(f"inference ERROR: {message}")
        self.status_label.setText(f"Inference error: {message.splitlines()[0] if message else ''}")
        self._set_inference_running(False)
        self._teardown_inference_thread()

    # ------------------------------------------------------------------
    # Explore Embeddings -- UMAP projection of a backbone's embeddings
    # over the pool's confirmed tiles, optionally plus a random sample of
    # unlabeled ones (see `core.classify.sample_unlabeled`). The forward
    # pass and UMAP run on a background thread; the pool is read up front.

    def _evaluate_embeddings(self):
        if self._embedding_task.is_running():
            return
        weights_path = self.embed_picker.weights_path()
        if weights_path is None:
            QMessageBox.warning(
                self, "yeastprep", "No VICReg backbone deployed yet -- choose one with Browse..."
            )
            return

        records = self.pool.gather_confirmed_records()
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
        confirmed_labels = list(labels)

        if self.include_unlabeled_cb.isChecked():
            pooled = self.pool.pooled_annotations()
            n = self.unlabeled_sample_size.value()
            unlabeled_paths = sample_unlabeled(pooled, n) if pooled is not None else []
            paths = paths + unlabeled_paths
            labels = labels + [UNLABELED_LABEL] * len(unlabeled_paths)

        umap_kwargs = self._umap_kwargs()

        def compute():
            import torch

            device = select_device()
            backbone = load_backbone(weights_path=weights_path, device=device)
            backbone.eval()
            with torch.no_grad():
                embeddings = extract_embeddings(paths, backbone, device)
            n_confirmed = len(confirmed_labels)
            acc = (
                knn_accuracy(embeddings[:n_confirmed], confirmed_labels)
                if len(set(confirmed_labels)) > 1
                else None
            )
            return embeddings, labels, paths, acc, umap_2d(embeddings, **umap_kwargs)

        self._embedding_done_text = "Embeddings evaluated."
        self.status_label.setText(f"Evaluating embeddings for {len(paths)} tile(s)...")
        self._set_embedding_running(True)
        self._embedding_task.start(compute)

    def _replot_embeddings(self):
        if self._last_embeddings is None or self._embedding_task.is_running():
            return
        embeddings, labels, paths, acc = self._last_embeddings
        umap_kwargs = self._umap_kwargs()
        self._embedding_done_text = "Re-plotted embeddings."
        self.status_label.setText("Re-running UMAP...")
        self._set_embedding_running(True)
        self._embedding_task.start(
            lambda: (embeddings, labels, paths, acc, umap_2d(embeddings, **umap_kwargs))
        )

    def _set_embedding_running(self, running: bool):
        self.evaluate_btn.setEnabled(not running)
        self.replot_btn.setEnabled(not running and self._last_embeddings is not None)

    def _on_embeddings_ready(self, result):
        embeddings, labels, paths, acc, xy = result
        self._last_embeddings = (embeddings, labels, paths, acc)
        self.embedding_scatter.show_embedding_scatter(xy, labels, paths=paths, knn_acc=acc)
        self._set_embedding_running(False)
        self.status_label.setText(self._embedding_done_text)

    def _on_embeddings_failed(self, message: str):
        self._set_embedding_running(False)
        self.status_label.setText("Embedding evaluation failed.")
        QMessageBox.critical(self, "yeastprep", f"Embedding evaluation failed:\n\n{message}")

    def _open_viewer_for_selection(self, paths: list[str]) -> None:
        """Opens tiles lasso-selected on the embedding scatter in a tile
        viewer window of their own, to check a suspicious cluster or outlier
        against the actual images (and annotate them there). In-process,
        unlike the other ways of opening the viewer, since it shows an
        arbitrary set of cells rather than whole FOVs. Kept referenced here
        so the parentless window isn't garbage-collected."""
        if not paths:
            return
        folders = sorted({str(Path(p).parent) for p in paths})
        window = MainWindow(folders, paths, tiles_per_page=len(paths))
        window.show()
        self._embedding_viewer_windows.append(window)
        window.destroyed.connect(
            lambda: self._embedding_viewer_windows.remove(window)
        )

    # ------------------------------------------------------------------

    def busy_description(self) -> str | None:
        jobs = []
        if self._inference_thread is not None:
            jobs.append("Inference on pool")
        if self._embedding_task.is_running():
            jobs.append("Embedding evaluation")
        return ", ".join(jobs) or None

    def shutdown(self):
        self._embedding_task.shutdown()
        if self._inference_thread is not None:
            self._inference_thread.quit()
            self._inference_thread.wait()
