"""Background workers for the field-flattening diagnostic tab.

`PipelineController` owns a long-lived `QThread` running `FocusPipelineWorker`
and debounces rapid parameter edits (a dragged spinbox shouldn't trigger a
recompute per tick). `FocusPipelineWorker` caches the expensive Stage A
(`compute_tile_variance_stack`, touches every pixel) separately from the
cheap Stages B-D (peak-finding, poly fit, resample), keyed on whichever
inputs actually changed.

`BatchProcessWorker` is unrelated and simpler: no caching needed, just
iterate files and call `core.pipeline.process_and_save`.

Structural reference (not API-identical, since pyvistra's ZProjectionWorker
is coupled to its own buffer/viewer types): pyvistra/widgets/z_projection_dialog.py's
QObject-worker-on-QThread + progress/finished/error/cancelled signal pattern.
"""

from dataclasses import dataclass
from pathlib import Path

from jssl_denoise.training import Trainer

from qtpy.QtCore import QObject, QThread, QTimer, Signal
from tileclass.training.supervised import TrainingCancelled, TrainingParams, train_classifier
from tileclass.training.vicreg import VICRegParams, pretrain_vicreg

from yeastprep.core.channels import ChannelSelection
from yeastprep.core.classify import classify_pool
from yeastprep.core.deconvolve import (
    DeconvolveParams,
    DeconvolveProcessResult,
    deconvolve_and_save,
    run_deconvolve,
)
from yeastprep.core.denoise import (
    DenoiseParams,
    DenoiseProcessResult,
    denoise_and_save,
    run_denoise,
)
from yeastprep.core.focus import (
    compute_tile_variance_stack,
    fit_focal_indices_to_poly2d,
    peaks_from_variance_stack,
)
from yeastprep.core.pipeline import (
    FlattenFieldParams,
    FocusDiagnostics,
    LoadedVolume,
    ProcessResult,
    compute_focal_slice,
    load_volume,
    process_and_save,
)
from yeastprep.core.project_scan import ProjectScanSnapshot, scan_project
from yeastprep.core.segmentation import (
    SegmentationParams,
    SegmentationResult,
    SegmentProcessResult,
    get_model,
    run_segmentation,
    segment_and_save,
)
from yeastprep.core.tiles import TileExportResult, TileParams, append_tile_index, export_tiles


class DebouncedController(QObject):
    """Shared main-thread-side manager for a `(payload, source_id, params,
    request_id)`-shaped pipeline worker: owns the worker's QThread,
    debounces rapid parameter changes, and tags each request with a
    monotonic id so the GUI can tell which params a given result
    corresponds to. Behaviorally identical to what used to be four
    hand-duplicated ~90-line Controller classes.

    `dedupe_key_fn(payload, source_id, params)`, if given, makes `schedule`
    a no-op when neither the upstream source nor the params changed since
    the last submit -- e.g. the user editing an unrelated param while on a
    different tab. `PipelineController` (the flatten-diagnostics
    controller) passes `dedupe_key_fn=None` to keep its original
    always-debounce behavior; the other three controllers pass
    `lambda payload, source_id, params: (source_id, params)`."""

    result_ready = Signal(object)
    error = Signal(str)
    _request_signal = Signal(object, object, object, int)

    def __init__(self, worker: QObject, debounce_ms: int = 300, dedupe_key_fn=None, parent=None):
        super().__init__(parent)
        self._dedupe_key_fn = dedupe_key_fn
        self._request_id = 0
        self._pending = None  # (payload, source_id, params) awaiting debounce flush
        self._last_submitted_key = None

        self._thread = QThread()
        self._worker = worker
        self._worker.moveToThread(self._thread)
        self._worker.result_ready.connect(self.result_ready)
        self._worker.error.connect(self.error)
        self._request_signal.connect(self._worker.request)
        self._thread.start()

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(debounce_ms)
        self._debounce_timer.timeout.connect(self._flush)

    def schedule(self, payload, source_id, params):
        """Debounced request -- coalesces rapid edits into one recompute
        after the user pauses; a no-op if `dedupe_key_fn` says nothing
        actually changed since the last submit."""
        if self._dedupe_key_fn is not None:
            cache_key = self._dedupe_key_fn(payload, source_id, params)
            if cache_key == self._last_submitted_key:
                return
        self._pending = (payload, source_id, params)
        self._debounce_timer.start()

    def recompute_now(self, payload, source_id, params):
        """Immediate request, bypassing both the debounce timer and the
        unchanged-input check."""
        self._debounce_timer.stop()
        self._pending = None
        self._submit(payload, source_id, params)

    def _flush(self):
        if self._pending is not None:
            payload, source_id, params = self._pending
            self._pending = None
            self._submit(payload, source_id, params)

    def _submit(self, payload, source_id, params):
        if self._dedupe_key_fn is not None:
            self._last_submitted_key = self._dedupe_key_fn(payload, source_id, params)
        self._request_id += 1
        self._request_signal.emit(payload, source_id, params, self._request_id)

    def latest_request_id(self) -> int:
        return self._request_id

    def shutdown(self):
        self._thread.quit()
        self._thread.wait()


class SimplePipelineWorker(QObject):
    """Shared worker for the three stage previews whose `request()` is
    just `result_cls(request_id, source_id, payload, compute_fn(payload,
    params))` wrapped in a try/except -- `SegmentationPipelineWorker`,
    `DenoisePipelineWorker`, and `DeconvolvePipelineWorker`'s bodies before
    this consolidation. `compute_fn(payload, params)` is one of
    `run_denoise`/`run_deconvolve`/a small segmentation adapter (see
    below); `result_cls` is that stage's existing 4-field result
    dataclass, unchanged."""

    result_ready = Signal(object)
    error = Signal(str)

    def __init__(self, compute_fn, result_cls):
        super().__init__()
        self._compute_fn = compute_fn
        self._result_cls = result_cls

    def request(self, payload, source_id, params, request_id: int):
        try:
            computed = self._compute_fn(payload, params)
            self.result_ready.emit(self._result_cls(request_id, source_id, payload, computed))
        except Exception as exc:
            self.error.emit(str(exc))


@dataclass
class ProjectScanResult:
    request_id: int
    source_id: str
    payload: str  # project root
    scan: ProjectScanSnapshot


def _run_project_scan(root: str, params) -> ProjectScanSnapshot:
    raw_pattern, segmentation_override, stage_keys = params
    return scan_project(root, raw_pattern, segmentation_override, stage_keys)


class ProjectScanController(DebouncedController):
    """Debounced controller for `ProjectTreePanel`'s background project
    scan (`core.project_scan.scan_project`) -- keeps the glob+stat walk of
    every stage folder off the GUI thread, since on a slow external/network
    drive with hundreds of raw files that walk alone can take long enough
    to make the app look hung (and get force-killed as "not responding").
    `dedupe_key_fn` collapses back-to-back triggers for the same project
    with unchanged params (e.g. `set_project_root` immediately writing a
    default raw pattern) into a single scan."""

    def __init__(self):
        super().__init__(
            SimplePipelineWorker(_run_project_scan, ProjectScanResult),
            debounce_ms=150,
            dedupe_key_fn=lambda payload, source_id, params: (payload, params),
        )


@dataclass
class PipelineResult:
    request_id: int
    volume: LoadedVolume
    diagnostics: FocusDiagnostics
    focal_slice: object


class FocusPipelineWorker(QObject):
    """Lives on its own QThread. `request()` is invoked via a queued
    connection from `PipelineController`, so calls execute one at a time,
    in submission order, on the worker thread."""

    stage_a_done = Signal(object)  # LoadedVolume
    result_ready = Signal(object)  # PipelineResult
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._cache_key = None
        self._cached_volume = None
        self._cached_variance_stack = None

    def request(self, path, channels: ChannelSelection, params: FlattenFieldParams, request_id: int):
        try:
            cache_key = (
                str(path),
                channels.brightfield,
                channels.projection,
                params.num_tiles_y,
                params.num_tiles_x,
            )
            if cache_key != self._cache_key:
                volume = load_volume(path, channels)
                variance_stack = compute_tile_variance_stack(
                    volume.img3d, params.num_tiles_y, params.num_tiles_x
                )
                self._cached_volume = volume
                self._cached_variance_stack = variance_stack
                self._cache_key = cache_key
                self.stage_a_done.emit(volume)
            else:
                volume = self._cached_volume
                variance_stack = self._cached_variance_stack

            coarse = peaks_from_variance_stack(
                variance_stack, params.inverted_variance_prominence
            )
            _Nz, Ny, Nx = volume.img3d.shape
            fine = fit_focal_indices_to_poly2d(
                coarse, Nx, Ny, variance_stack.tile_info, poly_degree=params.poly_degree
            )
            diagnostics = FocusDiagnostics(
                variance_stack=variance_stack,
                coarse_focal_indices=coarse,
                fine_focus_indices=fine,
            )
            focal_slice = compute_focal_slice(volume, diagnostics, params)

            self.result_ready.emit(
                PipelineResult(
                    request_id=request_id,
                    volume=volume,
                    diagnostics=diagnostics,
                    focal_slice=focal_slice,
                )
            )
        except Exception as exc:
            self.error.emit(str(exc))


class PipelineController(DebouncedController):
    """Debounced controller for the flatten-diagnostics pipeline. Sits on
    top of `FocusPipelineWorker`'s own Stage-A/B-D caching, so it always
    debounces rather than deduping on an unchanged-input check (no
    `dedupe_key_fn`) -- identical to its pre-consolidation behavior. Adds
    `stage_a_done`, the one signal `FocusPipelineWorker` has that the
    other three stage workers don't."""

    stage_a_done = Signal(object)

    def __init__(self, debounce_ms: int = 300, parent=None):
        worker = FocusPipelineWorker()
        super().__init__(worker, debounce_ms=debounce_ms, dedupe_key_fn=None, parent=parent)
        worker.stage_a_done.connect(self.stage_a_done)

    def schedule(self, path, channels: ChannelSelection, params: FlattenFieldParams):
        super().schedule(path, channels, params)

    def recompute_now(self, path, channels: ChannelSelection, params: FlattenFieldParams):
        super().recompute_now(path, channels, params)


class BatchProcessWorker(QObject):
    progress = Signal(int, int, str)  # done, total, current_name
    file_result = Signal(object)  # ProcessResult
    finished = Signal()
    cancelled = Signal()

    def __init__(
        self,
        paths: list[Path],
        outdir: Path,
        params: FlattenFieldParams,
        channels: ChannelSelection,
    ):
        super().__init__()
        self._paths = list(paths)
        self._outdir = outdir
        self._params = params
        self._channels = channels
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        total = len(self._paths)
        for done, path in enumerate(self._paths, start=1):
            if self._cancel_requested:
                self.cancelled.emit()
                return
            self.progress.emit(done - 1, total, Path(path).name)
            result: ProcessResult = process_and_save(
                path, self._outdir, self._params, self._channels
            )
            self.file_result.emit(result)
            self.progress.emit(done, total, Path(path).name)
        self.finished.emit()


@dataclass
class SegmentationPipelineResult:
    request_id: int
    source_id: int
    focal_slice: object
    result: SegmentationResult


def _run_segmentation_for_pipeline(focal_slice, params: SegmentationParams) -> SegmentationResult:
    """`SimplePipelineWorker`-shaped adapter over `run_segmentation`: unlike
    denoise/deconvolve, segmentation needs a model looked up from
    `params.model_path` before the forward pass."""
    model = get_model(params.model_path)
    return run_segmentation(model, focal_slice, params)


def _new_segmentation_worker() -> SimplePipelineWorker:
    return SimplePipelineWorker(_run_segmentation_for_pipeline, SegmentationPipelineResult)


class SegmentationController(DebouncedController):
    """Debounced controller for live segmentation preview. The cache key is
    (source_id, params) rather than a hash of the focal-slice array itself:
    main_window bumps a single monotonic counter every time the "current"
    focal slice changes for *any* reason -- a new flatten-pipeline result,
    or the user picking an already-processed tiff straight off disk -- so
    that counter is a free, exact stand-in for "did the upstream image
    actually change," regardless of which source it came from."""

    def __init__(self, debounce_ms: int = 300, parent=None):
        super().__init__(
            _new_segmentation_worker(),
            debounce_ms=debounce_ms,
            dedupe_key_fn=lambda payload, source_id, params: (source_id, params),
            parent=parent,
        )

    def schedule(self, focal_slice, source_id: int, params: SegmentationParams):
        """Debounced request -- a no-op if neither the upstream focal
        slice nor the segmentation params changed since the last submit
        (e.g. the user is editing a flatten-only param while on the
        Segmentation tab)."""
        super().schedule(focal_slice, source_id, params)

    def recompute_now(self, focal_slice, source_id: int, params: SegmentationParams):
        super().recompute_now(focal_slice, source_id, params)


class SegmentationBatchWorker(QObject):
    """Batch counterpart to BatchProcessWorker: segments every focal-slice
    tiff already saved by stage 1, writing cellpose's native `_seg.npy`
    next to each one."""

    progress = Signal(int, int, str)  # done, total, current_name
    file_result = Signal(object)  # SegmentProcessResult
    finished = Signal()
    cancelled = Signal()

    def __init__(self, paths: list[Path], params: SegmentationParams):
        super().__init__()
        self._paths = list(paths)
        self._params = params
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        model = get_model(self._params.model_path)
        total = len(self._paths)
        for done, path in enumerate(self._paths, start=1):
            if self._cancel_requested:
                self.cancelled.emit()
                return
            self.progress.emit(done - 1, total, Path(path).name)
            result: SegmentProcessResult = segment_and_save(path, model, self._params)
            self.file_result.emit(result)
            self.progress.emit(done, total, Path(path).name)
        self.finished.emit()


@dataclass
class DenoisePipelineResult:
    request_id: int
    source_id: int
    before: object
    after: object


def _new_denoise_worker() -> SimplePipelineWorker:
    return SimplePipelineWorker(run_denoise, DenoisePipelineResult)


class DenoiseController(DebouncedController):
    """Debounced controller for live denoise preview."""

    def __init__(self, debounce_ms: int = 300, parent=None):
        super().__init__(
            _new_denoise_worker(),
            debounce_ms=debounce_ms,
            dedupe_key_fn=lambda payload, source_id, params: (source_id, params),
            parent=parent,
        )

    def schedule(self, image, source_id: int, params: DenoiseParams):
        super().schedule(image, source_id, params)

    def recompute_now(self, image, source_id: int, params: DenoiseParams):
        """Immediate request, bypassing both the debounce timer and the
        unchanged-input check."""
        super().recompute_now(image, source_id, params)


class DenoiseBatchWorker(QObject):
    """Batch counterpart to BatchProcessWorker: denoises every
    combined-channel tiff already saved upstream, writing into `outdir`
    (02_denoised/)."""

    progress = Signal(int, int, str)  # done, total, current_name
    file_result = Signal(object)  # DenoiseProcessResult
    finished = Signal()
    cancelled = Signal()

    def __init__(self, paths: list[Path], outdir: Path, params: DenoiseParams):
        super().__init__()
        self._paths = list(paths)
        self._outdir = outdir
        self._params = params
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        total = len(self._paths)
        for done, path in enumerate(self._paths, start=1):
            if self._cancel_requested:
                self.cancelled.emit()
                return
            self.progress.emit(done - 1, total, Path(path).name)
            result: DenoiseProcessResult = denoise_and_save(path, self._outdir, self._params)
            self.file_result.emit(result)
            self.progress.emit(done, total, Path(path).name)
        self.finished.emit()


@dataclass
class DeconvolvePipelineResult:
    request_id: int
    source_id: int
    target_before: object
    target_after: object


def _new_deconvolve_worker() -> SimplePipelineWorker:
    return SimplePipelineWorker(run_deconvolve, DeconvolvePipelineResult)


class DeconvolveController(DebouncedController):
    """Debounced controller for live deconvolve preview."""

    def __init__(self, debounce_ms: int = 300, parent=None):
        super().__init__(
            _new_deconvolve_worker(),
            debounce_ms=debounce_ms,
            dedupe_key_fn=lambda payload, source_id, params: (source_id, params),
            parent=parent,
        )

    def schedule(self, target, source_id: int, params: DeconvolveParams):
        super().schedule(target, source_id, params)

    def recompute_now(self, target, source_id: int, params: DeconvolveParams):
        """Immediate request, bypassing both the debounce timer and the
        unchanged-input check."""
        super().recompute_now(target, source_id, params)


class DeconvolveBatchWorker(QObject):
    """Batch counterpart to BatchProcessWorker: deconvolves every
    combined-channel tiff already saved upstream (typically 02_denoised/ if
    that stage ran, else 01_reduced/), writing into `outdir`
    (03_deconvolved/)."""

    progress = Signal(int, int, str)  # done, total, current_name
    file_result = Signal(object)  # DeconvolveProcessResult
    finished = Signal()
    cancelled = Signal()

    def __init__(self, paths: list[Path], outdir: Path, params: DeconvolveParams):
        super().__init__()
        self._paths = list(paths)
        self._outdir = outdir
        self._params = params
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        total = len(self._paths)
        for done, path in enumerate(self._paths, start=1):
            if self._cancel_requested:
                self.cancelled.emit()
                return
            self.progress.emit(done - 1, total, Path(path).name)
            result: DeconvolveProcessResult = deconvolve_and_save(
                path, self._outdir, self._params
            )
            self.file_result.emit(result)
            self.progress.emit(done, total, Path(path).name)
        self.finished.emit()


class TileBatchWorker(QObject):
    """Batch counterpart to SegmentationBatchWorker: crops every cell out
    of each already-segmented tiff's saved masks, writing per-cell tiffs
    under `out_dir` and appending each file's rows to that folder's running
    tile index (core.tiles.append_tile_index) as it goes -- so a run that's
    cancelled partway still leaves a usable, consistent index."""

    progress = Signal(int, int, str)  # done, total, current_name
    file_result = Signal(object)  # TileExportResult
    finished = Signal()
    cancelled = Signal()

    def __init__(self, paths: list[Path], out_dir: Path, params: TileParams):
        super().__init__()
        self._paths = list(paths)
        self._out_dir = out_dir
        self._params = params
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        total = len(self._paths)
        for done, path in enumerate(self._paths, start=1):
            if self._cancel_requested:
                self.cancelled.emit()
                return
            self.progress.emit(done - 1, total, Path(path).name)
            result: TileExportResult = export_tiles(path, self._out_dir, self._params)
            if result.success and result.records is not None:
                append_tile_index(self._out_dir, result.records)
            self.file_result.emit(result)
            self.progress.emit(done, total, Path(path).name)
        self.finished.emit()


def _release_device_cache(device) -> None:
    """MPS/CUDA's caching allocator doesn't return freed blocks to the
    driver on its own -- worth releasing after a long training run so a
    long-lived GUI session doesn't accumulate pinned device memory across
    repeated Train runs (same justification as jssl_denoise's own
    pyvistra-plugin worker, which isn't importable here -- see
    TrainingWorker's docstring)."""
    if device is None:
        return
    try:
        import torch

        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


class TrainingWorker(QObject):
    """Trains a D-Net/N-Net denoiser pair on pooled frames via
    `jssl_denoise.training.Trainer.fit`, reporting progress via Qt signals.

    jssl_denoise ships its own pyvistra-plugin version of this class
    (`jssl_denoise.pyvistra_gui.train_worker.TrainingWorker`), but that
    submodule isn't part of the installed `jssl-denoise` package pin (see
    pyproject.toml) -- only `jssl_denoise.training` itself is guaranteed
    available, so this is a from-scratch, Qt-only re-implementation of
    that same thin wrapper: it duck-types
    `jssl_denoise.callbacks.TrainingCallback` (`on_step_end`/
    `on_epoch_end`/`on_epoch_metrics`/`on_early_stop`) and passes itself to
    `Trainer.fit` directly as `callback=self`.
    """

    step_progress = Signal(int, int, int, float)  # step, total_steps, epoch, loss
    epoch_finished = Signal(int, int, float, float)  # epoch, total_epochs, loss, lr
    epoch_metrics = Signal(int, dict)  # epoch, {"mu_mse": ..., "sigma_mean": ...}
    early_stopped = Signal(int, int, float)  # epoch, best_epoch, best_mu_mse
    finished = Signal(object)  # checkpoint dict
    cancelled = Signal(object)  # checkpoint dict -- see run()
    error = Signal(str)

    def __init__(self, images, config):
        super().__init__()
        self._images = images
        self._config = config
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    # ------------------------------------------------------------------
    # TrainingCallback protocol

    def on_step_end(self, step, total_steps, epoch, loss):
        self.step_progress.emit(step, total_steps, epoch, loss)

    def on_epoch_end(self, epoch, total_epochs, loss, lr):
        self.epoch_finished.emit(epoch, total_epochs, loss, lr)

    def on_epoch_metrics(self, epoch, metrics):
        self.epoch_metrics.emit(epoch, dict(metrics))

    def on_early_stop(self, epoch, best_epoch, best_mu_mse):
        self.early_stopped.emit(epoch, best_epoch, best_mu_mse)

    # ------------------------------------------------------------------

    def run(self):
        import torch

        device = (
            torch.device(self._config.device)
            if self._config.device
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        )
        try:
            checkpoint = Trainer(self._config).fit(
                self._images, callback=self, should_stop=lambda: self._cancel_requested
            )
        except Exception as exc:
            self.error.emit(str(exc))
            return
        finally:
            _release_device_cache(device)

        if self._cancel_requested:
            self.cancelled.emit(checkpoint)
        else:
            self.finished.emit(checkpoint)


class ClassifierTrainingWorker(QObject):
    """Wraps `tileclass.training.supervised.train_classifier` -- ported
    from tileclass's own `workers.TrainingWorker` (now removed there, see
    the Classifier Training page) verbatim except for the added
    `output_dir`: a yeastprep-driven run always writes its resulting
    checkpoint to a project-local session folder rather than tileclass's
    live inference slot (see `core.classify.supervised_output_dir` and
    `train_classifier`'s `output_dir` docstring) -- promoting it there is
    a separate, explicit "Deploy to Tile Classifier" step."""

    progress = Signal(object)  # tileclass.training.supervised.TrainingProgress
    finished = Signal(object)  # tileclass.training.supervised.TrainingResult
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        records,
        params: TrainingParams = TrainingParams(),
        output_dir: Path | None = None,
        backbone_weights_path=None,
        categories=None,
    ):
        super().__init__()
        self._records = list(records)
        self._params = params
        self._output_dir = output_dir
        self._backbone_weights_path = backbone_weights_path
        self._categories = categories
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            result = train_classifier(
                self._records,
                params=self._params,
                progress_callback=self.progress.emit,
                cancel_check=lambda: self._cancel_requested,
                backbone_weights_path=self._backbone_weights_path,
                categories=self._categories,
                output_dir=self._output_dir,
            )
        except TrainingCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.finished.emit(result)


class ClassifierInferenceWorker(QObject):
    """Runs `core.classify.classify_pool` off the GUI thread -- a pooled
    inference pass over every tile crop under the checked FOV folders can
    be thousands of tiles across several projects, unlike tileclass's own
    per-page Auto-Annotate (a single page, blocking is fine there). Only a
    finished/error signal, no epoch-by-epoch progress -- inference has no
    natural per-item checkpoint to report through, and no cancel: a batched
    forward pass isn't interruptible mid-flight the way a training loop's
    per-epoch boundary is."""

    finished = Signal(object)  # core.classify.ClassifyPoolResult
    error = Signal(str)

    def __init__(self, pooled, classifier):
        super().__init__()
        self._pooled = pooled
        self._classifier = classifier

    def run(self):
        try:
            result = classify_pool(self._pooled, self._classifier)
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.finished.emit(result)


class ClassifierVicregWorker(QObject):
    """Wraps `tileclass.training.vicreg.pretrain_vicreg` -- ported from
    tileclass's own `workers.VICRegWorker` (now removed there), with the
    same `output_dir` addition as `ClassifierTrainingWorker` above."""

    progress = Signal(object)  # tileclass.training.vicreg.VICRegProgress
    finished = Signal(object)  # tileclass.training.vicreg.VICRegResult
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, records, params: VICRegParams = VICRegParams(), output_dir: Path | None = None):
        super().__init__()
        self._records = list(records)
        self._params = params
        self._output_dir = output_dir
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            result = pretrain_vicreg(
                self._records,
                params=self._params,
                progress_callback=self.progress.emit,
                cancel_check=lambda: self._cancel_requested,
                output_dir=self._output_dir,
            )
        except TrainingCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:
            self.error.emit(str(exc))
            return
        self.finished.emit(result)
