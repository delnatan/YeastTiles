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

from qtpy.QtCore import QObject, QThread, QTimer, Signal

from yeastprep.core.channels import ChannelSelection
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
from yeastprep.core.segmentation import (
    SegmentationParams,
    SegmentationResult,
    SegmentProcessResult,
    get_model,
    run_segmentation,
    segment_and_save,
)
from yeastprep.core.tiles import TileExportResult, TileParams, append_tile_index, export_tiles


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


class PipelineController(QObject):
    """Main-thread-side manager: owns the worker's QThread, debounces
    parameter changes, and tags each request with a monotonic id so the
    GUI can tell which params a given result corresponds to."""

    stage_a_done = Signal(object)
    result_ready = Signal(object)
    error = Signal(str)
    _request_signal = Signal(object, object, object, int)

    def __init__(self, debounce_ms: int = 300, parent=None):
        super().__init__(parent)
        self._request_id = 0
        self._pending = None  # (path, channels, params) awaiting debounce flush

        self._thread = QThread()
        self._worker = FocusPipelineWorker()
        self._worker.moveToThread(self._thread)
        self._worker.stage_a_done.connect(self.stage_a_done)
        self._worker.result_ready.connect(self.result_ready)
        self._worker.error.connect(self.error)
        self._request_signal.connect(self._worker.request)
        self._thread.start()

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(debounce_ms)
        self._debounce_timer.timeout.connect(self._flush)

    def schedule(self, path, channels: ChannelSelection, params: FlattenFieldParams):
        """Debounced request -- coalesces rapid edits into one recompute
        after the user pauses."""
        self._pending = (path, channels, params)
        self._debounce_timer.start()

    def recompute_now(self, path, channels: ChannelSelection, params: FlattenFieldParams):
        """Immediate request, bypassing the debounce timer."""
        self._debounce_timer.stop()
        self._pending = None
        self._submit(path, channels, params)

    def _flush(self):
        if self._pending is not None:
            path, channels, params = self._pending
            self._pending = None
            self._submit(path, channels, params)

    def _submit(self, path, channels, params):
        self._request_id += 1
        self._request_signal.emit(path, channels, params, self._request_id)

    def latest_request_id(self) -> int:
        return self._request_id

    def shutdown(self):
        self._thread.quit()
        self._thread.wait()


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


class SegmentationPipelineWorker(QObject):
    """Lives on its own QThread, separate from FocusPipelineWorker's --
    cellpose inference shouldn't block the flatten pipeline's queue (or
    vice versa). No stage-A/B-D style caching split here: unlike focus
    diagnostics, every param in SegmentationParams feeds the network
    forward pass, so there's no cheap post-processing step worth
    separating out yet."""

    result_ready = Signal(object)  # SegmentationPipelineResult
    error = Signal(str)

    def request(
        self,
        focal_slice,
        source_id: int,
        params: SegmentationParams,
        request_id: int,
    ):
        try:
            model = get_model(params.model_path)
            result = run_segmentation(model, focal_slice, params)
            self.result_ready.emit(
                SegmentationPipelineResult(
                    request_id, source_id, focal_slice, result
                )
            )
        except Exception as exc:
            self.error.emit(str(exc))


class SegmentationController(QObject):
    """Debounced controller for live segmentation preview, structurally
    identical to PipelineController. The cache key is (source_id, params)
    rather than a hash of the focal-slice array itself: main_window bumps
    a single monotonic counter every time the "current" focal slice
    changes for *any* reason -- a new flatten-pipeline result, or the user
    picking an already-processed tiff straight off disk -- so that counter
    is a free, exact stand-in for "did the upstream image actually
    change," regardless of which source it came from."""

    result_ready = Signal(object)
    error = Signal(str)
    _request_signal = Signal(object, object, object, int)

    def __init__(self, debounce_ms: int = 300, parent=None):
        super().__init__(parent)
        self._request_id = 0
        self._pending = None
        self._last_submitted_key = None

        self._thread = QThread()
        self._worker = SegmentationPipelineWorker()
        self._worker.moveToThread(self._thread)
        self._worker.result_ready.connect(self.result_ready)
        self._worker.error.connect(self.error)
        self._request_signal.connect(self._worker.request)
        self._thread.start()

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(debounce_ms)
        self._debounce_timer.timeout.connect(self._flush)

    def schedule(self, focal_slice, source_id: int, params: SegmentationParams):
        """Debounced request -- a no-op if neither the upstream focal
        slice nor the segmentation params changed since the last submit
        (e.g. the user is editing a flatten-only param while on the
        Segmentation tab)."""
        cache_key = (source_id, params)
        if cache_key == self._last_submitted_key:
            return
        self._pending = (focal_slice, source_id, params)
        self._debounce_timer.start()

    def recompute_now(self, focal_slice, source_id: int, params: SegmentationParams):
        """Immediate request, bypassing both the debounce timer and the
        unchanged-input check."""
        self._debounce_timer.stop()
        self._pending = None
        self._submit(focal_slice, source_id, params)

    def _flush(self):
        if self._pending is not None:
            focal_slice, source_id, params = self._pending
            self._pending = None
            self._submit(focal_slice, source_id, params)

    def _submit(self, focal_slice, source_id, params):
        self._last_submitted_key = (source_id, params)
        self._request_id += 1
        self._request_signal.emit(focal_slice, source_id, params, self._request_id)

    def latest_request_id(self) -> int:
        return self._request_id

    def shutdown(self):
        self._thread.quit()
        self._thread.wait()


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
