"""Embedded vispy canvas for the raw-stack viewer tab.

Vendored (trimmed, adapted) from pyvistra/visuals/image.py's
CompositeImageVisual + the SceneCanvas setup in pyvistra/viewers/tiled.py.
Only what's needed for "decent multi-channel visual check" (design.md
Feature 1, step 1): pan/zoom + per-channel colormap/clim/visibility
compositing of a single (C, Y, X) plane. No ROI, line-profile, FFT,
timelapse, or rotation/alignment -- those are pyvistra features this tool
doesn't need.
"""

import numpy as np
from qtpy import API_NAME
from qtpy.QtWidgets import QWidget, QVBoxLayout
from vispy import app as _vispy_app
from vispy import scene
from vispy.visuals.transforms.linear import STTransform

from .channel_state import ChannelDisplayList

try:
    _vispy_app.use_app(API_NAME)
except Exception:
    pass

from . import colormaps as _colormaps  # noqa: E402

# GL internal formats exist for these under texture_format="auto";
# anything else needs an explicit cast before upload.
_GL_TEXTURE_DTYPES = (
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.float32),
)


def default_clim_for_dtype(dtype) -> tuple[float, float]:
    if dtype == np.uint16:
        return (0.0, 65535.0)
    if dtype.kind == "f":
        return (0.0, 1.0)
    return (0.0, 255.0)


class RawStackCanvas(QWidget):
    """A vispy SceneCanvas embedded as a plain QWidget, displaying one
    (C, Y, X) plane as an additive composite of per-channel image layers.

    Channel display state lives in `self.display` (a ChannelDisplayList);
    UI controls should mutate it directly (`set_clim`/`set_colormap`/
    `set_channel_visible`) rather than touching vispy layers.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = scene.SceneCanvas(keys=None, bgcolor="#1a1a1a", show=False)
        self.canvas.native.setMinimumSize(200, 200)
        layout.addWidget(self.canvas.native)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = "panzoom"
        self.view.camera.aspect = 1

        self.layers: list[scene.visuals.Image] = []
        self.display: ChannelDisplayList | None = None
        self._scale = (1.0, 1.0)
        self._shape_yx = (1, 1)
        self._current_plane = None

    def set_channels(
        self, n_channels: int, dtype, scale: tuple[float, float] = (1.0, 1.0)
    ):
        """(Re)build layers for a new stack -- call once per file load
        (channel count / dtype changed), not per-slice."""
        for layer in self.layers:
            layer.parent = None
        self.layers = []
        self._scale = scale

        default_clim = default_clim_for_dtype(np.dtype(dtype))
        self.display = ChannelDisplayList(n_channels)
        palette = list(_colormaps._TWO_COLOR)  # Orange, Green, Cyan, Magenta, Yellow, White
        for c in range(n_channels):
            self.display.set_clim(c, *default_clim)
            self.display.set_colormap_name(
                c, "White" if n_channels == 1 else palette[c % len(palette)]
            )

        for c in range(n_channels):
            state = self.display[c]
            cmap, _ = _colormaps.get(state.colormap_name)
            layer = scene.visuals.Image(
                cmap=cmap, parent=self.view.scene, method="auto", interpolation="nearest"
            )
            layer.clim = state.clim
            layer.transform = STTransform(scale=scale)
            layer.set_gl_state(
                preset="translucent", blend=True, blend_func=("one", "one"), depth_test=False
            )
            layer.order = -c
            self.layers.append(layer)

        self.display.subscribe(self._on_display_changed)

    def set_plane(self, plane: np.ndarray):
        """Display an already-sliced (C, Y, X) or (Y, X) plane."""
        if plane.ndim == 2:
            plane = plane[np.newaxis, :, :]
        if plane.dtype not in _GL_TEXTURE_DTYPES:
            plane = plane.astype(np.float32)
        self._current_plane = plane
        self._shape_yx = plane.shape[-2:]
        for c, layer in enumerate(self.layers):
            if c < plane.shape[0]:
                layer.set_data(plane[c])
        self._update_visibility()

    def reset_camera(self):
        Y, X = self._shape_yx
        sy, sx = self._scale
        self.view.camera.rect = (0, 0, X * sx, Y * sy)
        self.view.camera.flip = (False, True, False)

    def auto_contrast(self, pct_low=0.5, pct_high=99.98):
        from tileclass.contrast import compute_percentile_clim

        if self._current_plane is None:
            return
        for c in range(self._current_plane.shape[0]):
            if c < len(self.layers):
                vmin, vmax = compute_percentile_clim(
                    self._current_plane[c], pct_low, pct_high
                )
                self.display.set_clim(c, vmin, vmax)

    def _on_display_changed(self, idx, field):
        if idx >= len(self.layers):
            return
        state = self.display[idx]
        layer = self.layers[idx]
        if field == "clim":
            layer.clim = state.clim
        elif field == "colormap_name":
            cmap, _ = _colormaps.get(state.colormap_name)
            layer.cmap = cmap
        elif field == "visible":
            self._update_visibility()

    def _update_visibility(self):
        for i, layer in enumerate(self.layers):
            layer.visible = self.display[i].visible if self.display else True
