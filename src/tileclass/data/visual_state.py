"""Minimal global per-channel visibility/contrast state for the thumbnail
grid's own Qt-only compositor.

Mirrors pyvistra's ``TiledVisualProxy`` / ``ChannelDisplayList``, but
stripped to just what ``composite_to_rgb`` and ``ThumbnailColorsPanel``
actually use: clim + visibility. There's no colormap picker or gamma
control surface in this project (colors come from
``data/overlay_state.py``'s flat-color overlay state instead), so those
fields don't exist here -- gamma stays fixed at 1.0.
"""

from dataclasses import dataclass, replace


@dataclass
class ChannelDisplayState:
    clim: tuple = (0.0, 1.0)
    gamma: float = 1.0
    visible: bool = True


class ChannelDisplayList:
    """Ordered list of :class:`ChannelDisplayState`, one per channel."""

    def __init__(self, n_channels=0):
        self._states = [ChannelDisplayState() for _ in range(n_channels)]
        self._listeners = []

    def __len__(self):
        return len(self._states)

    def __getitem__(self, idx):
        return self._states[idx]

    def set_clim(self, idx, vmin, vmax):
        if 0 <= idx < len(self._states):
            self._states[idx] = replace(self._states[idx], clim=(vmin, vmax))
            self._notify(idx, "clim")

    def set_visible(self, idx, visible):
        if 0 <= idx < len(self._states):
            self._states[idx] = replace(self._states[idx], visible=visible)
            self._notify(idx, "visible")

    def subscribe(self, callback):
        """Register ``callback(channel_idx, field)``. Returns an
        unsubscribe function."""
        self._listeners.append(callback)

        def _unsubscribe():
            try:
                self._listeners.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    def _notify(self, idx, field):
        for cb in list(self._listeners):
            try:
                cb(idx, field)
            except Exception:
                pass


class VisualState:
    """Global per-channel visibility/contrast broadcast to the thumbnail
    grid compositor. Trimmed from pyvistra's ``TiledVisualProxy`` -- no
    colormap, no gamma control, no per-tile vispy renderers to fan
    settings out to; a change just invalidates the grid's pixmap cache
    so the next paint recomposites."""

    def __init__(self, viewer):
        self.viewer = viewer
        self._max_channels = 0
        # Channels whose clim was explicitly set via the Colors panel
        # (as opposed to left at the per-image auto-contrast default).
        self._custom_clim = set()
        self.display = ChannelDisplayList(0)
        self.display.subscribe(self._on_display_changed)

    def update_max_channels(self, max_c):
        if max_c == self._max_channels:
            return
        old = self.display
        new = ChannelDisplayList(max_c)
        for c in range(min(self._max_channels, max_c)):
            old_state = old[c]
            new.set_clim(c, *old_state.clim)
            new.set_visible(c, old_state.visible)
        new.subscribe(self._on_display_changed)
        self.display = new
        self._max_channels = max_c
        self._custom_clim = {c for c in self._custom_clim if c < max_c}

    def _on_display_changed(self, channel_idx, field):
        self.viewer.thumbnail_grid.invalidate_pixmaps()

    @property
    def custom_clim(self):
        """Live ``set[int]`` of channel indices explicitly overridden via
        the Colors panel. Mutating it in place (e.g. ``.clear()``) is how
        "Auto Contrast All" reverts every channel back to per-image
        auto-contrast."""
        return self._custom_clim

    def set_channel_visible(self, channel_idx, visible):
        self.display.set_visible(channel_idx, visible)

    def get_channel_visible(self, channel_idx):
        if channel_idx < len(self.display):
            return self.display[channel_idx].visible
        return True

    def set_clim(self, channel_idx, vmin, vmax):
        self._custom_clim.add(channel_idx)
        self.display.set_clim(channel_idx, vmin, vmax)
