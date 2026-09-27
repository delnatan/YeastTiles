"""Per-channel display state for the thumbnail grid's compositor
(``data/thumbnail_cache.py``'s ``composite_to_rgb``): flat color, blend
mode, and opacity -- persisted to the annotation sidecar's
``#channel_colors`` line -- plus visibility and an optional contrast
override, which last only for the session. Pure data, no Qt.
"""

from dataclasses import dataclass, replace

ADDITIVE = "additive"
OVERLAY = "overlay"


@dataclass(frozen=True)
class ChannelState:
    color_hex: str = "#ffffff"
    blend_mode: str = ADDITIVE
    opacity: float = 1.0
    visible: bool = True
    # None = each tile's own auto-contrast, computed at decode time.
    clim: tuple[float, float] | None = None

    @property
    def colors(self):
        """The persisted part: ``(color_hex, blend_mode, opacity)``."""
        return (self.color_hex, self.blend_mode, self.opacity)


# (color_hex, opacity) cycled through for new channels, blend mode always
# ADDITIVE. Channels 0-2 are tuned for this project's yeast-tile
# convention (core/tiles.py's CHANNEL_NAMES = brightfield, target, mask):
# 0 reads as plain grayscale, 1 as cyan fluorescence, 2 (the binary cell
# mask) as a translucent red highlight rather than a solid color covering
# the image underneath.
DEFAULT_CHANNEL_COLORS = [
    ("#5e5c64", 1.0),  # 0: brightfield -- neutral gray
    ("#00eaff", 1.0),  # 1: target/fluorescence -- cyan
    ("#c01c28", 0.33),  # 2: mask -- translucent red
    ("#ff00ea", 1.0),  # magenta
    ("#ffee00", 1.0),  # yellow
    ("#ffffff", 1.0),  # white
]


def default_channel_state(idx: int) -> ChannelState:
    color_hex, opacity = DEFAULT_CHANNEL_COLORS[idx % len(DEFAULT_CHANNEL_COLORS)]
    return ChannelState(color_hex=color_hex, opacity=opacity)


class ChannelStateList:
    """One :class:`ChannelState` per channel, grown in place as tiles with
    more channels finish decoding. Every mutation notifies subscribers
    with the channel index (``-1`` for all channels)."""

    def __init__(self, n_channels=0):
        self._states = [default_channel_state(i) for i in range(n_channels)]
        self._listeners = []

    def __len__(self):
        return len(self._states)

    def __getitem__(self, idx):
        return self._states[idx]

    def resize(self, n_channels):
        """Grow/shrink, keeping existing channels' state."""
        if n_channels == len(self._states):
            return
        del self._states[n_channels:]
        for i in range(len(self._states), n_channels):
            self._states.append(default_channel_state(i))
        self._notify(-1)

    def set(self, idx, **fields):
        """Replace the given fields of channel `idx`, e.g.
        ``set(0, opacity=0.5)``."""
        if 0 <= idx < len(self._states):
            self._states[idx] = replace(self._states[idx], **fields)
            self._notify(idx)

    def reset_clims(self):
        """Back to every tile's own auto-contrast on every channel."""
        self._states = [replace(state, clim=None) for state in self._states]
        self._notify(-1)

    def subscribe(self, callback):
        """Register ``callback(channel_idx)``. Returns an unsubscribe
        function."""
        self._listeners.append(callback)

        def _unsubscribe():
            if callback in self._listeners:
                self._listeners.remove(callback)

        return _unsubscribe

    def _notify(self, idx):
        for callback in list(self._listeners):
            callback(idx)
