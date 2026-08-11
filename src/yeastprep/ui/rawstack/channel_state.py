"""Per-channel display state for the raw-stack canvas.

Vendored (trimmed, adapted) from pyvistra/data/channel_state.py, kept as
plain Python (no Qt, no vispy) so the raw-stack viewer doesn't need a live
pyvistra dependency for rendering -- only pyvistra.io stays, for file
reading (see rawstack/panel.py).
"""

from dataclasses import dataclass, replace

from . import colormaps as _colormaps


@dataclass
class ChannelDisplayState:
    """Per-channel display parameters. Treat as immutable from outside
    ChannelDisplayList; mutations go through the list so listeners get
    notified."""

    clim: tuple = (0.0, 1.0)
    colormap_name: str = "White"
    visible: bool = True

    def display_color(self):
        _, color = _colormaps.get(self.colormap_name)
        return color


class ChannelDisplayList:
    """Ordered list of ChannelDisplayState, one per channel. Mutations go
    through set_clim/set_colormap_name/set_visible, each firing a
    (channel_idx, field) notification to every subscriber."""

    def __init__(self, n_channels, defaults=None):
        self._states = [
            replace(defaults) if defaults is not None else ChannelDisplayState()
            for _ in range(n_channels)
        ]
        self._listeners = []

    def __len__(self):
        return len(self._states)

    def __getitem__(self, idx):
        return self._states[idx]

    def set_clim(self, idx, vmin, vmax):
        if 0 <= idx < len(self._states):
            self._states[idx] = replace(self._states[idx], clim=(float(vmin), float(vmax)))
            self._notify(idx, "clim")

    def set_colormap_name(self, idx, name):
        if 0 <= idx < len(self._states):
            self._states[idx] = replace(self._states[idx], colormap_name=name)
            self._notify(idx, "colormap_name")

    def set_visible(self, idx, visible):
        if 0 <= idx < len(self._states):
            self._states[idx] = replace(self._states[idx], visible=bool(visible))
            self._notify(idx, "visible")

    def subscribe(self, callback):
        """Register callback(channel_idx, field). Returns an unsubscribe
        function."""
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
