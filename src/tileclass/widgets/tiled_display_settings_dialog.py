from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
    QVBoxLayout,
)


class TiledDisplaySettingsDialog(QDialog):
    """Per-window min/max limits for TiledViewer's tile-size slider and
    tiles-per-page spinbox -- not persisted, so a session can be tuned to
    the current monitor/machine without affecting other windows.

    Each min/max pair cross-clamps the other spinbox's range live, so the
    dialog can never be accepted with an inverted or degenerate range.
    """

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tile Size / Page Limits")

        cfg = dict(config or {})

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.tile_min = QSpinBox()
        self.tile_min.setRange(20, 2000)
        self.tile_max = QSpinBox()
        self.tile_max.setRange(20, 2000)
        self.tile_min.setValue(int(cfg.get("tile_size_min", 100)))
        self.tile_max.setValue(int(cfg.get("tile_size_max", 400)))
        self.tile_min.valueChanged.connect(lambda v: self.tile_max.setMinimum(v + 1))
        self.tile_max.valueChanged.connect(lambda v: self.tile_min.setMaximum(v - 1))
        form.addRow("Min Tile Size (px)", self.tile_min)
        form.addRow("Max Tile Size (px)", self.tile_max)

        self.page_min = QSpinBox()
        self.page_min.setRange(1, 1000)
        self.page_max = QSpinBox()
        self.page_max.setRange(1, 1000)
        self.page_min.setValue(int(cfg.get("tiles_per_page_min", 1)))
        self.page_max.setValue(int(cfg.get("tiles_per_page_max", 300)))
        self.page_min.valueChanged.connect(lambda v: self.page_max.setMinimum(v + 1))
        self.page_max.valueChanged.connect(lambda v: self.page_min.setMaximum(v - 1))
        form.addRow("Min Tiles / Page", self.page_min)
        form.addRow("Max Tiles / Page", self.page_max)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self):
        return {
            "tile_size_min": self.tile_min.value(),
            "tile_size_max": self.tile_max.value(),
            "tiles_per_page_min": self.page_min.value(),
            "tiles_per_page_max": self.page_max.value(),
        }
