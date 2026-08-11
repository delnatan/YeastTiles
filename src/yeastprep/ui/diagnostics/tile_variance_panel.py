"""Per-tile intensity-variance-vs-Z inspector, shown in a dock, driven by
clicks on CoarseHeatmapPanel/FocusSurfacePanel. Replays
troubleshoot_wildtype.ipynb's `plt.plot(-variance); plt.axvline(x=best_minima)`
for whichever tile was last clicked.
"""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QVBoxLayout, QWidget


class TileVarianceInspectorPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMaximumHeight(240)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Wide/short aspect -- this is a single line plot, not a square
        # image panel, so it shouldn't compete for vertical space with the
        # 2x2 diagnostic grid above it.
        self.figure = Figure(figsize=(6, 2), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Select a tile to inspect")
        layout.addWidget(self.canvas)

    def clear(self):
        self.ax.clear()
        self.ax.set_title("Select a tile to inspect")
        self.canvas.draw_idle()

    def show_tile(self, i: int, j: int, variance_stack, focal_index: int):
        variance = variance_stack.variances[i, j]
        self.ax.clear()
        self.ax.plot(-variance)
        self.ax.axvline(x=focal_index, color="red", linestyle="--", label="selected focus")
        self.ax.set_title(f"Tile ({i}, {j})")
        self.ax.set_xlabel("Z index")
        self.ax.set_ylabel("-variance")
        self.ax.legend(loc="upper right", fontsize="small")
        self.canvas.draw_idle()
