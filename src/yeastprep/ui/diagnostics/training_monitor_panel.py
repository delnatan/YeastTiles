"""Loss/mu_mse-vs-epoch plot + run log for the Train Denoiser page. A
minimal matplotlib-only equivalent of jssl_denoise's own (pyvistra-plugin-
only) `ConvergencePlotWidget`, since that widget is designed for pyvistra's
viewer chrome and isn't usable standalone here (see worker.TrainingWorker's
docstring for why jssl_denoise's own plugin code isn't importable at all)."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import QPlainTextEdit, QSplitter, QVBoxLayout, QWidget
from qtpy.QtCore import Qt


class TrainingMonitorPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._epochs: list[int] = []
        self._losses: list[float] = []
        self._mu_mses: list[float] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self.figure = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax_right = self.ax.twinx()
        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel("loss")
        self.ax_right.set_ylabel("mu_mse")

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.log_view)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    def clear(self):
        self._epochs = []
        self._losses = []
        self._mu_mses = []
        self.ax.clear()
        self.ax_right.clear()
        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel("loss")
        self.ax_right.set_ylabel("mu_mse")
        self.canvas.draw_idle()
        self.log_view.clear()

    def append_epoch(self, epoch: int, loss: float, mu_mse: float | None = None):
        self._epochs.append(epoch)
        self._losses.append(loss)
        if mu_mse is not None:
            self._mu_mses.append(mu_mse)
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        self.ax_right.clear()
        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel("loss")
        self.ax_right.set_ylabel("mu_mse")
        (loss_line,) = self.ax.plot(
            self._epochs, self._losses, color="tab:blue", label="loss"
        )
        lines = [loss_line]
        if self._mu_mses:
            (mu_mse_line,) = self.ax_right.plot(
                self._epochs[: len(self._mu_mses)],
                self._mu_mses,
                color="tab:orange",
                label="mu_mse",
            )
            lines.append(mu_mse_line)
        # One shared legend for both y-axes -- ax.legend() alone would only
        # list "loss", since mu_mse's Line2D lives on ax_right, not ax.
        self.ax.legend(lines, [line.get_label() for line in lines], loc="upper right")
        self.canvas.draw_idle()

    def log(self, text: str):
        self.log_view.appendPlainText(text)
