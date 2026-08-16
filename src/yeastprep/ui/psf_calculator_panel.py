"""PSF Calculator: a form over `core/psf_calc.py`'s widefield + 2D-sum PSF
computation (design.md 2b), living as a tab inside the Denoise/Deconvolve
page. Deliberately narrower than psfkit's own `PSFDialog` (widefield only,
no confocal/spinning-disk/Zernike, no pyvistra output-selector coupling) --
see core/psf_calc.py's module docstring for why.
"""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tileclass.contrast import compute_percentile_clim
from yeastprep.core.psf_calc import PSFCalcParams, compute_2d_psf, save_psf


class PSFCalculatorPanel(QWidget):
    """Emits `psf_saved(str)` with the saved PSF's path, so EnhancePage can
    offer to route it straight into the deconvolve PSF file picker."""

    psf_saved = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._psf = None
        self._spacing = None

        outer = QHBoxLayout(self)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._build_optics_group())
        left_layout.addWidget(self._build_grid_group())
        left_layout.addLayout(self._build_actions_row())
        left_layout.addStretch(1)
        left.setMinimumWidth(300)
        left.setMaximumWidth(380)
        outer.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(3, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("PSF preview (sum-projected)")
        self.ax.axis("off")
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, 1)

        self.status_label = QLabel("")
        right_layout.addWidget(self.status_label)

        outer.addWidget(right, 1)

    # ------------------------------------------------------------------

    def _build_optics_group(self) -> QGroupBox:
        group = QGroupBox("Optics")
        form = QFormLayout(group)
        defaults = PSFCalcParams()

        self.wavelength = QDoubleSpinBox()
        self.wavelength.setRange(0.3, 1.0)
        self.wavelength.setDecimals(4)
        self.wavelength.setSingleStep(0.005)
        self.wavelength.setValue(defaults.wavelength_um)
        form.addRow("Wavelength (um)", self.wavelength)

        self.na = QDoubleSpinBox()
        self.na.setRange(0.1, 1.7)
        self.na.setDecimals(3)
        self.na.setSingleStep(0.05)
        self.na.setValue(defaults.na)
        form.addRow("NA", self.na)

        self.ni = QDoubleSpinBox()
        self.ni.setRange(1.0, 1.7)
        self.ni.setDecimals(4)
        self.ni.setSingleStep(0.001)
        self.ni.setValue(defaults.ni)
        form.addRow("Immersion RI (ni)", self.ni)

        self.ns_auto = QCheckBox("Same as ni")
        self.ns_auto.setChecked(True)
        self.ns_auto.stateChanged.connect(lambda _: self.ns.setEnabled(not self.ns_auto.isChecked()))
        form.addRow(self.ns_auto)

        self.ns = QDoubleSpinBox()
        self.ns.setRange(1.0, 1.7)
        self.ns.setDecimals(4)
        self.ns.setSingleStep(0.001)
        self.ns.setValue(defaults.ni)
        self.ns.setEnabled(False)
        form.addRow("Sample RI (ns)", self.ns)

        return group

    def _build_grid_group(self) -> QGroupBox:
        group = QGroupBox("Grid")
        form = QFormLayout(group)
        defaults = PSFCalcParams()

        self.dxy_auto = QCheckBox("Nyquist-limited")
        self.dxy_auto.setChecked(True)
        self.dxy_auto.stateChanged.connect(lambda _: self.dxy.setEnabled(not self.dxy_auto.isChecked()))
        form.addRow(self.dxy_auto)

        self.dxy = QDoubleSpinBox()
        self.dxy.setRange(0.01, 1.0)
        self.dxy.setDecimals(4)
        self.dxy.setSingleStep(0.005)
        self.dxy.setEnabled(False)
        form.addRow("Lateral spacing (um)", self.dxy)

        self.nxy = QSpinBox()
        self.nxy.setRange(16, 1024)
        self.nxy.setValue(defaults.nxy)
        form.addRow("Lateral size (px)", self.nxy)

        self.dz = QDoubleSpinBox()
        self.dz.setRange(0.001, 2.0)
        self.dz.setDecimals(4)
        self.dz.setSingleStep(0.01)
        self.dz.setValue(defaults.dz_um)
        form.addRow("Axial spacing (um)", self.dz)

        self.axial_range = QDoubleSpinBox()
        self.axial_range.setRange(0.1, 200.0)
        self.axial_range.setDecimals(2)
        self.axial_range.setValue(defaults.axial_range_um)
        form.addRow("Axial range (um)", self.axial_range)

        self.axial_oversample = QSpinBox()
        self.axial_oversample.setRange(1, 10)
        self.axial_oversample.setValue(defaults.axial_oversample)
        form.addRow("Axial oversample (x Nyquist)", self.axial_oversample)

        self.pixel_integrated = QCheckBox("Pixel-integrated (exact camera-pixel intensity)")
        self.pixel_integrated.setChecked(defaults.pixel_integrated)
        form.addRow(self.pixel_integrated)

        return group

    def _build_actions_row(self) -> QVBoxLayout:
        column = QVBoxLayout()

        self.compute_btn = QPushButton("Compute")
        self.compute_btn.clicked.connect(self._compute)
        column.addWidget(self.compute_btn)

        self.save_btn = QPushButton("Save PSF As...")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)
        column.addWidget(self.save_btn)

        return column

    # ------------------------------------------------------------------

    def _params(self) -> PSFCalcParams:
        return PSFCalcParams(
            wavelength_um=self.wavelength.value(),
            na=self.na.value(),
            ni=self.ni.value(),
            ns=None if self.ns_auto.isChecked() else self.ns.value(),
            dxy_um=None if self.dxy_auto.isChecked() else self.dxy.value(),
            nxy=self.nxy.value(),
            dz_um=self.dz.value(),
            axial_range_um=self.axial_range.value(),
            axial_oversample=self.axial_oversample.value(),
            pixel_integrated=self.pixel_integrated.isChecked(),
        )

    def _compute(self):
        self.status_label.setText("Computing...")
        self.compute_btn.setEnabled(False)
        self.repaint()
        try:
            psf, spacing = compute_2d_psf(self._params())
            self._psf = psf
            self._spacing = spacing
            self._show_preview(psf[0])
            self.save_btn.setEnabled(True)
            self.status_label.setText(f"Done. Shape {psf.shape}, spacing {spacing} um.")
        except Exception as exc:
            self.status_label.setText(f"Error: {exc}")
        finally:
            self.compute_btn.setEnabled(True)

    def _show_preview(self, psf2d: np.ndarray):
        self.ax.clear()
        vmin, vmax = compute_percentile_clim(psf2d)
        self.ax.imshow(psf2d, cmap="gray", vmin=vmin, vmax=vmax)
        self.ax.set_title("PSF preview (sum-projected)")
        self.ax.axis("off")
        self.canvas.draw_idle()

    def _save(self):
        if self._psf is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Save PSF As", "", "TIFF (*.tif *.tiff)"
        )
        if not path:
            return
        try:
            save_psf(path, self._psf, self._spacing)
        except Exception as exc:
            QMessageBox.warning(self, "yeastprep", f"Failed to save PSF: {exc}")
            return
        self.status_label.setText(f"Saved: {path}")
        self.psf_saved.emit(path)
