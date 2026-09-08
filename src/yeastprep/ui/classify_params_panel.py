"""Training hyperparameter controls for the Classifier Training page's two
tabs, wrapping `tileclass.training.supervised.TrainingParams` and
`tileclass.training.vicreg.VICRegParams` -- same spinbox-per-field pattern
as `train_params_panel.TrainParamsPanel` (denoise's equivalent)."""

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tileclass.training.supervised import TrainingParams
from tileclass.training.vicreg import VICRegParams

from .common.collapsible_section import CollapsibleSection


def _dspin(value, minimum, maximum, step=0.01, decimals=4) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(value)
    return box


def _ispin(value, minimum, maximum) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    return box


class SupervisedTrainParamsPanel(QWidget):
    params_changed = Signal(object)  # TrainingParams

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._build_group())
        layout.addStretch(1)

    def _build_group(self) -> QGroupBox:
        group = QGroupBox("Supervised Training")
        outer = QVBoxLayout(group)
        defaults = TrainingParams()

        description = QLabel(
            "Two stages, run back to back: <b>Probe</b> freezes the backbone "
            "and trains only the classifier head; <b>Finetune</b> then "
            "unfreezes the whole network and trains end-to-end, with the "
            "backbone at a lower learning rate than the head. See the "
            "'Starting point' panel below to choose the backbone's initial "
            "weights -- a VICReg-pretrained backbone is recommended over "
            "generic ImageNet weights whenever one is available."
        )
        description.setWordWrap(True)
        description.setToolTip(
            "The classifier head is always freshly initialized and never "
            "warm-started from a deployed classifier, regardless of the "
            "backbone starting point chosen below: the train/validation "
            "split is recomputed fresh from whatever's currently annotated "
            "every run, so warm-starting the head would let a crop that "
            "trained last run land in this run's validation split and look "
            "artificially accurate."
        )
        outer.addWidget(description)

        # Basic: the knobs worth tweaking on nearly every run. Everything
        # else (learning rates, weight decay, val split, seed) has a
        # sensible default that's rarely worth touching -- tucked behind
        # "Advanced" below so the common case isn't lost among them.
        basic_widget = QWidget()
        basic_form = QFormLayout(basic_widget)
        basic_form.setContentsMargins(0, 0, 0, 0)
        basic_form.setVerticalSpacing(8)

        self.batch_size_spin = _ispin(defaults.batch_size, 1, 1024)
        self.probe_epochs_spin = _ispin(defaults.probe_epochs, 0, 2000)
        self.probe_epochs_spin.setToolTip(
            "Number of epochs for the probe stage, during which the backbone "
            "is frozen (requires_grad=False) and only the classifier head "
            "trains."
        )
        self.finetune_epochs_spin = _ispin(defaults.finetune_epochs, 0, 2000)
        self.finetune_epochs_spin.setToolTip(
            "Number of epochs for the finetune stage, during which the entire "
            "network -- backbone included -- is unfrozen (requires_grad=True) "
            "and trained end-to-end."
        )

        basic_form.addRow("Batch size:", self.batch_size_spin)
        basic_form.addRow("Probe epochs:", self.probe_epochs_spin)
        basic_form.addRow("Finetune epochs:", self.finetune_epochs_spin)
        outer.addWidget(basic_widget)

        advanced_widget = QWidget()
        advanced_form = QFormLayout(advanced_widget)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.setVerticalSpacing(8)

        self.val_frac_spin = _dspin(defaults.val_frac, 0.05, 0.9, 0.05, decimals=2)
        self.seed_spin = _ispin(defaults.seed, 0, 999999)
        self.probe_lr_spin = _dspin(defaults.probe_lr, 1e-6, 1.0, 1e-4, decimals=6)
        self.probe_lr_spin.setToolTip(
            "Learning rate for the classifier head during the probe stage. "
            "The backbone is frozen and unaffected by this value."
        )
        self.finetune_backbone_lr_spin = _dspin(
            defaults.finetune_backbone_lr, 1e-8, 1.0, 1e-6, decimals=8
        )
        self.finetune_backbone_lr_spin.setToolTip(
            "Learning rate applied to the backbone once it's unfrozen in the "
            "finetune stage. Kept low to avoid destroying the VICReg-pretrained "
            "features."
        )
        self.finetune_head_lr_spin = _dspin(
            defaults.finetune_head_lr, 1e-6, 1.0, 1e-5, decimals=6
        )
        self.finetune_head_lr_spin.setToolTip(
            "Learning rate applied to the classifier head during the finetune "
            "stage."
        )
        self.weight_decay_spin = _dspin(defaults.weight_decay, 0.0, 1.0, 1e-4, decimals=6)

        advanced_form.addRow("Validation fraction:", self.val_frac_spin)
        advanced_form.addRow("Seed:", self.seed_spin)
        advanced_form.addRow("Probe LR:", self.probe_lr_spin)
        advanced_form.addRow("Finetune backbone LR:", self.finetune_backbone_lr_spin)
        advanced_form.addRow("Finetune head LR:", self.finetune_head_lr_spin)
        advanced_form.addRow("Weight decay:", self.weight_decay_spin)
        outer.addWidget(CollapsibleSection("Advanced", advanced_widget))

        for box in (
            self.val_frac_spin,
            self.seed_spin,
            self.batch_size_spin,
            self.probe_epochs_spin,
            self.finetune_epochs_spin,
            self.probe_lr_spin,
            self.finetune_backbone_lr_spin,
            self.finetune_head_lr_spin,
            self.weight_decay_spin,
        ):
            box.valueChanged.connect(self._emit_params)

        return group

    def params(self) -> TrainingParams:
        return TrainingParams(
            val_frac=self.val_frac_spin.value(),
            seed=self.seed_spin.value(),
            batch_size=self.batch_size_spin.value(),
            probe_epochs=self.probe_epochs_spin.value(),
            finetune_epochs=self.finetune_epochs_spin.value(),
            probe_lr=self.probe_lr_spin.value(),
            finetune_backbone_lr=self.finetune_backbone_lr_spin.value(),
            finetune_head_lr=self.finetune_head_lr_spin.value(),
            weight_decay=self.weight_decay_spin.value(),
        )

    def _emit_params(self, _value=None):
        self.params_changed.emit(self.params())


class VicregTrainParamsPanel(QWidget):
    params_changed = Signal(object)  # VICRegParams

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._build_group())
        layout.addStretch(1)

    def _build_group(self) -> QGroupBox:
        group = QGroupBox("VICReg Pretraining")
        outer = QVBoxLayout(group)
        defaults = VICRegParams()

        description = QLabel(
            "Self-supervised backbone pretraining: learns an embedding space "
            "directly from tile crops, no annotations required. Always "
            "starts from an ImageNet-pretrained stem; optionally warm-starts "
            "from the deployed backbone (see below) to keep building on what "
            "a prior run already learned. Deploying a backbone from this tab "
            "feeds both <i>future VICReg runs</i> here and Supervised "
            "Training's 'Starting Point' backbone picker on that tab -- "
            "pretrain and deploy a backbone here first if you want "
            "Supervised Training to build on it instead of generic "
            "ImageNet weights."
        )
        description.setWordWrap(True)
        outer.addWidget(description)

        # Basic: run length + the two toggles almost every run cares about.
        # The VICReg loss coefficients, learning rate, and other tuning
        # knobs below have well-tested defaults from the NN_workflow
        # scripts this training was ported from -- tucked behind "Advanced"
        # so they don't crowd out the handful of settings worth changing
        # per run.
        basic_widget = QWidget()
        basic_form = QFormLayout(basic_widget)
        basic_form.setContentsMargins(0, 0, 0, 0)
        basic_form.setVerticalSpacing(8)

        self.epochs_spin = _ispin(defaults.epochs, 1, 2000)
        self.batch_size_spin = _ispin(defaults.batch_size, 1, 1024)
        self.warm_start_cb = QCheckBox("Warm start from deployed backbone")
        self.warm_start_cb.setChecked(defaults.warm_start)
        self.warm_start_cb.setToolTip(
            "Checked (default): initialize from the currently deployed VICReg "
            "backbone if one exists, so this run builds on what it's already "
            "learned. Unchecked: always start cold from an ImageNet-pretrained "
            "stem, discarding the deployed backbone as a starting point (it "
            "stays deployed until this run's result is explicitly promoted)."
        )
        self.balanced_sampling_cb = QCheckBox("Balanced sampling")
        self.balanced_sampling_cb.setChecked(defaults.balanced_sampling)

        basic_form.addRow("Epochs:", self.epochs_spin)
        basic_form.addRow("Batch size:", self.batch_size_spin)
        basic_form.addRow(self.warm_start_cb)
        basic_form.addRow(self.balanced_sampling_cb)
        outer.addWidget(basic_widget)

        advanced_widget = QWidget()
        advanced_form = QFormLayout(advanced_widget)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.setVerticalSpacing(8)

        self.lr_spin = _dspin(defaults.lr, 1e-6, 1.0, 1e-4, decimals=6)
        self.weight_decay_spin = _dspin(defaults.weight_decay, 0.0, 1.0, 1e-4, decimals=6)
        self.proj_dim_spin = _ispin(defaults.proj_dim, 8, 8192)
        self.sim_coeff_spin = _dspin(defaults.sim_coeff, 0.0, 1000.0, 1.0, decimals=2)
        self.std_coeff_spin = _dspin(defaults.std_coeff, 0.0, 1000.0, 1.0, decimals=2)
        self.cov_coeff_spin = _dspin(defaults.cov_coeff, 0.0, 1000.0, 0.1, decimals=2)
        self.num_workers_spin = _ispin(defaults.num_workers, 0, 32)
        self.seed_spin = _ispin(defaults.seed, 0, 999999)

        advanced_form.addRow("Learning rate:", self.lr_spin)
        advanced_form.addRow("Weight decay:", self.weight_decay_spin)
        advanced_form.addRow("Projection dim:", self.proj_dim_spin)
        advanced_form.addRow("Invariance (sim) coeff:", self.sim_coeff_spin)
        advanced_form.addRow("Variance (std) coeff:", self.std_coeff_spin)
        advanced_form.addRow("Covariance coeff:", self.cov_coeff_spin)
        advanced_form.addRow("Dataloader workers:", self.num_workers_spin)
        advanced_form.addRow("Seed:", self.seed_spin)
        outer.addWidget(CollapsibleSection("Advanced", advanced_widget))

        for box in (
            self.epochs_spin,
            self.batch_size_spin,
            self.lr_spin,
            self.weight_decay_spin,
            self.proj_dim_spin,
            self.sim_coeff_spin,
            self.std_coeff_spin,
            self.cov_coeff_spin,
            self.num_workers_spin,
            self.seed_spin,
        ):
            box.valueChanged.connect(self._emit_params)
        self.warm_start_cb.stateChanged.connect(self._emit_params)
        self.balanced_sampling_cb.stateChanged.connect(self._emit_params)

        return group

    def params(self) -> VICRegParams:
        return VICRegParams(
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_size_spin.value(),
            warm_start=self.warm_start_cb.isChecked(),
            balanced_sampling=self.balanced_sampling_cb.isChecked(),
            lr=self.lr_spin.value(),
            weight_decay=self.weight_decay_spin.value(),
            proj_dim=self.proj_dim_spin.value(),
            sim_coeff=self.sim_coeff_spin.value(),
            std_coeff=self.std_coeff_spin.value(),
            cov_coeff=self.cov_coeff_spin.value(),
            num_workers=self.num_workers_spin.value(),
            seed=self.seed_spin.value(),
        )

    def _emit_params(self, _value=None):
        self.params_changed.emit(self.params())
