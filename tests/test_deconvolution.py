"""Smoke tests for the vendored yeastprep.core.deconvolution submodule.

Trimmed analogues of resolvde's own test_nlcg.py/test_photon_calibration.py:
finite-difference checks of the closed-form gradient/Hessian against the
objective, end-to-end recovery on small synthetic problems, exact
step-length optimality, and a photon-calibration round trip. No
regularizer/background-order cases here -- this vendored copy dropped both.
"""

from __future__ import annotations

import torch

from yeastprep.core.deconvolution import (
    NLCGResult,
    PhotonCalibration,
    SolveUnits,
    deconvolve,
    make_forward_model,
)
from yeastprep.core.deconvolution.nlcg import (
    nlcg_gradient,
    nlcg_hessian_quadform,
    nlcg_objective,
    nlcg_solver,
    nlcg_step_length,
    nlcg_with_operator,
)
from yeastprep.core.deconvolution.operators import BlurFFT
from yeastprep.core.deconvolution.photon_calibration import estimate_photon_calibration


def test_nlcg_gradient_matches_finite_difference():
    torch.manual_seed(0)
    shape = (12, 14)
    psf = torch.rand(5, 5, dtype=torch.float32) + 0.2
    blur_op = BlurFFT.build(psf, shape)
    s = torch.rand(*shape, dtype=torch.float32) + 0.5
    observed = torch.rand(*shape, dtype=torch.float32) * 10 + 1.0
    background = 0.5

    grad, _ = nlcg_gradient(s, blur_op, observed, background=background)
    d = torch.randn(*shape, dtype=torch.float32)

    h = 1e-3
    phi_plus = nlcg_objective(s + h * d, blur_op, observed, background=background)
    phi_minus = nlcg_objective(s - h * d, blur_op, observed, background=background)
    fd = (phi_plus - phi_minus) / (2 * h)
    analytic = float(torch.sum(grad * d).item())

    assert abs(fd - analytic) < 5e-2 * (abs(analytic) + 1e-6)


def test_nlcg_hessian_quadform_matches_finite_difference():
    torch.manual_seed(0)
    shape = (10, 10)
    psf = torch.rand(5, 5, dtype=torch.float32) + 0.2
    blur_op = BlurFFT.build(psf, shape)
    s = torch.rand(*shape, dtype=torch.float32) + 0.5
    observed = torch.rand(*shape, dtype=torch.float32) * 10 + 1.0
    background = 0.5

    _, aux = nlcg_gradient(s, blur_op, observed, background=background)
    d = torch.randn(*shape, dtype=torch.float32)
    quad = nlcg_hessian_quadform(s, d, blur_op, observed, background=background, aux=aux)

    h = 5e-3
    phi0 = nlcg_objective(s, blur_op, observed, background=background)
    phi_plus = nlcg_objective(s + h * d, blur_op, observed, background=background)
    phi_minus = nlcg_objective(s - h * d, blur_op, observed, background=background)
    fd_second = (phi_plus - 2 * phi0 + phi_minus) / (h * h)

    assert abs(fd_second - quad) < 5e-2 * (abs(quad) + 1e-6)


def test_nlcg_step_length_decreases_objective_more_than_neighbors():
    """The exact step length should (near-)minimize phi along d: nearby
    step sizes must not do better."""
    torch.manual_seed(0)
    shape = (10, 10)
    psf = torch.rand(5, 5, dtype=torch.float32) + 0.2
    blur_op = BlurFFT.build(psf, shape)
    s = torch.rand(*shape, dtype=torch.float32) + 0.5
    observed = torch.rand(*shape, dtype=torch.float32) * 10 + 1.0
    background = 0.5

    f = s * s
    _, aux = nlcg_gradient(s, blur_op, observed, background=background)
    d = -aux[1]  # steepest-descent direction w.r.t s (up to the 2*s factor)

    lam, _ = nlcg_step_length(s, d, f, aux[0], blur_op, observed, background=background)
    lam_val = float(lam.item())

    def phi_at(lam_scalar: float) -> float:
        return nlcg_objective(s + lam_scalar * d, blur_op, observed, background=background)

    phi_lam = phi_at(lam_val)
    for delta in (-1e-2, 1e-2, -1e-1, 1e-1):
        assert phi_lam <= phi_at(lam_val + delta) + 1e-3


def test_nlcg_recovers_identity_blur_1d():
    """K == identity (near-delta kernel): the ML solution should recover
    the observed data almost exactly."""
    torch.manual_seed(0)
    shape = (64,)
    psf = torch.zeros(3, dtype=torch.float32)
    psf[1] = 1.0  # centered delta
    blur_op = BlurFFT.build(psf, shape)

    truth = torch.rand(*shape, dtype=torch.float32) * 50.0 + 5.0
    observed = truth.clone()

    result = nlcg_with_operator(observed, blur_op, num_iter=100, background=0.0)
    torch.testing.assert_close(result.restored, truth, rtol=1e-2, atol=1e-1)
    assert result.iterations > 0


def test_nlcg_recovers_gaussian_blur_2d():
    """Small 2D Gaussian-blur bead image: reconstruction should correlate
    strongly with ground truth and the objective should not increase."""
    torch.manual_seed(0)
    shape = (32, 32)
    ky, kx = torch.meshgrid(torch.linspace(-1, 1, 7), torch.linspace(-1, 1, 7), indexing="ij")
    psf = torch.exp(-(ky**2 + kx**2) / (2 * 0.3**2)).to(torch.float32)
    blur_op = BlurFFT.build(psf, shape)

    truth = torch.zeros(shape, dtype=torch.float32)
    truth[10, 10] = 100.0
    truth[20, 22] = 150.0
    truth[15, 24] = 80.0
    observed = blur_op.forward(truth) + 2.0  # background

    result = nlcg_with_operator(observed, blur_op, num_iter=100, background=2.0)

    for prev, curr in zip(result.loss_history, result.loss_history[1:]):
        assert curr <= prev + 1e-3

    correlation = torch.corrcoef(torch.stack([result.restored.flatten(), truth.flatten()]))[0, 1]
    assert correlation.item() > 0.9


def test_nlcg_solver_adapter_shape():
    torch.manual_seed(0)
    psf = torch.rand(7, 7, dtype=torch.float32) + 0.2
    model = make_forward_model(psf, (24, 24), zoom=1.0)
    truth = torch.rand(*model.padded_shape, dtype=torch.float32) * 10.0 + 1.0
    data = model.op.forward(truth) + 0.5

    solve = nlcg_solver(num_iter=20, background=0.5)
    restored = solve(data, model)
    assert restored.shape == model.visible_shape


def test_nlcg_result_is_dataclass_with_expected_fields():
    fields = NLCGResult.__dataclass_fields__
    assert set(fields) == {
        "restored",
        "pred",
        "iterations",
        "loss_history",
        "converged",
        "background",
        "data_scale",
        "full_shape",
    }


def test_deconvolve_end_to_end_recovers_gaussian_blur_2d():
    """The package's top-level convenience function: build the forward
    model, edge-taper-init, solve, crop -- all in one call."""
    torch.manual_seed(0)
    shape = (40, 40)
    coords = torch.arange(9, dtype=torch.float32) - 4.0
    g1 = torch.exp(-(coords**2) / (2 * 1.2**2))
    psf = (g1[:, None] * g1[None, :])
    psf = psf / psf.sum()

    truth = torch.zeros(shape, dtype=torch.float32)
    truth[12, 15] = 120.0
    truth[27, 24] = 90.0

    model = make_forward_model(psf, shape, device="cpu")
    padded_truth = torch.zeros(model.padded_shape, dtype=torch.float32)
    padded_truth[model.valid_slices] = truth
    observed = model.op.forward(padded_truth) + 3.0

    result = deconvolve(observed, psf, background=3.0, num_iter=80, device="cpu")
    assert result.restored.shape == shape
    assert torch.all(torch.isfinite(result.restored))

    correlation = torch.corrcoef(torch.stack([result.restored.flatten(), truth.flatten()]))[0, 1]
    assert correlation.item() > 0.9


def test_photon_calibration_round_trip():
    cal = PhotonCalibration(gain=2.5, offset=100.0, read_variance=4.0, offset_was_measured=True)
    photons = torch.tensor([0.0, 10.0, 50.0])
    adu = cal.to_adu(photons)
    torch.testing.assert_close(cal.to_photons(adu), photons)
    assert cal.pedestal > 0.0


def test_estimate_photon_calibration_recovers_known_gain():
    """Synthetic camera model: Poisson-distributed rate scaled by a known
    gain and offset, structured (not flat) so the estimator's
    difference-variance approach is actually exercised."""
    torch.manual_seed(0)
    gain, offset = 3.0, 200.0
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, 256), torch.linspace(0, 1, 256), indexing="ij"
    )
    rate = 5.0 + 60.0 * torch.exp(-((yy - 0.5) ** 2 + (xx - 0.5) ** 2) / 0.05)
    photons = torch.poisson(rate)
    adu = offset + gain * photons

    cal = estimate_photon_calibration(adu)
    assert cal.r_squared > 0.5
    assert abs(cal.gain - gain) / gain < 0.25


def test_solve_units_raw_counts_discrepancy_scale_matches_gain():
    cal = PhotonCalibration(gain=4.0, offset=100.0, offset_was_measured=True)
    photon_units = SolveUnits(calibration=cal)
    raw_units = SolveUnits(calibration=cal, raw_counts=True)

    assert photon_units.discrepancy_scale() == 1.0
    assert raw_units.discrepancy_scale() == 4.0
