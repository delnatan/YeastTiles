"""Accelerated maximum-likelihood restoration by nonlinear conjugate gradients.

Vendored (simplified) from resolvde/nlcg.py -- itself a direct port of
``deconlib.deconvolution.nlcg`` -- Schaefer, Schuster & Herz (2001),
"Generalized approach for accelerated maximum likelihood based image
restoration" (J. Microsc. 204:99-107). There is no autodiff anywhere in
this module (closed-form gradient/Hessian/step-length), so the port is
mechanical; the whole solve runs under ``torch.inference_mode()``.

Simplified from the source relative to two things resolvde also supports:

- **No regularizer.** resolvde's version accepts an arbitrary linear
  operator ``C`` and a ``beta ||C x||^2`` penalty (with a whole
  curvature-normalization scheme to make ``beta`` portable across
  datasets). This package's use case is a single well-conditioned 2D
  deconvolution with a known PSF, where the discrepancy principle
  (``slack``, below) already gives a well-defined, parameter-light
  stopping rule -- so that machinery, and the extra convolutions it costs
  per iteration, is dropped entirely. If a project using this vendored
  copy later needs a smoothness prior, pull ``nlcg.py`` and
  ``regularizers.py`` back in from resolvde rather than growing this file
  again.
- **No jointly-fit background.** resolvde can refine a low-frequency
  background field alongside the object (``background_order``). Only the
  plain scalar ``background`` is kept here.

Objective (Eq. 10)::

    phi(s) = sum(Kf) - sum(g * ln(Kf + b)),   f = s^2

Positivity is implicit via ``f = s^2`` -- there is no clamping anywhere in
the solve.

Gradient (Eq. 12)::

    grad phi = 2 s * K^T(1 - g/(Kf+b))

Hessian quadratic form (Eq. 13), via the 3-convolution-per-iteration trick
(``aux = (m, KTw)`` reused across objective/gradient/Hessian/step-length;
``m = Kf + b``)::

    <d, A(s) d> = 4 sum (g/m^2) (K(s*d))^2 + 2 sum K^T(1-g/m) d^2

Step length: because ``f(lambda) = (s + lambda*d)^2`` and K is linear,
``K f(lambda)`` is *exactly* quadratic in lambda -- not just locally, so
Newton's method on its derivative converges to the true 1-D minimizer with
no backtracking and no extra convolutions per refinement
(:func:`nlcg_step_length`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from . import padding
from .protocol import LinearOperator

__all__ = [
    "NLCGResult",
    "nlcg_objective",
    "nlcg_gradient",
    "nlcg_hessian_quadform",
    "nlcg_step_length",
    "i_divergence_between",
    "nlcg_with_operator",
    "nlcg_solver",
]


def _poisson_i_divergence(observed: Tensor, model: Tensor, eps: float = 1e-6) -> float:
    """Mean Poisson I-divergence -- a monitoring metric, not the objective."""
    observed_safe = torch.clamp(observed, min=eps)
    model_safe = torch.clamp(model, min=eps)
    div = torch.mean(observed * torch.log(observed_safe / model_safe) - (observed - model_safe))
    return float(div.item())


@dataclass
class NLCGResult:
    """Result from nonlinear conjugate-gradient deconvolution.

    Attributes:
        restored: Deconvolved image (``f = s^2``).
        pred: Forward-predicted data, ``blur_op.forward(restored) +
            background``.
        iterations: Number of iterations performed.
        loss_history: Mean data-vs-model Poisson I-divergence at each
            ``eval_interval``.
        converged: Whether a convergence test was met (discrepancy
            principle or Eq. 17), as opposed to exhausting ``num_iter``.
        background: Background level used (in the original data units).
        data_scale: Amplitude the data was divided by before solving; see
            ``normalize`` in :func:`nlcg_with_operator`.
        full_shape: Shape of the internal reconstruction before any output
            crop.
    """

    restored: Tensor
    pred: Tensor
    iterations: int
    loss_history: list[float]
    converged: bool = False
    background: float = 0.0
    data_scale: float = 1.0
    full_shape: Optional[tuple[int, ...]] = None


def _forward_model(s: Tensor, blur_op: LinearOperator, background: float) -> tuple[Tensor, Tensor]:
    """Reconstruction ``f = s^2`` and model ``m = K f + b`` (one forward conv)."""
    f = s * s
    m = blur_op.forward(f) + background
    return f, m


def _objective_from_m(m: Tensor, observed: Tensor, eps: float) -> Tensor:
    """Objective phi from a precomputed model ``m`` (no forward convolution)."""
    return torch.sum(m) - torch.sum(observed * torch.log(torch.clamp(m, min=eps)))


def _gradient_from_m(
    s: Tensor,
    m: Tensor,
    observed: Tensor,
    blur_op: LinearOperator,
    eps: float,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Gradient from a precomputed model ``m`` (one adjoint conv).

    ``aux = (m, KTw)``, reused by the objective/Hessian/step-length at this
    same point to keep the iteration at three convolutions.
    """
    w = 1.0 - observed / torch.clamp(m, min=eps)
    KTw = blur_op.adjoint(w)
    grad = 2.0 * s * KTw
    return grad, (m, KTw)


def nlcg_objective(
    s: Tensor,
    blur_op: LinearOperator,
    observed: Tensor,
    background: float = 0.0,
    eps: float = 1e-10,
) -> float:
    """Restoration functional phi(s) (Eq. 10), ``f = s^2``."""
    _, m = _forward_model(s, blur_op, background)
    return float(_objective_from_m(m, observed, eps).item())


def nlcg_gradient(
    s: Tensor,
    blur_op: LinearOperator,
    observed: Tensor,
    background: float = 0.0,
    eps: float = 1e-10,
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Gradient of phi at s (Eq. 12).

    Returns ``(grad, aux)`` where ``aux = (m, KTw)`` holds intermediates
    reused by the objective/Hessian/step-length to keep the iteration at
    three convolutions.
    """
    _, m = _forward_model(s, blur_op, background)
    return _gradient_from_m(s, m, observed, blur_op, eps)


def nlcg_hessian_quadform(
    s: Tensor,
    d: Tensor,
    blur_op: LinearOperator,
    observed: Tensor,
    background: float = 0.0,
    aux: Optional[tuple[Tensor, Tensor]] = None,
    eps: float = 1e-10,
) -> float:
    """Hessian quadratic form ``<d, A(s) d>`` (Eq. 13).

    Pass ``aux`` from :func:`nlcg_gradient` to reuse ``m``/``KTw`` (one
    extra forward convolution total); omit it to recompute standalone.
    """
    if aux is None:
        _, aux = nlcg_gradient(s, blur_op, observed, background, eps)
    m, KTw = aux
    sd = s * d
    Ksd = blur_op.forward(sd)
    quad = 4.0 * torch.sum((observed / torch.clamp(m * m, min=eps)) * (Ksd * Ksd)) + 2.0 * torch.sum(
        KTw * (d * d)
    )
    return float(quad.item())


def nlcg_step_length(
    s: Tensor,
    d: Tensor,
    f: Tensor,
    m: Tensor,
    blur_op: LinearOperator,
    observed: Tensor,
    background: float | Tensor = 0.0,
    newton_iters: int = 3,
    eps: float = 1e-10,
) -> tuple[Tensor, Tensor]:
    """Exact step size lambda minimizing phi(s + lambda*d) along direction d.

    Because ``f(lambda) = (s + lambda*d)^2`` and K is linear,

        ``K f(lambda) = Kss + 2*lambda*K(s*d) + lambda^2*K(d*d)``

    is *exactly* quadratic in lambda -- not just locally, so Newton's
    method on its derivative converges to the true 1-D minimizer with no
    additional convolutions per refinement and no backtracking.

    ``K(s*d)`` and ``K(d*d)`` are fused into a single batched ``forward``
    call rather than two separate convolutions.

    Returns:
        ``(lam, m_lam)`` -- the step size (0-dim tensor) and the resulting
        model ``blur_op.forward((s + lam*d)**2) + background``, computed
        algebraically (no extra forward convolution).
    """
    g = observed
    b = background

    Kss = m - b
    sd = s * d
    dd = d * d

    Kbatched = blur_op.forward(torch.stack([sd, dd], dim=0))
    Ksd, Kdd = Kbatched[0], Kbatched[1]

    lam = torch.zeros((), dtype=s.dtype, device=s.device)
    for _ in range(newton_iters):
        m_lam_raw = Kss + (2.0 * lam) * Ksd + (lam * lam) * Kdd + b
        m_lam = torch.clamp(m_lam_raw, min=eps)
        Ksd_lam = Ksd + lam * Kdd
        w_lam = 1.0 - g / m_lam
        dphi1 = 2.0 * torch.sum(Ksd_lam * w_lam)
        dphi2 = 2.0 * torch.sum(Kdd * w_lam) + 4.0 * torch.sum(
            g * (Ksd_lam * Ksd_lam) / (m_lam * m_lam)
        )

        lam_candidate = lam - dphi1 / dphi2
        ok = torch.isfinite(dphi2) & (dphi2.abs() >= eps) & torch.isfinite(lam_candidate)
        if not bool(ok.item()):
            break
        lam = lam_candidate

    m_lam = Kss + (2.0 * lam) * Ksd + (lam * lam) * Kdd + b
    return lam, m_lam


def i_divergence_between(f_prev: Tensor, f_curr: Tensor, eps: float = 1e-10) -> float:
    """Csiszar I-divergence between successive iterates (Eq. 17).

    ``I(f_k, f_{k+1}) = sum[ f_k ln(f_k / f_{k+1}) - f_k + f_{k+1} ]``,
    componential. Used as a convergence measure: the distance between
    successive estimates, independent of the restoration functional.
    """
    fp = torch.clamp(f_prev, min=eps)
    fc = torch.clamp(f_curr, min=eps)
    div = torch.sum(f_prev * (torch.log(fp) - torch.log(fc)) - f_prev + f_curr)
    return float(div.item())


@torch.inference_mode()
def nlcg_with_operator(
    observed: Tensor,
    blur_op: LinearOperator,
    num_iter: int = 50,
    background: float = 0.0,
    normalize: bool = True,
    data_scale: Optional[float] = None,
    init: Optional[Tensor] = None,
    callback: Optional[Callable[[int, Tensor, float], Optional[bool]]] = None,
    eval_interval: int = 10,
    slack: float = 1.0,
    tol: float = 1e-4,
    min_iter: int = 10,
    restart_interval: Optional[int] = None,
    newton_iters: int = 3,
    verbose: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> NLCGResult:
    """Accelerated Poisson-ML deconvolution with a pre-built positive operator.

    Nonlinear conjugate gradients (Fletcher-Reeves) with the
    Poisson-likelihood gradient of Schaefer et al. (2001) and the exact
    step-length solve in place of a backtracking line search (see
    :func:`nlcg_step_length`). Positivity is implicit via ``f = s^2``.

    Always returns the full reconstruction domain (``NLCGResult.restored``).
    Callers who padded the domain for wrap-free convolution should crop
    the result themselves (or use :func:`nlcg_solver`, which does).

    Convergence: stop via Morozov's discrepancy principle once the mean
    per-pixel data-model I-divergence reaches ``0.5 * slack`` (``slack``
    interprets ``Var[data] = mean`` -- adjust it, e.g. via
    :meth:`.photon_calibration.SolveUnits.discrepancy_scale`, if solving in
    units where that doesn't hold). Eq. 17 -- the relative (mass-normalized,
    5-iteration averaged) I-divergence between successive iterates dropping
    below ``tol`` -- runs alongside as a fallback, in case the discrepancy
    target is unreachable (``slack=0`` disables the discrepancy test and
    leaves Eq. 17 as the sole stopping rule).

    Args:
        observed: Observed detector image.
        blur_op: Positive forward operator with ``forward``/``adjoint``.
        num_iter: Maximum number of iterations.
        background: Constant background level, in the same units as
            ``observed``.
        normalize / data_scale: Amplitude normalization dividing the data
            (and the reconstruction, correspondingly) by
            ``c = max(observed)`` before solving, so absolute intensity
            scale doesn't perturb the solve's numerics. ``data_scale``
            overrides the auto ``max(observed)`` scale. Purely a numerics
            choice here (no regularizer weight depends on it, unlike
            resolvde's version) -- leave at the default unless reproducing
            a specific raw-units behavior.
        init: Optional initial estimate (of f) on the full reconstruction
            domain, original data units. Defaults to an edge-tapered
            extension of ``observed`` itself when this solver is reached
            via :func:`nlcg_solver` / the package's top-level
            ``deconvolve()``; a plain adjoint-based init when called
            directly with no ``init``.
        callback: Optional ``(iter, f, loss) -> Optional[bool]``; truthy
            stops early. ``loss`` (mean Poisson I-divergence) is the
            current iteration's value, already computed regardless of
            ``eval_interval``.
        eval_interval: Interval for logging mean data-vs-model
            I-divergence.
        slack: Multiplier on the discrepancy principle's target of 0.5.
            ``0`` disables it, falling through to Eq. 17 alone.
        tol: Eq. 17 threshold, a fallback stopping test alongside the
            discrepancy principle. ``0`` disables it.
        min_iter: Minimum iterations before any convergence test can
            trigger.
        restart_interval: If set, force a steepest-descent restart every
            this many iterations.
        newton_iters: Newton-Raphson refinements for the step length;
            3 (the default) matches the original COSM implementation.
        verbose: Print progress.
        log: Optional ``(str) -> None`` sink for the ``verbose`` progress
            lines, used instead of ``print`` when given.
    """
    eps = 1e-10

    # Normalize the data amplitude so the solve's numerics don't depend on
    # the data's raw photon-count scale.
    if data_scale is not None:
        c = float(data_scale)
    elif normalize:
        c = float(torch.max(observed).item())
    else:
        c = 1.0
    if not math.isfinite(c) or c <= 0.0:
        c = 1.0

    g = observed / c if c != 1.0 else observed
    if isinstance(background, Tensor):
        b = background / c if c != 1.0 else background
    else:
        b = float(background) / c

    # Optimize over s with f = s^2; a zero (or exactly-zero-floored) init
    # is a fixed point here (grad phi is proportional to s), so floor away
    # from 0.
    if init is None:
        f0 = torch.clamp(blur_op.adjoint(torch.clamp(g - b, min=1e-6)), min=1e-6)
    else:
        init_norm = init / c if c != 1.0 else init
        f0 = torch.clamp(init_norm, min=1e-6)
    s = torch.sqrt(f0)

    f, m = _forward_model(s, blur_op, b)

    grad, aux = _gradient_from_m(s, m, g, blur_op, eps)
    r = -grad
    d = r
    rr = torch.sum(r * r)

    loss_history: list[float] = []
    idiv_window: list[float] = []
    converged = False
    k = 0

    def _step(direction: Tensor, phi0: Tensor):
        """One Newton-exact-step-length trial along ``direction``.

        Returns ``(s_new, f_new, m_new, lam)`` on success, or ``None`` if
        the step is degenerate or fails to decrease phi.
        """
        lam, m_lam = nlcg_step_length(
            s, direction, f, aux[0], blur_op, g, background=b, newton_iters=newton_iters, eps=eps,
        )
        if not bool(torch.isfinite(lam).item()):
            return None
        s_new = s + lam * direction
        f_new = s_new * s_new
        phi_new = _objective_from_m(m_lam, g, eps)
        if bool((phi_new >= phi0).item()):
            return None
        return s_new, f_new, m_lam, lam

    for k in range(num_iter):
        phi0 = _objective_from_m(aux[0], g, eps)

        # Exact step size; fall back to steepest descent if the CG
        # direction fails to decrease phi.
        step = _step(d, phi0)
        if step is None and d is not r:
            d = r
            step = _step(d, phi0)
        if step is None:
            converged = True
            loss_history.append(c * _poisson_i_divergence(g, aux[0]))
            break

        f_prev = f
        s, f, m, lam = step
        grad, aux = _gradient_from_m(s, m, g, blur_op, eps)

        r_new = -grad
        rr_new = torch.sum(r_new * r_new)

        gamma = torch.where(rr > eps, rr_new / rr, torch.zeros_like(rr))
        if restart_interval is not None and (k + 1) % restart_interval == 0:
            gamma = torch.zeros_like(gamma)
        d = r_new + gamma * d
        r = r_new
        rr = rr_new

        # Rescale back to original-count units so loss/slack/verbose output
        # mean the same thing regardless of normalize -- _poisson_i_divergence
        # is a per-pixel mean, and both g/m are already divided by c, so the
        # raw mean comes back divided by c too; undo that here.
        loss = c * _poisson_i_divergence(g, m)
        logged = k % eval_interval == 0 or k == num_iter - 1
        if logged:
            loss_history.append(loss)
            if verbose:
                (log or print)(
                    f"  Iter {k:4d}: mean I-div = {loss:.6g}, "
                    f"lambda = {float(lam.item()):.4g}, gamma = {float(gamma.item()):.4g}"
                )

        # Eq. 17: relative (mass-normalized) I-divergence between
        # successive iterates, averaged over the last 5. Fallback stopping
        # test alongside the discrepancy principle.
        rel = i_divergence_between(f_prev, f) / max(float(torch.sum(f).item()), eps)
        idiv_window.append(rel)
        if len(idiv_window) > 5:
            idiv_window.pop(0)
        rel_avg = sum(idiv_window) / len(idiv_window)

        stop = False
        stop_reason = None
        if callback is not None:
            stop = bool(callback(k, f * c if c != 1.0 else f, loss))

        ready = not stop and k + 1 >= min_iter
        if ready and slack > 0.0 and loss <= 0.5 * slack:
            converged = True
            stop = True
            stop_reason = "discrepancy principle"
        elif ready and tol > 0.0 and len(idiv_window) == 5 and rel_avg < tol:
            converged = True
            stop = True
            stop_reason = "relative divergence change (Eq. 17)"

        if stop:
            if not logged:  # record the true loss at the stopping iteration
                loss_history.append(loss)
            if verbose and stop_reason is not None:
                (log or print)(f"  Stopped at iter {k}: {stop_reason}")
            break

    # Back to original data units: restored = c * s^2,
    # pred = c * (K s^2 + b_norm) = K(restored) + background.
    x = (s * s) * c
    full_shape = tuple(x.shape)
    pred = blur_op.forward(x) + (float(background) if not isinstance(background, Tensor) else background)
    background_scalar = float(background) if not isinstance(background, Tensor) else float(background.mean().item())

    return NLCGResult(
        restored=x,
        pred=pred,
        iterations=k + 1,
        loss_history=loss_history,
        converged=converged,
        background=background_scalar,
        data_scale=c,
        full_shape=full_shape,
    )


def nlcg_solver(
    num_iter: int = 50,
    background: float = 0.0,
    init_value: Optional[float] = None,
    **nlcg_kwargs,
) -> Callable[[Tensor, "ForwardModel"], Tensor]:  # noqa: F821 (ForwardModel: model.py, no import cycle)
    """Adapt the NLCG solver to the ``solve(data, model)`` contract.

    Returns a callable that deconvolves one image against a
    :class:`.model.ForwardModel` and returns the visible-space result.

    Args:
        num_iter: Maximum iterations.
        background: Constant background in data-space counts.
        init_value: Optional flat initial estimate on the padded
            reconstruction domain. Defaults to an edge-tapered extension
            of ``data`` itself (see :func:`.padding.default_init`).
        **nlcg_kwargs: Extra keyword arguments forwarded to
            :func:`nlcg_with_operator` (e.g. ``tol``, ``min_iter``,
            ``eval_interval``, ``normalize``, ``data_scale``).
    """

    def solve(data: Tensor, model) -> Tensor:
        if init_value is not None:
            init = torch.full(model.padded_shape, float(init_value), dtype=data.dtype, device=data.device)
        else:
            init = padding.default_init(data, model, background=background)
        result = nlcg_with_operator(
            observed=data,
            blur_op=model.op,
            num_iter=num_iter,
            background=background,
            init=init,
            **nlcg_kwargs,
        )
        return result.restored[model.valid_slices]

    return solve
