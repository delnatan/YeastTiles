"""Recover a camera's gain and baseline, and use them without leaving ADU.

Vendored (near-verbatim) from resolvde/photon_calibration.py -- fully
self-contained (only depends on ``torch``), so nothing needed trimming
beyond a few docstring references to resolvde modules this package
doesn't carry (its wider set of Poisson-likelihood solvers).

A camera returns

    y = O + g * n + r,      n ~ Poisson(rate),   Var(r) = R,

so ``Var(y) = g * (E[y] - O) + R``: a deterministic baseline ``O`` carrying no
shot noise, a gain ``g`` inflating the variance by ``g`` relative to the mean,
and a read variance ``R`` that is there at any light level. Feeding raw ADU to
:mod:`.nlcg` (whose Poisson likelihood asserts ``Var = mean``) is wrong on all
three counts -- and wrong in a way that is easy to miss, because a solve on
mis-scaled data still converges, it just weights the discrepancy principle
(``slack``) as if the data were a factor ``g`` noisier or quieter than they
are.

Prefer a measured offset
------------------------
``O`` and ``R`` are properties of the camera, not of the sample, and both fall
out of a dark-frame series in one line each -- see
:func:`dark_frame_statistics`. Supplying them is strictly better than fitting
them, and not only for accuracy: from a *single* image the model above is two
unknowns against one fitted intercept, so ``O`` and ``R`` are not separately
identifiable at all. Fixing ``O`` collapses that degeneracy and makes ``R``
fall out of the same fit as a real, separately reported number.

Everything here therefore takes ``offset`` as an optional argument and says
plainly, via :attr:`PhotonCalibration.offset_was_measured`, which regime
produced a given result.

Working in ADU
--------------
Most users want reconstructions in the data's own units, to lay next to the
raw frame. Convert on the way in, solve in photons where the likelihood is
exact, convert back on the way out -- :meth:`PhotonCalibration.to_photons`,
:meth:`PhotonCalibration.signal_to_adu`. The round trip is exact up to
floating point, so nothing about the displayed units changes; only the
statistical weighting inside the solve, which is the part that was wrong.

``background`` must be converted too, and it splits: the ``O`` part is
deterministic and gets subtracted, while the remainder is real background
fluorescence with its own shot noise and stays in ``background``. That is
:meth:`PhotonCalibration.split_background`.

The estimator
-------------
:func:`estimate_photon_calibration` regresses local variance on local mean --
the classic photon transfer curve, with two departures that matter for a
single image of real structure rather than a flat-field series:

* Variance comes from **nearest-neighbour differences**, not from the variance
  within a block. A block on a gradient or an edge has a large block-variance
  that is entirely structure; differencing cancels any locally-linear
  component.
* Per mean-bin it takes a **low quantile** of those difference variances.
  Structure only ever *adds* variance, so the low quantile tracks the noise
  floor where a median still drifts upward wherever texture is dense. The
  resulting downward bias is a fixed property of the block geometry, not of
  the data, and is divided out by :func:`_quantile_bias`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

__all__ = [
    "PhotonCalibration",
    "SolveUnits",
    "dark_frame_statistics",
    "estimate_photon_calibration",
]


@dataclass(frozen=True)
class PhotonCalibration:
    """A camera's ADU-to-photon conversion.

    Attributes:
        gain: ADU per photon (``g``). ``1.0`` means the data are already
            photon counts.
        offset: Camera baseline in ADU (``O``).

            When it was **measured** (``offset_was_measured``), this is the
            sensor's actual pedestal. When it was **fitted** from one frame it
            is the x-intercept of the variance-vs-mean line, which is
            ``O - R/g`` rather than ``O``: with read variance in play, one
            frame identifies only that combination. It is still the right
            thing to subtract -- doing so makes the converted data exactly
            Poisson, absorbing read noise as a small negative pedestal -- but
            it sits below the nominal baseline by ``R/g``, so do not read it
            as a sensor spec. Measure a dark frame if you need that.
        read_variance: Read noise variance ``R`` in ADU^2, or ``0.0`` when it
            could not be identified (a fitted offset absorbs it instead).
        offset_was_measured: ``True`` when ``offset`` came from the caller
            rather than from the variance fit. Determines which of the two
            readings above applies.
        dark_variance: Measured (not fitted) variance in the dimmest mean-bin,
            in ADU^2. Diagnostic for whether Poisson applies at all: compare
            it to ``gain * signal`` over the range you care about. If it
            dominates, the frame is read-noise limited and no unit conversion
            will make a Poisson likelihood correct.
        r_squared: Goodness of the variance-vs-mean fit, in ``[0, 1]``. Low
            values mean the photon transfer curve did not hold, so treat
            ``gain`` as unreliable rather than as a measurement.
    """

    gain: float
    offset: float
    read_variance: float = 0.0
    offset_was_measured: bool = False
    dark_variance: float = 0.0
    r_squared: float = 1.0

    @property
    def pedestal(self) -> float:
        """Shifted-Poisson ``c = R / g^2``, in photons^2.

        ``Var(n) = E[n] + c`` once read noise is folded into the photon-unit
        model this way (the standard sCMOS variance-stabilization result,
        Huang et al. 2013). :mod:`.nlcg`'s plain Poisson likelihood has no
        pedestal term to feed this into, so it is exposed here purely as a
        diagnostic / for solvers added later that do.

        Falls back to ``1.0`` when the read variance was never identified,
        rather than asserting a noiseless camera.
        """
        if self.read_variance <= 0.0:
            return 1.0
        return self.read_variance / (self.gain * self.gain)

    def to_photons(self, image: Tensor) -> Tensor:
        """``(y - offset) / gain``. Not clamped: negative pixels are real
        (read noise around a dark background), and the solver's likelihood
        clamps where it needs to."""
        return (image - self.offset) / self.gain

    def to_adu(self, photons: Tensor) -> Tensor:
        """Inverse of :meth:`to_photons` -- for a *measured-like* quantity
        that should carry the camera pedestal, such as a forward prediction
        laid over the raw frame."""
        return photons * self.gain + self.offset

    def signal_to_adu(self, photons: Tensor) -> Tensor:
        """Scale a background-free quantity -- a reconstruction ``f``, which
        is an object estimate with no pedestal in it -- back to ADU.

        Distinct from :meth:`to_adu` on purpose: adding ``offset`` to a
        deconvolved object would reintroduce a camera pedestal that was never
        part of the object.
        """
        return photons * self.gain

    def split_background(self, background_adu: float) -> float:
        """Photon-unit background for a measured total background level.

        ``background_adu`` is what you read off a blank region of the raw
        frame -- baseline plus background fluorescence together. Only the
        fluorescent part is Poisson-distributed, so this returns
        ``(background_adu - offset) / gain``, which is what belongs in
        :func:`.nlcg.nlcg_with_operator`'s ``background`` argument once the
        data are in photons.
        """
        return (float(background_adu) - self.offset) / self.gain


def dark_frame_statistics(dark: Tensor) -> tuple[float, float]:
    """``(offset, read_variance)`` from one or more dark frames, in ADU.

    A dark frame has no light on it, so its mean *is* the baseline and its
    variance *is* the read noise -- no fitting involved. Pass the results
    straight to :func:`estimate_photon_calibration` as ``offset`` and
    ``read_variance``.

    Args:
        dark: A dark frame, or a stack of them with frames along the leading
            axis. A stack is treated as one pooled sample, which is the right
            thing for a spatially uniform sensor; for a per-pixel sCMOS map,
            take ``dark.mean(0)`` / ``dark.var(0)`` yourself -- this module's
            calibration takes scalar pedestals, so the pooled scalar is what
            it can use.
    """
    d = dark.to(dtype=torch.float64)
    return float(d.mean()), float(d.var(unbiased=True))


def _blocks(x: Tensor, win: int) -> Tensor:
    n0 = (x.shape[-2] // win) * win
    n1 = (x.shape[-1] // win) * win
    if n0 == 0 or n1 == 0:
        raise ValueError(f"image {tuple(x.shape)} is smaller than block_size={win}")
    b = x[..., :n0, :n1].reshape(n0 // win, win, n1 // win, win)
    return b.permute(0, 2, 1, 3).reshape(-1, win * win)


def _difference_variance(x: Tensor, win: int) -> Tensor:
    """Per-block noise variance from nearest-neighbour differences.

    ``Var(y[i] - y[i+1]) = 2 Var(y)`` for independent noise, and the
    difference cancels any locally-linear structure. Averaging the two axes
    halves the estimator's own variance. Padding with a zero difference
    (rather than wrapping) leaves the last row/column slightly under-weighted,
    which is preferable to inventing a large spurious difference at the edge.
    """
    dh = torch.nn.functional.pad(x[:, 1:] - x[:, :-1], (0, 1))
    dv = torch.nn.functional.pad(x[1:, :] - x[:-1, :], (0, 0, 0, 1))
    return 0.25 * (_blocks(dh, win).pow(2).mean(1) + _blocks(dv, win).pow(2).mean(1))


_BIAS_CACHE: dict[tuple[int, float], float] = {}


def _quantile_bias(win: int, quantile: float, seed: int = 0) -> float:
    """How much the low-quantile noise-floor estimator under-reads, as a
    multiplicative factor on variance.

    Taking a low quantile is what makes the estimator immune to structure, but
    it is not free: even on pure noise a per-block variance estimate has
    sampling scatter, so its 20th percentile sits *below* the true variance.
    The shortfall is a fixed property of the block geometry and the quantile
    -- it does not depend on the data -- so it is measured once on synthetic
    noise of known unit variance and divided out.

    Left uncorrected this is an underestimate of ``gain`` that moves with
    ``block_size`` (measured: 1.67 / 2.08 / 2.35 at win = 4 / 8 / 16 against a
    true gain of 2.5), which would make the fitted gain an artifact of the
    estimator's own tuning rather than a property of the camera.

    The *offset* needs no such correction when it is fitted: a multiplicative
    bias common to every bin scales slope and intercept alike, and the
    x-intercept is their ratio.

    Gaussian noise is used rather than Poisson. The two differ only at very
    low counts, and at counts low enough to matter the shot noise is no longer
    what limits a gain fit anyway.
    """
    key = (win, round(float(quantile), 6))
    if key not in _BIAS_CACHE:
        g = torch.Generator().manual_seed(seed)
        n = win * 64  # 4096 blocks -- the sample quantile has settled by here
        z = torch.randn((n, n), generator=g, dtype=torch.float64)
        _BIAS_CACHE[key] = float(torch.quantile(_difference_variance(z, win), quantile))
    return _BIAS_CACHE[key]


@torch.inference_mode()
def estimate_photon_calibration(
    image: Tensor,
    offset: Optional[float] = None,
    read_variance: Optional[float] = None,
    block_size: int = 8,
    n_bins: int = 12,
    noise_quantile: float = 0.2,
    min_blocks_per_bin: int = 20,
    fit_fraction: float = 1.0,
    gain_bounds: tuple[float, float] = (1e-3, 1e4),
    min_r_squared: float = 0.5,
) -> PhotonCalibration:
    """Fit ``Var(y) = gain * (E[y] - offset) + read_variance`` on one 2D frame.

    Args:
        image: A single 2D frame, in raw ADU, with its structure intact -- do
            **not** background-subtract or denoise it first, since both
            operations destroy the variance this reads.
        offset: Camera baseline in ADU, if you know it. **Supply this when you
            can** -- see the module docstring. Given it, only ``gain`` (and
            ``read_variance``, if not also given) is fitted, the fit is far
            better conditioned, and ``read_variance`` becomes identifiable.
            :func:`dark_frame_statistics` produces it from a dark frame.
        read_variance: Read noise variance in ADU^2, if known. Only usable
            alongside a known ``offset``; with a fitted offset the two are not
            separately identifiable and this is ignored.
        block_size: Side of the square blocks the frame is tiled into. Each
            block yields one ``(mean, variance)`` pair, trading the number of
            points against how well a block's mean represents it.
        n_bins: Quantile bins in mean, along which the noise floor is taken.
        noise_quantile: Quantile of the per-block difference variance taken
            within each bin. Low, because structure only adds variance; its
            bias is corrected by :func:`_quantile_bias`.
        min_blocks_per_bin: Bins with fewer blocks than this are dropped.
        fit_fraction: Fraction of the bins, from the dim end, used in the
            regression. ``1.0`` uses all of them; lower it if the brightest
            bins are structure-dominated enough to bend the curve.
        gain_bounds: Sanity range for the fitted gain.
        min_r_squared: Minimum fit quality to accept. Below it -- or outside
            ``gain_bounds`` -- the result is reported as ``gain = 1.0`` with
            ``r_squared = 0.0``, i.e. "no calibration", rather than silently
            rescaling a whole dataset by a slope fitted to noise. A
            read-noise-dominated frame lands here, correctly.

    Returns:
        :class:`PhotonCalibration`. Check ``r_squared`` before trusting it.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2D frame, got shape {tuple(image.shape)}")
    if read_variance is not None and offset is None:
        raise ValueError(
            "read_variance requires a known offset: from a single frame the two "
            "are not separately identifiable (see the module docstring). Measure "
            "both with dark_frame_statistics(), or pass neither."
        )

    x = image.to(dtype=torch.float64)
    win = int(block_size)
    mean_b = _blocks(x, win).mean(1)
    var_b = _difference_variance(x, win) / _quantile_bias(win, noise_quantile)

    edges = torch.quantile(
        mean_b, torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    )
    xs: list[float] = []
    ys: list[float] = []
    for i in range(n_bins):
        sel = (mean_b >= edges[i]) & (mean_b <= edges[i + 1])
        if int(sel.sum()) < min_blocks_per_bin:
            continue
        xs.append(float(mean_b[sel].median()))
        ys.append(float(torch.quantile(var_b[sel], noise_quantile)))

    if len(xs) < 3:
        raise ValueError(
            f"only {len(xs)} usable mean-bins from a {tuple(image.shape)} frame at "
            f"block_size={block_size}; use a larger frame or a smaller block_size"
        )

    k = max(3, int(round(len(xs) * float(fit_fraction))))
    mx = torch.tensor(xs[:k], dtype=torch.float64)
    my = torch.tensor(ys[:k], dtype=torch.float64)
    dark_variance = ys[0]

    def _refuse() -> PhotonCalibration:
        return PhotonCalibration(
            gain=1.0,
            offset=float(offset) if offset is not None else 0.0,
            read_variance=0.0,
            offset_was_measured=offset is not None,
            dark_variance=dark_variance,
            r_squared=0.0,
        )

    if offset is None:
        # Two free parameters; the intercept absorbs read noise into the
        # x-intercept, which is the identifiable combination O - R/g.
        mxc, myc = mx - mx.mean(), my - my.mean()
        denom = float((mxc * mxc).sum())
        if denom <= 0.0:
            raise ValueError(
                "all mean-bins are identical; the frame has no dynamic range"
            )
        gain = float((mxc * myc).sum() / denom)
        intercept = float(my.mean() - gain * mx.mean())
        fitted_offset = -intercept / gain if gain != 0.0 else 0.0
        fitted_read = 0.0
        ss_tot = float((myc * myc).sum())
    elif read_variance is None:
        # Offset known: regress on (mean - O), so the intercept IS the read
        # variance rather than a degenerate mixture.
        sx = mx - float(offset)
        sxc, myc = sx - sx.mean(), my - my.mean()
        denom = float((sxc * sxc).sum())
        if denom <= 0.0:
            raise ValueError(
                "all mean-bins are identical; the frame has no dynamic range"
            )
        gain = float((sxc * myc).sum() / denom)
        intercept = float(my.mean() - gain * sx.mean())
        fitted_offset = float(offset)
        fitted_read = max(intercept, 0.0)
        ss_tot = float((myc * myc).sum())
    else:
        # Both known: one free parameter, fitted through the origin on
        # (mean - O, var - R). The best-conditioned case there is.
        sx = mx - float(offset)
        sy = my - float(read_variance)
        denom = float((sx * sx).sum())
        if denom <= 0.0:
            raise ValueError(
                "all mean-bins are identical; the frame has no dynamic range"
            )
        gain = float((sx * sy).sum() / denom)
        intercept = float(read_variance)
        fitted_offset = float(offset)
        fitted_read = float(read_variance)
        ss_tot = float((sy * sy).sum())

    lo, hi = gain_bounds
    if not (lo <= gain <= hi):
        return _refuse()

    pred = gain * (mx - fitted_offset) + fitted_read
    ss_res = float(((my - pred) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    if not (r2 >= min_r_squared):
        return _refuse()

    return PhotonCalibration(
        gain=gain,
        offset=fitted_offset,
        read_variance=fitted_read,
        offset_was_measured=offset is not None,
        dark_variance=dark_variance,
        r_squared=max(0.0, min(1.0, r2)),
    )


@dataclass(frozen=True)
class SolveUnits:
    """The unit policy for running a Poisson solve on calibrated data.

    Two unit conventions, selected by ``raw_counts``:

    * **Photon units** (``raw_counts=False``, default): data converts with
      ``(y - offset) / gain``, a reconstruction converts back with ``*
      gain``. The likelihood is exact here.
    * **Raw counts** (``raw_counts=True``): data converts with just ``y -
      offset`` -- no gain division. This is not an approximation of the
      photon-unit solve, it is algebraically the *same* MLE point estimate
      in different units: for an (unshifted) Poisson data term, dividing
      the whole objective by ``gain`` doesn't move its minimizer, and
      dropping a ``gain``-only additive term doesn't either -- see
      :meth:`discrepancy_scale` for the one place ``gain`` still has to
      enter explicitly. The reconstruction then comes back *without* a ``*
      gain`` step -- it's already ``gain`` times the true photon count,
      i.e. on the camera's own ADU-equivalent scale, which is the point of
      this mode: no round-trip conversion, everything (data, background,
      restored) stays nameable in one unit throughout.

    In both modes the **background** *splits*: the deterministic baseline is
    subtracted and only the background fluorescence, which has its own shot
    noise, stays in the forward model.

    ``calibration=None`` makes every method the identity, so a caller can
    hold one of these unconditionally rather than branching at each use.

    Attributes:
        calibration: The camera calibration, or ``None`` for "already in
            photons, or not calibrated".
        background_adu: Total background in the data's own units -- baseline
            plus background fluorescence, i.e. what you read off a blank
            region.
        raw_counts: Solve in offset-subtracted ADU instead of photons (see
            above). Has no effect when ``calibration`` is ``None``.
    """

    calibration: Optional[PhotonCalibration] = None
    background_adu: float = 0.0
    raw_counts: bool = False

    @property
    def active(self) -> bool:
        return self.calibration is not None

    def data(self, image: Tensor) -> Tensor:
        """Measured data into the units the likelihood is true in."""
        if self.calibration is None:
            return image
        if self.raw_counts:
            return image - self.calibration.offset
        return self.calibration.to_photons(image)

    def background(self) -> float:
        """``background`` in whatever units the solve is running in --
        i.e. what to pass as :func:`.nlcg.nlcg_with_operator`'s
        ``background``."""
        if self.calibration is None:
            return float(self.background_adu)
        if self.raw_counts:
            return float(self.background_adu) - self.calibration.offset
        return self.calibration.split_background(self.background_adu)

    def restored(
        self, f: Tensor, add_background: bool = False, zoom: float = 1.0
    ) -> Tensor:
        """A reconstruction back in the data's own units.

        ``add_background`` re-adds the *total* background, for a
        background-free ``restored`` that should display on the raw data's
        scale. It is added after the unit conversion, never before: the
        background is an ADU quantity and the object is not.

        ``zoom`` is the reconstruction grid's voxel count per data pixel --
        the product of the per-axis zoom factors (``zoom**ndim`` for an
        isotropic zoom, ``1.0`` at the default/no-zoom resolution). The
        forward operator's downsample stage *sums* the ``zoom`` fine voxels
        that fall inside one data pixel (flux-conserving, see
        :class:`.operators.Downsample`), so painting the *whole*
        ``background_adu`` onto every one of those voxels would inflate the
        total background flux by a factor of ``zoom`` once summed back down
        to the data grid. Dividing by ``zoom`` first keeps that sum exact;
        at ``zoom == 1`` it is a no-op.
        """
        if self.calibration is None or self.raw_counts:
            # raw_counts: f is already gain*photons, i.e. ADU-equivalent --
            # no further scaling, unlike the photon-units signal_to_adu step.
            out = f
        else:
            out = self.calibration.signal_to_adu(f)
        if not add_background:
            return out
        return out + float(self.background_adu) / float(zoom)

    def discrepancy_scale(self) -> float:
        """Multiplier for :func:`.nlcg.nlcg_with_operator`'s ``slack``
        (its Morozov-discrepancy-principle noise-floor threshold), which is
        calibrated assuming ``Var[data] = mean``.

        That holds in photon units (returns ``1.0``) but not in
        ``raw_counts`` mode, where ``Var[data] = gain*mean + read_var`` --
        the expected per-pixel Poisson I-divergence at the true solution is
        ``gain/2``, not ``0.5``, so a threshold tuned in photon units needs
        multiplying by ``gain`` here to mean the same physical stopping
        point. Left uncorrected the solver chases the data down to a
        ``gain``-times-tighter fit than the noise supports -- measured
        overfitting (rel-RMSE 1.43 unscaled vs. 0.22 scaled on a real test
        scene), not a cosmetic difference.
        """
        if self.calibration is None or not self.raw_counts:
            return 1.0
        return float(self.calibration.gain)

    def describe(self) -> str:
        """One line for a solver log, or ``""`` when inactive."""
        cal = self.calibration
        if cal is None:
            return ""
        if self.raw_counts:
            tail = f"; discrepancy thresholds scaled by gain={cal.gain:.5g}"
            return (
                f"photon calibration: gain={cal.gain:.5g} ADU/photon, "
                f"offset={cal.offset:.6g} ADU -> solving in RAW COUNTS "
                f"(offset-subtracted, no gain division); background "
                f"{self.background_adu:.5g} ADU = offset + "
                f"{self.background():.5g} ADU-equivalent fluorescence" + tail
            )
        return (
            f"photon calibration: gain={cal.gain:.5g} ADU/photon, "
            f"offset={cal.offset:.6g} ADU -> solving in photons; "
            f"background {self.background_adu:.5g} ADU = offset + "
            f"{self.background():.5g} photons of fluorescence"
        )
