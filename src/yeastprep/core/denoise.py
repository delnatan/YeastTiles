"""Denoise driver: an optional self-supervised pass over a combined-channel
tiff, writing into 02_denoised/.

Each channel of each image gets its own small network, fitted to that image
alone and then discarded -- no training data, no checkpoints. The method is
Noise2Self-style blind-spot denoising (Batson & Royer 2019): every step hides
a sparse grid of pixels behind the average of their 8 neighbors and scores
the prediction only at the hidden pixels. Camera noise is independent from
pixel to pixel, so nothing in a hidden pixel's neighborhood predicts its
noise, and the best the network can do is predict the clean signal. The
output is J-invariant too: each pixel's value is predicted while that pixel
is hidden, so its own noise can't leak through.

Tested on 40x yeast data against the jssl-denoise D-Net/N-Net it replaces:
similar brightfield, no intensity bias on the target channel (jssl lowered
it by ~10% of the signal above offset), and the residual (raw - denoised)
is white -- it removes noise, not structure. A learned noise model, a
Bayesian posterior, a wider network and fitting across several images were
tried as well and didn't help.

torch is imported only inside the functions that fit, so `DenoiseParams`
stays importable without it (core/project.py needs it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tileclass.classifiers.device import select_device

from .combined_tiff import (
    BRIGHTFIELD_CHANNEL,
    TARGET_CHANNEL,
    read_combined_tiff,
    read_z_count,
    write_combined_tiff,
)

CROP = 256  # training crop side, px
BATCH = 2  # crops per step
MASK_SPACING = (3, 5)  # hidden-grid spacing, drawn per step (inclusive)
INFERENCE_SPACING = 4  # J-invariant output: INFERENCE_SPACING**2 passes
LEARNING_RATE = 1e-3
SEED = 0


@dataclass
class DenoiseParams:
    channels: tuple[int, ...] = (BRIGHTFIELD_CHANNEL, TARGET_CHANNEL)
    steps: int = 2000  # optimizer steps per channel per image (~30 s on an M5)


def _neighbor_kernel(torch):
    """Gaussian (sigma=1) weights over the 8 neighbors, center excluded."""
    edge, corner = math.exp(-0.5), math.exp(-1.0)
    k = torch.tensor([[corner, edge, corner], [edge, 0.0, edge], [corner, edge, corner]])
    return (k / k.sum()).view(1, 1, 3, 3)


def _build_net(torch, width: int = 32):
    """2-level U-Net: 3x3 convs, LeakyReLU, max-pool down, nearest up with
    skips. ~70k parameters; its receptive field covers a yeast cell."""
    nn, F = torch.nn, torch.nn.functional

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.LeakyReLU(0.2))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.b = block(1, width), block(width, width), block(width, width)
            self.d2, self.d1 = block(2 * width, width), block(2 * width, width)
            self.out = nn.Conv2d(width, 1, 1)

        def forward(self, x):
            h, w = x.shape[-2:]
            ph, pw = (-h) % 4, (-w) % 4
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            e1 = self.e1(x)
            e2 = self.e2(F.max_pool2d(e1, 2))
            b = self.b(F.max_pool2d(e2, 2))
            d2 = self.d2(torch.cat([F.interpolate(b, scale_factor=2), e2], 1))
            d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
            return self.out(d1)[..., :h, :w]

    return UNet()


def _hide(torch, x, mask, kernel):
    """`x` with the `mask` pixels replaced by their neighbors' weighted average."""
    F = torch.nn.functional
    avg = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), kernel)
    return torch.where(mask, avg, x)


def denoise_image(image: np.ndarray, steps: int = 2000, device=None) -> np.ndarray:
    """Fit a blind-spot network to `image` (2D) and return its J-invariant
    denoised version, float32, in the input's units. Deterministic for a
    given image and device."""
    import torch

    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    if h < CROP or w < CROP:
        raise ValueError(f"image {image.shape} is smaller than the {CROP}px training crop")
    device = device or select_device()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    lo, hi = (float(v) for v in np.percentile(image, [0.1, 99.9]))
    scale = (hi - lo) or 1.0
    x = torch.from_numpy((image - lo) / scale)[None, None].to(device)
    kernel = _neighbor_kernel(torch).to(device)

    net = _build_net(torch).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, steps * 3 // 4), gamma=0.5)
    for _ in range(steps):
        tops = rng.integers(0, h - CROP + 1, BATCH)
        lefts = rng.integers(0, w - CROP + 1, BATCH)
        batch = torch.cat([x[..., t : t + CROP, l : l + CROP] for t, l in zip(tops, lefts)])
        batch = torch.rot90(batch, int(rng.integers(0, 4)), dims=(-2, -1))
        if rng.random() < 0.5:
            batch = torch.flip(batch, dims=[-1])

        spacing = int(rng.integers(MASK_SPACING[0], MASK_SPACING[1] + 1))
        mask = torch.zeros((CROP, CROP), dtype=torch.bool, device=device)
        mask[int(rng.integers(0, spacing)) :: spacing, int(rng.integers(0, spacing)) :: spacing] = True
        prediction = net(_hide(torch, batch, mask, kernel))
        loss = ((prediction - batch) ** 2)[..., mask].mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    out = torch.empty_like(x)
    s = INFERENCE_SPACING
    with torch.no_grad():
        for oy in range(s):
            for ox in range(s):
                mask = torch.zeros((h, w), dtype=torch.bool, device=device)
                mask[oy::s, ox::s] = True
                out[..., mask] = net(_hide(torch, x, mask, kernel))[..., mask]
    result = out.squeeze().cpu().numpy() * scale + lo
    _release_device_cache(torch, device)
    return result.astype(np.float32)


def _release_device_cache(torch, device) -> None:
    """MPS/CUDA keep freed blocks cached; release them between fits so a
    long batch doesn't hold onto device memory."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def run_denoise(image: np.ndarray, params: DenoiseParams) -> np.ndarray:
    """Denoise one channel's image, returning it in the input's dtype so
    02_denoised/ stays a drop-in replacement for 01_reduced/: uint16
    brightfield is cast back to uint16, the float32 target (raw summed
    counts, see core/pipeline.py's `sum_project`) stays float32 -- clipped at
    zero, not 65535, so deconvolution still gets the full range."""
    denoised = denoise_image(image, steps=params.steps)
    if np.asarray(image).dtype.kind == "f":
        return np.maximum(denoised, 0).astype(np.float32)
    return np.clip(np.rint(denoised), 0, 65535).astype(np.uint16)


@dataclass
class DenoiseProcessResult:
    path: Path
    success: bool
    output_path: Path | None = None
    error: str | None = None


def denoise_and_save(
    path: Path, outdir: Path, params: DenoiseParams = DenoiseParams()
) -> DenoiseProcessResult:
    """Denoise `params.channels` of one combined-channel tiff and write the
    result (same 2-channel layout) into `outdir` (02_denoised/). A channel
    not being denoised this run is carried over from that file's existing
    output if there is one, else from the source -- so denoising one channel
    after the other doesn't lose either result. Catches any exception so a
    batch run doesn't abort on the first bad file (matches
    core/pipeline.py's process_and_save)."""
    path = Path(path)
    outdir = Path(outdir)
    try:
        if not params.channels:
            raise ValueError("no channels selected to denoise")
        outdir.mkdir(parents=True, exist_ok=True)
        output_path = outdir / f"{path.stem}.tiff"

        brightfield, target, scale = read_combined_tiff(path)
        if output_path.exists():
            existing_brightfield, existing_target, _ = read_combined_tiff(output_path)
        else:
            existing_brightfield, existing_target = brightfield, target

        brightfield = (
            run_denoise(brightfield, params)
            if BRIGHTFIELD_CHANNEL in params.channels
            else existing_brightfield
        )
        target = run_denoise(target, params) if TARGET_CHANNEL in params.channels else existing_target

        write_combined_tiff(output_path, brightfield, target, scale, n_z=read_z_count(path))
        return DenoiseProcessResult(path=path, success=True, output_path=output_path)
    except Exception as exc:
        return DenoiseProcessResult(path=path, success=False, error=str(exc))
