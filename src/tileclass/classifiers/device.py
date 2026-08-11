"""Platform-aware torch device selection, shared by every classifier.

macOS has no CUDA, so "prefer CUDA, fall back to MPS" happens to land on
the right device there by accident -- but it doesn't express intent, and
would silently pick a slower device if that accident ever stopped
holding (e.g. some future PyTorch build exposing a CPU-side CUDA stub).
Branching on the actual platform says what's meant: MPS on macOS, CUDA
on Linux (and anywhere else that has it), CPU otherwise.
"""

import platform


def select_device():
    import torch

    system = platform.system()
    if system == "Darwin":
        if torch.backends.mps.is_available():
            return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
