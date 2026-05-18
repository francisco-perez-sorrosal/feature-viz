"""Device selection with a runtime capability probe.

The feature-visualisation render needs two ops that have historically lagged on
the Apple-Silicon MPS backend:

  - ``torch.fft.irfft2`` — the Fourier image parameterisation;
  - the backward of ``grid_sample`` — used by the rotation in
    ``random_transform`` (transformation robustness).

Rather than hard-code an assumption about either, this module *probes* MPS at
runtime: it runs both ops, forward and backward, and only selects MPS if every
one succeeds. As of PyTorch 2.12 the FFT path works but ``grid_sampler_2d_backward``
is unimplemented on MPS, so the probe fails and CPU is chosen — measured the
fastest option for this workload on an M2 anyway. If a future PyTorch adds the
missing op, the probe passes and MPS is selected automatically — no code change.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

__all__ = ["probe_mps_render", "best_device"]


@lru_cache(maxsize=1)
def probe_mps_render() -> bool:
    """Return True if MPS can run every op the feature-vis render needs.

    Probes the FFT round-trip and the ``grid_sample`` round-trip, each forward
    *and* backward — gradient ascent needs the backward passes. Result is
    cached; the probe runs at most once per process.
    """
    if not torch.backends.mps.is_available():
        return False
    try:
        # FFT round-trip (FourierImage.forward), forward + backward.
        spec = torch.randn(1, 3, 32, 17, 2, device="mps", requires_grad=True)
        img = torch.fft.irfft2(torch.view_as_complex(spec.contiguous()), s=(32, 32))
        img.sum().backward()
        # grid_sample round-trip (the rotation in random_transform), forward + backward.
        x = torch.randn(1, 3, 16, 16, device="mps", requires_grad=True)
        grid = F.affine_grid(
            torch.eye(2, 3, device="mps")[None], (1, 3, 16, 16), align_corners=False
        )
        F.grid_sample(x, grid, align_corners=False).sum().backward()
        torch.mps.synchronize()
        return True
    except (NotImplementedError, RuntimeError):
        return False


def best_device(requested: str = "auto") -> torch.device:
    """Resolve a device string to a concrete ``torch.device``.

    ``"auto"`` prefers CUDA, then MPS (only if :func:`probe_mps_render` passes),
    then CPU. Any explicit value (``"cpu"``, ``"cuda"``, ``"mps"``) is honoured
    as-is — the caller has opted out of the probe and owns the consequences.
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if probe_mps_render():
        return torch.device("mps")
    return torch.device("cpu")
