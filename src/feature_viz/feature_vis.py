"""Feature visualisation — the early-vision question.

Distill, *An Overview of Early Vision in InceptionV1* (2020):
https://distill.pub/2020/circuits/early-vision/
Method: *Feature Visualization* (Olah, Mordvintsev & Schubert, Distill 2017):
https://distill.pub/2017/feature-visualization/

The DxD picture used as a neuron's identity in the Distill articles is not a
slice of weights and not an activation map — it is a *synthetic input image*,
one element of the ``3 x 224 x 224`` input space, found by activation-
maximisation gradient ascent through the frozen network. The canonical 'lucid'
recipe has four parts:

  (a) Parameterise the image in a decorrelated Fourier basis with 1/f
      frequency scaling — a natural-image prior that suppresses adversarial
      high-frequency noise.
  (b) Apply a learned ImageNet colour-correlation matrix, then a sigmoid, so
      RGB values stay in (0, 1) and match natural-image colour statistics.
  (c) Each step, randomly transform the image (jitter / scale / rotate) before
      the forward pass — transformation robustness.
  (d) Forward through InceptionV1, read the spatial-mean activation of the
      target channel, gradient-ascend on the image parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from .model import capture_activation

__all__ = ["FourierImage", "random_transform", "render_neuron", "RenderResult"]


# ImageNet colour statistics, from lucid (Olah et al.). The columns are the SVD
# components of the ImageNet colour covariance; multiplying decorrelated channel
# values by this matrix yields natural-photograph colour distributions.
_COLOR_CORRELATION = np.asarray(
    [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]], dtype=np.float32
)
_COLOR_CORRELATION /= float(np.max(np.linalg.norm(_COLOR_CORRELATION, axis=0)))

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FourierImage(nn.Module):
    """An image parameterised in a decorrelated Fourier basis — steps (a) + (b).

    The learnable parameters are the real/imaginary spectrum coefficients, not
    the pixels. ``forward()`` applies 1/f frequency scaling, inverse-FFTs to
    pixel space, recorrelates to ImageNet colour, and squashes through a sigmoid.
    """

    def __init__(
        self, h: int = 224, w: int = 224, *, decay_power: float = 1.0, sd: float = 0.01
    ):
        super().__init__()
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.rfftfreq(w)[None, :]
        freqs = np.sqrt(fx * fx + fy * fy)

        # 1/f scaling, floored so the DC term is not infinite.
        scale = 1.0 / np.maximum(freqs, 1.0 / max(h, w)) ** decay_power
        scale = scale * np.sqrt(w * h)
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

        fh, fw = freqs.shape  # h, w // 2 + 1
        # Trailing dim of 2 is reinterpreted as a complex spectrum.
        self.params = nn.Parameter(torch.randn(1, 3, fh, fw, 2) * sd)
        self.h, self.w = h, w
        self.register_buffer("color_correlation", torch.tensor(_COLOR_CORRELATION))

    def forward(self) -> torch.Tensor:
        scaled = self.params * self.scale[None, None, :, :, None]
        spectrum = torch.view_as_complex(scaled.contiguous())  # [1, 3, h, w//2+1]
        img = torch.fft.irfft2(spectrum, s=(self.h, self.w))  # [1, 3, h, w]
        img = img / 4.0  # match lucid magnitude

        b, c, h, w = img.shape
        flat = img.permute(0, 2, 3, 1).reshape(-1, 3)  # [B*H*W, 3]
        flat = flat @ self.color_correlation.T  # mix channels to ImageNet colour
        img = flat.reshape(b, h, w, 3).permute(0, 3, 1, 2)
        return torch.sigmoid(img)  # values in (0, 1)


def random_transform(
    img: torch.Tensor,
    *,
    max_jitter: int = 16,
    scale_pct: float = 0.15,
    max_rot_deg: float = 10.0,
) -> torch.Tensor:
    """Apply random jitter / scale / rotation — step (c), transformation robustness."""
    _, _, h, w = img.shape
    # Pad then random-crop = translation jitter.
    padded = F.pad(img, [max_jitter] * 4, mode="constant", value=0.5)
    dy = int(torch.randint(0, 2 * max_jitter + 1, (1,)).item())
    dx = int(torch.randint(0, 2 * max_jitter + 1, (1,)).item())
    out = padded[:, :, dy : dy + h, dx : dx + w]
    # Random scale.
    s = 1.0 + (torch.rand(1).item() * 2 - 1) * scale_pct
    out = F.interpolate(
        out,
        size=(max(1, int(h * s)), max(1, int(w * s))),
        mode="bilinear",
        align_corners=False,
    )
    out = TF.center_crop(out, [h, w])
    # Random rotation.
    angle = (torch.rand(1).item() * 2 - 1) * max_rot_deg
    return TF.rotate(out, angle)


def _imagenet_normalise(img: torch.Tensor, device: torch.device) -> torch.Tensor:
    return (img - _IMAGENET_MEAN.to(device)) / _IMAGENET_STD.to(device)


def _decode(img_param: FourierImage) -> np.ndarray:
    """Decode the current image parameters to a ``uint8 [H, W, 3]`` RGB array."""
    with torch.no_grad():
        arr = img_param().detach().cpu().squeeze(0).clamp(0, 1).numpy()
    return (arr.transpose(1, 2, 0) * 255).astype(np.uint8)


@dataclass(frozen=True)
class RenderResult:
    """Output of :func:`render_neuron`."""

    image: np.ndarray  # uint8 [H, W, 3], the final synthesised image
    activations: list[float]  # target-channel activation per step (the ascent curve)
    snapshots: list[
        tuple[int, np.ndarray]
    ]  # (step, image) — the optimisation trajectory

    @property
    def final_activation(self) -> float:
        return self.activations[-1] if self.activations else float("nan")


def render_neuron(
    model: nn.Module,
    layer_name: str,
    channel: int,
    *,
    steps: int = 512,
    lr: float = 0.05,
    size: int = 224,
    seed: int | None = None,
    n_snapshots: int = 0,
    progress: Callable[[int, float], None] | None = None,
) -> RenderResult:
    """Synthesise the input image that maximally activates ``layer_name:channel``.

    ``model`` must already be on the target device, in eval mode, with frozen
    parameters (see :func:`feature_viz.model.load_inception`). ``progress`` is
    called as ``progress(step, activation)`` each step — a notebook can use it
    to drive a progress bar.

    When ``n_snapshots > 0``, that many evenly-spaced decoded images are kept in
    ``RenderResult.snapshots`` — from step 0 (the initial random noise) to the
    final image — so a caller can show the optimisation trajectory. Each
    snapshot is one extra forward pass through the tiny image parameterisation,
    so the cost is negligible next to the render itself.
    """
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)

    img_param = FourierImage(size, size).to(device)
    optimiser = torch.optim.Adam(img_param.parameters(), lr=lr)
    activations: list[float] = []
    snapshots: list[tuple[int, np.ndarray]] = []

    # Step indices to snapshot at — 0 (initial noise) ... steps (final image).
    if n_snapshots > 1:
        snap_at = {round(i * steps / (n_snapshots - 1)) for i in range(n_snapshots)}
    elif n_snapshots == 1:
        snap_at = {steps}
    else:
        snap_at = set()
    if 0 in snap_at:
        snapshots.append((0, _decode(img_param)))

    with capture_activation(model, layer_name) as store:
        for step in range(steps):
            optimiser.zero_grad()
            img_aug = random_transform(img_param())
            model(_imagenet_normalise(img_aug, device))
            loss = -store["activation"][:, channel].mean()
            loss.backward()
            optimiser.step()

            activation = -float(loss.item())
            activations.append(activation)
            if progress is not None:
                progress(step, activation)
            if (step + 1) in snap_at:
                snapshots.append((step + 1, _decode(img_param)))

    return RenderResult(
        image=_decode(img_param), activations=activations, snapshots=snapshots
    )
