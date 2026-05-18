"""What a neuron *is* — the zoom-in question.

Distill, *Zoom In: An Introduction to Circuits* (2020):
https://distill.pub/2020/circuits/zoom-in/

A neuron is not an object stored anywhere. It is a channel index into the
activation tensor of a convolutional layer. Push an image through the network,
capture the tensor ``[B, C, H, W]`` at a layer, and "neuron N" is the slice
``act[:, N, :, :]`` — a single ``H x W`` scalar map. This module produces that
map and summarises it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .model import capture_activation

__all__ = [
    "preprocess_image",
    "preprocess_display",
    "gray_placeholder",
    "neuron_activation",
    "channel_slice",
    "neuron_stats",
    "NeuronStats",
]

# ImageNet normalisation the pretrained network expects.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_RESIZE_CROP = T.Compose([T.Resize(256), T.CenterCrop(224)])
_PREPROCESS = T.Compose(
    [
        _RESIZE_CROP,
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ]
)


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Resize / crop / normalise a PIL image into a ``[1, 3, 224, 224]`` batch."""
    return _PREPROCESS(img.convert("RGB")).unsqueeze(0)


def preprocess_display(img: Image.Image) -> np.ndarray:
    """Resize / crop to the exact ``224 x 224`` pixels the network sees — but
    *un-normalised*, as ``uint8`` RGB.

    Use this (not :func:`preprocess_image`) whenever pixels must line up
    spatially with the activation grid: an activation cell ``(gy, gx)`` in an
    ``H x W`` map corresponds to image region ``(gy/H, gx/W)`` of this array.
    """
    return np.asarray(_RESIZE_CROP(img.convert("RGB")))


def gray_placeholder() -> Image.Image:
    """A neutral 256x256 image, for demos run without an input photo."""
    return Image.new("RGB", (256, 256), color=(128, 128, 128))


def neuron_activation(
    model: torch.nn.Module, layer_name: str, image: torch.Tensor
) -> torch.Tensor:
    """Return the activation tensor ``[B, C, H, W]`` at ``layer_name`` for ``image``."""
    device = next(model.parameters()).device
    with capture_activation(model, layer_name) as store, torch.no_grad():
        model(image.to(device))
    return store["activation"]


def channel_slice(activation: torch.Tensor, channel: int) -> torch.Tensor:
    """Extract neuron ``channel`` from a ``[B, C, H, W]`` tensor as an ``H x W`` map."""
    return activation[0, channel].detach().cpu()


@dataclass(frozen=True)
class NeuronStats:
    """Summary of one neuron's response to one input."""

    mean: float
    max: float
    min: float
    argmax_yx: tuple[int, int]


def neuron_stats(channel_map: torch.Tensor) -> NeuronStats:
    """Compute mean / max / min and the peak location of an ``H x W`` neuron map."""
    width = channel_map.shape[-1]
    argmax = int(torch.argmax(channel_map.flatten()).item())
    return NeuronStats(
        mean=float(channel_map.mean()),
        max=float(channel_map.max()),
        min=float(channel_map.min()),
        argmax_yx=divmod(argmax, width),
    )
