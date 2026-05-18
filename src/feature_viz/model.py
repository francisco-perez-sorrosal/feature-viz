"""Loading InceptionV1 and reaching inside it.

A "neuron" in the Circuits papers is an address: ``(layer_name, channel_index)``.
To inspect one you need two things this module provides — the model itself,
and a way to read the activation tensor at a named layer. Activation capture
uses a forward hook wrapped in a context manager so the hook is always removed,
even on error.

Model: ``torchvision.models.googlenet`` with ``IMAGENET1K_V1`` weights — a
faithful re-implementation of InceptionV1. Structural and algorithmic claims
from the Distill papers carry over; channel *indices* do not match OpenAI
Microscope's TensorFlow checkpoint.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn
from torchvision.models import GoogLeNet_Weights, googlenet

__all__ = [
    "load_inception",
    "get_layer",
    "inception_blocks",
    "layer_channels",
    "capture_activation",
]


def load_inception(
    device: torch.device | str = "cpu", *, frozen: bool = True
) -> nn.Module:
    """Return InceptionV1 (GoogLeNet) with pretrained ImageNet weights, in eval mode.

    When ``frozen`` is True (the default) every parameter has ``requires_grad``
    cleared — the feature-visualisation loop optimises the *input image*, not
    the network, so gradients on weights would only waste memory.
    """
    model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1).to(device).eval()
    if frozen:
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def get_layer(model: nn.Module, name: str) -> nn.Module:
    """Resolve a named submodule, e.g. ``"inception4b"`` or ``"inception4b.branch2"``."""
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(
            f"no module named {name!r}; try one of {inception_blocks(model)}"
        )
    return modules[name]


def inception_blocks(model: nn.Module) -> list[str]:
    """List the top-level Inception block names, in forward order."""
    return [name for name, _ in model.named_children() if name.startswith("inception")]


def layer_channels(model: nn.Module, layer_name: str) -> int:
    """Channel count (= neuron count) of ``layer_name``.

    Determined by a single forward pass of a zero image — robust to the
    branch-concatenation structure of Inception blocks, where the count is not
    a single attribute to read off.
    """
    device = next(model.parameters()).device
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with capture_activation(model, layer_name) as store, torch.no_grad():
        model(dummy)
    return int(store["activation"].shape[1])


@contextmanager
def capture_activation(model: nn.Module, layer_name: str) -> Iterator[dict]:
    """Context manager yielding a dict that fills with the layer's output.

    Inside the ``with`` block, run a forward pass; afterwards ``store["activation"]``
    holds the captured tensor. The hook is registered on entry and removed on
    exit regardless of how the block ends.

        with capture_activation(model, "inception3b") as store:
            model(x)
        act = store["activation"]   # [B, C, H, W]
    """
    store: dict = {}
    layer = get_layer(model, layer_name)

    def hook(_module, _inp, output):
        store["activation"] = output

    handle = layer.register_forward_hook(hook)
    try:
        yield store
    finally:
        handle.remove()
