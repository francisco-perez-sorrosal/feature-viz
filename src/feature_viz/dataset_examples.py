"""Dataset examples — showing what a neuron *detects*, the way Distill does.

Distill, [*Zoom In*](https://distill.pub/2020/circuits/zoom-in/) and
[*Early Vision*](https://distill.pub/2020/circuits/early-vision/) (2020).

A neuron's *activation map* (see :mod:`feature_viz.neuron`) tells you **where**
it fires on one input — it is not the neuron's identity. To see **what** the
neuron detects, the Circuits articles use *dataset examples*: for each image in
a dataset, crop the small patch around the spot where the neuron fires hardest.
A curve detector then yields curve-shaped patches; a fur detector, fur.

This module computes those crops. It works over the bundled `sample_images/`
set (~196 images) — tiny next to the millions the papers search, so it
demonstrates the *method*, not the paper's polish.

Two products:

  - :func:`top_examples` — for one channel, the peak crops of the images that
    activate it most (its "feature");
  - :func:`channel_cards` — one peak crop per channel, for browsing every
    channel at once (the "channel scout"). Channel indices in this re-implemented
    InceptionV1 do **not** match the Distill model, so browsing — not guessing a
    number — is how you find an interesting detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .model import capture_activation
from .neuron import preprocess_display, preprocess_image

__all__ = [
    "SampleImage",
    "DatasetActivations",
    "Example",
    "ChannelCard",
    "load_bundled_images",
    "compute_dataset_activations",
    "peak_crop",
    "top_examples",
    "channel_cards",
]


@dataclass(frozen=True)
class SampleImage:
    """One bundled image and its class name (the filename stem)."""

    name: str
    image: Image.Image


@dataclass(frozen=True)
class DatasetActivations:
    """Every bundled image's activations at one layer — computed once, reused.

    ``activations`` is ``[N, C, H, W]``; ``display[n]`` is the ``224 x 224``
    un-normalised image whose pixels line up with the activation grid.
    """

    names: list[str]
    display: list[np.ndarray]  # each [224, 224, 3] uint8
    activations: torch.Tensor  # [N, C, H, W], on CPU

    @property
    def scores(self) -> torch.Tensor:
        """Per-image, per-channel spatial-max activation — ``[N, C]``."""
        return self.activations.amax(dim=(2, 3))


@dataclass(frozen=True)
class Example:
    """One dataset image's peak crop for a particular neuron."""

    name: str
    score: float  # the neuron's peak activation on this image
    crop: np.ndarray  # uint8 [crop, crop, 3] — the receptive-field patch


@dataclass(frozen=True)
class ChannelCard:
    """A channel's "calling card": the single strongest peak crop across the set."""

    channel: int
    name: str  # which image produced it
    score: float
    crop: np.ndarray  # uint8 [crop, crop, 3]


def load_bundled_images() -> list[SampleImage]:
    """Load the bundled `sample_images/` set, sorted by name."""
    image_dir = files("feature_viz") / "sample_images"
    out: list[SampleImage] = []
    for entry in sorted(image_dir.iterdir(), key=lambda p: p.name):
        if not entry.name.endswith(".jpg"):
            continue
        with entry.open("rb") as fh:
            img = Image.open(fh).convert("RGB")
            img.load()
        out.append(SampleImage(name=entry.name[:-4], image=img))
    return out


def compute_dataset_activations(
    model: nn.Module, layer_name: str, images: list[SampleImage]
) -> DatasetActivations:
    """Run every bundled image through the model and capture ``layer_name``.

    One batched forward pass. The returned object holds the full ``[N, C, H, W]``
    activation tensor, so any channel can be inspected afterwards without
    touching the model again.
    """
    device = next(model.parameters()).device
    batch = torch.cat([preprocess_image(s.image) for s in images]).to(device)
    with capture_activation(model, layer_name) as store, torch.no_grad():
        model(batch)
    return DatasetActivations(
        names=[s.name for s in images],
        display=[preprocess_display(s.image) for s in images],
        activations=store["activation"].detach().cpu(),
    )


def peak_crop(
    display: np.ndarray, act_map: torch.Tensor, crop_size: int = 96
) -> np.ndarray:
    """Crop the ``crop_size x crop_size`` patch centred where ``act_map`` peaks.

    ``act_map`` is one neuron's ``H x W`` response; ``display`` is the matching
    ``224 x 224`` image. The crop approximates the neuron's receptive field at
    its strongest spatial location — the patch of input it is "looking at".
    """
    h, w = act_map.shape
    full = display.shape[0]  # 224
    crop_size = min(crop_size, full)
    flat_argmax = int(torch.argmax(act_map))
    gy, gx = divmod(flat_argmax, w)
    # Grid cell (gy, gx) -> centre pixel in the 224x224 image.
    cy = int((gy + 0.5) * full / h)
    cx = int((gx + 0.5) * full / w)
    half = crop_size // 2
    y0 = min(max(cy - half, 0), full - crop_size)
    x0 = min(max(cx - half, 0), full - crop_size)
    return np.ascontiguousarray(display[y0 : y0 + crop_size, x0 : x0 + crop_size])


def top_examples(
    da: DatasetActivations, channel: int, *, top_k: int = 8, crop_size: int = 96
) -> list[Example]:
    """The ``top_k`` dataset images that activate ``channel`` most, as peak crops.

    Ordered strongest first. These crops are the closest this module gets to the
    Distill "what the neuron detects" panel.
    """
    column = da.scores[:, channel]
    order = torch.argsort(column, descending=True).tolist()[:top_k]
    return [
        Example(
            name=da.names[n],
            score=float(column[n]),
            crop=peak_crop(da.display[n], da.activations[n, channel], crop_size),
        )
        for n in order
    ]


def channel_cards(da: DatasetActivations, *, crop_size: int = 64) -> list[ChannelCard]:
    """One peak crop per channel — the strongest example across the bundled set.

    Feeds the channel scout: a glance over these cards reveals which channels
    look like edge / curve / texture / object detectors, without guessing
    indices that would not match the Distill model anyway.
    """
    scores = da.scores  # [N, C]
    best_image = torch.argmax(scores, dim=0)  # [C]
    cards: list[ChannelCard] = []
    for channel in range(scores.shape[1]):
        n = int(best_image[channel])
        cards.append(
            ChannelCard(
                channel=channel,
                name=da.names[n],
                score=float(scores[n, channel]),
                crop=peak_crop(da.display[n], da.activations[n, channel], crop_size),
            )
        )
    return cards
