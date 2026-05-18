"""feature-viz — an educational reimplementation of Distill's Circuits work.

Three concepts, each tied to a Distill article:

  - :mod:`feature_viz.neuron`           — what a neuron *is*           (zoom-in)
  - :mod:`feature_viz.weights`          — the weights between neurons  (visualizing-weights)
  - :mod:`feature_viz.feature_vis`      — the synthesised DxD icon     (early-vision)
  - :mod:`feature_viz.dataset_examples` — what a neuron detects, via crops (zoom-in / early-vision)

:mod:`feature_viz.model` loads InceptionV1 and reads activations;
:mod:`feature_viz.device` picks CPU / CUDA / MPS via a runtime capability probe;
:mod:`feature_viz.plotting` holds the matplotlib helpers the notebooks call.
"""

from __future__ import annotations

from .dataset_examples import (
    ChannelCard,
    DatasetActivations,
    Example,
    SampleImage,
    channel_cards,
    compute_dataset_activations,
    load_bundled_images,
    top_examples,
)
from .device import best_device, probe_mps_render
from .feature_vis import RenderResult, render_neuron
from .model import capture_activation, inception_blocks, layer_channels, load_inception
from .neuron import (
    channel_slice,
    neuron_activation,
    neuron_stats,
    preprocess_display,
    preprocess_image,
)
from .weights import effective_kernel, fold_bn_into_conv, weight_matrix

__all__ = [
    "best_device",
    "probe_mps_render",
    "load_inception",
    "inception_blocks",
    "layer_channels",
    "capture_activation",
    "preprocess_image",
    "preprocess_display",
    "neuron_activation",
    "channel_slice",
    "neuron_stats",
    "fold_bn_into_conv",
    "effective_kernel",
    "weight_matrix",
    "render_neuron",
    "RenderResult",
    "load_bundled_images",
    "compute_dataset_activations",
    "top_examples",
    "channel_cards",
    "SampleImage",
    "DatasetActivations",
    "Example",
    "ChannelCard",
]
