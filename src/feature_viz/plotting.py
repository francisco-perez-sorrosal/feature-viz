"""Matplotlib helpers shared by the marimo notebooks.

Every function returns a ``matplotlib.figure.Figure`` — marimo (and Jupyter)
render a returned figure inline, so notebook cells stay one-liners and all the
drawing logic is tested here, not duplicated across notebooks.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .weights import weight_matrix

__all__ = [
    "show_channel_map",
    "show_activation_overlay",
    "show_image",
    "show_image_grid",
    "show_weight_matrix",
    "show_weight_grid",
    "show_ascent_curve",
    "show_activation_ranking",
    "show_feature_directions",
    "show_circuit_rows",
]


def show_channel_map(channel_map: torch.Tensor, title: str | None = None) -> Figure:
    """Heatmap of one neuron's ``H x W`` activation map."""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(channel_map.numpy(), cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def show_image_grid(items: list[tuple[str, object]], ncols: int = 6) -> Figure:
    """Display a labelled row/grid of images — ``items`` is ``(label, image)`` pairs.

    Each image may be a PIL image or an array; used for the top-activating
    dataset examples in notebook 01.
    """
    n = len(items)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(1.7 * ncols, 1.9 * nrows), squeeze=False
    )
    for ax, (label, image) in zip(axes.flat, items):
        ax.imshow(image)
        ax.set_title(label, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def show_activation_overlay(
    image: np.ndarray,
    act_map: torch.Tensor,
    *,
    title: str | None = None,
    crop_size: int = 96,
) -> Figure:
    """Overlay a neuron's activation heatmap on the input image it responded to.

    ``image`` is the ``224 x 224`` un-normalised input (see
    :func:`feature_viz.neuron.preprocess_display`); ``act_map`` is that neuron's
    ``H x W`` response. The small map is upscaled and blended on top, and a box
    marks the receptive-field patch at the firing peak — so *where the neuron
    fires* is read directly off the user's own image.
    """
    img = np.asarray(image)
    full = img.shape[0]
    grid = (
        act_map.detach().cpu().numpy()
        if isinstance(act_map, torch.Tensor)
        else np.asarray(act_map)
    )
    h, w = grid.shape

    fig, ax = plt.subplots(figsize=(4, 4))
    extent = (0, full, full, 0)  # row 0 at top, matching the image
    ax.imshow(img, extent=extent)
    ax.imshow(grid, cmap="inferno", alpha=0.5, extent=extent, interpolation="bilinear")

    # Box the receptive-field patch at the activation peak.
    gy, gx = divmod(int(np.argmax(grid)), w)
    cy, cx = (gy + 0.5) * full / h, (gx + 0.5) * full / w
    half = crop_size / 2
    x0 = min(max(cx - half, 0), full - crop_size)
    y0 = min(max(cy - half, 0), full - crop_size)
    ax.add_patch(
        Rectangle((x0, y0), crop_size, crop_size, fill=False, edgecolor="cyan", lw=2)
    )

    ax.set_xlim(0, full)
    ax.set_ylim(full, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def show_image(image: np.ndarray, title: str | None = None) -> Figure:
    """Display an RGB image array (the synthesised feature visualisation)."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def show_weight_matrix(matrix: torch.Tensor, title: str | None = None) -> Figure:
    """Heatmap of one ``k x k`` weight matrix, each cell annotated with its value.

    Red excites the downstream neuron, blue inhibits; the colour scale is
    symmetric about zero so the sign is read directly off the hue.
    """
    w = (
        matrix.detach().cpu().numpy()
        if isinstance(matrix, torch.Tensor)
        else np.asarray(matrix)
    )
    vmax = max(abs(float(w.min())), abs(float(w.max())), 1e-9)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    im = ax.imshow(w, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for (yy, xx), value in np.ndenumerate(w):
        ax.text(xx, yy, f"{value:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def show_weight_grid(
    effective: torch.Tensor, downstream_j: int, n_upstream: int = 25
) -> Figure:
    """Grid of effective ``k x k`` weight matrices into one downstream neuron.

    Red entries excite the downstream neuron, blue inhibit; each cell is a
    different upstream channel. This is the panel the Distill weight explorer
    renders.
    """
    n_rows = int(math.sqrt(n_upstream))
    n_cols = (n_upstream + n_rows - 1) // n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))
    for i, ax in enumerate(axes.flat):
        if i >= n_upstream:
            ax.axis("off")
            continue
        w = weight_matrix(effective, downstream_j, i).numpy()
        vmax = max(abs(w.min()), abs(w.max()), 1e-9)
        ax.imshow(w, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"i={i}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    k = effective.shape[-1]
    fig.suptitle(
        f"effective {k}x{k} weights into downstream channel j={downstream_j}\n"
        f"red = excitatory   blue = inhibitory   (per relative spatial offset)",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def show_ascent_curve(activations: list[float]) -> Figure:
    """Plot the target-channel activation over gradient-ascent steps."""
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(activations, color="tab:blue")
    ax.set_xlabel("step")
    ax.set_ylabel("target activation")
    ax.set_title("activation-maximisation curve", fontsize=10)
    fig.tight_layout()
    return fig


def show_feature_directions() -> Figure:
    """Schematic — neurons are axes of activation space; a feature is a direction.

    A teaching figure for Claim 1: a layer's activations at one position form a
    vector; each neuron is one axis (a *privileged* direction), and a feature is
    a direction that is usually — but not always — a single neuron's axis.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    arrow = dict(arrowstyle="-|>", lw=2)
    # Two neuron axes.
    ax.annotate(
        "", xy=(1.1, 0), xytext=(0, 0), arrowprops={**arrow, "color": "steelblue"}
    )
    ax.annotate(
        "", xy=(0, 1.1), xytext=(0, 0), arrowprops={**arrow, "color": "steelblue"}
    )
    ax.text(1.12, 0.02, "neuron 1", color="steelblue", fontsize=9, va="bottom")
    ax.text(0.02, 1.12, "neuron 2", color="steelblue", fontsize=9)
    # An off-axis feature direction.
    ax.annotate(
        "", xy=(0.78, 0.62), xytext=(0, 0), arrowprops={**arrow, "color": "crimson"}
    )
    ax.text(0.80, 0.64, "a feature\n(a direction)", color="crimson", fontsize=9)
    ax.set_xlim(-0.15, 1.45)
    ax.set_ylim(-0.15, 1.35)
    ax.set_aspect("equal")
    ax.set_title("neurons are axes — a feature is a direction", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def show_circuit_rows(
    inputs: list[tuple[int, float, torch.Tensor, list[np.ndarray]]],
    *,
    n_crops: int = 4,
    synth_thumbnails: list[np.ndarray | None] | None = None,
) -> Figure:
    """One row per upstream input — weight matrix on the left, feature crops on the right.

    Each tuple is ``(channel_index, signed_sum, k_by_k_weight, list_of_crops)``.
    The weight is rendered with the symmetric red/blue colormap used elsewhere
    (red excites the downstream neuron, blue inhibits); the crops alongside it
    are the upstream neuron's peak dataset examples — its *feature*. Reading
    the two together turns a column of weights into a *circuit*.

    When ``synth_thumbnails`` is provided (one image per row, or ``None`` for
    "not yet rendered"), an extra column is inserted between the weight matrix
    and the dataset crops showing the upstream neuron's synthesised
    feature-visualisation thumbnail — a second, model-internal representation
    of the same feature, complementing the real-world crops.
    """
    n_rows = len(inputs)
    if n_rows == 0:
        fig, ax = plt.subplots(figsize=(2, 1))
        ax.axis("off")
        return fig
    has_synth = synth_thumbnails is not None
    n_cols = 1 + (1 if has_synth else 0) + n_crops
    width_ratios = [1.4] + ([1.1] if has_synth else []) + [1.0] * n_crops
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(1.4 * n_cols + 0.6, 1.5 * n_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )
    for r, (ch, score, w, crops) in enumerate(inputs):
        ax_w = axes[r, 0]
        w_np = (
            w.detach().cpu().numpy() if isinstance(w, torch.Tensor) else np.asarray(w)
        )
        vmax = max(abs(float(w_np.min())), abs(float(w_np.max())), 1e-9)
        ax_w.imshow(w_np, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for (yy, xx), value in np.ndenumerate(w_np):
            ax_w.text(xx, yy, f"{value:+.1f}", ha="center", va="center", fontsize=7)
        ax_w.set_title(f"ch {ch}   sum={score:+.2f}", fontsize=8)
        ax_w.set_xticks([])
        ax_w.set_yticks([])
        col_offset = 1
        if has_synth:
            ax_s = axes[r, 1]
            thumb = synth_thumbnails[r]
            if thumb is not None:
                ax_s.imshow(thumb)
            else:
                ax_s.text(
                    0.5,
                    0.5,
                    "(click to\nrender)",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="gray",
                    transform=ax_s.transAxes,
                )
            if r == 0:
                ax_s.set_title("synth", fontsize=8)
            ax_s.set_xticks([])
            ax_s.set_yticks([])
            col_offset = 2
        for c in range(n_crops):
            ax = axes[r, col_offset + c]
            if c < len(crops):
                ax.imshow(crops[c])
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    return fig


def show_activation_ranking(
    names: list[str],
    scores: list[float],
    user_score: float,
    *,
    user_label: str = "★ your image",
) -> Figure:
    """Rank dataset images by a neuron's activation, with the user's image slotted in.

    The bridge between the two panels: it places the user's input on the same
    axis as the dataset, so changing the input visibly moves its bar.
    """
    items = [(n, float(s)) for n, s in zip(names, scores)]
    items.append((user_label, float(user_score)))
    items.sort(key=lambda pair: pair[1], reverse=True)

    labels = [n for n, _ in items]
    values = [v for _, v in items]
    colors = ["crimson" if n == user_label else "steelblue" for n in labels]

    fig, ax = plt.subplots(figsize=(5, 0.30 * len(items) + 0.6))
    ax.barh(range(len(items)), values, color=colors)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()  # strongest at the top
    ax.set_xlabel("peak activation of this neuron")
    ax.set_title("which images fire this neuron hardest", fontsize=10)
    fig.tight_layout()
    return fig
