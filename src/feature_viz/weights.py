"""The weights between two neurons — the visualizing-weights question.

Distill, *Visualizing Weights* (2020):
https://distill.pub/2020/circuits/visualizing-weights/

The weight from upstream neuron ``i`` to downstream neuron ``j`` between two
conv layers is a ``k x k`` spatial matrix: the slice ``W[j, i, :, :]`` of a
conv kernel tensor of shape ``[C_out, C_in, k, k]``.

Inception branches have a bottleneck — a ``1x1`` channel reduction followed by
a ``k x k`` conv. The Distill weight panels show the *multiplied-out* effective
kernel across the bottleneck axis ``b``:

    W_eff[j, i, u, v] = sum_b  W_1x1[b, i, 0, 0] * W_kxk[j, b, u, v]

This ignores the ReLU between the two convs — the same approximation the paper
makes. BatchNorm is folded into each conv first, so the effective weights are
the ones the network actually applies at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model import get_layer

__all__ = ["fold_bn_into_conv", "effective_kernel", "weight_matrix"]


def fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> torch.Tensor:
    """Return the conv weight folded with the BatchNorm that immediately follows it.

    For ``Conv -> BN`` with BN parameters ``(gamma, beta, mu, var, eps)``:

        y = gamma * (W x - mu) / sqrt(var + eps) + beta

    so the effective weight is ``W' = W * gamma / sqrt(var + eps)``, a per
    output-channel rescaling. Bias does not affect the spatial *shape* of the
    weight pattern, so it is dropped.
    """
    w = conv.weight.detach()  # [C_out, C_in, k, k]
    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)  # [C_out]
    return w * scale.view(-1, 1, 1, 1)


def effective_kernel(
    model: nn.Module, block_name: str, branch_name: str = "branch2"
) -> torch.Tensor:
    """Multiply a bottleneck branch out into a single effective kernel.

    ``branch2`` of an Inception block is ``[1x1 bottleneck, k x k conv]``.
    Returns ``W_eff`` of shape ``[C_out, C_in, k, k]`` — the effective weight
    from every input channel to every output channel, BatchNorm folded in.
    """
    branch = getattr(get_layer(model, block_name), branch_name)
    w_1x1 = fold_bn_into_conv(branch[0].conv, branch[0].bn)  # [C_b, C_in, 1, 1]
    w_kxk = fold_bn_into_conv(branch[1].conv, branch[1].bn)  # [C_out, C_b, k, k]
    w_1x1_sq = w_1x1.squeeze(-1).squeeze(-1)  # [C_b, C_in]
    # W_eff[j, i, u, v] = sum_b W_1x1[b, i] * W_kxk[j, b, u, v]
    return torch.einsum("bi,jbuv->jiuv", w_1x1_sq, w_kxk)  # [C_out, C_in, k, k]


def weight_matrix(
    effective: torch.Tensor, downstream_j: int, upstream_i: int
) -> torch.Tensor:
    """Pull the single ``k x k`` matrix from upstream neuron ``i`` to downstream ``j``.

    Positive entries mean the upstream channel firing at that relative spatial
    offset *excites* the downstream neuron; negative entries *inhibit* it.
    """
    return effective[downstream_j, upstream_i].detach().cpu()
