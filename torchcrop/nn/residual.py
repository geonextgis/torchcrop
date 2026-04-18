"""Neural residual correction networks.

A ``NeuralResidual`` adds a small MLP-based correction to the output of a
mechanistic process, scaled by ``tanh`` to keep corrections bounded.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NeuralResidual(nn.Module):
    r"""Bounded additive residual from an MLP.

    .. math::
        f_\theta(\mathbf{x}) = \text{scale} \cdot \tanh(\text{MLP}(\mathbf{x}))
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 32,
        n_hidden: int = 2,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
            d = hidden_dim
        layers.append(nn.Linear(d, output_dim))
        self.mlp = nn.Sequential(*layers)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.tanh(self.mlp(x))
