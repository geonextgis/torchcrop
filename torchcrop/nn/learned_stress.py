"""Learned stress-factor module.

Replaces the empirical water-stress curve (``TRANRF``) with a small MLP that
maps soil-water descriptors to a stress factor in :math:`[0, 1]` via a
sigmoid output.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedStressFactor(nn.Module):
    """MLP-based replacement for empirical water-stress curves."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 32,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, tranrf: torch.Tensor, nstress: torch.Tensor, *extra: torch.Tensor) -> torch.Tensor:
        """Return a combined stress factor in ``[0, 1]``.

        The mechanistic inputs (``tranrf``, ``nstress``) are concatenated with
        any extra context tensors and passed through the MLP, which outputs
        a single sigmoid-scaled factor.
        """
        stacked = torch.stack([tranrf, nstress, *extra], dim=-1)
        return torch.sigmoid(self.mlp(stacked)).squeeze(-1)
