"""Learned stress-factor module.

Replaces the empirical water-stress curve (``TRANRF``) with a small MLP that
maps soil-water descriptors to a stress factor in $[0, 1]$ via a
sigmoid output.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedStressFactor(nn.Module):
    """MLP-based replacement for empirical water-stress curves.

    Args:
        input_dim: Number of scalar inputs concatenated along the feature
            axis — typically ``2 + len(extra)`` for the mechanistic
            ``tranrf``/``nstress`` pair plus optional context tensors.
        hidden_dim: Hidden-layer width.
        n_hidden: Number of hidden layers (each followed by ``Tanh``).
    """

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

        The mechanistic inputs (``tranrf``, ``nstress``) are stacked with any
        extra context tensors along the last axis and passed through the MLP,
        whose scalar output is squashed by a sigmoid.

        Args:
            tranrf: Water-stress factor in ``[0, 1]`` from
                `WaterBalance`, shape ``[B]``.
            nstress: Nutrient-stress factor in ``[0, 1]`` from
                `NutrientDemand`, shape ``[B]``.
            *extra: Optional additional context tensors, each of shape
                ``[B]``.

        Returns:
            Combined stress factor in ``[0, 1]``, shape ``[B]``.
        """
        stacked = torch.stack([tranrf, nstress, *extra], dim=-1)
        return torch.sigmoid(self.mlp(stacked)).squeeze(-1)
