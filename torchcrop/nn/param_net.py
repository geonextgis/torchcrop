"""Parameter net — predict crop parameters from site/weather embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn


class ParameterNet(nn.Module):
    """MLP that maps a context embedding to a vector of crop parameters.

    This is useful when you want to condition parameters on site/cultivar
    metadata (e.g., a learned embedding for each cultivar identifier).

    Args:
        embedding_dim: Size of the input context embedding.
        n_params: Number of crop parameters to predict.
        hidden_dim: Hidden-layer width.
        n_hidden: Number of hidden layers (each followed by ``Tanh``).
    """

    def __init__(
        self,
        embedding_dim: int,
        n_params: int,
        hidden_dim: int = 64,
        n_hidden: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = embedding_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden_dim), nn.Tanh()]
            d = hidden_dim
        layers.append(nn.Linear(d, n_params))
        self.mlp = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Predict crop parameters from a context embedding.

        Args:
            embedding: Context embedding tensor of shape
                ``[..., embedding_dim]``.

        Returns:
            Predicted parameter vector of shape ``[..., n_params]``.
        """
        return self.mlp(embedding)
