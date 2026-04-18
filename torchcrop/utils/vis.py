"""Plotting helpers for state trajectories.

``matplotlib`` is an optional dependency — the functions here import it
lazily and raise a helpful error if it is not installed.
"""

from __future__ import annotations

from typing import Iterable

import torch


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for torchcrop.utils.vis. "
            "Install with `pip install matplotlib`."
        ) from e
    return plt


def plot_trajectories(
    traj: torch.Tensor,
    title: str = "",
    ylabel: str = "",
    labels: Iterable[str] | None = None,
):
    """Plot one or more trajectories of shape ``[B, T]`` or ``[T]``."""
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 4))
    y = traj.detach().cpu().numpy()
    if y.ndim == 1:
        ax.plot(y, label=None)
    else:
        labels = labels if labels is not None else [f"batch {i}" for i in range(y.shape[0])]
        for i, row in enumerate(y):
            ax.plot(row, label=labels[i])
        ax.legend(fontsize=8)
    ax.set_xlabel("day")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax
