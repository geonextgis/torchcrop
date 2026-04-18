"""PyTorch ports of SIMPLACE ``FSTFunctions``.

These helpers mirror the FST language primitives used inside Lintul5 so that
the Java ↔ Python translation is as syntactically close as possible.
"""

from __future__ import annotations

import torch

from torchcrop.functions.interpolation import interpolate
from torchcrop.functions.smoothing import soft_if


def limit(
    lo: float | torch.Tensor,
    hi: float | torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Clamp ``x`` to ``[lo, hi]`` — FST ``LIMIT``."""
    if not isinstance(lo, torch.Tensor):
        lo = torch.tensor(lo, dtype=x.dtype, device=x.device)
    if not isinstance(hi, torch.Tensor):
        hi = torch.tensor(hi, dtype=x.dtype, device=x.device)
    return torch.minimum(torch.maximum(x, lo), hi)


def insw(
    x: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    smooth: bool = False,
    k: float = 50.0,
) -> torch.Tensor:
    """FST ``INSW`` — input switch.

    Returns ``y2`` when ``x >= 0`` and ``y1`` otherwise. With ``smooth=True``
    a sigmoid blend is used so that gradient flows through both branches.
    """
    if smooth:
        return soft_if(x, y2, y1, k=k)
    return torch.where(x >= 0.0, y2, y1)


def notnul(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """FST ``NOTNUL`` — guarded denominator.

    Returns ``x`` where ``|x| > eps`` else ``1``, for safe division.
    """
    return torch.where(x.abs() > eps, x, torch.ones_like(x))


def reaand(*conditions: torch.Tensor) -> torch.Tensor:
    """FST ``REAAND`` — real-valued logical AND.

    Returns the element-wise product of real-valued indicator tensors. Useful
    to keep conditions differentiable as soft masks.
    """
    if not conditions:
        raise ValueError("reaand requires at least one condition")
    out = conditions[0]
    for c in conditions[1:]:
        out = out * c
    return out


def afgen(table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """FST ``AFGEN`` alias — piecewise-linear interpolation of ``x`` in ``table``."""
    return interpolate(table, x)
