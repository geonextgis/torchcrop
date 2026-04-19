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
    """Clamp ``x`` to ``[lo, hi]`` — FST ``LIMIT``.

    Args:
        lo: Lower bound (scalar or tensor broadcastable against ``x``).
        hi: Upper bound (scalar or tensor broadcastable against ``x``).
        x: Input tensor.

    Returns:
        ``x`` clamped element-wise to ``[lo, hi]``.
    """
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

    Args:
        x: Selector tensor; the sign determines the branch.
        y1: Value returned when ``x < 0``.
        y2: Value returned when ``x >= 0``.
        smooth: If ``True``, use a sigmoid blend of sharpness ``k`` instead
            of the hard :func:`torch.where` switch.
        k: Sharpness of the sigmoid blend (only used when ``smooth=True``).

    Returns:
        Element-wise selected tensor.
    """
    if smooth:
        return soft_if(x, y2, y1, k=k)
    return torch.where(x >= 0.0, y2, y1)


def notnul(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """FST ``NOTNUL`` — guarded denominator.

    Returns ``x`` where ``|x| > eps`` else ``1``, for safe division.

    Args:
        x: Input tensor.
        eps: Threshold below which ``x`` is replaced by ``1``.

    Returns:
        Tensor of the same shape as ``x`` with near-zero entries replaced.
    """
    return torch.where(x.abs() > eps, x, torch.ones_like(x))


def reaand(*conditions: torch.Tensor) -> torch.Tensor:
    """FST ``REAAND`` — real-valued logical AND.

    Returns the element-wise product of real-valued indicator tensors. Useful
    to keep conditions differentiable as soft masks.

    Args:
        *conditions: One or more real-valued indicator tensors
            (element-wise in ``[0, 1]``).

    Returns:
        Element-wise product of all arguments (shape of the first).

    Raises:
        ValueError: If called with no conditions.
    """
    if not conditions:
        raise ValueError("reaand requires at least one condition")
    out = conditions[0]
    for c in conditions[1:]:
        out = out * c
    return out


def afgen(table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """FST ``AFGEN`` alias — piecewise-linear interpolation of ``x`` in ``table``.

    Args:
        table: Breakpoint table of shape ``[N, 2]`` or ``[B, N, 2]``.
        x: Query values.

    Returns:
        Interpolated values; see :func:`interpolate` for details.
    """
    return interpolate(table, x)
