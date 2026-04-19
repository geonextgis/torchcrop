"""Smooth replacements for discontinuous operations.

These are optional drop-ins that preserve gradient flow through both
branches of a discontinuity, useful for second-order-smooth optimization
(L-BFGS, Hessian-based methods) and for differentiating through threshold
events in the model.
"""

from __future__ import annotations

import torch


def smooth_step(
    x: torch.Tensor,
    x0: float | torch.Tensor = 0.0,
    k: float = 50.0,
) -> torch.Tensor:
    """Differentiable Heaviside step.

    Returns $\sigma(k (x - x_0))$ which tends to 0 for
    $x \ll x_0$ and to 1 for $x \gg x_0$.

    Args:
        x: Input tensor.
        x0: Step location (scalar or broadcastable tensor).
        k: Sharpness of the transition; larger ``k`` yields a sharper step.

    Returns:
        Tensor with the same shape as ``x``, element-wise in ``(0, 1)``.
    """
    return torch.sigmoid(k * (x - x0))


def soft_clamp(
    x: torch.Tensor,
    lo: float | torch.Tensor,
    hi: float | torch.Tensor,
    k: float = 50.0,
) -> torch.Tensor:
    """Smooth clamp between ``lo`` and ``hi`` using softplus.

    The output asymptotes to ``lo`` below the lower bound and to ``hi`` above
    the upper bound, with a smooth transition of sharpness ``k``.

    Args:
        x: Input tensor.
        lo: Lower bound (scalar or broadcastable tensor).
        hi: Upper bound (scalar or broadcastable tensor).
        k: Sharpness of the transition at the bounds.

    Returns:
        Tensor with the same shape as ``x``, smoothly clamped into
        ``[lo, hi]``.
    """
    softplus = torch.nn.functional.softplus
    upper = hi - softplus(k * (hi - x)) / k
    return lo + softplus(k * (upper - lo)) / k


def soft_min(a: torch.Tensor, b: torch.Tensor, k: float = 50.0) -> torch.Tensor:
    """Differentiable minimum using the log-sum-exp trick.

    $\text{softmin}(a,b) = -\frac{1}{k}\log(e^{-k a} + e^{-k b})$.

    Args:
        a: First input tensor.
        b: Second input tensor (same shape as ``a``).
        k: Sharpness; larger ``k`` approximates the hard minimum more
            closely.

    Returns:
        Element-wise smooth minimum of ``a`` and ``b``.
    """
    stacked = torch.stack([a, b], dim=-1)
    return -torch.logsumexp(-k * stacked, dim=-1) / k


def soft_max(a: torch.Tensor, b: torch.Tensor, k: float = 50.0) -> torch.Tensor:
    """Differentiable maximum using the log-sum-exp trick.

    Args:
        a: First input tensor.
        b: Second input tensor (same shape as ``a``).
        k: Sharpness; larger ``k`` approximates the hard maximum more
            closely.

    Returns:
        Element-wise smooth maximum of ``a`` and ``b``.
    """
    stacked = torch.stack([a, b], dim=-1)
    return torch.logsumexp(k * stacked, dim=-1) / k


def soft_if(
    condition: torch.Tensor,
    true_val: torch.Tensor,
    false_val: torch.Tensor,
    k: float = 50.0,
) -> torch.Tensor:
    """Smooth replacement for ``torch.where(condition >= 0, true_val, false_val)``.

    Mirrors the FST ``INSW`` function but keeps gradient flow through both
    branches via a sigmoid blend of sharpness ``k``.

    Args:
        condition: Selector tensor; the sign (and magnitude) controls the
            sigmoid blend.
        true_val: Value used when ``condition >= 0``.
        false_val: Value used when ``condition < 0``.
        k: Sharpness of the sigmoid blend.

    Returns:
        Sigmoid-blended combination
        ``sigmoid(k * condition) * true_val +
        (1 - sigmoid(k * condition)) * false_val``.
    """
    alpha = torch.sigmoid(k * condition)
    return alpha * true_val + (1.0 - alpha) * false_val
