"""Comparison utilities for validating torchcrop against reference outputs."""

from __future__ import annotations

import torch


def relative_error(
    y: torch.Tensor,
    y_ref: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute element-wise relative error.

    $$
    e_i = \\frac{|y_i - y_{\\text{ref},i}|}{\\max(|y_{\\text{ref},i}|,
    \\epsilon)}
    $$

    Args:
        y: Tensor of computed values.
        y_ref: Tensor of reference values broadcastable to ``y``.
        eps: Lower bound on the denominator to avoid division by zero.

    Returns:
        Tensor of relative errors with the broadcast shape of ``y`` and
        ``y_ref``.
    """
    return (y - y_ref).abs() / torch.clamp(y_ref.abs(), min=eps)


def compare_trajectories(
    y: torch.Tensor,
    y_ref: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dict[str, torch.Tensor | bool]:
    """Compare trajectories and report max/mean error and pass/fail flag.

    Args:
        y: Computed trajectory tensor.
        y_ref: Reference trajectory tensor broadcastable to ``y``.
        rtol: Relative tolerance forwarded to `torch.allclose`.
        atol: Absolute tolerance forwarded to `torch.allclose`.

    Returns:
        Dict with the following entries:

            * ``max_abs_error`` — scalar tensor, maximum absolute error.
            * ``mean_abs_error`` — scalar tensor, mean absolute error.
            * ``max_rel_error`` — scalar tensor, maximum relative error
              (using `relative_error`).
            * ``passed`` — ``bool``, result of
              ``torch.allclose(y, y_ref, rtol=rtol, atol=atol)``.
    """
    diff = (y - y_ref).abs()
    max_err = diff.max()
    mean_err = diff.mean()
    ok = torch.allclose(y, y_ref, rtol=rtol, atol=atol)
    return {
        "max_abs_error": max_err,
        "mean_abs_error": mean_err,
        "max_rel_error": relative_error(y, y_ref).max(),
        "passed": bool(ok),
    }
