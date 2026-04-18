"""Comparison utilities for validating torchcrop against reference outputs."""

from __future__ import annotations

import torch


def relative_error(
    y: torch.Tensor,
    y_ref: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Element-wise relative error :math:`|y - y_\\text{ref}| / \\max(|y_\\text{ref}|, \\epsilon)`."""
    return (y - y_ref).abs() / torch.clamp(y_ref.abs(), min=eps)


def compare_trajectories(
    y: torch.Tensor,
    y_ref: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-6,
) -> dict[str, torch.Tensor | bool]:
    """Compare trajectories and report max/mean error and pass/fail flag."""
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
