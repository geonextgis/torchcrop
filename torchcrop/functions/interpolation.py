"""Differentiable piecewise-linear interpolation.

Replaces the SIMPLACE ``AFGEN``/``InterpolationTable`` lookup used throughout
Lintul5. The implementation is a pure PyTorch piecewise-linear interpolator
that is differentiable with respect to both the query ``x`` and the table
``y`` values.
"""

from __future__ import annotations

import torch


def interpolate(
    table: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    r"""Piecewise-linear interpolation of ``x`` against a breakpoint table.

    Equivalent to the Lintul5 ``AFGEN`` function. Outside the support of the
    table the output is clamped to the first/last y-value (flat extrapolation),
    matching the SIMPLACE ``InterpolationTable`` behaviour.

    Parameters
    ----------
    table : torch.Tensor
        Breakpoint table of shape ``[N, 2]`` with columns ``(x_k, y_k)``
        sorted by increasing ``x_k``. May also be of shape ``[B, N, 2]``
        for batch-specific tables.
    x : torch.Tensor
        Query values of shape ``[B]`` (or scalar).

    Returns
    -------
    torch.Tensor
        Interpolated values with the broadcast shape of ``x``.
    """
    if table.dim() == 2:
        xs = table[:, 0].contiguous()
        ys = table[:, 1].contiguous()
        return _interp_1d(xs, ys, x)
    if table.dim() == 3:
        return _interp_batched(table, x)
    raise ValueError(
        f"table must be of shape [N, 2] or [B, N, 2]; got {tuple(table.shape)}"
    )


def _interp_1d(
    xs: torch.Tensor,
    ys: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Piecewise-linear interpolation with a single shared table."""
    n = xs.shape[0]
    if n < 2:
        raise ValueError("interpolation table needs at least 2 breakpoints")

    x_flat = x.reshape(-1).contiguous()

    idx = torch.bucketize(x_flat, xs, right=False)
    idx = torch.clamp(idx, min=1, max=n - 1)

    x0 = xs[idx - 1]
    x1 = xs[idx]
    y0 = ys[idx - 1]
    y1 = ys[idx]

    dx = x1 - x0
    safe_dx = torch.where(dx.abs() > 1e-12, dx, torch.ones_like(dx))
    t = (x_flat - x0) / safe_dx
    t = torch.clamp(t, 0.0, 1.0)

    y = y0 + t * (y1 - y0)
    return y.reshape(x.shape)


def _interp_batched(
    table: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Piecewise-linear interpolation with a per-batch table ``[B, N, 2]``."""
    b, n, _ = table.shape
    if n < 2:
        raise ValueError("interpolation table needs at least 2 breakpoints")
    if x.shape[0] != b:
        raise ValueError(
            f"batch size mismatch between table ({b}) and x ({x.shape[0]})"
        )

    xs = table[..., 0]
    ys = table[..., 1]

    x_expanded = x.unsqueeze(-1)
    idx = (xs < x_expanded).sum(dim=-1)
    idx = torch.clamp(idx, min=1, max=n - 1)

    gather_idx_lo = (idx - 1).unsqueeze(-1)
    gather_idx_hi = idx.unsqueeze(-1)

    x0 = torch.gather(xs, -1, gather_idx_lo).squeeze(-1)
    x1 = torch.gather(xs, -1, gather_idx_hi).squeeze(-1)
    y0 = torch.gather(ys, -1, gather_idx_lo).squeeze(-1)
    y1 = torch.gather(ys, -1, gather_idx_hi).squeeze(-1)

    dx = x1 - x0
    safe_dx = torch.where(dx.abs() > 1e-12, dx, torch.ones_like(dx))
    t = (x - x0) / safe_dx
    t = torch.clamp(t, 0.0, 1.0)

    return y0 + t * (y1 - y0)
