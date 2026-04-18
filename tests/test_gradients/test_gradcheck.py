"""Gradient checks (torch.autograd.gradcheck) for the differentiable primitives."""

from __future__ import annotations

import torch

from torchcrop.functions import interpolate
from torchcrop.functions.smoothing import smooth_step, soft_clamp


def test_gradcheck_interpolate():
    table = torch.tensor(
        [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 5.0]], dtype=torch.float64
    )
    x = torch.tensor([0.3, 1.1, 2.5], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda xi: interpolate(table, xi), (x,), eps=1e-6, atol=1e-4)


def test_gradcheck_smooth_step():
    x = torch.tensor([-0.5, 0.1, 0.7], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda xi: smooth_step(xi, 0.0, 5.0), (x,), eps=1e-6, atol=1e-4)


def test_gradcheck_soft_clamp():
    x = torch.tensor([-0.5, 0.5, 1.5], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda xi: soft_clamp(xi, 0.0, 1.0, 5.0), (x,), eps=1e-6, atol=1e-4)
