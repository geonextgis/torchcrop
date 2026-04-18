"""Tests for FST-style primitives."""

from __future__ import annotations

import torch

from torchcrop.functions import insw, limit, notnul, soft_clamp, soft_if


def test_limit_basic():
    x = torch.tensor([-1.0, 0.5, 2.0])
    assert torch.allclose(limit(0.0, 1.0, x), torch.tensor([0.0, 0.5, 1.0]))


def test_insw_hard():
    x = torch.tensor([-1.0, 0.0, 1.0])
    y1 = torch.tensor([10.0, 10.0, 10.0])
    y2 = torch.tensor([20.0, 20.0, 20.0])
    # x >= 0 → y2, else y1
    assert torch.allclose(insw(x, y1, y2), torch.tensor([10.0, 20.0, 20.0]))


def test_insw_smooth_gradients():
    x = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    y1 = torch.tensor(1.0, dtype=torch.float64)
    y2 = torch.tensor(2.0, dtype=torch.float64)
    y = insw(x, y1, y2, smooth=True, k=10.0)
    y.backward()
    assert x.grad is not None and x.grad.abs() > 0


def test_notnul():
    x = torch.tensor([0.0, 1e-15, 0.5])
    y = notnul(x)
    # zero-like values replaced by 1 for safe division
    assert y[0] == 1.0 and y[1] == 1.0 and y[2] == 0.5


def test_soft_clamp_bounds():
    x = torch.tensor([-5.0, 0.5, 10.0])
    y = soft_clamp(x, 0.0, 1.0, k=20.0)
    assert (y >= -0.05).all() and (y <= 1.05).all()


def test_soft_if_blend():
    cond = torch.tensor(0.0)
    a = torch.tensor(1.0)
    b = torch.tensor(0.0)
    y = soft_if(cond, a, b, k=1.0)
    assert torch.isclose(y, torch.tensor(0.5), atol=1e-4)
