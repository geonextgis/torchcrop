"""Tests for the differentiable interpolation primitive."""

from __future__ import annotations

import torch

from torchcrop.functions import interpolate


def test_interpolate_exact_breakpoints():
    table = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
    x = torch.tensor([0.0, 1.0, 2.0])
    y = interpolate(table, x)
    assert torch.allclose(y, torch.tensor([0.0, 2.0, 4.0]))


def test_interpolate_linear_midpoint():
    table = torch.tensor([[0.0, 0.0], [2.0, 4.0]])
    x = torch.tensor([1.0])
    y = interpolate(table, x)
    assert torch.allclose(y, torch.tensor([2.0]))


def test_interpolate_flat_extrapolation():
    table = torch.tensor([[0.0, 1.0], [1.0, 3.0]])
    x = torch.tensor([-1.0, 2.0])
    y = interpolate(table, x)
    assert torch.allclose(y, torch.tensor([1.0, 3.0]))


def test_interpolate_gradient_flows():
    table = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], dtype=torch.float64)
    x = torch.tensor([0.5, 1.5], dtype=torch.float64, requires_grad=True)
    y = interpolate(table, x).sum()
    y.backward()
    # dy/dx = slope = 2 for both points
    assert torch.allclose(x.grad, torch.tensor([2.0, 2.0], dtype=torch.float64))


def test_interpolate_batched_table():
    table = torch.stack([
        torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        torch.tensor([[0.0, 0.0], [1.0, 2.0]]),
    ])
    x = torch.tensor([0.5, 0.5])
    y = interpolate(table, x)
    assert torch.allclose(y, torch.tensor([0.5, 1.0]))
