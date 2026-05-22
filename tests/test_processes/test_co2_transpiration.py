"""Tests for the CO2-influence-on-transpiration process module.

Mirrors the SIMPLACE ``Co2InfluenceOnTranspiration.java`` component:
``PTRAN_reduced = PTRAN * (m * CO2 + b)``.
"""

from __future__ import annotations

import torch

from torchcrop.processes.co2_transpiration import Co2Transpiration


def test_factor_matches_linear_formula():
    ct = Co2Transpiration(transpiration_m=-0.0003, transpiration_b=1.1)
    co2 = torch.tensor([350.0, 700.0])
    expected = -0.0003 * co2 + 1.1
    assert torch.allclose(ct.factor(co2), expected)


def test_elevated_co2_reduces_transpiration():
    """With the default slope, higher CO2 must give a smaller factor."""
    ct = Co2Transpiration()
    f_low = ct.factor(torch.tensor(350.0))
    f_high = ct.factor(torch.tensor(700.0))
    assert f_high < f_low
    assert torch.isclose(f_low, torch.tensor(0.995), atol=1e-6)


def test_forward_shapes_and_scaling():
    ct = Co2Transpiration()
    ptran = torch.tensor([2.0, 3.0, 4.0])
    out = ct(ptran, co2=torch.tensor(420.0))
    assert out["ptran"].shape == (3,)
    assert out["co2_factor"].shape == (3,)
    assert torch.allclose(out["ptran"], ptran * ct.factor(torch.tensor(420.0)))


def test_factor_clamped_nonnegative():
    """An extreme CO2 value cannot flip transpiration negative."""
    ct = Co2Transpiration(transpiration_m=-0.0003, transpiration_b=1.1)
    # b/|m| = 3666 ppm is the zero-crossing; beyond it the raw factor is < 0.
    assert torch.allclose(ct.factor(torch.tensor(1e5)), torch.tensor(0.0))


def test_gradcheck():
    ct = Co2Transpiration()
    ptran = torch.tensor([1.5, 2.5], dtype=torch.float64, requires_grad=True)
    co2 = torch.tensor(410.0, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda p, c: ct(p, c)["ptran"], (ptran, co2), eps=1e-6, atol=1e-4
    )
