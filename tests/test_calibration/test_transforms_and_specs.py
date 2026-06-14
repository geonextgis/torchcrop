"""Unit tests for calibration transforms and spec validation."""

from __future__ import annotations

import pytest
import torch

from torchcrop.calibration import (
    ConstraintGroup,
    ParameterSpec,
    available_transforms,
    build_transform,
    round_ste,
)


def test_affine_sigmoid_stays_in_bounds_and_roundtrips():
    bij = build_transform("affine_sigmoid", (1000.0, 2000.0))
    z = torch.linspace(-8, 8, 50, dtype=torch.float64)
    v = bij.forward(z)
    assert (v > 1000.0).all() and (v < 2000.0).all()
    # inverse(forward(z)) == z
    assert torch.allclose(bij.inverse(v), z, atol=1e-6)


def test_softplus_lower_bound_and_roundtrip():
    bij = build_transform("softplus", (0.5, 0.0))  # lo=0.5; hi ignored
    z = torch.linspace(-5, 5, 30, dtype=torch.float64)
    v = bij.forward(z)
    assert (v > 0.5).all()
    assert torch.allclose(bij.forward(bij.inverse(v)), v, atol=1e-6)


def test_round_ste_forward_integral_backward_identity():
    z = torch.tensor(3.7, requires_grad=True)
    y = round_ste(z)
    assert y.item() == 4.0
    y.backward()
    assert z.grad.item() == pytest.approx(1.0)


def test_available_transforms_registered():
    names = available_transforms()
    for n in ("identity", "affine_sigmoid", "softplus", "log"):
        assert n in names


def test_spec_rejects_bad_bounds_and_init():
    with pytest.raises(ValueError):
        ParameterSpec("crop.tsum1", bounds=(2000.0, 1000.0))
    with pytest.raises(ValueError):
        ParameterSpec("crop.tsum1", bounds=(1000.0, 2000.0), init=50.0)


def test_categorical_requires_categories():
    with pytest.raises(ValueError):
        ParameterSpec("crop.iopt", kind="categorical")
    ParameterSpec("crop.iopt", kind="categorical", categories=(1.0, 2.0))


def test_constraint_group_validates_members():
    with pytest.raises(ValueError):
        ConstraintGroup(("a",), order="ascending")
    with pytest.raises(ValueError):
        ConstraintGroup(("a", "a"), order="ascending")
