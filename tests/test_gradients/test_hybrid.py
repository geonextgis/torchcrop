"""Tests for the constraint-aware neural residual framework."""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop import Lintul5Model
from torchcrop.nn import HybridManager, ResidualHead, ResidualSpec, default_slots
from torchcrop.utils.io import make_constant_weather


def _randomize(module: nn.Module) -> None:
    """Fill every parameter with non-trivial values (breaks zero-init)."""
    for p in module.parameters():
        nn.init.normal_(p, mean=0.0, std=1.0)


# --------------------------------------------------------------------------- #
# Constraint projections
# --------------------------------------------------------------------------- #


def test_rate_factor_is_positive_and_identity_at_init():
    spec = ResidualSpec("x", "rate_factor", context=("a", "b"), scale=0.2)
    head = ResidualHead(spec)
    base = torch.tensor([0.0, 1.0, 5.0, 100.0])
    ctx = torch.randn(4, 2)
    value, delta = head(base, ctx)
    # Zero-init MLP -> delta == 0 -> exp(0) == 1 -> exact identity.
    assert torch.allclose(value, base)
    assert torch.allclose(delta, torch.zeros_like(delta))
    # With random weights the multiplicative correction stays non-negative
    # and bounded by exp(±scale).
    _randomize(head)
    value, _ = head(base, ctx)
    assert (value >= 0).all()
    assert (value <= base * torch.exp(torch.tensor(0.2)) + 1e-6).all()


def test_unit_interval_stays_in_open_unit_and_identity_at_init():
    spec = ResidualSpec("x", "unit_interval", context=("a",), scale=0.3)
    head = ResidualHead(spec)
    base = torch.tensor([0.0, 0.25, 0.5, 0.9, 1.0])
    ctx = torch.randn(5, 1)
    value, _ = head(base, ctx)
    # logit/sigmoid round-trip is the identity on the clamped interior;
    # the 0 and 1 endpoints map to the 1e-6 clamp, so use a loose tol.
    assert torch.allclose(value, base, atol=1e-5)
    _randomize(head)
    value, _ = head(base, ctx)
    assert (value > 0).all() and (value < 1).all()


def test_simplex_sums_to_one():
    spec = ResidualSpec("x", "simplex", context=("a",), output_dim=3, scale=0.4)
    head = ResidualHead(spec)
    base = torch.tensor([[0.3, 0.5, 0.2], [0.0, 0.7, 0.3], [1.0, 0.0, 0.0]])
    ctx = torch.randn(3, 1)
    value, _ = head(base, ctx)
    assert torch.allclose(value.sum(-1), torch.ones(3), atol=1e-6)
    # Conservation must hold even with arbitrary (trained) weights.
    _randomize(head)
    value, _ = head(base, ctx)
    assert torch.allclose(value.sum(-1), torch.ones(3), atol=1e-6)
    assert (value >= 0).all()


# --------------------------------------------------------------------------- #
# Model integration
# --------------------------------------------------------------------------- #


def test_no_specs_matches_baseline_exactly():
    weather = make_constant_weather(batch_size=2, n_days=60, dtype=torch.float64)
    baseline = Lintul5Model().double()
    hybrid = Lintul5Model(residual_specs=[]).double()
    out_b = baseline(weather, start_doy=60)
    out_h = hybrid(weather, start_doy=60)
    assert torch.allclose(out_b.yield_, out_h.yield_)
    assert torch.allclose(out_b.biomass, out_h.biomass)


def test_zero_init_slots_reproduce_mechanistic_trajectory():
    weather = make_constant_weather(batch_size=2, n_days=60, dtype=torch.float64)
    baseline = Lintul5Model().double()
    # All default slots, freshly constructed (zero-init) -> identity maps,
    # so the full hybrid model reproduces the mechanistic trajectory. The
    # simplex projection carries a ~1e-8 bias, hence the loose tolerance.
    hybrid = Lintul5Model(residual_specs=default_slots()).double()
    out_b = baseline(weather, start_doy=60)
    out_h = hybrid(weather, start_doy=60)
    assert torch.allclose(out_b.yield_, out_h.yield_, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_b.lai, out_h.lai, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out_b.dvs, out_h.dvs, atol=1e-5, rtol=1e-5)


def test_random_weights_stay_finite():
    weather = make_constant_weather(batch_size=2, n_days=80, dtype=torch.float32)
    model = Lintul5Model(residual_specs=default_slots())
    _randomize(model.hybrid)
    out = model(weather, start_doy=60)
    assert torch.isfinite(out.yield_).all()
    assert torch.isfinite(out.lai).all()
    assert (out.lai >= 0).all()


def test_gradient_flows_to_residual_head():
    weather = make_constant_weather(batch_size=1, n_days=80, dtype=torch.float64)
    spec = ResidualSpec(
        "photosynthesis.gtotal",
        "rate_factor",
        context=("lai", "dvs", "davtmp", "tranrf", "nstress"),
        scale=0.15,
    )
    model = Lintul5Model(residual_specs=[spec]).double()
    out = model(weather, start_doy=60)
    out.biomass.sum().backward()
    # The final (zero-initialised) layer still receives a non-trivial
    # gradient on the first step, proving the chain is connected.
    last_layer = model.hybrid.head("photosynthesis.gtotal").net[-1]
    assert last_layer.weight.grad is not None
    assert last_layer.weight.grad.abs().sum().item() > 0.0


def test_penalty_accumulates_in_training_and_resets():
    weather = make_constant_weather(batch_size=2, n_days=40, dtype=torch.float32)
    model = Lintul5Model(residual_specs=default_slots())
    _randomize(model.hybrid)
    model.train()
    model(weather, start_doy=60)
    penalty = model.hybrid.penalty()
    assert penalty.item() > 0.0
    # A fresh forward resets the accumulator (no unbounded growth).
    model(weather, start_doy=60)
    assert torch.isfinite(model.hybrid.penalty())
    # In eval mode nothing is accumulated.
    model.eval()
    model(weather, start_doy=60)
    assert model.hybrid.penalty().item() == 0.0
