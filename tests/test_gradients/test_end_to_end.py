"""End-to-end gradient tests for the full Lintul5 model."""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop import Lintul5Model
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.utils.io import make_constant_weather


def test_forward_runs_and_shapes_match():
    weather = make_constant_weather(batch_size=2, n_days=60, dtype=torch.float32)
    model = Lintul5Model()
    out = model(weather, start_doy=60)
    assert out.lai.shape == (2, 61)
    assert out.dvs.shape == (2, 61)
    assert out.yield_.shape == (2,)
    assert torch.isfinite(out.yield_).all()


def test_gradient_wrt_rue():
    weather = make_constant_weather(batch_size=1, n_days=80, dtype=torch.float64)
    crop_params = CropParameters().to(dtype=torch.float64)
    # RUE is now sourced from ruetb(DVS) per SIMPLACE RadiationUseEfficiency
    crop_params.ruetb = nn.Parameter(crop_params.ruetb.clone().detach())
    model = Lintul5Model(crop_params=crop_params)
    model = model.double()
    out = model(weather, start_doy=60)
    # Use total biomass (wlv + wst + wso) as the gradient sink: it is
    # non-zero whenever any growth happens, regardless of whether DVS
    # crossed anthesis (which gates storage-organ allocation) within the
    # 80-day window.
    out.biomass.sum().backward()
    assert crop_params.ruetb.grad is not None
    assert torch.isfinite(crop_params.ruetb.grad).all()
    assert crop_params.ruetb.grad.abs().sum().item() > 0.0


def test_gradient_wrt_tsum1():
    weather = make_constant_weather(batch_size=1, n_days=80, dtype=torch.float64)
    crop_params = CropParameters().to(dtype=torch.float64)
    crop_params.tsum1 = nn.Parameter(torch.tensor(900.0, dtype=torch.float64))
    model = Lintul5Model(crop_params=crop_params).double()
    out = model(weather, start_doy=60)
    # DVS trajectory depends on tsum1 — gradient should exist
    out.dvs.sum().backward()
    assert crop_params.tsum1.grad is not None
    assert torch.isfinite(crop_params.tsum1.grad)


def test_batch_consistency():
    weather = make_constant_weather(batch_size=3, n_days=50, dtype=torch.float32)
    model = Lintul5Model()
    out_batch = model(weather, start_doy=60)
    for i in range(weather.batch_size):
        single_w = weather.data[i : i + 1]
        out_single = model(single_w, start_doy=60)
        assert torch.allclose(out_batch.yield_[i], out_single.yield_[0], atol=1e-5)
        assert torch.allclose(out_batch.lai[i], out_single.lai[0], atol=1e-5)


def test_external_fertilizer_changes_soil_nitrogen():
    """An external [B, T, 3] fertiliser driver feeds the inorganic N pool.

    Applying N on a span of days must raise the ``nmint`` trajectory
    relative to the default (table-driven) run, with the recovery
    fraction ``nrf`` applied to the raw amount.
    """
    weather = make_constant_weather(batch_size=2, n_days=60, dtype=torch.float64)
    model = Lintul5Model().double()

    out0 = model(weather, start_doy=60)

    fert = torch.zeros(2, 60, 3, dtype=torch.float64)
    fert[:, 10:15, 0] = 2.0  # 2 g N m-2 d-1 on days 10..14, N channel only
    out1 = model(weather, start_doy=60, fertilizer=fert)

    nmint0 = torch.stack([s.nmint for s in out0.states], dim=1)
    nmint1 = torch.stack([s.nmint for s in out1.states], dim=1)
    # Inorganic N rises; P/K pools are untouched (only the N channel set).
    assert (nmint1 - nmint0).abs().max() > 1e-9
    assert (nmint1 >= nmint0 - 1e-9).all()
    pmint0 = torch.stack([s.pmint for s in out0.states], dim=1)
    pmint1 = torch.stack([s.pmint for s in out1.states], dim=1)
    assert torch.allclose(pmint0, pmint1)


def test_gradient_wrt_external_fertilizer():
    """Yield/state gradients flow back to the fertiliser driver."""
    weather = make_constant_weather(batch_size=1, n_days=60, dtype=torch.float64)
    model = Lintul5Model().double()

    fert = torch.zeros(1, 60, 3, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        fert[:, 10:15, :] = 2.0
    out = model(weather, start_doy=60, fertilizer=fert)

    # Differentiate a target directly downstream of the fertiliser input
    # (the accumulated inorganic N pool) so connectivity is exercised
    # regardless of whether N is the yield-limiting factor.
    torch.stack([s.nmint for s in out.states], dim=1).sum().backward()
    assert fert.grad is not None
    assert torch.isfinite(fert.grad).all()
    # The N channel on applied days carries the recovery fraction (nrf).
    assert fert.grad[:, 10:15, 0].abs().sum() > 0.0


def test_external_fertilizer_shape_validation():
    """A mis-shaped fertiliser driver is rejected before the run."""
    import pytest

    weather = make_constant_weather(batch_size=2, n_days=60, dtype=torch.float64)
    model = Lintul5Model().double()
    with pytest.raises(ValueError, match=r"fertilizer must have shape"):
        model(weather, start_doy=60, fertilizer=torch.zeros(2, 60, 2))
