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


def test_batch_consistency_per_element_params():
    """Batch-varying soil/site/crop params match the per-location loop.

    A per-site dataloader supplies ``[B]`` soil/site parameters (one value
    per batch element). Running all elements together must reproduce, to
    floating-point tolerance, the result of running each element on its own
    with the corresponding scalar parameters — i.e. the batched
    ``initialize`` builds a genuinely per-element initial state rather than
    broadcasting element 0 across the batch.
    """
    from torchcrop.parameters.site_params import SiteParameters
    from torchcrop.parameters.soil_params import SoilParameters

    b = 3
    weather = make_constant_weather(batch_size=b, n_days=80, dtype=torch.float64)

    # Per-element parameters: each batch element gets a distinct value so a
    # broadcast bug (using element 0 for all) would change the answer.
    wci = torch.tensor([0.20, 0.30, 0.36], dtype=torch.float64)
    wcwp = torch.tensor([0.10, 0.12, 0.11], dtype=torch.float64)
    wcfc = torch.tensor([0.36, 0.36, 0.36], dtype=torch.float64)
    rdmso = torch.tensor([1.10, 1.20, 1.30], dtype=torch.float64)
    lat = torch.tensor([50.0, 52.0, 54.0], dtype=torch.float64)

    soil = SoilParameters(wci=wci, wcwp=wcwp, wcfc=wcfc, rdmso=rdmso)
    site = SiteParameters(latitude=lat)
    model = Lintul5Model(soil_params=soil, site_params=site).double()

    out_batch = model(weather, start_doy=60)
    assert torch.isfinite(out_batch.yield_).all()
    # The distinct initial water contents must produce distinct states.
    assert not torch.allclose(out_batch.yield_[0], out_batch.yield_[1])

    for i in range(b):
        soil_i = SoilParameters(
            wci=wci[i], wcwp=wcwp[i], wcfc=wcfc[i], rdmso=rdmso[i]
        )
        site_i = SiteParameters(latitude=lat[i])
        model_i = Lintul5Model(soil_params=soil_i, site_params=site_i).double()
        out_i = model_i(weather.data[i : i + 1], start_doy=60)
        assert torch.allclose(out_batch.yield_[i], out_i.yield_[0], atol=1e-8)
        assert torch.allclose(out_batch.lai[i], out_i.lai[0], atol=1e-8)
        assert torch.allclose(out_batch.dvs[i], out_i.dvs[0], atol=1e-8)


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
