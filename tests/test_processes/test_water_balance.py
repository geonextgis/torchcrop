"""Tests for the soil water balance."""

from __future__ import annotations

import torch

from torchcrop.parameters.soil_params import SoilParameters
from torchcrop.processes.water_balance import WaterBalance
from torchcrop.states.model_state import ModelState


def test_tranrf_zero_at_wilting_point():
    wb = WaterBalance()
    params = SoilParameters()
    state = ModelState.initial(batch_size=2, rootdi=0.5)
    # Force WA to wilting-point content
    wwp = 1000.0 * params.wcwp * state.rootd
    state = state.replace(wa=wwp)
    out = wb(
        state,
        rain=torch.zeros(2),
        pevap=torch.zeros(2),
        ptran=torch.tensor([3.0, 3.0]),
        params=params,
    )
    assert torch.allclose(out["tranrf"], torch.zeros(2), atol=1e-6)
    assert torch.allclose(out["tran"], torch.zeros(2), atol=1e-6)


def test_tranrf_one_at_field_capacity():
    wb = WaterBalance()
    params = SoilParameters()
    state = ModelState.initial(batch_size=2, rootdi=0.5)
    wfc = 1000.0 * params.wcfc * state.rootd
    state = state.replace(wa=wfc)
    out = wb(
        state,
        rain=torch.zeros(2),
        pevap=torch.zeros(2),
        ptran=torch.tensor([3.0, 3.0]),
        params=params,
    )
    assert torch.allclose(out["tranrf"], torch.ones(2), atol=1e-6)


def test_rain_increases_water():
    wb = WaterBalance()
    params = SoilParameters()
    state = ModelState.initial(batch_size=1, rootdi=0.5, wai=60.0)
    out = wb(
        state,
        rain=torch.tensor([10.0]),
        pevap=torch.tensor([1.0]),
        ptran=torch.tensor([2.0]),
        params=params,
    )
    # wa_rate should be positive (net gain)
    assert out["wa_rate"].item() > 0
