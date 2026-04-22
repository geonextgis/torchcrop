"""Tests for the two-zone soil water balance (SIMPLACE port)."""

from __future__ import annotations

import torch

from torchcrop.parameters.soil_params import SoilParameters
from torchcrop.processes.water_balance import WaterBalance
from torchcrop.states.model_state import ModelState


def _rdm(params: SoilParameters, state: ModelState, rdmcr: float = 1.2) -> torch.Tensor:
    return torch.full_like(state.rootd, min(float(params.rdmso), rdmcr))


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
        rdm=_rdm(params, state),
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
        rdm=_rdm(params, state),
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
        rdm=_rdm(params, state),
    )
    # Net change across both zones should be positive (net gain from rain)
    total = out["wa_rate"] + out["wa_lower_rate"]
    assert total.item() > 0


def test_mass_balance_residual_near_zero():
    """Water balance should be mass-conservative each day."""
    wb = WaterBalance()
    params = SoilParameters()
    state = ModelState.initial(batch_size=3, rootdi=0.4)
    out = wb(
        state,
        rain=torch.tensor([5.0, 12.0, 0.0]),
        pevap=torch.tensor([1.5, 2.0, 0.8]),
        ptran=torch.tensor([2.0, 3.0, 1.0]),
        params=params,
        rdm=_rdm(params, state),
    )
    assert torch.allclose(out["wbal"], torch.zeros(3), atol=1e-4)


def test_dslr_resets_on_heavy_rain():
    wb = WaterBalance()
    params = SoilParameters()
    state = ModelState.initial(batch_size=2, rootdi=0.4, dslri=5.0)
    out = wb(
        state,
        rain=torch.tensor([10.0, 0.0]),  # heavy / dry
        pevap=torch.tensor([2.0, 2.0]),
        ptran=torch.tensor([1.0, 1.0]),
        params=params,
        rdm=_rdm(params, state),
    )
    new_dslr = state.dslr + out["dslr_rate"]
    # batch 0 → heavy rain, DSLR → 1; batch 1 → dry, DSLR → 6
    assert torch.allclose(new_dslr, torch.tensor([1.0, 6.0]), atol=1e-6)


def test_dsos_accumulates_under_saturation():
    """DSOS should grow when the soil is above SMAIR."""
    wb = WaterBalance()
    params = SoilParameters()
    state = ModelState.initial(batch_size=1, rootdi=0.4)
    # Saturate the rooted zone
    wst = 1000.0 * params.wcst * state.rootd
    state = state.replace(wa=wst, dsos=torch.zeros(1))
    out = wb(
        state,
        rain=torch.zeros(1),
        pevap=torch.zeros(1),
        ptran=torch.tensor([1.0]),
        params=params,
        rdm=_rdm(params, state),
    )
    assert (state.dsos + out["dsos_rate"]).item() > 0.0


def test_oxygen_stress_reduces_tranrf_for_non_rice():
    """Saturated, non-rice crop → TRANRF < 1 via RWET."""
    wb = WaterBalance()
    params = SoilParameters()
    # Ensure non-aquatic
    params.iairdu = torch.tensor(0.0)
    state = ModelState.initial(batch_size=1, rootdi=0.4)
    # Soak the rooted zone
    wst = 1000.0 * params.wcst * state.rootd
    state = state.replace(wa=wst, dsos=torch.tensor([4.0]))  # max DSOS
    out = wb(
        state,
        rain=torch.zeros(1),
        pevap=torch.zeros(1),
        ptran=torch.tensor([3.0]),
        params=params,
        rdm=_rdm(params, state),
    )
    assert out["rwet"].item() < 1.0


def test_root_front_transfers_water_from_lower_zone():
    wb = WaterBalance()
    params = SoilParameters()
    state = ModelState.initial(batch_size=1, rootdi=0.3, wa_lower_i=200.0)
    rr = torch.tensor([0.01])  # 1 cm d⁻¹ root growth
    out = wb(
        state,
        rain=torch.zeros(1),
        pevap=torch.zeros(1),
        ptran=torch.tensor([1.0]),
        params=params,
        rdm=_rdm(params, state),
        rr=rr,
    )
    # WDR should be positive (water moves from lower → rooted zone)
    assert out["wdr"].item() > 0
    # Lower-zone rate takes the hit, rooted-zone rate benefits
    assert out["wa_lower_rate"].item() < 0
    assert out["wa_rate"].item() > 0


def test_automatic_irrigation_triggered_below_smcr():
    wb = WaterBalance()
    params = SoilParameters()
    # IRRI = 1 (automatic)
    params.irri = torch.tensor(1.0)
    state = ModelState.initial(batch_size=1, rootdi=0.5)
    # Set WA just above wilting point → below SMCR
    wwp = 1000.0 * params.wcwp * state.rootd
    state = state.replace(wa=wwp + 1.0)
    out = wb(
        state,
        rain=torch.tensor([0.0]),
        pevap=torch.tensor([1.0]),
        ptran=torch.tensor([2.0]),
        params=params,
        rdm=_rdm(params, state),
    )
    assert out["rirr"].item() > 0.0


def test_runoff_captures_saturation_excess():
    wb = WaterBalance()
    params = SoilParameters()
    params.runfr = torch.tensor(0.1)
    state = ModelState.initial(batch_size=1, rootdi=0.2, wa_lower_i=500.0)
    # Near-saturate the rooted zone
    wst = 1000.0 * params.wcst * state.rootd
    state = state.replace(wa=wst * 0.99)
    out = wb(
        state,
        rain=torch.tensor([100.0]),  # massive input
        pevap=torch.tensor([0.0]),
        ptran=torch.tensor([0.1]),
        params=params,
        rdm=_rdm(params, state),
    )
    # Runoff must exceed the preliminary runfr * rain share
    assert out["runoff"].item() > 0.1 * 100.0
