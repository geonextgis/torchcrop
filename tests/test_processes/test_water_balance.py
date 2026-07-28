"""Tests for the two-zone soil water balance (SIMPLACE port)."""

from __future__ import annotations

import pytest
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
    # Rain infiltrates into the rooted zone, which starts well below field
    # capacity, so its storage must rise.
    assert out["wa_rate"].item() > 0
    # The two zones combined need not gain: the default lower zone starts
    # above field capacity and drains at KSUB regardless of the rain, so
    # only the per-zone rate carries the intended meaning here.
    assert out["runoff"].item() == pytest.approx(0.0, abs=1e-6)


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
        iairdu=torch.tensor(0.0),  # crop trait — non-aquatic
    )
    assert out["rwet"].item() < 1.0


def test_root_front_transfers_water_from_lower_zone():
    wb = WaterBalance()
    params = SoilParameters()
    state = ModelState.initial(batch_size=1, rootdi=0.3, wa_lower_i=200.0)
    out = wb(
        state,
        rain=torch.zeros(1),
        pevap=torch.zeros(1),
        ptran=torch.tensor([1.0]),
        params=params,
        rdm=_rdm(params, state),
        rr_lag=torch.tensor([0.01]),  # prior-day root front (drives WDR)
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


def test_table_irrigation_holds_over_excess_above_delivery_limit():
    """A 20 mm scheduled application is delivered as 10 mm on two days.

    The system applies at most 10 mm d⁻¹; the remainder must be carried in
    ``ModelState.dirro`` and applied the next day rather than discarded.
    """
    wb = WaterBalance()
    params = SoilParameters()
    params.irri = torch.tensor(2.0)
    # 20 mm scheduled on DOY 100, bracketed by zeros so the piecewise-linear
    # look-up returns a single-day spike.
    params.irrtab = torch.tensor(
        [[0.0, 0.0], [99.0, 0.0], [100.0, 20.0], [101.0, 0.0], [365.0, 0.0]]
    )
    state = ModelState.initial(batch_size=1, rootdi=0.5)
    kwargs = dict(
        rain=torch.zeros(1),
        pevap=torch.zeros(1),
        ptran=torch.tensor([1.0]),
        params=params,
        rdm=_rdm(params, state),
    )

    day100 = wb(state, doy=torch.tensor([100.0]), **kwargs)
    assert day100["rirr"].item() == pytest.approx(10.0)
    assert day100["dirro_next"].item() == pytest.approx(10.0)

    # Carry the surplus forward; the schedule itself is empty on DOY 101.
    state = state.replace(dirro=day100["dirro_next"])
    day101 = wb(state, doy=torch.tensor([101.0]), **kwargs)
    assert day101["rirr"].item() == pytest.approx(10.0)
    assert day101["dirro_next"].item() == pytest.approx(0.0)

    # Full scheduled depth reached the soil.
    assert day100["rirr"].item() + day101["rirr"].item() == pytest.approx(20.0)


def test_root_zone_above_field_capacity_drains_to_lower_zone():
    """Water stored above field capacity is released, not stranded.

    The storage headroom ``CAP`` is signed: when the rooted zone holds more
    than field capacity it is negative and the excess percolates down until
    the zone relaxes to field capacity.
    """
    wb = WaterBalance()
    params = SoilParameters()
    state = ModelState.initial(batch_size=1, rootdi=0.5, wa_lower_i=100.0)
    # Rooted zone midway between field capacity and saturation.
    wfc = 1000.0 * params.wcfc * state.rootd
    wst = 1000.0 * params.wcst * state.rootd
    state = state.replace(wa=0.5 * (wfc + wst))
    out = wb(
        state,
        rain=torch.zeros(1),
        pevap=torch.zeros(1),
        ptran=torch.zeros(1),
        params=params,
        rdm=_rdm(params, state),
    )
    # Nothing enters or leaves the profile today, so the whole excess above
    # field capacity moves down (it is well inside the KSUB + CAPL0 limit)
    # and the rooted zone lands exactly on field capacity.
    excess = float(state.wa - wfc)
    assert out["perc2"].item() == pytest.approx(excess, abs=1e-3)
    assert out["wa_rate"].item() == pytest.approx(-excess, abs=1e-3)
    assert (state.wa + out["wa_rate"]).item() == pytest.approx(float(wfc), abs=1e-3)
    assert out["wbal"].item() == pytest.approx(0.0, abs=1e-4)


def test_automatic_irrigation_only_runs_while_a_crop_is_present():
    """Demand-driven irrigation serves a crop, so a bare field gets none.

    The soil is dry enough to trigger in every case; only the presence of
    a crop decides whether water is applied.
    """
    wb = WaterBalance()
    params = SoilParameters()
    params.irri = torch.tensor(1.0)
    state = ModelState.initial(batch_size=1, rootdi=0.5)
    wwp = 1000.0 * params.wcwp * state.rootd
    state = state.replace(wa=wwp + 1.0)
    kwargs = dict(
        rain=torch.zeros(1),
        pevap=torch.tensor([1.0]),
        ptran=torch.tensor([2.0]),
        params=params,
        rdm=_rdm(params, state),
    )

    # Before sowing and after maturity — no crop, no irrigation.
    assert wb(state, crop_present=torch.zeros(1), **kwargs)["rirr"].item() == 0.0
    # Between sowing and maturity — irrigation runs.
    assert wb(state, crop_present=torch.ones(1), **kwargs)["rirr"].item() > 0.0
    # A standalone call that says nothing about the crop keeps irrigating,
    # so the module stays usable on its own.
    assert wb(state, **kwargs)["rirr"].item() > 0.0


def test_table_irrigation_is_not_gated_by_crop_presence():
    """A scheduled application is management, not a response to the crop."""
    wb = WaterBalance()
    params = SoilParameters()
    params.irri = torch.tensor(2.0)
    params.irrtab = torch.tensor(
        [[0.0, 0.0], [99.0, 0.0], [100.0, 8.0], [101.0, 0.0], [365.0, 0.0]]
    )
    state = ModelState.initial(batch_size=1, rootdi=0.5)
    out = wb(
        state,
        rain=torch.zeros(1),
        pevap=torch.zeros(1),
        ptran=torch.tensor([1.0]),
        params=params,
        rdm=_rdm(params, state),
        doy=torch.tensor([100.0]),
        crop_present=torch.zeros(1),  # pre-sowing seedbed irrigation
    )
    assert out["rirr"].item() == pytest.approx(8.0)


def test_potential_mode_transpires_at_the_unreduced_rate():
    """``iopt = 1`` bypasses RDRY/RWET and the available-water ceiling.

    SIMPLACE ``WaterBalance.java`` special-cases potential production with
    ``ActualTranspiration = PotentialTranspiration``, so the crop transpires
    at the full potential rate and ``TRANRF`` is identically ``1`` even on a
    soil dried to wilting point — where the water-limited branch would give
    ``TRANRF`` close to ``0``.
    """
    wb = WaterBalance()
    params = SoilParameters()
    state = ModelState.initial(batch_size=1, rootdi=0.4)
    # Dry the rooted zone down to wilting point: no available water at all.
    state = state.replace(wa=1000.0 * params.wcwp * state.rootd)
    ptran = torch.tensor([4.0])
    kwargs = dict(
        state=state,
        rain=torch.zeros(1),
        pevap=torch.zeros(1),
        ptran=ptran,
        params=params,
        rdm=_rdm(params, state),
    )

    limited = wb(**kwargs, iopt=torch.tensor(2.0))
    assert limited["tran"].item() < 1e-6
    assert limited["tranrf"].item() < 1e-6

    potential = wb(**kwargs, iopt=torch.tensor(1.0))
    assert torch.allclose(potential["tran"], ptran)
    assert torch.allclose(potential["tranrf"], torch.ones(1))
    # The soil balance itself still runs: the extra uptake is drawn from the
    # root zone, which is what lets a potential-mode profile dry down.
    assert potential["wa_rate"].item() < limited["wa_rate"].item()


def test_potential_mode_defaults_to_water_limited_behaviour():
    """Omitting ``iopt`` keeps the reducing branch (backward compatible)."""
    wb = WaterBalance()
    params = SoilParameters()
    state = ModelState.initial(batch_size=1, rootdi=0.4)
    state = state.replace(wa=1000.0 * params.wcwp * state.rootd)
    kwargs = dict(
        state=state,
        rain=torch.zeros(1),
        pevap=torch.zeros(1),
        ptran=torch.tensor([4.0]),
        params=params,
        rdm=_rdm(params, state),
    )
    default = wb(**kwargs)["tran"]
    explicit = wb(**kwargs, iopt=torch.tensor(2.0))["tran"]
    assert torch.equal(default, explicit)


def test_potential_mode_is_selected_per_batch_element():
    """``iopt`` is honoured elementwise, not collapsed to a single branch."""
    wb = WaterBalance()
    params = SoilParameters()
    state = ModelState.initial(batch_size=2, rootdi=0.4)
    state = state.replace(wa=1000.0 * params.wcwp * state.rootd)
    ptran = torch.tensor([4.0, 4.0])
    out = wb(
        state=state,
        rain=torch.zeros(2),
        pevap=torch.zeros(2),
        ptran=ptran,
        params=params,
        rdm=_rdm(params, state),
        iopt=torch.tensor([1.0, 2.0]),
    )
    assert torch.allclose(out["tranrf"], torch.tensor([1.0, 0.0]), atol=1e-6)
