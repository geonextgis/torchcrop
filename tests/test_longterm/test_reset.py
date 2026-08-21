"""Harvest must leave a clean, physically consistent field behind.

The reset is the load-bearing piece of the long-term framework: it is
what stops the thermal clocks, the standing dead canopy and the mature
root zone from leaking into next year's crop, and it must do so without
inventing or destroying soil water.
"""

from __future__ import annotations

import torch

from torchcrop import CropParameters, Lintul5Model, SiteParameters, SoilParameters
from torchcrop.longterm import CarryOverPolicy, CropCalendar, LongTermSimulator
from torchcrop.longterm.carryover import (
    _ACCUMULATOR_FIELDS,
    _CROP_ONE_FIELDS,
    _CROP_ZERO_FIELDS,
    reset_at_harvest,
)
from torchcrop.states.model_state import ModelState
from torchcrop.utils.io import make_constant_weather


def _mature_state(batch_size: int = 3) -> ModelState:
    """A stand-in for a ripe crop on a partly dried soil."""
    state = ModelState.initial(
        batch_size=batch_size, wai=250.0, rootdi=1.25, wa_lower_i=180.0
    )
    return state.replace(
        dvs=torch.full((batch_size,), 2.0),
        tsum=torch.full((batch_size,), 2100.0),
        tsump=torch.full((batch_size,), 2200.0),
        vern=torch.full((batch_size,), 45.0),
        sown=torch.ones(batch_size),
        lai=torch.full((batch_size,), 0.6),
        wlv=torch.full((batch_size,), 120.0),
        wst=torch.full((batch_size,), 300.0),
        wso=torch.full((batch_size,), 450.0),
        wrt=torch.full((batch_size,), 90.0),
        anlv=torch.full((batch_size,), 2.0),
        anst=torch.full((batch_size,), 1.5),
        anrt=torch.full((batch_size,), 0.8),
        nlossl=torch.full((batch_size,), 1.2),
        tran_cum=torch.full((batch_size,), 210.0),
    )


def test_crop_state_is_cleared():
    state = _mature_state()
    crop, soil = CropParameters(crop_name="wheat"), SoilParameters()
    harvest = torch.ones(3)

    new = reset_at_harvest(state, harvest, crop, soil)

    for name in _CROP_ZERO_FIELDS:
        assert torch.all(getattr(new, name) == 0.0), name
    for name in _CROP_ONE_FIELDS:
        assert torch.all(getattr(new, name) == 1.0), name
    assert torch.allclose(new.dvs, crop.dvsi.expand_as(new.dvs))
    assert torch.allclose(new.rootd, crop.rdi.expand_as(new.rootd))


def test_root_zone_reset_conserves_profile_water():
    # ``wa`` is the water *in the rooted zone*, so shrinking the zone must
    # push the water it no longer covers into the lower store rather than
    # deleting it.
    state = _mature_state()
    crop, soil = CropParameters(crop_name="wheat"), SoilParameters()

    new = reset_at_harvest(state, torch.ones(3), crop, soil)

    before = state.wa + state.wa_lower
    after = new.wa + new.wa_lower
    assert torch.allclose(after, before, atol=1e-4)
    # The volumetric content of the new shallow zone is unchanged.
    theta_before = state.wa / (1000.0 * state.rootd)
    theta_after = new.wa / (1000.0 * new.rootd)
    assert torch.allclose(theta_after, theta_before, atol=1e-6)


def test_reset_is_per_batch_element():
    # Batch elements harvest on different days of different years, so the
    # reset has to leave the non-harvesting ones completely alone.
    state = _mature_state()
    crop, soil = CropParameters(crop_name="wheat"), SoilParameters()

    new = reset_at_harvest(state, torch.tensor([1.0, 0.0, 1.0]), crop, soil)

    assert float(new.tsum[1]) == 2100.0
    assert float(new.wso[1]) == 450.0
    assert float(new.rootd[1]) == float(state.rootd[1])
    assert float(new.tsum[0]) == 0.0 and float(new.tsum[2]) == 0.0


def test_soil_water_and_minerals_carry_by_default():
    state = _mature_state().replace(
        nmin=torch.full((3,), 5.0), nmint=torch.full((3,), 0.7)
    )
    crop, soil = CropParameters(crop_name="wheat"), SoilParameters()

    new = reset_at_harvest(state, torch.ones(3), crop, soil, CarryOverPolicy())

    assert torch.equal(new.nmin, state.nmin)
    assert torch.equal(new.nmint, state.nmint)


def test_soil_minerals_reset_policy_reseeds_from_parameters():
    state = _mature_state().replace(
        nmin=torch.full((3,), 5.0), nmint=torch.full((3,), 0.7)
    )
    crop, soil = CropParameters(crop_name="wheat"), SoilParameters()
    policy = CarryOverPolicy(soil_minerals="reset")

    new = reset_at_harvest(state, torch.ones(3), crop, soil, policy)

    assert torch.allclose(new.nmin, soil.nmini.expand_as(new.nmin))
    assert torch.allclose(new.nmint, soil.nminti.expand_as(new.nmint))


def test_accumulators_reset_by_default_and_carry_on_request():
    state = _mature_state()
    crop, soil = CropParameters(crop_name="wheat"), SoilParameters()

    cleared = reset_at_harvest(state, torch.ones(3), crop, soil)
    for name in _ACCUMULATOR_FIELDS:
        assert torch.all(getattr(cleared, name) == 0.0), name

    kept = reset_at_harvest(
        state, torch.ones(3), crop, soil, CarryOverPolicy(accumulators="carry")
    )
    assert torch.equal(kept.tran_cum, state.tran_cum)


def test_residue_return_adds_nutrients_to_the_organic_pool():
    state = _mature_state().replace(nmin=torch.full((3,), 5.0))
    crop, soil = CropParameters(crop_name="wheat"), SoilParameters()

    off = reset_at_harvest(state, torch.ones(3), crop, soil, CarryOverPolicy())
    on = reset_at_harvest(
        state,
        torch.ones(3),
        crop,
        soil,
        CarryOverPolicy(residue_fraction=0.5),
    )

    # Residue is leaves + roots + stems + senesced tissue, never the
    # storage organs (those leave the field as yield).
    expected = 0.5 * (
        state.anlv + state.anrt + state.anst + state.nlossl + state.nlossr
        + state.nlosss
    )
    assert torch.allclose(on.nmin - off.nmin, expected, atol=1e-6)

    stripped = reset_at_harvest(
        state,
        torch.ones(3),
        crop,
        soil,
        CarryOverPolicy(residue_fraction=0.5, residue_removes_stems=True),
    )
    assert torch.all(stripped.nmin < on.nmin)


def test_fallow_field_is_dormant_and_thermal_clocks_stay_at_zero():
    # After harvest the state is an ordinary pre-sowing condition: with
    # ``tsump = 0`` the EMERG gates shut, so a bare field neither grows
    # nor intercepts light — only the soil balances evolve.
    crop = CropParameters(crop_name="wheat")
    model = Lintul5Model(crop, SoilParameters(), SiteParameters())
    n_days = 500
    weather = make_constant_weather(
        batch_size=1, n_days=n_days, davtmp=14.0, start_doy=1
    )
    calendar = CropCalendar(sow_days=torch.tensor([[10]]), n_days=n_days)
    output = LongTermSimulator(model).run(
        weather, calendar, collect=("tsum", "tsump", "lai", "wso", "dvs", "rootd")
    )

    harvest_day = int(output.seasons.harvest_day[0, 0])
    assert bool(output.seasons.reached_maturity[0, 0])
    assert harvest_day < n_days - 1  # a real fallow period follows

    fallow = slice(harvest_day + 2, None)  # +1 for the leading initial state
    for name in ("tsum", "tsump", "lai", "wso"):
        assert torch.all(output.daily[name][0, fallow] == 0.0), name
    assert torch.all(output.daily["rootd"][0, fallow] == crop.rdi)

    # Without the reset, TSUM would keep climbing for the rest of the run:
    # its peak is the harvest day's value and nothing after it exceeds that.
    # (Index t + 1 holds day t, since index 0 is the initial condition, and
    # the harvest day still shows the ripe crop — the reset lands the day
    # after.)
    in_season = output.daily["tsum"][0, : harvest_day + 2]
    assert float(output.daily["tsum"].max()) == float(in_season.max())
