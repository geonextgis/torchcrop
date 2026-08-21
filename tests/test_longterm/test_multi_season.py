"""Season sequencing over a multi-year run.

Covers the properties that only appear once more than one crop is grown:
seasons stay disjoint, each one re-sows the seed reserve from scratch,
sowing dates may differ per year *and* per batch element, and a run can be
stopped and resumed across weather files without changing the answer.
"""

from __future__ import annotations

import torch

from torchcrop import CropParameters, Lintul5Model, SiteParameters, SoilParameters
from torchcrop.longterm import CarryOverPolicy, CropCalendar, LongTermSimulator
from torchcrop.utils.io import make_constant_weather


def _model(iopt: float = 2.0) -> Lintul5Model:
    crop = CropParameters(crop_name="wheat")
    crop.iopt = torch.tensor(float(iopt))
    return Lintul5Model(crop, SoilParameters(), SiteParameters())


def _weather(n_days: int, batch_size: int = 1):
    return make_constant_weather(
        batch_size=batch_size, n_days=n_days, davtmp=14.0, rain=3.5, start_doy=1
    )


def test_three_seasons_run_and_are_disjoint():
    n_days = 620
    model = _model()
    calendar = CropCalendar(sow_days=torch.tensor([[10, 210, 410]]), n_days=n_days)

    seasons = LongTermSimulator(model).run(_weather(n_days), calendar).seasons

    assert torch.all(seasons.valid > 0)
    assert torch.all(seasons.reached_maturity > 0)
    sow, harvest = seasons.sow_day[0], seasons.harvest_day[0]
    assert torch.all(harvest > sow)
    # Each harvest lands before the next sowing — no overlap anywhere.
    assert torch.all(harvest[:-1] < sow[1:])
    assert torch.all(seasons.yield_ > 0)


def test_each_season_re_sows_the_seed_reserve():
    # The seed-reserve bootstrap is the model's own; the calendar just has
    # to re-arm it. Every sowing day must therefore deposit exactly TDWI
    # of biomass, in every year, from a field that was bare the day before.
    n_days = 620
    model = _model()
    sow_days = [10, 210, 410]
    calendar = CropCalendar(sow_days=torch.tensor([sow_days]), n_days=n_days)

    output = LongTermSimulator(model).run(
        _weather(n_days), calendar, collect=("wlv", "wst", "wrt", "wso", "lai")
    )

    tdwi = float(model.crop_params.tdwi)
    for day in sow_days:
        # Index t + 1 holds day t (index 0 is the initial condition).
        pools = ("wlv", "wst", "wrt", "wso")
        before = sum(float(output.daily[n][0, day]) for n in pools)
        after = sum(float(output.daily[n][0, day + 1]) for n in pools)
        assert before == 0.0, f"field not bare before sowing on day {day}"
        assert abs(after - tdwi) < 1e-4, f"seed reserve wrong on day {day}"


def test_sowing_dates_may_differ_per_batch_element():
    n_days = 420
    model = _model()
    calendar = CropCalendar(
        sow_days=torch.tensor([[10, 210], [40, 240]]), n_days=n_days
    )

    seasons = LongTermSimulator(model).run(_weather(n_days, 2), calendar).seasons

    assert seasons.sow_day[0].tolist() == [10.0, 210.0]
    assert seasons.sow_day[1].tolist() == [40.0, 240.0]
    # A later sowing is harvested later, and the two differ.
    assert torch.all(seasons.harvest_day[1] > seasons.harvest_day[0])


def test_ragged_calendars_leave_unused_slots_invalid():
    # Batch elements may run different numbers of seasons.
    n_days = 420
    model = _model()
    calendar = CropCalendar(
        sow_days=torch.tensor([[10, 210], [10, -1]]), n_days=n_days
    )

    seasons = LongTermSimulator(model).run(_weather(n_days, 2), calendar).seasons

    assert seasons.valid[0].tolist() == [1.0, 1.0]
    assert seasons.valid[1].tolist() == [1.0, 0.0]
    assert float(seasons.sow_day[1, 1]) == -1.0
    assert float(seasons.yield_[1, 1]) == 0.0


def test_crop_that_never_matures_is_forced_off_before_the_next_sowing():
    # A cold year (or too short a season) must not leave a crop standing
    # into the next one — the deadline clears the field regardless.
    n_days = 400
    crop = CropParameters(crop_name="wheat")
    crop.iopt = torch.tensor(2.0)
    model = Lintul5Model(crop, SoilParameters(), SiteParameters())
    weather = make_constant_weather(
        batch_size=1, n_days=n_days, davtmp=2.0, rain=3.5, start_doy=1
    )
    calendar = CropCalendar(sow_days=torch.tensor([[10, 260]]), n_days=n_days)

    seasons = LongTermSimulator(model).run(weather, calendar).seasons

    assert torch.all(seasons.valid > 0)
    assert float(seasons.reached_maturity[0, 0]) == 0.0
    assert float(seasons.harvest_day[0, 0]) == 259.0  # day before next sowing


def test_fixed_harvest_dates_are_honoured():
    n_days = 420
    model = _model()
    calendar = CropCalendar(
        sow_days=torch.tensor([[10, 210]]),
        harvest_days=torch.tensor([[120, 350]]),
        n_days=n_days,
    )

    seasons = LongTermSimulator(model).run(_weather(n_days), calendar).seasons

    assert seasons.harvest_day[0].tolist() == [120.0, 350.0]
    # Cut early, before maturity, so the storage organs are still light.
    assert float(seasons.reached_maturity[0, 0]) == 0.0


def test_after_maturity_days_delays_the_harvest():
    n_days = 300
    model = _model()
    sow = torch.tensor([[10]])
    prompt = LongTermSimulator(model).run(
        _weather(n_days), CropCalendar(sow_days=sow, n_days=n_days)
    )
    delayed = LongTermSimulator(model).run(
        _weather(n_days),
        CropCalendar(sow_days=sow, n_days=n_days, after_maturity_days=14),
    )

    assert (
        float(delayed.seasons.harvest_day[0, 0])
        == float(prompt.seasons.harvest_day[0, 0]) + 14.0
    )
    # Yield is unaffected: growth already stopped at maturity.
    assert torch.allclose(delayed.seasons.yield_, prompt.seasons.yield_)


def test_run_can_be_resumed_from_a_previous_final_state():
    # Continuing from ``final_state`` over the next slice of weather must
    # give the same answer as running the whole horizon in one go — the
    # property that makes decade-by-decade streaming safe.
    n_days, split = 620, 350
    model = _model()
    weather = _weather(n_days)
    whole = LongTermSimulator(model).run(
        weather, CropCalendar(sow_days=torch.tensor([[10, 410]]), n_days=n_days)
    )

    first = LongTermSimulator(model).run(
        weather.data[:, :split],
        CropCalendar(sow_days=torch.tensor([[10]]), n_days=split),
    )
    second = LongTermSimulator(model).run(
        weather.data[:, split:],
        CropCalendar(sow_days=torch.tensor([[410 - split]]), n_days=n_days - split),
        initial_state=first.final_state,
        start_doy=split + 1,
        use_weather_doy=True,
    )

    assert torch.allclose(
        first.seasons.yield_[:, 0], whole.seasons.yield_[:, 0], atol=1e-4
    )
    assert torch.allclose(
        second.seasons.yield_[:, 0], whole.seasons.yield_[:, 1], atol=1e-4
    )


def test_soil_minerals_reset_policy_isolates_seasons():
    # Under "reset" every season starts from the same soil fertility, so a
    # long run cannot drift through nutrient depletion.
    n_days = 620
    model = _model(iopt=3.0)
    calendar = CropCalendar(sow_days=torch.tensor([[10, 210, 410]]), n_days=n_days)

    carried = LongTermSimulator(
        model, carryover=CarryOverPolicy(soil_minerals="carry")
    ).run(_weather(n_days), calendar).seasons
    isolated = LongTermSimulator(
        model, carryover=CarryOverPolicy(soil_minerals="reset")
    ).run(_weather(n_days), calendar).seasons

    # Season 1 is identical either way — the policies only differ from the
    # first harvest onwards.
    assert torch.allclose(carried.yield_[:, 0], isolated.yield_[:, 0])
    # Under "reset" later seasons see a replenished pool, so they take up
    # at least as much N as the depleting "carry" run.
    assert torch.all(isolated.nuptr_cum[:, 1:] >= carried.nuptr_cum[:, 1:] - 1e-6)
