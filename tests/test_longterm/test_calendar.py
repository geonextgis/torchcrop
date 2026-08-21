"""Calendar construction, validation, and management-schedule expansion.

These are pure bookkeeping units — no simulation — so they are cheap and
cover the edge cases that would otherwise only surface as a wrong sowing
date thirty years into a run.
"""

from __future__ import annotations

from datetime import date

import pytest
import torch

from torchcrop.longterm import (
    CropCalendar,
    ManagementEvent,
    ManagementSchedule,
)


def test_annual_converts_day_of_year_across_leap_years():
    calendar = CropCalendar.annual(
        sow_doy=285, n_years=3, start_date="2000-01-01", n_days=1461
    )
    # 2000 is a leap year, so the second gap is 366 days and the third 365.
    assert calendar.sow_days[0].tolist() == [284, 650, 1015]
    assert calendar.start_date == date(2000, 1, 1)


def test_annual_accepts_a_different_day_of_year_per_year():
    calendar = CropCalendar.annual(
        sow_doy=[280, 290, 275],
        n_years=3,
        start_date="2001-01-01",
        n_days=1200,
    )
    days = calendar.sow_days[0].tolist()
    assert days[1] - days[0] == 375  # 365 + (290 - 280)
    assert days[2] - days[1] == 350  # 365 + (275 - 290)


def test_annual_can_leave_a_spin_up_year():
    calendar = CropCalendar.annual(
        sow_doy=285,
        n_years=2,
        start_date="2000-01-01",
        n_days=1200,
        first_year_offset=1,
    )
    assert calendar.sow_days[0, 0] == 650  # first sowing is in 2001


def test_annual_rejects_dates_outside_the_weather_series():
    with pytest.raises(ValueError, match="outside the"):
        CropCalendar.annual(
            sow_doy=285, n_years=5, start_date="2000-01-01", n_days=400
        )


def test_from_dates_handles_per_element_schedules_of_different_lengths():
    calendar = CropCalendar.from_dates(
        [["2000-04-01", "2001-04-01"], ["2000-05-01"]],
        start_date="2000-01-01",
        n_days=800,
    )
    assert calendar.sow_days[0].tolist() == [91, 456]
    assert calendar.sow_days[1].tolist() == [121, -1]
    assert calendar.season_mask[1].tolist() == [True, False]


def test_from_dates_broadcasts_a_shared_schedule_over_the_batch():
    calendar = CropCalendar.from_dates(
        ["2000-04-01", "2001-04-01"],
        start_date="2000-01-01",
        n_days=800,
        batch_size=4,
    )
    assert calendar.batch_size == 4
    assert torch.equal(calendar.sow_days[0], calendar.sow_days[3])


def test_sowing_days_must_increase():
    with pytest.raises(ValueError, match="increase strictly"):
        CropCalendar(sow_days=torch.tensor([[100, 50]]), n_days=400)


def test_padding_must_be_trailing():
    with pytest.raises(ValueError, match="trailing"):
        CropCalendar(sow_days=torch.tensor([[-1, 50]]), n_days=400)


def test_fixed_harvest_must_follow_sowing():
    with pytest.raises(ValueError, match="strictly after"):
        CropCalendar(
            sow_days=torch.tensor([[100]]),
            harvest_days=torch.tensor([[100]]),
            n_days=400,
        )


def test_deadline_signal_marks_the_day_before_each_next_sowing():
    calendar = CropCalendar(sow_days=torch.tensor([[10, 210, 410]]), n_days=600)
    days = calendar.deadline_signal().nonzero()[:, 1].tolist()
    assert days == [209, 409, 599]  # last season runs to the end of the series


def test_sow_signal_is_a_single_pulse_per_season():
    calendar = CropCalendar(sow_days=torch.tensor([[10, 210]]), n_days=400)
    signal = calendar.sow_signal()
    assert signal.sum().item() == 2
    assert signal[0, 10] == 1.0 and signal[0, 210] == 1.0


def test_season_dates_round_trips_day_indices():
    calendar = CropCalendar.annual(
        sow_doy=100, n_years=2, start_date="2010-01-01", n_days=800
    )
    dates = calendar.season_dates(calendar.sow_days)
    assert dates[0][0] == date(2010, 4, 10)
    assert dates[0][1] == date(2011, 4, 10)


# ---------------------------------------------------------------------- #
# Management schedules
# ---------------------------------------------------------------------- #


def _calendar() -> CropCalendar:
    return CropCalendar(sow_days=torch.tensor([[10, 210]]), n_days=400)


def test_sowing_relative_events_follow_each_season():
    schedule = ManagementSchedule(
        fertilizer=[ManagementEvent(amount=6.0, days_after_sowing=20)]
    )
    _, fertilizer = schedule.expand(_calendar())

    assert fertilizer.shape == (1, 400, 3)
    assert fertilizer[0, 30, 0] == 6.0
    assert fertilizer[0, 230, 0] == 6.0
    assert fertilizer[..., 1:].sum() == 0.0  # N only
    assert fertilizer.sum() == 12.0


def test_events_on_the_same_day_accumulate():
    schedule = ManagementSchedule(
        fertilizer=[
            ManagementEvent(amount=6.0, days_after_sowing=20, nutrient="n"),
            ManagementEvent(amount=2.0, days_after_sowing=20, nutrient="p"),
            ManagementEvent(amount=1.0, days_after_sowing=20, nutrient="n"),
        ]
    )
    _, fertilizer = schedule.expand(_calendar())

    assert fertilizer[0, 30].tolist() == [7.0, 2.0, 0.0]


def test_events_can_target_selected_seasons():
    schedule = ManagementSchedule(
        irrigation=[ManagementEvent(amount=25.0, days_after_sowing=30, seasons=[1])]
    )
    irrigation, _ = schedule.expand(_calendar())

    assert irrigation[0, 40] == 0.0
    assert irrigation[0, 240] == 25.0


def test_offsets_that_fall_outside_the_series_are_dropped():
    # A pre-sowing dressing 20 days before a season sown on day 10 has
    # nowhere to land; it is simply not applied.
    schedule = ManagementSchedule(
        fertilizer=[ManagementEvent(amount=5.0, days_after_sowing=-20)]
    )
    _, fertilizer = schedule.expand(_calendar())

    assert fertilizer.sum() == 5.0  # only the second season's application
    assert fertilizer[0, 190, 0] == 5.0


def test_date_anchored_events_need_a_start_date():
    schedule = ManagementSchedule(
        irrigation=[ManagementEvent(amount=10.0, date="2000-03-01")]
    )
    with pytest.raises(ValueError, match="start_date"):
        schedule.expand(_calendar())


def test_amounts_keep_their_gradient_path():
    # A schedule built from parameters must stay differentiable, so an
    # N strategy can be optimised through the simulation.
    amount = torch.nn.Parameter(torch.tensor(6.0))
    schedule = ManagementSchedule(
        fertilizer=[ManagementEvent(amount=amount, days_after_sowing=20)]
    )
    _, fertilizer = schedule.expand(_calendar())

    fertilizer.sum().backward()
    assert amount.grad is not None
    assert float(amount.grad) == 2.0  # applied in both seasons


def test_event_requires_exactly_one_anchor():
    with pytest.raises(ValueError, match="exactly one"):
        ManagementEvent(amount=1.0)
    with pytest.raises(ValueError, match="exactly one"):
        ManagementEvent(amount=1.0, day=5, days_after_sowing=5)
