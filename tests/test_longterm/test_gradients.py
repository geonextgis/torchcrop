"""Differentiability across a multi-season run.

The whole point of torchcrop is that a simulation is differentiable, and
the long-term layer must not break that: a loss built from thirty years of
yields has to produce a usable gradient on crop parameters and on the
management schedule itself.

Two regimes are covered. Under ``truncate_bptt=True`` each season's yield
carries a gradient back to its own sowing, and the finished season's graph
is released. Under ``truncate_bptt=False`` one graph spans the whole run,
so a late season also feels parameters through the carried soil state.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop import CropParameters, Lintul5Model, SiteParameters, SoilParameters
from torchcrop.longterm import (
    CropCalendar,
    LongTermSimulator,
    ManagementEvent,
    ManagementSchedule,
)
from torchcrop.utils.io import make_constant_weather

N_DAYS = 420
SOW_DAYS = torch.tensor([[10, 210]])


def _model(
    learnable: str = "scale_factor_rue", soil_n: float | None = None
) -> tuple[Lintul5Model, nn.Parameter]:
    """A wheat model with one crop parameter marked learnable.

    Args:
        learnable: Crop-parameter field to wrap as an `nn.Parameter`.
        soil_n: Optional initial soil N [g N m⁻²] for both the organic and
            inorganic pools. The default soil is N-replete, so uptake is
            demand-limited and added fertiliser has no effect at all;
            passing a small value puts the crop into the N-limited regime
            where nutrient supply actually drives yield.
    """
    crop = CropParameters(crop_name="wheat")
    crop.iopt = torch.tensor(3.0)
    param = nn.Parameter(getattr(crop, learnable).clone())
    setattr(crop, learnable, param)
    soil = SoilParameters()
    if soil_n is not None:
        soil.nmini = torch.tensor(soil_n)
        soil.nminti = torch.tensor(soil_n)
    return Lintul5Model(crop, soil, SiteParameters()), param


def _weather():
    return make_constant_weather(
        batch_size=1, n_days=N_DAYS, davtmp=14.0, rain=3.5, start_doy=1
    )


def test_every_season_yield_has_a_gradient_on_a_crop_parameter():
    model, rue = _model()
    calendar = CropCalendar(sow_days=SOW_DAYS, n_days=N_DAYS)

    output = LongTermSimulator(model).run(_weather(), calendar)
    output.seasons.yield_.sum().backward()

    assert rue.grad is not None
    assert torch.isfinite(rue.grad).all()
    assert float(rue.grad.abs()) > 1e-8


def test_a_single_late_season_alone_carries_a_gradient():
    # Truncation cuts *between* seasons, never inside one, so the last
    # season's yield must still be differentiable on its own.
    model, rue = _model()
    calendar = CropCalendar(sow_days=SOW_DAYS, n_days=N_DAYS)

    output = LongTermSimulator(model).run(_weather(), calendar)
    output.seasons.yield_[0, 1].backward()

    assert rue.grad is not None
    assert float(rue.grad.abs()) > 1e-8


def _season_one_fertiliser(truncate: bool):
    """Fertilise only the first season of an N-limited two-season run.

    Nitrogen applied to the first crop that the first crop does not use
    stays in the soil and feeds the second one, which is a real carry-over
    pathway crossing the season boundary — the thing ``truncate_bptt``
    decides whether to differentiate through.

    Returns:
        Tuple ``(output, amount)`` of the run and the learnable N amount.
    """
    model, _ = _model(soil_n=0.05)
    amount = nn.Parameter(torch.tensor(30.0))
    schedule = ManagementSchedule(
        fertilizer=[
            ManagementEvent(amount=amount, days_after_sowing=20, seasons=[0])
        ]
    )
    simulator = LongTermSimulator(model, truncate_bptt=truncate)
    calendar = CropCalendar(sow_days=SOW_DAYS, n_days=N_DAYS)
    return simulator.run(_weather(), calendar, management=schedule), amount


def test_residual_soil_nitrogen_carries_into_the_next_season():
    # Establish the physical effect the gradient tests below rely on:
    # fertilising only season 1 still lifts season 2's yield, because the
    # unused nitrogen is carried across the harvest boundary.
    fed, _ = _season_one_fertiliser(truncate=True)

    model, _ = _model(soil_n=0.05)
    starved = LongTermSimulator(model).run(
        _weather(), CropCalendar(sow_days=SOW_DAYS, n_days=N_DAYS)
    )

    assert float(fed.seasons.yield_[0, 1]) > float(starved.seasons.yield_[0, 1])


def test_truncation_cuts_the_gradient_at_the_season_boundary():
    # Truncated: season 2 cannot see season 1's management at all.
    truncated, amount_t = _season_one_fertiliser(truncate=True)
    truncated.seasons.yield_[0, 1].backward()
    assert float(amount_t.grad) == 0.0

    # Full graph: the same yield is differentiable through the residual
    # soil nitrogen the previous crop left behind.
    full, amount_f = _season_one_fertiliser(truncate=False)
    full.seasons.yield_[0, 1].backward()
    assert torch.isfinite(amount_f.grad).all()
    assert float(amount_f.grad) > 1e-6


def test_truncation_does_not_change_the_simulated_values():
    # ``truncate_bptt`` is a memory setting, not a physics setting: the
    # numbers must be identical either way.
    calendar = CropCalendar(sow_days=SOW_DAYS, n_days=N_DAYS)
    model, _ = _model()

    truncated = LongTermSimulator(model, truncate_bptt=True).run(
        _weather(), calendar, collect=("dvs", "lai", "wso")
    )
    full = LongTermSimulator(model, truncate_bptt=False).run(
        _weather(), calendar, collect=("dvs", "lai", "wso")
    )

    assert torch.equal(truncated.seasons.yield_, full.seasons.yield_.detach())
    for name in ("dvs", "lai", "wso"):
        assert torch.equal(truncated.daily[name], full.daily[name].detach()), name


def test_collected_daily_output_is_detached_when_truncating():
    # Holding graphs on the collected trajectory would defeat truncation:
    # the released season would stay alive through the output tensors.
    calendar = CropCalendar(sow_days=SOW_DAYS, n_days=N_DAYS)
    model, _ = _model()

    truncated = LongTermSimulator(model, truncate_bptt=True).run(
        _weather(), calendar, collect=("lai",)
    )
    full = LongTermSimulator(model, truncate_bptt=False).run(
        _weather(), calendar, collect=("lai",)
    )

    assert not truncated.daily["lai"].requires_grad
    assert full.daily["lai"].requires_grad


def test_fertiliser_schedule_is_differentiable_through_the_simulation():
    # Optimising a management strategy means back-propagating into the
    # applied amounts, so the schedule has to stay in the graph. The soil
    # must be N-limited for the gradient to be non-zero: on an N-replete
    # soil, uptake is demand-limited and extra fertiliser correctly has no
    # effect whatsoever.
    model, _ = _model(soil_n=0.05)
    amount = nn.Parameter(torch.tensor(10.0))
    schedule = ManagementSchedule(
        fertilizer=[ManagementEvent(amount=amount, days_after_sowing=20)]
    )
    calendar = CropCalendar(sow_days=SOW_DAYS, n_days=N_DAYS)

    output = LongTermSimulator(model).run(_weather(), calendar, management=schedule)
    output.seasons.yield_.sum().backward()

    assert amount.grad is not None
    assert torch.isfinite(amount.grad).all()
    assert float(amount.grad) > 0.0  # more N, more yield


def test_model_state_detach_preserves_values():
    model, _ = _model()
    state = model.initialize(batch_size=2)
    state = state.replace(wa=state.wa.clone().requires_grad_(True))

    detached = state.detach()

    assert not detached.wa.requires_grad
    assert torch.equal(detached.wa, state.wa.detach())
    assert torch.equal(detached.rootd, state.rootd)
