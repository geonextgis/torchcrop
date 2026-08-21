"""The long-term layer must not perturb the mechanistic model.

torchcrop reproduces SIMPLACE Lintul5, so the multi-season framework is
only acceptable if it changes *nothing* about how a day is simulated.
These tests assert that with ``torch.equal`` — bit-for-bit, not
``allclose`` — at three levels:

1. the ``sowing`` / ``doy`` hooks are inert when unused, and reproduce the
   default path exactly when they encode it;
2. a one-season `LongTermSimulator` run matches `Lintul5Model.forward` on
   every state field of every day;
3. that holds in all four production modes (``IOPT`` 1-4).
"""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache

import pytest
import torch

from torchcrop import (
    CropParameters,
    Lintul5Model,
    SiteParameters,
    SoilParameters,
)
from torchcrop.longterm import CropCalendar, LongTermSimulator
from torchcrop.states.model_state import ModelState
from torchcrop.utils.io import make_constant_weather

N_DAYS = 300
IDPL = 60
STATE_FIELDS = [f.name for f in fields(ModelState)]


def _setup(iopt: float, batch_size: int = 2):
    """Build a matched (model, weather) pair for a production mode."""
    crop = CropParameters(crop_name="wheat")
    crop.iopt = torch.tensor(float(iopt))
    site = SiteParameters(idpl=torch.tensor(float(IDPL)))
    model = Lintul5Model(crop, SoilParameters(), site)
    weather = make_constant_weather(
        batch_size=batch_size, n_days=N_DAYS, davtmp=14.0, start_doy=1
    )
    return model, weather


def _trajectory(states) -> dict[str, torch.Tensor]:
    """Stack a list of `ModelState` into ``{field: [B, T + 1]}``."""
    return {
        name: torch.stack([getattr(s, name) for s in states], dim=1)
        for name in STATE_FIELDS
    }


@lru_cache(maxsize=None)
def _reference(iopt: float) -> dict[str, torch.Tensor]:
    """Cached single-season reference trajectory for a production mode.

    Every test in this module compares against a plain
    `Lintul5Model.forward` run, so the reference is computed once per
    ``iopt`` instead of once per test.
    """
    model, weather = _setup(iopt)
    return _trajectory(model(weather, start_doy=1).states)


@pytest.mark.parametrize("iopt", [1, 2, 3, 4])
def test_sowing_and_doy_hooks_reproduce_the_default_path(iopt):
    # The hooks exist to let a caller *re-express* the default behaviour,
    # so encoding it explicitly must land on the identical trajectory.
    model, weather = _setup(iopt)
    reference = _reference(iopt)

    doy = weather.channel("doy")
    sowing = (doy >= float(IDPL)).to(weather.data.dtype)
    overridden = _trajectory(
        model(weather, start_doy=1, sowing=sowing, doy=doy).states
    )

    for name in STATE_FIELDS:
        assert torch.equal(overridden[name], reference[name]), name


@pytest.mark.parametrize("iopt", [1, 2, 3, 4])
def test_single_season_run_matches_forward_exactly(iopt):
    # One season through the long-term machinery is still one season: the
    # scheduler adds a sowing pulse and a harvest event, neither of which
    # may move a single number in between.
    model, weather = _setup(iopt)
    reference = _reference(iopt)

    # ``harvest_rule="deadline"`` keeps the crop standing to the final day,
    # matching a plain forward run that simply stops at the end of the
    # weather. Sowing on day index IDPL - 1 is the day whose DOY is IDPL.
    calendar = CropCalendar(
        sow_days=torch.full((weather.batch_size, 1), IDPL - 1),
        n_days=N_DAYS,
        harvest_rule="deadline",
    )
    output = LongTermSimulator(model).run(weather, calendar, collect=STATE_FIELDS)

    for name in STATE_FIELDS:
        assert torch.equal(output.daily[name], reference[name]), name


def test_single_season_summary_matches_forward():
    model, weather = _setup(iopt=3)
    reference = model(weather, start_doy=1)
    calendar = CropCalendar(
        sow_days=torch.full((weather.batch_size, 1), IDPL - 1),
        n_days=N_DAYS,
        harvest_rule="deadline",
    )
    seasons = LongTermSimulator(model).run(weather, calendar).seasons

    assert torch.equal(seasons.yield_[:, 0], reference.yield_)
    assert torch.equal(seasons.adjusted_yield[:, 0], reference.adjusted_yield)
    assert torch.equal(
        seasons.heat_stress_factor[:, 0], reference.heat_stress_factor
    )


def test_diagnostics_match_forward():
    model, weather = _setup(iopt=2)
    reference = model(weather, start_doy=1)
    calendar = CropCalendar(
        sow_days=torch.full((weather.batch_size, 1), IDPL - 1),
        n_days=N_DAYS,
        harvest_rule="deadline",
    )
    names = ("tranrf", "gtotal", "nni", "tran", "evap", "smact")
    output = LongTermSimulator(model).run(weather, calendar, collect=names)

    for name in names:
        expected = torch.stack(
            [getattr(d, name) for d in reference.diagnostics], dim=1
        )
        assert torch.equal(output.daily[name], expected), name
