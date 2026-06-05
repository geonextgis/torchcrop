"""Tests for the ``idpl``-driven sowing gate (pre-sowing spin-up).

The model equates ``t = 0`` with sowing unless ``site.idpl`` delays it.
With ``idpl > start_doy`` the emergence and vernalisation thermal clocks
must stay frozen until the sowing day, while the soil water balance is
free to evolve — giving the crop a stable spin-up period. The latch must
also survive the day-of-year wraparound when a run spans a calendar-year
boundary (autumn sowing, summer harvest).
"""

from __future__ import annotations

import torch

from torchcrop import (
    CropParameters,
    Lintul5Model,
    SiteParameters,
    SoilParameters,
)
from torchcrop.utils.io import make_constant_weather


def _run(idpl: float, n_days: int = 731, start_doy: int = 1):
    weather = make_constant_weather(batch_size=1, n_days=n_days, davtmp=12.0)
    site = SiteParameters(idpl=torch.tensor(float(idpl)))
    model = Lintul5Model(CropParameters(), SoilParameters(), site)
    out = model(weather, start_doy=start_doy)
    sown = torch.stack([s.sown for s in out.states], dim=1)[0]  # [T+1]
    tsump = torch.stack([s.tsump for s in out.states], dim=1)[0]
    wa = torch.stack([s.wa for s in out.states], dim=1)[0]
    return out, sown, tsump, wa


def test_crop_dormant_until_sowing_day():
    # Sowing at doy 270 with the run starting on doy 1 → days 1..269 are
    # a spin-up period during which no crop development occurs.
    out, sown, tsump, _ = _run(idpl=270)

    # State index t corresponds to doy = t (start_doy=1). The latch turns
    # on exactly on the sowing day and never before it.
    assert float(sown[269]) == 0.0  # doy 269 — still bare soil
    assert float(sown[270]) == 1.0  # doy 270 — sown

    # Emergence clock and development stay frozen through the spin-up.
    assert float(tsump[269]) == 0.0
    assert float(out.dvs[0, 269]) == 0.0
    assert float(out.lai[0, 269]) == 0.0

    # After sowing the crop actually develops.
    assert float(tsump[-1]) > 0.0
    assert float(out.dvs[0, -1]) > 0.0


def test_sowing_latch_survives_year_wraparound():
    # The run spans two years; once sown on doy 270 the latch must remain
    # 1 even when doy cycles back below 270 in the second year.
    _, sown, _, _ = _run(idpl=270)
    assert torch.all(sown[270:] == 1.0)


def test_soil_water_evolves_during_spinup():
    # The spin-up is only useful if the soil water balance is live before
    # sowing: wa must change over days 1..269 even though the crop is off.
    _, _, _, wa = _run(idpl=270)
    assert float((wa[1:270] - wa[0]).abs().max()) > 0.0


def test_default_idpl_reproduces_sown_at_start():
    # Default idpl=0 ⇒ sown from the first day (legacy behaviour): the
    # emergence clock accumulates immediately.
    _, sown, tsump, _ = _run(idpl=0, n_days=10)
    assert float(sown[1]) == 1.0
    assert float(tsump[1]) > 0.0
