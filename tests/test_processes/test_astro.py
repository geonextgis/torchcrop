"""Tests for astronomical calculations."""

from __future__ import annotations

import torch

from torchcrop.processes.astro import Astro


def test_daylength_equinox():
    astro = Astro()
    # DOY 80 (~ spring equinox) at latitude 0° → ~12 h daylength
    doy = torch.tensor([80.0])
    lat = torch.tensor([0.0])
    out = astro(doy, lat)
    assert torch.isclose(out["daylength"], torch.tensor([12.0]), atol=0.3)


def test_daylength_summer_north():
    astro = Astro()
    doy = torch.tensor([172.0])  # ~ June 21
    lat = torch.tensor([52.0])
    out = astro(doy, lat)
    # At 52°N midsummer, expect >15 h daylight
    assert out["daylength"].item() > 15.0


def test_daylength_batch_shape():
    astro = Astro()
    doy = torch.tensor([80.0, 172.0, 355.0])
    lat = torch.tensor([52.0, 52.0, 52.0])
    out = astro(doy, lat)
    assert out["daylength"].shape == (3,)
