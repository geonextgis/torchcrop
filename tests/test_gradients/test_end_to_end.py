"""End-to-end gradient tests for the full Lintul5 model."""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop import Lintul5Model
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.utils.io import make_constant_weather


def test_forward_runs_and_shapes_match():
    weather = make_constant_weather(batch_size=2, n_days=60, dtype=torch.float32)
    model = Lintul5Model()
    out = model(weather, start_doy=60)
    assert out.lai.shape == (2, 61)
    assert out.dvs.shape == (2, 61)
    assert out.yield_.shape == (2,)
    assert torch.isfinite(out.yield_).all()


def test_gradient_wrt_rue():
    weather = make_constant_weather(batch_size=1, n_days=80, dtype=torch.float64)
    crop_params = CropParameters().to(dtype=torch.float64)
    crop_params.rue = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    model = Lintul5Model(crop_params=crop_params)
    model = model.double()
    out = model(weather, start_doy=60)
    out.yield_.sum().backward()
    assert crop_params.rue.grad is not None
    assert torch.isfinite(crop_params.rue.grad)
    assert crop_params.rue.grad.abs().item() >= 0.0


def test_gradient_wrt_tsum1():
    weather = make_constant_weather(batch_size=1, n_days=80, dtype=torch.float64)
    crop_params = CropParameters().to(dtype=torch.float64)
    crop_params.tsum1 = nn.Parameter(torch.tensor(900.0, dtype=torch.float64))
    model = Lintul5Model(crop_params=crop_params).double()
    out = model(weather, start_doy=60)
    # DVS trajectory depends on tsum1 — gradient should exist
    out.dvs.sum().backward()
    assert crop_params.tsum1.grad is not None
    assert torch.isfinite(crop_params.tsum1.grad)


def test_batch_consistency():
    weather = make_constant_weather(batch_size=3, n_days=50, dtype=torch.float32)
    model = Lintul5Model()
    out_batch = model(weather, start_doy=60)
    for i in range(weather.batch_size):
        single_w = weather.data[i : i + 1]
        out_single = model(single_w, start_doy=60)
        assert torch.allclose(out_batch.yield_[i], out_single.yield_[0], atol=1e-5)
        assert torch.allclose(out_batch.lai[i], out_single.lai[0], atol=1e-5)
