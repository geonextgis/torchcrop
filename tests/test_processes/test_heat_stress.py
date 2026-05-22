"""Tests for the heat-stress process modules.

Covers `HeatStressOnLeafSenescence` (per-day RDR multiplier) and
`HeatStressOnGrain` (trajectory-level yield penalty), mirroring the
SIMPLACE ``HeatStressOnLeafSenescence.java`` / ``HeatStressOnGrain.java``
components. Both modules read their crop constants from `CropParameters`.
"""

from __future__ import annotations

import torch

from torchcrop.parameters.crop_params import CropParameters
from torchcrop.processes.heat_stress import (
    HeatStressOnGrain,
    HeatStressOnLeafSenescence,
)


# --------------------------------------------------------------------------- #
# HeatStressOnLeafSenescence
# --------------------------------------------------------------------------- #


def test_leaf_senescence_inactive_regime_returns_one():
    """Below Tc or below DVS_c the factor must be exactly 1.0."""
    hl = HeatStressOnLeafSenescence()
    params = CropParameters()
    tmax = torch.tensor([30.0, 40.0, 20.0])
    dvs = torch.tensor([1.5, 0.5, 1.5])  # cold / pre-DVSc / cold
    out = hl(tmax=tmax, dvs=dvs, params=params)
    assert torch.allclose(out, torch.ones(3))


def test_leaf_senescence_active_matches_formula():
    """Inside the regime: F = max(0, m*(Tmax-Tc) + F_Tc)."""
    hl = HeatStressOnLeafSenescence()
    # Defaults: leaf_heat_temp_critical=34, factor_at=3.0, slope=0.5.
    params = CropParameters()
    tmax = torch.tensor([34.0, 36.0, 40.0])
    dvs = torch.tensor([1.5, 1.5, 1.5])
    out = hl(tmax=tmax, dvs=dvs, params=params)
    # 34 -> 3.0 ; 36 -> 0.5*2 + 3 = 4.0 ; 40 -> 0.5*6 + 3 = 6.0
    assert torch.allclose(out, torch.tensor([3.0, 4.0, 6.0]))


def test_leaf_senescence_per_crop_params_override():
    """Crop-specific values flow in via CropParameters fields."""
    hl = HeatStressOnLeafSenescence()
    params = CropParameters(
        leaf_heat_temp_critical=torch.tensor(30.0),
        leaf_heat_factor_at_temp_critical=torch.tensor(2.0),
        leaf_heat_factor_slope=torch.tensor(1.0),
    )
    tmax = torch.tensor([32.0])
    dvs = torch.tensor([1.5])
    out = hl(tmax=tmax, dvs=dvs, params=params)
    # 1.0 * (32 - 30) + 2.0 = 4.0
    assert torch.allclose(out, torch.tensor([4.0]))


def test_leaf_senescence_shapes_and_smooth_gradcheck():
    hl_hard = HeatStressOnLeafSenescence()
    params = CropParameters()
    out = hl_hard(tmax=torch.zeros(5), dvs=torch.zeros(5), params=params)
    assert out.shape == (5,)

    params64 = CropParameters().to(dtype=torch.float64)
    hl_smooth = HeatStressOnLeafSenescence(smooth=True)
    tmax = torch.tensor([35.0, 38.0], dtype=torch.float64, requires_grad=True)
    dvs = torch.tensor([1.2, 1.6], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t, d: hl_smooth(tmax=t, dvs=d, params=params64),
        (tmax, dvs),
        eps=1e-6,
        atol=1e-4,
    )


# --------------------------------------------------------------------------- #
# HeatStressOnGrain
# --------------------------------------------------------------------------- #


def _grain_inputs(dtype=torch.float32):
    # Two batch elements, 6 days. DVS window [0.8, 1.3] open on days 2-4.
    tmin = torch.tensor([[10.0] * 6, [10.0] * 6], dtype=dtype)
    tmax = torch.tensor(
        [[20.0, 20.0, 44.0, 44.0, 44.0, 20.0], [20.0] * 6], dtype=dtype
    )
    dvs = torch.tensor(
        [[0.5, 0.7, 0.9, 1.1, 1.2, 1.5], [0.5, 0.7, 0.9, 1.1, 1.2, 1.5]],
        dtype=dtype,
    )
    return tmin, tmax, dvs


def test_grain_no_heat_gives_zero_factor():
    """A cool season yields a zero heat-stress factor and unchanged yield."""
    hg = HeatStressOnGrain()
    params = CropParameters()
    tmin, _, dvs = _grain_inputs()
    tmax = torch.full_like(tmin, 22.0)
    out = hg(
        tmin=tmin,
        tmax=tmax,
        dvs=dvs,
        params=params,
        yield_=torch.tensor([100.0, 100.0]),
    )
    assert torch.allclose(out["heat_stress_factor"], torch.zeros(2))
    assert torch.allclose(out["adjusted_yield"], torch.tensor([100.0, 100.0]))


def test_grain_factor_matches_window_average():
    """HSF = mean daily intensity over the DVS window only."""
    hg = HeatStressOnGrain()
    # Defaults: grain_heat_temp_critical=27, grain_heat_temp_limit=40.
    params = CropParameters()
    tmin, tmax, dvs = _grain_inputs()
    out = hg(
        tmin=tmin,
        tmax=tmax,
        dvs=dvs,
        params=params,
        yield_=torch.tensor([200.0, 200.0]),
    )
    # Window = days 2,3,4 (DVS in [0.8,1.3]). On those days TDay =
    # 44 - (44-10)/4 = 35.5 -> saturated daily factor clip((35.5-27)/13)=0.6538.
    # Days 2-4 hot for batch 0 -> HSF = 0.6538; batch 1 is cool -> HSF = 0.
    expected = (35.5 - 27.0) / (40.0 - 27.0)
    assert torch.allclose(out["heat_stress_factor"][0], torch.tensor(expected))
    assert torch.allclose(out["heat_stress_factor"][1], torch.tensor(0.0))
    assert torch.allclose(out["window_days"], torch.tensor([3.0, 3.0]))
    assert torch.allclose(
        out["adjusted_yield"][0], torch.tensor((1.0 - expected) * 200.0)
    )


def test_grain_per_crop_params_override():
    """A different crop config shifts both the threshold and the window."""
    hg = HeatStressOnGrain()
    params = CropParameters(
        grain_heat_begin_devstage=torch.tensor(0.0),
        grain_heat_end_devstage=torch.tensor(0.6),
    )
    tmin, tmax, dvs = _grain_inputs()
    out = hg(tmin=tmin, tmax=tmax, dvs=dvs, params=params)
    # Window now day 0 only (DVS 0.5 in [0.0, 0.6]; 0.7 is out); both
    # cool -> HSF = 0.
    assert torch.allclose(out["window_days"], torch.tensor([1.0, 1.0]))
    assert torch.allclose(out["heat_stress_factor"], torch.zeros(2))


def test_grain_empty_window_is_safe():
    """When DVS never reaches the window the factor is 0 (no divide-by-zero)."""
    hg = HeatStressOnGrain()
    params = CropParameters()
    tmin, tmax, _ = _grain_inputs()
    dvs = torch.zeros_like(tmin)  # never enters [0.8, 1.3]
    out = hg(tmin=tmin, tmax=tmax, dvs=dvs, params=params)
    assert torch.allclose(out["heat_stress_factor"], torch.zeros(2))
    assert torch.allclose(out["window_days"], torch.zeros(2))


def test_grain_gradcheck():
    hg = HeatStressOnGrain()
    params = CropParameters().to(dtype=torch.float64)
    tmin, tmax, dvs = _grain_inputs(torch.float64)
    tmin = tmin.clone().requires_grad_(True)
    tmax = tmax.clone().requires_grad_(True)
    yield_ = torch.tensor([150.0, 150.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b, y: hg(
            tmin=a, tmax=b, dvs=dvs, params=params, yield_=y
        )["adjusted_yield"],
        (tmin, tmax, yield_),
        eps=1e-6,
        atol=1e-4,
    )
