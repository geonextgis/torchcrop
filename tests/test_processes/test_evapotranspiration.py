"""Tests for ``PotentialEvapoTranspiration``, focusing on the site/crop
drivers that flow in through ``forward``: the Penman ET0 CO₂ correction
table ``fpenmtb``, site ``altitude``, and the crop transpiration factor
``cfet``.

Reference: SIMPLACE ``PotentialEvapoTranspiration.java`` line 194
(``ETC = ET0 · FPENMTB_interpol.getValueAt(CO2)``) and line 202
(``PTRAN = CFET · ETC · FINT``).
"""

from __future__ import annotations

import torch

from torchcrop.processes.evapotranspiration import PotentialEvapoTranspiration


def _inputs(b: int = 1, dtype: torch.dtype = torch.float64) -> dict:
    z = torch.zeros(b, dtype=dtype)
    return {
        "tmin": z + 5.0,
        "tmax": z + 15.0,
        "wind": z + 2.0,
        "vap": z + 1.0,  # kPa
        "avrad": z + 15.0e6,  # J m-2 d-1
        "atmtr": z + 0.5,
        "frac_int": z + 0.6,
    }


def _flat_table(value: float, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor([[40.0, value], [2000.0, value]], dtype=dtype)


# ----------------------- shape ----------------------- #


def test_et_output_shapes():
    et = PotentialEvapoTranspiration()
    out = et(**_inputs(b=4))
    for key in ("e0", "es0", "etc", "ptran", "pevap"):
        assert out[key].shape == (4,), f"{key} shape mismatch"


# ------------- fpenmtb drives the ET0 CO2 correction ------------- #


def test_default_fpenmtb_matches_explicit_table():
    """Omitting fpenmtb uses the standard C3 table (factor 1.0 at 360 ppm)."""
    et = PotentialEvapoTranspiration()
    base = _inputs()
    out_default = et(**base, co2=360.0)
    out_flat1 = et(**base, co2=360.0, fpenmtb=_flat_table(1.0))
    # Default table has factor 1.00 at 360 ppm -> ETC == uncorrected ET0.
    assert torch.allclose(out_default["etc"], out_flat1["etc"], atol=1e-10)


def test_fpenmtb_scales_etc_linearly():
    """ETC scales linearly with the interpolated FPENMTB factor."""
    et = PotentialEvapoTranspiration()
    base = _inputs()
    etc_1 = et(**base, co2=500.0, fpenmtb=_flat_table(1.0))["etc"]
    etc_half = et(**base, co2=500.0, fpenmtb=_flat_table(0.5))["etc"]
    assert torch.allclose(etc_half, 0.5 * etc_1, atol=1e-10)


def test_fpenmtb_dvs_like_interpolation_at_breakpoint():
    """A non-flat table is interpolated at the queried CO2 concentration."""
    et = PotentialEvapoTranspiration()
    base = _inputs()
    table = torch.tensor([[360.0, 1.0], [760.0, 0.8]], dtype=torch.float64)
    # Midpoint 560 ppm -> factor 0.9.
    etc = et(**base, co2=560.0, fpenmtb=table)["etc"]
    etc_ref = et(**base, co2=560.0, fpenmtb=_flat_table(0.9))["etc"]
    assert torch.allclose(etc, etc_ref, atol=1e-10)


# ------------- cfet scales potential transpiration ------------- #


def test_cfet_scales_ptran():
    """PTRAN = CFET · ETC · FINT scales linearly with cfet (above the floor)."""
    et = PotentialEvapoTranspiration()
    base = _inputs()
    ptran_1 = et(**base, cfet=1.0)["ptran"]
    ptran_2 = et(**base, cfet=2.0)["ptran"]
    assert bool((ptran_1 > 1e-3).all())  # comfortably above the 1e-4 clamp
    assert torch.allclose(ptran_2, 2.0 * ptran_1, atol=1e-10)


# ------------- altitude enters the barometric pressure ------------- #


def test_altitude_changes_output():
    """A non-zero altitude must propagate (via barometric pressure) to ETC."""
    et = PotentialEvapoTranspiration()
    base = _inputs()
    etc_sea = et(**base, altitude=0.0)["etc"]
    etc_high = et(**base, altitude=2000.0)["etc"]
    assert not torch.allclose(etc_sea, etc_high)
    assert torch.isfinite(etc_high).all()


# ------------- differentiability ------------- #


def test_gradient_flows_through_drivers():
    """ETC/PTRAN must be differentiable w.r.t. co2, cfet, and fpenmtb."""
    et = PotentialEvapoTranspiration()
    base = _inputs()
    co2 = torch.tensor([500.0], dtype=torch.float64, requires_grad=True)
    cfet = torch.tensor([1.2], dtype=torch.float64, requires_grad=True)
    fpenmtb = torch.tensor(
        [[40.0, 1.05], [2000.0, 0.9]], dtype=torch.float64, requires_grad=True
    )
    out = et(**base, co2=co2, cfet=cfet, fpenmtb=fpenmtb)
    (out["etc"].sum() + out["ptran"].sum()).backward()
    assert co2.grad is not None and torch.isfinite(co2.grad).all()
    assert cfet.grad is not None and torch.isfinite(cfet.grad).all()
    assert cfet.grad.abs() > 0
    assert fpenmtb.grad is not None and torch.isfinite(fpenmtb.grad).all()
