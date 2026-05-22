"""Tests for the photosynthesis (RadiationUseEfficiency) process module.

Mirrors the SIMPLACE ``RadiationUseEfficiency.process()`` outputs (RUE and
RTMCO only); the GTOTAL formula belongs to ``LintulFunctions.GROWTH`` and is
applied downstream in the model — exercised here for end-to-end sanity.
"""

from __future__ import annotations

import torch

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.processes.photosynthesis import Photosynthesis


def _inputs():
    tmax = torch.tensor([20.0, 25.0, 12.0])
    tmin = torch.tensor([8.0, 12.0, -2.0])
    dvs = torch.tensor([0.5, 1.0, 1.8])
    return tmax, tmin, dvs


def test_photosynthesis_shapes_and_keys():
    photo = Photosynthesis()
    tmax, tmin, dvs = _inputs()
    out = photo(tmax, tmin, dvs, CropParameters())
    for key in ("rue", "rtmco", "rco", "rtmp", "dtemp"):
        assert key in out, f"missing output {key}"
        assert out[key].shape == (3,)
    assert "gtotal" not in out, "RadiationUseEfficiency must not emit GTOTAL"


def test_dtemp_matches_simplace_formula():
    photo = Photosynthesis()
    tmax, tmin, dvs = _inputs()
    params = CropParameters()
    out = photo(tmax, tmin, dvs, params)
    expected = tmax - params.day_temp_factor * (tmax - tmin)
    assert torch.allclose(out["dtemp"], expected)


def test_rue_uses_dvs_interpolation():
    """RUE must drop after DVS=1.3 as encoded in the default ruetb."""
    photo = Photosynthesis()
    params = CropParameters()
    tmax = torch.tensor([20.0, 20.0])
    tmin = torch.tensor([10.0, 10.0])
    dvs = torch.tensor([0.5, 1.9])
    out = photo(tmax, tmin, dvs, params)
    assert out["rue"][0] > out["rue"][1]
    assert torch.allclose(out["rue"][0], torch.tensor(3.0))
    assert torch.allclose(
        out["rue"], params.scale_factor_rue * interpolate(params.ruetb, dvs)
    )


def test_rtmco_combines_temp_and_co2():
    photo = Photosynthesis()
    params = CropParameters()
    tmax, tmin, dvs = _inputs()
    # photo() is called without an explicit co2, so the default 370 ppm
    # (the SIMPLACE reference) is used for the COTB look-up.
    out = photo(tmax, tmin, dvs, params)
    expected_rco = interpolate(params.cotb, torch.full_like(tmax, 370.0))
    expected_rtmp = interpolate(params.tmpftb, out["dtemp"]) * interpolate(
        params.tmnftb, tmin
    )
    assert torch.allclose(out["rco"], expected_rco)
    assert torch.allclose(out["rtmp"], expected_rtmp)
    assert torch.allclose(out["rtmco"], expected_rtmp * expected_rco)


def test_cold_minimum_temperature_zeroes_rtmco():
    """tmin at/below tmnftb left edge must drive RTMCO (and hence GTOTAL) to 0."""
    photo = Photosynthesis()
    params = CropParameters()
    tmax = torch.tensor([10.0])
    tmin = torch.tensor([-5.0])
    dvs = torch.tensor([0.5])
    out = photo(tmax, tmin, dvs, params)
    assert torch.allclose(out["rtmco"], torch.zeros(1))
    # Downstream GTOTAL formula collapses to zero under any PARINT / stress
    parint = torch.tensor([10.0])
    gtotal = out["rue"] * out["rtmco"] * parint * torch.ones(1) * torch.ones(1)
    assert torch.allclose(gtotal, torch.zeros(1))


def test_gradient_flows_to_rue_table():
    """∂(downstream GTOTAL)/∂RUETB[:,1] must be non-zero."""
    photo = Photosynthesis()
    params = CropParameters().to(dtype=torch.float64)
    params.ruetb = params.ruetb.clone().detach().requires_grad_(True)
    tmax = torch.tensor([20.0], dtype=torch.float64)
    tmin = torch.tensor([10.0], dtype=torch.float64)
    dvs = torch.tensor([0.5], dtype=torch.float64)
    out = photo(tmax, tmin, dvs, params)
    parint = torch.tensor([10.0], dtype=torch.float64)
    gtotal = out["rue"] * out["rtmco"] * parint
    gtotal.sum().backward()
    assert params.ruetb.grad is not None
    assert params.ruetb.grad.abs().sum() > 0


if __name__ == "__main__":
    photo = Photosynthesis()
    params = CropParameters()
    tmax, tmin, dvs = _inputs()
    out = photo(tmax, tmin, dvs, params)
    parint = torch.tensor([8.0, 10.0, 6.0])
    tranrf = torch.ones(3)
    nstress = torch.ones(3)
    # GTOTAL is computed downstream (LintulFunctions.GROWTH), shown here for context
    gtotal = out["rue"] * out["rtmco"] * parint * tranrf * nstress
    print("=== RadiationUseEfficiency demonstration ===")
    print(f"  TMAX                = {tmax.tolist()}")
    print(f"  TMIN                = {tmin.tolist()}")
    print(f"  DVS                 = {dvs.tolist()}")
    print(f"  DTEMP               = {out['dtemp'].tolist()}")
    print(f"  RUE  (= RUETB(DVS)) = {out['rue'].tolist()}")
    print(f"  RCO  (= COTB(CO2))  = {out['rco'].tolist()}")
    print(f"  RTMP (T correction) = {out['rtmp'].tolist()}")
    print(f"  RTMCO               = {out['rtmco'].tolist()}")
    print("  --- downstream (Lintul5 biomass block) ---")
    print(f"  PARINT              = {parint.tolist()}")
    print(f"  GTOTAL [g/m2/d]     = {gtotal.tolist()}")
