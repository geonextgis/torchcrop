"""Tests for ``Irradiation`` PAR interception, focusing on the
DVS-dependent diffuse-PAR extinction coefficient.

Reference: SIMPLACE ``Lintul5.java`` line 1391
(``KDIF = cScaleFactorKDIF · KDIFTB_interpol.getValueAt(DVS)``) and the
Beer–Lambert interception ``FINT = 1 − exp(−KDIF · LAI)``.
"""

from __future__ import annotations

import math

import torch

from torchcrop.parameters.crop_params import CropParameters
from torchcrop.states.model_state import ModelState


def _state(
    b: int = 1,
    lai: float = 3.0,
    dvs: float = 0.5,
    dtype: torch.dtype = torch.float64,
) -> ModelState:
    s = ModelState.initial(batch_size=b, dtype=dtype)
    return s.replace(
        lai=torch.full_like(s.lai, lai),
        dvs=torch.full_like(s.dvs, dvs),
    )


def _geo(b: int = 1, dtype: torch.dtype = torch.float64):
    """Arbitrary but valid solar-geometry inputs (cosld != 0)."""
    z = torch.zeros(b, dtype=dtype)
    return {
        "doy": z + 180.0,
        "dayl": z + 12.0,
        "sinld": z + 0.3,
        "cosld": z + 0.5,
        "dtr": z + 20.0,  # MJ m-2 d-1
    }


def _params() -> CropParameters:
    return CropParameters().to(dtype=torch.float64)


def _run(state: ModelState, params: CropParameters):
    from torchcrop.processes.irradiation import Irradiation

    return Irradiation()(state=state, params=params, **_geo(b=state.lai.shape[0]))


# ----------------------- shape ----------------------- #


def test_irradiation_output_shapes():
    out = _run(_state(b=3), _params())
    for key in ("avrad", "atmtr", "par", "parint", "frac_intercepted"):
        assert out[key].shape == (3,), f"{key} shape mismatch"


# ------------- default KDIFTB equals scalar k=0.6 ------------- #


def test_default_kdiftb_is_flat_constant():
    """Default KDIFTB is flat 0.6 -> FINT = 1 - exp(-0.6 * LAI) at any DVS."""
    p = _params()
    for dvs in (0.0, 0.5, 1.3, 2.0):
        out = _run(_state(lai=3.0, dvs=dvs), p)
        expected = 1.0 - math.exp(-0.6 * 3.0)
        assert torch.allclose(
            out["frac_intercepted"],
            torch.tensor([expected], dtype=torch.float64),
            atol=1e-12,
        ), f"DVS={dvs}"


# ------------- DVS-varying KDIFTB is honoured ------------- #


def test_dvs_varying_kdiftb_changes_interception():
    """A non-flat KDIFTB makes interception DVS-dependent."""
    p = _params()
    # Extinction rises from 0.4 (DVS 0) to 0.8 (DVS 2).
    p.kdiftb = torch.tensor([[0.0, 0.4], [2.0, 0.8]], dtype=torch.float64)
    lai = 2.5

    out_young = _run(_state(lai=lai, dvs=0.0), p)
    out_old = _run(_state(lai=lai, dvs=2.0), p)

    assert torch.allclose(
        out_young["frac_intercepted"],
        torch.tensor([1.0 - math.exp(-0.4 * lai)], dtype=torch.float64),
        atol=1e-12,
    )
    assert torch.allclose(
        out_old["frac_intercepted"],
        torch.tensor([1.0 - math.exp(-0.8 * lai)], dtype=torch.float64),
        atol=1e-12,
    )
    # Higher extinction -> more interception.
    assert bool(out_old["frac_intercepted"] > out_young["frac_intercepted"])


# ------------- scale_factor_kdif scales the coefficient ------------- #


def test_scale_factor_kdif_scales_extinction():
    """cScaleFactorKDIF multiplies the KDIFTB y-value."""
    p = _params()
    p.scale_factor_kdif = torch.tensor(0.5, dtype=torch.float64)
    lai = 4.0
    out = _run(_state(lai=lai, dvs=0.5), p)
    # kdif = 0.5 * 0.6 = 0.3
    expected = 1.0 - math.exp(-0.3 * lai)
    assert torch.allclose(
        out["frac_intercepted"],
        torch.tensor([expected], dtype=torch.float64),
        atol=1e-12,
    )


# ------------- differentiability ------------- #


def test_gradient_flows_through_kdif_and_lai():
    """parint must be differentiable w.r.t. LAI and the KDIFTB y-values."""
    p = _params()
    p.kdiftb = torch.tensor(
        [[0.0, 0.4], [2.0, 0.8]], dtype=torch.float64, requires_grad=True
    )
    s = ModelState.initial(batch_size=1, dtype=torch.float64)
    lai = torch.tensor([2.5], dtype=torch.float64, requires_grad=True)
    s = s.replace(lai=lai, dvs=torch.full_like(s.dvs, 1.0))

    from torchcrop.processes.irradiation import Irradiation

    out = Irradiation()(state=s, params=p, **_geo(b=1))
    out["parint"].sum().backward()

    assert lai.grad is not None and torch.isfinite(lai.grad).all()
    assert lai.grad.abs() > 0
    assert p.kdiftb.grad is not None and torch.isfinite(p.kdiftb.grad).all()
