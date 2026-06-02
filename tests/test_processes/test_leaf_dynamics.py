"""Tests for ``LeafDynamics`` against the Java ``GLA`` / ``DEATHL`` spec.

Reference: ``simplace/sim/components/models/lintul5/LintulFunctions.java``
(``GLA`` lines 998–1019, ``DEATHL`` lines 938–977) and ``Lintul5.java``
line 1408 (SLA = SLATB · exp(−NSLA·(1−NPKI))).
"""

from __future__ import annotations

import math

import torch

from torchcrop.parameters.crop_params import CropParameters
from torchcrop.processes.leaf_dynamics import LeafDynamics
from torchcrop.states.model_state import ModelState


def _state(
    b: int = 2,
    dvs: float = 0.5,
    lai: float = 1.0,
    wlv: float = 50.0,
    dtype: torch.dtype = torch.float64,
    tsump: float = 500.0,
) -> ModelState:
    """Helper state. ``tsump`` defaults above ``tsumem=110`` so EMERG=true."""
    s = ModelState.initial(batch_size=b, dtype=dtype)
    return s.replace(
        dvs=torch.full_like(s.dvs, dvs),
        lai=torch.full_like(s.lai, lai),
        wlv=torch.full_like(s.wlv, wlv),
        tsump=torch.full_like(s.tsump, tsump),
    )


def _params() -> CropParameters:
    return CropParameters().to(dtype=torch.float64)


# ----------------------- shape ----------------------- #


def test_leaf_dynamics_output_shapes():
    leaf = LeafDynamics()
    s = _state(b=3)
    p = _params()
    z = torch.zeros(3, dtype=torch.float64)
    out = leaf(
        state=s,
        g_lv=z + 2.0,
        davtmp=z + 15.0,
        tranrf=z + 1.0,
        nstress=z + 1.0,
        params=p,
    )
    for key in ("lai_rate", "wlv_rate", "wlvd_rate", "lai_growth", "lai_sen", "rdr", "sla"):
        assert out[key].shape == (3,), f"{key} shape mismatch"


# ------------------- GLA: emergence branch ------------------- #


def test_emergence_branch_uses_laii():
    """When LAI == 0, GLAI must equal LAII (DELT = 1 d)."""
    leaf = LeafDynamics()
    s = _state(b=2, dvs=0.0, lai=0.0, wlv=0.0)
    p = _params()
    z = torch.zeros(2, dtype=torch.float64)
    out = leaf(
        state=s,
        g_lv=z + 0.0,
        davtmp=z + 15.0,
        tranrf=z + 1.0,
        nstress=z + 1.0,
        params=p,
    )
    expected = float(p.laii.item())
    assert torch.allclose(out["lai_growth"], torch.full((2,), expected, dtype=torch.float64))


# ------------------- GLA: juvenile branch ------------------- #


def test_juvenile_branch_formula():
    """DVS < 0.2 and LAI < 0.75 → exponential growth with TRANRF, NPKI.

    GLAI = LAI · (exp(RGRLAI · DTEFF) − 1) · TRANRF · exp(−NLAI · (1−NPKI))
    """
    leaf = LeafDynamics()
    dvs = 0.1
    lai = 0.5
    s = _state(b=1, dvs=dvs, lai=lai, wlv=20.0)
    p = _params()
    tranrf = torch.tensor([0.7], dtype=torch.float64)
    nstress = torch.tensor([0.8], dtype=torch.float64)
    out = leaf(
        state=s,
        g_lv=torch.tensor([1.5], dtype=torch.float64),
        davtmp=torch.tensor([15.0], dtype=torch.float64),
        tranrf=tranrf,
        nstress=nstress,
        params=p,
    )
    rgrl = float(p.rgrl.item())
    nlai = float(p.nlai.item())
    # DTEFF = max(0, davtmp - tbase); tbase defaults to 0 -> dteff = 15.
    dteff = 15.0 - float(p.tbase.item())
    expected = (
        lai
        * (math.exp(rgrl * dteff) - 1.0)
        * 0.7
        * math.exp(-nlai * (1.0 - 0.8))
    )
    assert torch.allclose(
        out["lai_growth"], torch.tensor([expected], dtype=torch.float64), atol=1e-12
    )


def test_juvenile_thermal_time_uses_tbase():
    """Juvenile DTEFF = max(0, davtmp - tbase) (Lintul5.java:1436).

    With a non-zero base temperature the effective thermal time — and
    therefore the exponential LAI growth — must shrink accordingly, and
    clamp to zero when davtmp <= tbase.
    """
    leaf = LeafDynamics()
    dvs, lai = 0.1, 0.5
    s = _state(b=1, dvs=dvs, lai=lai, wlv=20.0)
    p = _params()
    p.tbase = torch.tensor(8.0, dtype=torch.float64)  # e.g. maize-like base
    rgrl = float(p.rgrl.item())

    # davtmp = 20 -> dteff = 12.
    out = leaf(
        state=s,
        g_lv=torch.tensor([1.5], dtype=torch.float64),
        davtmp=torch.tensor([20.0], dtype=torch.float64),
        tranrf=torch.tensor([1.0], dtype=torch.float64),
        nstress=torch.tensor([1.0], dtype=torch.float64),
        params=p,
    )
    expected = lai * (math.exp(rgrl * 12.0) - 1.0)
    assert torch.allclose(
        out["lai_growth"], torch.tensor([expected], dtype=torch.float64), atol=1e-12
    )

    # davtmp = 5 < tbase=8 -> dteff clamps to 0 -> no juvenile growth.
    out_cold = leaf(
        state=s,
        g_lv=torch.tensor([1.5], dtype=torch.float64),
        davtmp=torch.tensor([5.0], dtype=torch.float64),
        tranrf=torch.tensor([1.0], dtype=torch.float64),
        nstress=torch.tensor([1.0], dtype=torch.float64),
        params=p,
    )
    assert torch.allclose(
        out_cold["lai_growth"], torch.zeros(1, dtype=torch.float64), atol=1e-12
    )


# ------------------- GLA: mature branch ------------------- #


def test_mature_branch_is_sla_times_glv():
    """Once past juvenile, GLAI = SLA · GLV."""
    leaf = LeafDynamics()
    s = _state(b=1, dvs=1.0, lai=3.0, wlv=100.0)
    p = _params()
    glv = torch.tensor([2.5], dtype=torch.float64)
    nstress = torch.tensor([1.0], dtype=torch.float64)
    out = leaf(
        state=s,
        g_lv=glv,
        davtmp=torch.tensor([15.0], dtype=torch.float64),
        tranrf=torch.tensor([1.0], dtype=torch.float64),
        nstress=nstress,
        params=p,
    )
    # SLA at NPKI=1 reduces to scale_factor * SLATB(DVS); SLATB is flat 0.0212.
    sla = float(p.scale_factor_sla.item()) * 0.0212
    assert torch.allclose(
        out["lai_growth"], torch.tensor([sla * 2.5], dtype=torch.float64), atol=1e-12
    )


# ------------------- DEATHL: RDRDV gating ------------------- #


def test_rdrdv_zero_before_dvsdlt():
    """RDRDV = 0 while DVS < DVSDLT, regardless of temperature."""
    leaf = LeafDynamics()
    s = _state(b=1, dvs=0.5, lai=2.0, wlv=100.0)  # below default dvsdlt=1.0
    p = _params()
    out = leaf(
        state=s,
        g_lv=torch.tensor([0.0], dtype=torch.float64),
        davtmp=torch.tensor([50.0], dtype=torch.float64),  # extreme T → RDRTMP large
        tranrf=torch.tensor([1.0], dtype=torch.float64),   # no drought
        nstress=torch.tensor([1.0], dtype=torch.float64),  # no NPK death
        params=p,
    )
    # LAI=2 < LAICR=4 → RDRSH=0; TRANRF=1 → RDRDRY=0; DVS<DVSDLT → RDRDV=0
    assert torch.allclose(out["rdr"], torch.zeros(1, dtype=torch.float64))
    assert torch.allclose(out["lai_sen"], torch.zeros(1, dtype=torch.float64))


def test_rdrdv_active_after_dvsdlt_uses_temperature_table():
    """After DVSDLT, RDRDV = RDRLTB(TMPA)."""
    leaf = LeafDynamics()
    s = _state(b=1, dvs=1.5, lai=2.0, wlv=100.0)
    p = _params()
    out = leaf(
        state=s,
        g_lv=torch.tensor([0.0], dtype=torch.float64),
        davtmp=torch.tensor([30.0], dtype=torch.float64),  # RDRLTB(30) = 0.05
        tranrf=torch.tensor([1.0], dtype=torch.float64),
        nstress=torch.tensor([1.0], dtype=torch.float64),
        params=p,
    )
    expected_rdr = 0.05  # from rdrltb default at T=30 °C
    assert torch.allclose(
        out["rdr"], torch.tensor([expected_rdr], dtype=torch.float64), atol=1e-12
    )
    # dlais = LAI * RDR; no NPK extra (NPKI=1)
    assert torch.allclose(
        out["lai_sen"],
        torch.tensor([2.0 * expected_rdr], dtype=torch.float64),
        atol=1e-12,
    )


# ------------------- DEATHL: RDRSH (self-shading) ------------------- #


def test_rdrsh_above_laicr():
    """RDRSH = RDRSHM · (LAI − LAICR) / LAICR when LAI > LAICR."""
    leaf = LeafDynamics()
    s = _state(b=1, dvs=0.5, lai=6.0, wlv=200.0)
    p = _params()
    out = leaf(
        state=s,
        g_lv=torch.tensor([0.0], dtype=torch.float64),
        davtmp=torch.tensor([15.0], dtype=torch.float64),
        tranrf=torch.tensor([1.0], dtype=torch.float64),
        nstress=torch.tensor([1.0], dtype=torch.float64),
        params=p,
    )
    rdrsh = float(p.rdrshm.item()) * (6.0 - float(p.laicr.item())) / float(p.laicr.item())
    assert torch.allclose(
        out["rdr"], torch.tensor([rdrsh], dtype=torch.float64), atol=1e-12
    )


# ------------------- DEATHL: RDRDRY (drought) ------------------- #


def test_rdrdry_under_water_stress():
    """RDRDRY = (1 − TRANRF) · RDRL dominates when other drivers are zero."""
    leaf = LeafDynamics()
    s = _state(b=1, dvs=0.5, lai=2.0, wlv=100.0)  # LAI<LAICR, DVS<DVSDLT
    p = _params()
    out = leaf(
        state=s,
        g_lv=torch.tensor([0.0], dtype=torch.float64),
        davtmp=torch.tensor([15.0], dtype=torch.float64),
        tranrf=torch.tensor([0.5], dtype=torch.float64),
        nstress=torch.tensor([1.0], dtype=torch.float64),
        params=p,
    )
    expected = 0.5 * float(p.rdrl.item())  # (1 - 0.5) * 0.05 = 0.025
    assert torch.allclose(
        out["rdr"], torch.tensor([expected], dtype=torch.float64), atol=1e-12
    )


# ------------------- DEATHL: additive NPK death ------------------- #


def test_npk_death_is_additive_and_uses_sla_for_area():
    """DLVNS = WLVG · RDRNS · (1 − NPKI); DLAINS = DLVNS · SLA (mass→area)."""
    leaf = LeafDynamics()
    s = _state(b=1, dvs=0.5, lai=2.0, wlv=100.0)
    p = _params()
    nstress = torch.tensor([0.6], dtype=torch.float64)
    out = leaf(
        state=s,
        g_lv=torch.tensor([0.0], dtype=torch.float64),
        davtmp=torch.tensor([15.0], dtype=torch.float64),
        tranrf=torch.tensor([1.0], dtype=torch.float64),
        nstress=nstress,
        params=p,
    )
    # RDR drivers are all zero here (LAI<LAICR, TRANRF=1, DVS<DVSDLT)
    rdrns = float(p.rdrns.item())
    dlvns = 100.0 * rdrns * (1.0 - 0.6)
    sla = (
        float(p.scale_factor_sla.item())
        * 0.0212
        * math.exp(-float(p.nsla.item()) * (1.0 - 0.6))
    )
    dlains = dlvns * sla
    assert torch.allclose(
        out["wlvd_rate"],
        torch.tensor([dlvns], dtype=torch.float64),
        atol=1e-12,
    )
    assert torch.allclose(
        out["lai_sen"],
        torch.tensor([dlains], dtype=torch.float64),
        atol=1e-12,
    )


# ------------------- end-to-end: differentiability ------------------- #


def test_gradient_flow_through_leaf_dynamics():
    """Gradients must flow through GLA and DEATHL to inputs."""
    leaf = LeafDynamics()
    s = _state(b=1, dvs=0.5, lai=2.0, wlv=100.0)
    p = _params()
    g_lv = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    tranrf = torch.tensor([0.8], dtype=torch.float64, requires_grad=True)
    nstress = torch.tensor([0.7], dtype=torch.float64, requires_grad=True)
    davtmp = torch.tensor([25.0], dtype=torch.float64, requires_grad=True)
    out = leaf(
        state=s,
        g_lv=g_lv,
        davtmp=davtmp,
        tranrf=tranrf,
        nstress=nstress,
        params=p,
    )
    (out["lai_rate"].sum() + out["wlv_rate"].sum()).backward()
    assert g_lv.grad is not None and torch.isfinite(g_lv.grad).all()
    assert tranrf.grad is not None and torch.isfinite(tranrf.grad).all()
    assert nstress.grad is not None and torch.isfinite(nstress.grad).all()


# ------------------- precedence: emergence > juvenile ------------------- #


def test_emergence_wins_over_juvenile_when_lai_zero():
    """If LAI == 0 even with DVS<0.2, the emergence branch must fire."""
    leaf = LeafDynamics()
    s = _state(b=1, dvs=0.1, lai=0.0, wlv=0.0)
    p = _params()
    out = leaf(
        state=s,
        g_lv=torch.tensor([3.0], dtype=torch.float64),
        davtmp=torch.tensor([15.0], dtype=torch.float64),
        tranrf=torch.tensor([1.0], dtype=torch.float64),
        nstress=torch.tensor([1.0], dtype=torch.float64),
        params=p,
    )
    expected = float(p.laii.item())
    assert torch.allclose(
        out["lai_growth"], torch.tensor([expected], dtype=torch.float64), atol=1e-12
    )
