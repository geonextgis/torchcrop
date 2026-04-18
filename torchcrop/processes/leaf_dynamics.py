"""Leaf area growth and senescence.

Reference
---------
``Lintul5.java`` (leaf block).

Early exponential LAI growth is driven by temperature (``RGRL``); after
canopy closure LAI follows leaf weight × SLA. Senescence is driven by self-
shading, ageing (``RDRTB``), and stress.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.states.model_state import ModelState


class LeafDynamics(nn.Module):
    """Leaf area growth, senescence, dead-leaf accumulation."""

    def forward(
        self,
        state: ModelState,
        g_lv: torch.Tensor,
        dtsu: torch.Tensor,
        tranrf: torch.Tensor,
        nstress: torch.Tensor,
        params: CropParameters,
    ) -> dict[str, torch.Tensor]:
        lai = state.lai
        wlv = state.wlv

        # Exponential LAI growth in the juvenile phase, limited by the critical LAI
        juvenile = (lai < params.laicr).to(lai.dtype)
        glai_exp = lai * (torch.exp(params.rgrl * dtsu) - 1.0) * juvenile

        # Source-limited LAI growth once canopy is established
        glai_src = g_lv * params.sla

        lai_growth = torch.minimum(glai_exp + glai_src, g_lv * params.sla + glai_exp)

        # Senescence: self-shading above LAI_cr, plus developmental senescence
        rdr_shade = params.rdrshm * torch.clamp((lai - params.laicr) / _safe(params.laicr), min=0.0)
        rdr_age = interpolate(params.rdrtb, state.dvs)
        rdr = torch.maximum(rdr_shade, rdr_age)

        # Add water and N stress amplification (bounded to 0.1 d-1 to avoid blow-up)
        rdr_stress = rdr * (1.0 + 0.5 * (1.0 - tranrf) + 0.5 * (1.0 - nstress))
        rdr_stress = torch.clamp(rdr_stress, 0.0, 0.1)

        lai_sen = lai * rdr_stress
        wlv_sen = wlv * rdr_stress

        lai_rate = lai_growth - lai_sen
        wlv_rate = g_lv - wlv_sen

        return {
            "lai_rate": lai_rate,
            "wlv_rate": wlv_rate,
            "wlvd_rate": wlv_sen,
            "lai_growth": lai_growth,
            "lai_sen": lai_sen,
            "rdr": rdr_stress,
        }


def _safe(t: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    return torch.where(t.abs() > eps, t, torch.ones_like(t))
