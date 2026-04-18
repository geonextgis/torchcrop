"""Crop nutrient demand, uptake, translocation and stress indices.

Reference
---------
NPK block of ``Lintul5.java``. A minimal, batch-compatible formulation is
implemented: daily demand from maximum concentrations, supply from the soil
reservoir, and a simple nutrient-stress index tied to leaf N.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.parameters.crop_params import CropParameters
from torchcrop.parameters.soil_params import SoilParameters
from torchcrop.states.model_state import ModelState


class NutrientDemand(nn.Module):
    """Daily NPK demand, uptake and stress indices."""

    def forward(
        self,
        state: ModelState,
        g_lv: torch.Tensor,
        g_st: torch.Tensor,
        g_rt: torch.Tensor,
        g_so: torch.Tensor,
        crop_params: CropParameters,
        soil_params: SoilParameters,
    ) -> dict[str, torch.Tensor]:
        # Maximum-demand uptake per organ (fraction of DM growth)
        n_demand = (
            g_lv * crop_params.nmaxlv
            + g_st * crop_params.nmaxst
            + g_rt * crop_params.nmaxrt
            + g_so * crop_params.nmaxso
        )
        p_demand = (
            g_lv * crop_params.pmaxlv
            + g_st * crop_params.pmaxst
            + g_rt * crop_params.pmaxrt
            + g_so * crop_params.pmaxso
        )
        k_demand = (
            g_lv * crop_params.kmaxlv
            + g_st * crop_params.kmaxst
            + g_rt * crop_params.kmaxrt
            + g_so * crop_params.kmaxso
        )

        n_uptake = torch.minimum(n_demand, soil_params.nmins.expand_as(n_demand))
        p_uptake = torch.minimum(p_demand, soil_params.pmins.expand_as(p_demand))
        k_uptake = torch.minimum(k_demand, soil_params.kmins.expand_as(k_demand))

        # Per-organ allocation proportional to per-organ demand shares
        def split(uptake: torch.Tensor, shares: list[torch.Tensor]) -> list[torch.Tensor]:
            total = sum(shares)
            total = torch.where(total > 1e-10, total, torch.ones_like(total))
            return [uptake * s / total for s in shares]

        n_lv, n_st, n_rt, n_so = split(
            n_uptake,
            [
                g_lv * crop_params.nmaxlv,
                g_st * crop_params.nmaxst,
                g_rt * crop_params.nmaxrt,
                g_so * crop_params.nmaxso,
            ],
        )
        p_lv, p_st, p_rt, p_so = split(
            p_uptake,
            [
                g_lv * crop_params.pmaxlv,
                g_st * crop_params.pmaxst,
                g_rt * crop_params.pmaxrt,
                g_so * crop_params.pmaxso,
            ],
        )
        k_lv, k_st, k_rt, k_so = split(
            k_uptake,
            [
                g_lv * crop_params.kmaxlv,
                g_st * crop_params.kmaxst,
                g_rt * crop_params.kmaxrt,
                g_so * crop_params.kmaxso,
            ],
        )

        # Stress index = uptake / demand (min across N, P, K)
        def ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.clamp(a / torch.where(b > 1e-10, b, torch.ones_like(b)), 0.0, 1.0)

        nstress = torch.minimum(
            torch.minimum(ratio(n_uptake, n_demand), ratio(p_uptake, p_demand)),
            ratio(k_uptake, k_demand),
        )
        # When there is no demand (e.g. DVS=0, no growth), stress = 1 (no stress).
        no_demand = ((n_demand + p_demand + k_demand) < 1e-10).to(nstress.dtype)
        nstress = nstress * (1.0 - no_demand) + no_demand

        return {
            "nstress": nstress,
            "n_uptake": n_uptake,
            "p_uptake": p_uptake,
            "k_uptake": k_uptake,
            "n_lv_rate": n_lv,
            "n_st_rate": n_st,
            "n_rt_rate": n_rt,
            "n_so_rate": n_so,
            "p_lv_rate": p_lv,
            "p_st_rate": p_st,
            "p_rt_rate": p_rt,
            "p_so_rate": p_so,
            "k_lv_rate": k_lv,
            "k_st_rate": k_st,
            "k_rt_rate": k_rt,
            "k_so_rate": k_so,
        }
