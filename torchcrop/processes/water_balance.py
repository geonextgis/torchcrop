"""Soil water balance — single-layer bucket model.

Reference
---------
SIMPLACE ``WaterBalance.java``. The simplified bucket implementation below
tracks a single effective root-zone water storage ``WA`` [mm]. It produces
actual transpiration from potential transpiration via a linear water-stress
response ``TRANRF`` in :math:`[0, 1]`.

Equations
---------
Available water, capacity limits at field capacity and wilting point:

.. math::
    W_\\text{fc}  = 1000 \\cdot \\theta_\\text{fc} \\cdot D_\\text{root}
    \\qquad
    W_\\text{wp}  = 1000 \\cdot \\theta_\\text{wp} \\cdot D_\\text{root}

Water stress reduction factor (linear between wilting and critical point):

.. math::
    \\text{TRANRF} = \\text{clip}\\!\\left(\\frac{W_a - W_\\text{wp}}
    {W_\\text{fc} - W_\\text{wp}},\\ 0,\\ 1\\right)

Actual transpiration:

.. math::
    T_a = \\text{PTRAN} \\cdot \\text{TRANRF}

Mass balance update:

.. math::
    W_{a,t+1} = W_{a,t} + P - T_a - E_a - \\text{drain} - \\text{runoff}
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.parameters.soil_params import SoilParameters
from torchcrop.states.model_state import ModelState


class WaterBalance(nn.Module):
    """Single-layer bucket water balance."""

    def forward(
        self,
        state: ModelState,
        rain: torch.Tensor,
        pevap: torch.Tensor,
        ptran: torch.Tensor,
        params: SoilParameters,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        rain : torch.Tensor
            Daily precipitation [mm d-1], shape ``[B]``.
        pevap : torch.Tensor
            Potential soil evaporation [mm d-1], shape ``[B]``.
        ptran : torch.Tensor
            Potential transpiration [mm d-1], shape ``[B]``.

        Returns
        -------
        dict
            ``wa_rate`` [mm d-1], ``tran`` actual transpiration,
            ``evap`` actual evap, ``tranrf`` water-stress factor in [0, 1],
            ``drain`` drainage losses, ``runoff``.
        """
        wfc = 1000.0 * params.wcfc * state.rootd
        wwp = 1000.0 * params.wcwp * state.rootd
        wst_capacity = 1000.0 * params.wcst * state.rootd

        denom = torch.clamp(wfc - wwp, min=1e-6)
        tranrf = torch.clamp((state.wa - wwp) / denom, min=0.0, max=1.0)

        tran = ptran * tranrf
        # Evaporation limited by water content above the air-dry value
        wad = 1000.0 * params.wcad * state.rootd
        evap_capacity = torch.clamp(state.wa - wad, min=0.0)
        evap = torch.minimum(pevap, evap_capacity)

        runoff = params.runfr * rain
        infiltration = rain - runoff

        # Available water after inputs and losses — then drain the excess above WFC
        wa_tentative = state.wa + infiltration - tran - evap
        excess = torch.clamp(wa_tentative - wfc, min=0.0)
        drain = torch.minimum(excess, params.drate.expand_as(excess))

        wa_rate = infiltration - tran - evap - drain - runoff * 0.0  # runoff already removed

        # Cap at saturation capacity (soft by using a drain-like outflow)
        over_sat = torch.clamp(state.wa + wa_rate - wst_capacity, min=0.0)
        wa_rate = wa_rate - over_sat

        return {
            "wa_rate": wa_rate,
            "tran": tran,
            "evap": evap,
            "tranrf": tranrf,
            "drain": drain + over_sat,
            "runoff": runoff,
        }
