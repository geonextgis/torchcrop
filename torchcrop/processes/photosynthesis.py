"""Radiation use efficiency and gross assimilation.

References:
    SIMPLACE ``RadiationUseEfficiency.java`` and biomass block of
    ``Lintul5.java``.

Equations:
    Gross biomass growth rate (daily):

    .. math::
        \\text{GTOTAL} = \\text{RUE} \\cdot f_T(T) \\cdot f_{CO_2} \\cdot
        \\text{NSTRESS} \\cdot \\text{TRANRF} \\cdot \\text{PARINT}

    where ``RUE`` has units of g dry matter per MJ intercepted PAR.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters


class Photosynthesis(nn.Module):
    """Radiation use efficiency and biomass assimilation."""

    def forward(
        self,
        parint: torch.Tensor,
        davtmp: torch.Tensor,
        tranrf: torch.Tensor,
        nstress: torch.Tensor,
        params: CropParameters,
    ) -> dict[str, torch.Tensor]:
        """Compute gross daily biomass production.

        Args:
            parint: PAR intercepted by the canopy [MJ m⁻² d⁻¹], shape
                ``[B]``.
            davtmp: Mean daily air temperature [°C], shape ``[B]``.
            tranrf: Water-stress reduction factor in ``[0, 1]``, shape
                ``[B]``.
            nstress: Nutrient-stress reduction factor in ``[0, 1]``, shape
                ``[B]``.
            params: Crop parameters; uses ``rue``, ``tmpf_tb``, ``co2``.

        Returns:
            Dict of ``[B]`` tensors grouped as follows.

            Rate variables (consumed by :class:`Partitioning`):

                * ``gtotal`` [g DM m⁻² d⁻¹] — Gross total biomass
                  production rate
                  ``= rue_eff * parint * tranrf * nstress``. Split into
                  organ-specific growth rates ``g_lv``, ``g_st``,
                  ``g_root``, ``g_so`` by the partitioning module.

            Diagnostics:

                * ``rue_eff`` [g MJ⁻¹] — Effective radiation use
                  efficiency after temperature and CO₂ adjustments.
                * ``tmp_factor`` [-] — Temperature reduction factor on RUE
                  from ``AFGEN(tmpf_tb, T_avg)``, in ``[0, 1]``.
                * ``co2_factor`` [-] — CO₂ enhancement factor (1.0 at 360
                  ppm, up to 1.3 at 700 ppm).
        """
        tmp_factor = interpolate(params.tmpf_tb, davtmp)
        # Simple CO2 response (linear around reference 360 ppm, saturates at 700)
        co2_factor = 1.0 + 0.3 * torch.clamp((params.co2 - 360.0) / (700.0 - 360.0), 0.0, 1.0)
        rue_eff = params.rue * tmp_factor * co2_factor
        gtotal = rue_eff * parint * tranrf * nstress
        return {
            "rue_eff": rue_eff,
            "gtotal": gtotal,
            "tmp_factor": tmp_factor,
            "co2_factor": co2_factor,
        }
