"""Radiation use efficiency and gross assimilation.

Reference
---------
SIMPLACE ``RadiationUseEfficiency.java`` and biomass block of ``Lintul5.java``.

Equations
---------
Gross biomass growth rate (daily) is:

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
