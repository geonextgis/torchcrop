"""Radiation interception by the canopy.

References:
    SIMPLACE ``Irradiation.java``.

Equations:
    Daily total irradiation (AVRAD) computed from solar geometry and
    atmospheric transmission:

    $$
    SC = 1370 \cdot (1 + 0.033 \cos(2\pi \text{DOY}/365))
    $$

    $$
    A_0 = \text{LIMIT}(-1, 1, \text{SINLD}/\text{COSLD})
    $$

    $$
    \text{DSINB} = 3600 \cdot (\text{DAYL} \cdot \text{SINLD} +
    24 \cdot \text{COSLD} \cdot \sqrt{1 - A_0^2} / \pi)
    $$

    $$
    \text{ANGOT} = \text{max}(0.0001, SC \cdot \text{DSINB})
    $$

    $$
    \text{AVRAD} = \text{min}(0.80 \cdot \text{ANGOT}, \text{DTR})
    $$

    Photosynthetically active radiation is 0.50 of global radiation:

    $$
    \text{PAR} = 0.5 \cdot \text{AVRAD}
    $$

    Fraction intercepted by canopy (Beer-Lambert law):

    $$
    \text{frac} = 1 - \exp(-K \cdot \text{LAI})
    $$

    Intercepted PAR:

    $$
    \text{PARINT} = \text{PAR} \cdot \text{frac}
    $$
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from torchcrop.parameters.crop_params import CropParameters
from torchcrop.states.model_state import ModelState


class Irradiation(nn.Module):
    """Daily total irradiation and PAR interception by canopy.

    Computes AVRAD (daily total irradiation) from solar geometry per
    SIMPLACE Irradiation.java, then calculates PAR interception using
    Beer-Lambert extinction law.
    """

    def forward(
        self,
        state: ModelState,
        doy: torch.Tensor,
        dayl: torch.Tensor,
        sinld: torch.Tensor,
        cosld: torch.Tensor,
        dtr: torch.Tensor,
        params: CropParameters,
    ) -> dict[str, torch.Tensor]:
        """Compute daily irradiation and canopy PAR interception.

        Args:
            state: Current model state (uses ``state.lai``).
            doy: Day of year [1-365], shape ``[B]``.
            dayl: Daylength [hours], shape ``[B]``.
            sinld: sin(declination) [dimensionless], shape ``[B]``.
            cosld: cos(declination) [dimensionless], shape ``[B]``.
            dtr: Daily total radiation [MJ m⁻² d⁻¹], shape ``[B]``.
                Will be converted to J m⁻² d⁻¹ for PENMAN calculation.
            params: Crop parameters; uses ``params.k`` (extinction
                coefficient).

        Returns:
            Dict of ``[B]`` tensors:

                * ``avrad`` [J m⁻² d⁻¹] — Daily total irradiation
                  (computed from solar geometry; converted from input MJ m⁻² d⁻¹).
                * ``atmtr`` [-] — Atmospheric transmission fraction.
                * ``par`` [J m⁻² d⁻¹] — Photosynthetically active
                  radiation (0.5 * avrad).
                * ``parint`` [J m⁻² d⁻¹] — PAR intercepted by canopy.
                * ``frac_intercepted`` [-] — Beer–Lambert interception
                  fraction.
        """
        # Convert DTR from MJ m⁻² d⁻¹ to J m⁻² d⁻¹ for PENMAN calculation
        dtr_j = dtr * 1e6

        # Daily total irradiation (SIMPLACE DailyTotalIrradiation logic)
        aob = torch.clamp(sinld / cosld, min=-1.0, max=1.0)
        dsinb = 3600.0 * (
            dayl * sinld
            + 24.0 * cosld * torch.sqrt(torch.clamp(1.0 - aob * aob, min=0.0)) / math.pi
        )

        # Solar constant [W m⁻²] as function of day of year
        sc = 1370.0 * (1.0 + 0.033 * torch.cos(2.0 * math.pi * doy / 365.0))

        # Extraterrestrial radiation [J m⁻² d⁻¹]
        angot = torch.clamp(sc * dsinb, min=0.0001)

        # Daily total irradiation (minimum of 80% extraterrestrial and measured)
        avrad = torch.min(0.80 * angot, dtr_j)

        # Atmospheric transmission
        atmtr = avrad / angot

        # PAR (50% of global radiation)
        par = 0.5 * avrad

        # Beer-Lambert interception by canopy
        frac = 1.0 - torch.exp(-params.k * state.lai)
        parint = par * frac

        return {
            "avrad": avrad,
            "atmtr": atmtr,
            "par": par,
            "parint": parint,
            "frac_intercepted": frac,
        }
