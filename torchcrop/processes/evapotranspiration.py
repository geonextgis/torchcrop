"""Potential evapotranspiration.

A light-weight Priestley–Taylor approximation is implemented by default; this
is adequate for differentiable experimentation. For operational use, swap in
a full FAO-56 Penman–Monteith variant (kept as a TODO hook).

Equations
---------
.. math::
    ET_0 = \\alpha \\cdot \\frac{\\Delta}{\\Delta + \\gamma} \\cdot \\frac{R_n}{\\lambda}

with :math:`\\alpha = 1.26`, :math:`\\lambda = 2.45\\ \\text{MJ kg}^{-1}`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PotentialEvapoTranspiration(nn.Module):
    """Priestley–Taylor reference evapotranspiration."""

    def __init__(self, alpha_pt: float = 1.26) -> None:
        super().__init__()
        self.alpha_pt = alpha_pt

    def forward(
        self,
        davtmp: torch.Tensor,
        irrad: torch.Tensor,
        lai: torch.Tensor,
        k_ext: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        davtmp : torch.Tensor
            Mean daily air temperature [°C], shape ``[B]``.
        irrad : torch.Tensor
            Global daily radiation [MJ m-2 d-1], shape ``[B]``.
        lai : torch.Tensor
            Leaf area index, shape ``[B]``.
        k_ext : torch.Tensor
            Canopy extinction coefficient.

        Returns
        -------
        dict
            ``et0`` [mm d-1], ``pevap`` (potential soil evap), ``ptran``
            (potential transpiration).
        """
        # Slope of saturation vapour pressure curve [kPa K-1]
        delta = (
            4098.0
            * (0.6108 * torch.exp(17.27 * davtmp / (davtmp + 237.3)))
            / torch.clamp((davtmp + 237.3) ** 2, min=1e-6)
        )
        gamma = torch.full_like(davtmp, 0.067)  # psychrometric constant [kPa K-1]
        lambda_v = 2.45  # latent heat of vaporization [MJ kg-1]

        rn = 0.75 * irrad  # crude net radiation estimate (albedo 0.25)
        et0 = self.alpha_pt * (delta / (delta + gamma)) * rn / lambda_v
        et0 = torch.clamp(et0, min=0.0)

        frac = 1.0 - torch.exp(-k_ext * lai)
        ptran = et0 * frac
        pevap = et0 * (1.0 - frac)

        return {"et0": et0, "ptran": ptran, "pevap": pevap}
