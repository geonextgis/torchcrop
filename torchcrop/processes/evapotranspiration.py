r"""Potential evapotranspiration.

Implements the PENMAN formula for computing potential evaporation from water
and soil surfaces, and potential transpiration of a closed crop canopy,
accounting for CO2 effects.

References:
    SIMPLACE ``PotentialEvapoTranspiration.java`` and ``LintulFunctions.PENMAN``.
    Based on Penman (1948) with modifications by van Kraalingen.

Equations:

    See PENMAN method for detailed derivations. Key outputs:

    - E0: Potential evaporation from open water surface [mm d⁻¹]
    - ES0: Potential evaporation from bare soil [mm d⁻¹]
    - ETC: Potential transpiration of closed canopy (CO2-corrected) [mm d⁻¹]

    Potential transpiration rate:

    $$
    P_t = \text{CFET} \cdot ETC \cdot F_{INT}
    $$

    Potential soil evaporation:

    $$
    P_s = ES_0 \cdot (1 - F_{INT})
    $$

    where CFET is a crop-specific correction factor and F_INT is fractional
    light interception.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PotentialEvapoTranspiration(nn.Module):
    """PENMAN-based potential evapotranspiration calculation.

    Implements the full PENMAN formula per SIMPLACE PotentialEvapoTranspiration.java.
    Outputs reference ET (E0, ES0, ETC) which are then split into potential
    transpiration and soil evaporation based on light interception.

    Args:
        altitude: Station altitude [m] (default 0).
        cfet: Crop-specific correction factor for transpiration (default 1.0).
        co2: Atmospheric CO2 concentration [ppm] (default 370).
    """

    def __init__(
        self,
        altitude: float = 0.0,
        cfet: float = 1.0,
        co2: float = 370.0,
    ) -> None:
        super().__init__()
        self.altitude = altitude
        self.cfet = cfet
        self.co2 = co2

        # CO2 correction table (x: CO2 ppm, y: correction factor)
        # From SIMPLACE cET0CorrectionTableCo2 and cET0CorrectionTableFactor
        self.co2_table_x = torch.tensor([40.0, 360.0, 720.0, 1000.0, 2000.0])
        self.co2_table_y = torch.tensor([1.05, 1.00, 0.95, 0.92, 0.92])

    def _interpolate_co2_factor(self, co2_val: torch.Tensor) -> torch.Tensor:
        """Linear interpolation of CO2 correction factor.

        Args:
            co2_val: CO2 concentration [ppm], shape ``[B]``.

        Returns:
            CO2 correction factor, shape ``[B]``.
        """
        # Simple linear interpolation between table points
        # Find bracketing indices
        x = self.co2_table_x
        y = self.co2_table_y

        # Clamp to table bounds
        co2_clamped = torch.clamp(co2_val, min=x[0], max=x[-1])

        # Find indices for interpolation
        factor = torch.ones_like(co2_clamped)
        for i in range(len(x) - 1):
            in_range = (co2_clamped >= x[i]) & (co2_clamped <= x[i + 1])
            if in_range.any():
                # Linear interpolation
                alpha = (co2_clamped[in_range] - x[i]) / (x[i + 1] - x[i])
                factor[in_range] = (1.0 - alpha) * y[i] + alpha * y[i + 1]

        return factor

    def forward(
        self,
        tmin: torch.Tensor,
        tmax: torch.Tensor,
        wind: torch.Tensor,
        vap: torch.Tensor,
        avrad: torch.Tensor,
        atmtr: torch.Tensor,
        frac_int: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute PENMAN potential ET and split into canopy/soil fluxes.

        Args:
            tmin: Minimum daily air temperature [°C], shape ``[B]``.
            tmax: Maximum daily air temperature [°C], shape ``[B]``.
            wind: Average wind speed [m s⁻¹], shape ``[B]``.
            vap: Vapour pressure [kPa], shape ``[B]``.
            avrad: Daily total irradiation [J m⁻² d⁻¹], shape ``[B]``.
            atmtr: Atmospheric transmission fraction [-], shape ``[B]``.
            frac_int: Fractional light interception [-], shape ``[B]``.

        Returns:
            Dict of ``[B]`` tensors:

                * ``e0`` [mm d⁻¹] — Potential evaporation from open water.
                * ``es0`` [mm d⁻¹] — Potential evaporation from bare soil.
                * ``etc`` [mm d⁻¹] — Potential transpiration (CO2-corrected).
                * ``ptran`` [mm d⁻¹] — Potential canopy transpiration
                  (= CFET * ETC * frac_int).
                * ``pevap`` [mm d⁻¹] — Potential soil evaporation
                  (= ES0 * (1 - frac_int)).
        """
        # Constants from PENMAN formula (SIMPLACE LintulFunctions.PENMAN)
        A = 0.20
        B = 0.56
        REFCFW = 0.05  # Albedo for water
        REFCFS = 0.15  # Albedo for soil
        REFCFC = 0.25  # Albedo for canopy
        LHVAP = 2.45e6  # Latent heat of evaporation [J kg-1]
        STBC = 4.9e-3  # Stefan-Boltzmann constant [J m-2 d-1 K-4]
        PSYCON = 0.000662  # Psychrometric constant [K-1]

        # Convert vapour pressure from kPa to mbar (1 kPa = 10 mbar)
        vap_mbar = vap * 10.0

        # Average daily temperature
        tmpa = (tmin + tmax) / 2.0

        # Temperature difference
        tdif = tmax - tmin

        # Wind function coefficient (depends on temperature range)
        bu = 0.54 + 0.35 * torch.clamp((tdif - 12.0) / 4.0, min=0.0, max=1.0)

        # Barometric pressure [mbar]
        pbar = 1013.0 * torch.exp(-0.034 * self.altitude / (tmpa + 273.0))

        # Psychrometric constant [mbar K-1]
        gamma = PSYCON * pbar

        # Saturated vapour pressure [mbar] per Goudriaan (1977)
        svap = 6.11 * torch.exp(17.4 * tmpa / (tmpa + 239.0))

        # Measured vapour pressure should not exceed saturated vapour pressure
        vap_clamped = torch.clamp(vap_mbar, max=svap)

        # Slope of saturation vapour pressure curve [mbar K-1]
        delta = 239.0 * 17.4 * svap / torch.clamp((tmpa + 239.0) ** 2, min=1e-6)

        # Relative sunshine duration (from Angstrom formula)
        relssd = torch.clamp((atmtr - A) / B, min=0.0, max=1.0)

        # Net outgoing long-wave radiation [J m-2 d-1]
        rb = (
            STBC
            * (tmpa + 273.0) ** 4
            * (0.56 - 0.079 * torch.sqrt(torch.clamp(vap_clamped, min=0.0)))
            * (0.1 + 0.9 * relssd)
        )

        # Net absorbed radiation [J m-2 d-1]
        rnw = avrad * (1.0 - REFCFW) - rb
        rns = avrad * (1.0 - REFCFS) - rb
        rnc = avrad * (1.0 - REFCFC) - rb

        # Evaporative demand of atmosphere [mm d-1]
        ea = 0.26 * (svap - vap_clamped) * (0.5 + bu * wind)
        eac = 0.26 * (svap - vap_clamped) * (1.0 + bu * wind)

        # PENMAN formula [mm d-1]
        e0 = (delta * (rnw / LHVAP) + gamma * ea) / torch.clamp(delta + gamma, min=1e-6)
        es0 = (delta * (rns / LHVAP) + gamma * ea) / torch.clamp(
            delta + gamma, min=1e-6
        )
        et0 = (delta * (rnc / LHVAP) + gamma * eac) / torch.clamp(
            delta + gamma, min=1e-6
        )

        # Ensure non-negative
        e0 = torch.clamp(e0, min=0.0)
        es0 = torch.clamp(es0, min=0.0)
        et0 = torch.clamp(et0, min=0.0)

        # CO2 correction for ET0
        co2_factor = self._interpolate_co2_factor(torch.full_like(e0, self.co2))
        etc = et0 * co2_factor

        # Potential transpiration and soil evaporation split by light interception
        ptran = torch.clamp(self.cfet * etc * frac_int, min=0.0001)
        pevap = es0 * (1.0 - frac_int)

        return {
            "e0": e0,
            "es0": es0,
            "etc": etc,
            "ptran": ptran,
            "pevap": pevap,
        }
