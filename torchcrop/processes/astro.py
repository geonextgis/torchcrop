"""Astronomical parameters: solar declination, daylength, extraterrestrial radiation.

References:
    Goudriaan, J. & van Laar, H.H. (1994). *Modelling potential crop growth
    processes* — the daylength formulation used by the SIMPLACE
    ``AstronomicParametersTransformer``.

Equations:
    $$
    \\delta = -\\arcsin\\left[\\sin(23.45^\\circ) \\cos\\left(2\\pi \\frac{DOY + 10}{365}\\right)\\right]
    $$

    $$
    \\cos(H_0) = -\\tan(\\phi)\\tan(\\delta)
    $$

    where $\phi$ is latitude in radians.

    Daylength ``DDLP`` is returned in hours; the photoperiodic daylength uses
    a civil-twilight inclusion angle of $-4^\\circ$ as in Lintul5.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

_DEG2RAD = math.pi / 180.0


class Astro(nn.Module):
    """Compute solar declination and astronomical daylength."""

    def forward(
        self,
        doy: torch.Tensor,
        latitude: torch.Tensor,
        inclination: float | torch.Tensor = -4.0,
    ) -> dict[str, torch.Tensor]:
        """Compute astronomical parameters.

        Args:
            doy: Day of year, shape ``[B]``.
            latitude: Latitude in degrees, shape ``[B]`` or ``[]``
                (broadcastable).
            inclination: Sun inclination angle in degrees used for the
                photoperiodic daylength calculation. Defaults to ``-4.0``
                (civil-twilight value used in Lintul5).

        Returns:
            Dict with keys ``declination`` [rad], ``daylength`` [h],
            ``ddlp`` [h] (photoperiod), and ``sinld``/``cosld``/``dsinbe``
            (used by irradiation).
        """
        lat_rad = latitude * _DEG2RAD
        dec = -torch.asin(
            torch.sin(
                torch.tensor(23.45 * _DEG2RAD, dtype=doy.dtype, device=doy.device)
            )
            * torch.cos(2.0 * math.pi * (doy + 10.0) / 365.0)
        )

        sinld = torch.sin(lat_rad) * torch.sin(dec)
        cosld = torch.cos(lat_rad) * torch.cos(dec)
        aob = torch.clamp(
            sinld / torch.where(cosld.abs() > 1e-12, cosld, torch.ones_like(cosld)),
            -1.0,
            1.0,
        )

        daylength = 12.0 * (1.0 + 2.0 * torch.asin(aob) / math.pi)

        # Photoperiodic daylength — configurable sun inclination angle
        inclination_tensor = torch.as_tensor(
            inclination, dtype=doy.dtype, device=doy.device
        )
        aob_phot = torch.clamp(
            (-torch.sin(inclination_tensor * _DEG2RAD) + sinld)
            / torch.where(cosld.abs() > 1e-12, cosld, torch.ones_like(cosld)),
            -1.0,
            1.0,
        )
        ddlp = 12.0 * (1.0 + 2.0 * torch.asin(aob_phot) / math.pi)

        dsinbe = 3600.0 * (
            daylength * (sinld + 0.4 * (sinld * sinld + cosld * cosld * 0.5))
            + 12.0
            * cosld
            * (2.0 + 3.0 * 0.4 * sinld)
            * torch.sqrt(torch.clamp(1.0 - aob * aob, min=0.0))
            / math.pi
        )

        return {
            "declination": dec,
            "daylength": daylength,
            "ddlp": ddlp,
            "sinld": sinld,
            "cosld": cosld,
            "dsinbe": dsinbe,
        }
