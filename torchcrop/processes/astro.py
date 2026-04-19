"""Astronomical parameters: solar declination, daylength, extraterrestrial radiation.

References:
    Goudriaan, J. & van Laar, H.H. (1994). *Modelling potential crop growth
    processes* — the daylength formulation used by the SIMPLACE
    ``AstronomicParametersTransformer``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

_DEG2RAD = math.pi / 180.0


class Astro(nn.Module):
    r"""Compute solar declination and astronomical daylength.

    Equations:

        $$
        \delta = -\arcsin\!\left[\sin(23.45^\circ) \cos\!\left(2\pi \frac{DOY + 10}{365}\right)\right]
        $$

        $$
        \cos(H_0) = -\tan(\phi)\tan(\delta)
        $$

        where $\phi$ is latitude in radians.

    Daylength ``DDLP`` is returned in hours; the photoperiodic daylength uses
    a civil-twilight inclusion angle of $-4^\circ$ as in Lintul5.
    """

    def forward(
        self,
        doy: torch.Tensor,
        latitude: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute astronomical parameters.

        Args:
            doy: Day of year, shape ``[B]``.
            latitude: Latitude in degrees, shape ``[B]`` or ``[]``
                (broadcastable).

        Returns:
            Dict with keys ``declination`` [rad], ``daylength`` [h],
            ``ddlp`` [h] (photoperiod), and ``sinld``/``cosld``/``dsinbe``
            (used by irradiation).
        """
        lat_rad = latitude * _DEG2RAD
        dec = -torch.asin(
            torch.sin(torch.tensor(23.45 * _DEG2RAD, dtype=doy.dtype, device=doy.device))
            * torch.cos(2.0 * math.pi * (doy + 10.0) / 365.0)
        )

        sinld = torch.sin(lat_rad) * torch.sin(dec)
        cosld = torch.cos(lat_rad) * torch.cos(dec)
        aob = torch.clamp(sinld / torch.where(cosld.abs() > 1e-12, cosld, torch.ones_like(cosld)), -1.0, 1.0)

        daylength = 12.0 * (1.0 + 2.0 * torch.asin(aob) / math.pi)

        # Photoperiodic daylength — civil twilight angle of -4°
        aob_phot = torch.clamp(
            (-torch.sin(torch.tensor(-4.0 * _DEG2RAD, dtype=doy.dtype, device=doy.device)) + sinld)
            / torch.where(cosld.abs() > 1e-12, cosld, torch.ones_like(cosld)),
            -1.0,
            1.0,
        )
        ddlp = 12.0 * (1.0 + 2.0 * torch.asin(aob_phot) / math.pi)

        dsinbe = 3600.0 * (
            daylength * (sinld + 0.4 * (sinld * sinld + cosld * cosld * 0.5))
            + 12.0 * cosld * (2.0 + 3.0 * 0.4 * sinld) * torch.sqrt(torch.clamp(1.0 - aob * aob, min=0.0)) / math.pi
        )

        return {
            "declination": dec,
            "daylength": daylength,
            "ddlp": ddlp,
            "sinld": sinld,
            "cosld": cosld,
            "dsinbe": dsinbe,
        }
