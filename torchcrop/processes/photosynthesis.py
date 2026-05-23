"""Radiation use efficiency.

This module produces only ``RUE`` and the overall correction factor
``RTMCO``. The gross biomass growth rate (``GTOTAL``) is produced
downstream in the partitioning / model-level code, not here.

Equations
---------
Effective daytime temperature:

$$
\\text{DTEMP} = \\text{TMAX} - f \\cdot (\\text{TMAX} - \\text{TMIN})
$$

where ``f`` is ``day_temp_factor`` (default ``0.25``, i.e. daytime
mean).

Temperature × CO\\(_2\\) correction factor on RUE:

$$
\\text{RTMCO} = \\text{TMPFTB}(\\text{DTEMP}) \\cdot
                \\text{TMNFTB}(\\text{TMIN}) \\cdot
                \\text{COTB}(\\text{CO}_2)
$$

DVS-dependent radiation use efficiency:

$$
\\text{RUE} = \\text{scale\\_factor\\_rue} \\cdot \\text{RUETB}(\\text{DVS})
$$

``RUE`` is in units of g dry matter per MJ intercepted PAR.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters


class Photosynthesis(nn.Module):
    """Radiation use efficiency component.

    Outputs ``RUE`` from ``ruetb(DVS)`` and an overall correction
    factor ``RTMCO`` that combines daytime-temperature,
    minimum-temperature and CO\\(_2\\) responses. ``parint``,
    ``tranrf`` and ``nstress`` are *not* consumed here — they enter
    the ``GTOTAL`` formula in the downstream biomass block.
    """

    def forward(
        self,
        tmax: torch.Tensor,
        tmin: torch.Tensor,
        dvs: torch.Tensor,
        params: CropParameters,
        co2: torch.Tensor | float = 370.0,
    ) -> dict[str, torch.Tensor]:
        """Compute RUE and the RTMCO correction factor.

        Args:
            tmax: Daily maximum air temperature [°C], shape ``[B]``.
            tmin: Daily minimum air temperature [°C], shape ``[B]``.
            dvs: Development stage [-] (0–2), shape ``[B]``.
            params: Crop parameters; uses ``ruetb``, ``scale_factor_rue``,
                ``tmpftb``, ``tmnftb``, ``cotb``, ``day_temp_factor``.
            co2: Atmospheric CO₂ concentration [ppm], a scalar or
                shape broadcastable to ``[B]`` (default ``370``).
                Supplied by ``SiteParameters.co2`` in the model.

        Returns:
            Dict of ``[B]`` tensors:

            * ``rue`` [g MJ⁻¹] — DVS-dependent radiation use
              efficiency ``scale_factor_rue · RUETB(DVS)``.
            * ``rtmco`` [-] — Overall correction factor
              ``RTMP · RCO``.
            * ``rco`` [-] — CO₂ correction factor ``COTB(CO₂)``.
            * ``rtmp`` [-] — Temperature correction factor
              ``TMPFTB(DTEMP) · TMNFTB(TMIN)``.
            * ``dtemp`` [°C] — Effective daytime temperature
              ``TMAX − f · (TMAX − TMIN)``.
        """
        # DVS-dependent base RUE.
        rue = params.scale_factor_rue * interpolate(params.ruetb, dvs)

        # CO₂ correction.
        co2_t = torch.as_tensor(co2, dtype=tmax.dtype, device=tmax.device)
        co2_b = co2_t.expand_as(tmax) if co2_t.dim() == 0 else co2_t
        rco = interpolate(params.cotb, co2_b)

        # Effective daytime temperature; tuned via day_temp_factor.
        dtemp = tmax - params.day_temp_factor * (tmax - tmin)

        # Temperature reduction: daytime-temp × low-min-temp.
        rtmp = interpolate(params.tmpftb, dtemp) * interpolate(params.tmnftb, tmin)

        rtmco = rtmp * rco

        return {
            "rue": rue,
            "rtmco": rtmco,
            "rco": rco,
            "rtmp": rtmp,
            "dtemp": dtemp,
        }
