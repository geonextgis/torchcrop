"""Radiation interception by the canopy.

Reference
---------
SIMPLACE ``Irradiation.java``; Monsi–Saeki extinction law.

Equations
---------
Photosynthetically active radiation fraction of global radiation is 0.50:

.. math::
    \\text{PAR} = 0.5 \\cdot \\text{IRRAD}

Fraction intercepted by a canopy of index ``LAI`` with extinction ``K``:

.. math::
    \\text{frac} = 1 - \\exp(-K \\cdot \\text{LAI})

So intercepted PAR is ``PARINT = PAR * frac`` [MJ m⁻² d⁻¹].
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.parameters.crop_params import CropParameters
from torchcrop.states.model_state import ModelState


class Irradiation(nn.Module):
    """PAR interception by canopy."""

    def forward(
        self,
        state: ModelState,
        irrad: torch.Tensor,
        params: CropParameters,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        irrad : torch.Tensor
            Global daily radiation [MJ m-2 d-1], shape ``[B]``.
        """
        par = 0.5 * irrad
        frac = 1.0 - torch.exp(-params.k * state.lai)
        parint = par * frac
        return {"par": par, "parint": parint, "frac_intercepted": frac}
