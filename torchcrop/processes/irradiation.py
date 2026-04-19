"""Radiation interception by the canopy.

References:
    SIMPLACE ``Irradiation.java``; Monsi–Saeki extinction law.

Equations:
    Photosynthetically active radiation fraction of global radiation is
    0.50:

    $$
    \\text{PAR} = 0.5 \\cdot \\text{IRRAD}
    $$

    Fraction intercepted by a canopy of index ``LAI`` with extinction
    ``K``:

    $$
    \\text{frac} = 1 - \\exp(-K \\cdot \\text{LAI})
    $$

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
        """Compute canopy PAR interception for one day.

        Args:
            state: Current model state (uses ``state.lai``).
            irrad: Global daily radiation [MJ m⁻² d⁻¹], shape ``[B]``.
            params: Crop parameters; uses ``params.k`` (canopy extinction
                coefficient).

        Returns:
            Dict of ``[B]`` tensors (diagnostics only — no state is directly
            integrated here, they feed photosynthesis and ET):

                * ``par`` [MJ m⁻² d⁻¹] — Photosynthetically active
                  radiation at the top of the canopy (``= 0.5 * irrad``).
                * ``parint`` [MJ m⁻² d⁻¹] — PAR intercepted by the canopy
                  (``= par * frac_intercepted``); used by
                  `Photosynthesis` to drive ``gtotal``.
                * ``frac_intercepted`` [-] — Beer–Lambert interception
                  fraction ``1 - exp(-k * lai)`` in ``[0, 1]``.
        """
        par = 0.5 * irrad
        frac = 1.0 - torch.exp(-params.k * state.lai)
        parint = par * frac
        return {"par": par, "parint": parint, "frac_intercepted": frac}
