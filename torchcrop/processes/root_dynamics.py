"""Root depth growth and root biomass accumulation.

Reference
---------
``Lintul5.java`` (root block). Root depth grows linearly at ``rrdmax`` until
the maximum rooting depth ``rootdm`` is reached. Growth is gated by water
availability (``TRANRF``) and halts at maturity (``DVS >= 2``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.parameters.crop_params import CropParameters
from torchcrop.states.model_state import ModelState


class RootDynamics(nn.Module):
    """Root depth and root biomass rates."""

    def forward(
        self,
        state: ModelState,
        g_root: torch.Tensor,
        tranrf: torch.Tensor,
        params: CropParameters,
    ) -> dict[str, torch.Tensor]:
        max_rootd = params.rootdm
        growing = (state.rootd < max_rootd).to(state.rootd.dtype)
        pre_mat = (state.dvs < 2.0).to(state.rootd.dtype)
        rootd_rate = params.rrdmax * tranrf * growing * pre_mat

        # Avoid overshoot: clamp increment to remaining headroom
        headroom = torch.clamp(max_rootd - state.rootd, min=0.0)
        rootd_rate = torch.minimum(rootd_rate, headroom)

        return {
            "rootd_rate": rootd_rate,
            "wrt_rate": g_root,
        }
