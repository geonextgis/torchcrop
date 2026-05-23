"""Root depth growth, root biomass and root senescence.

Root depth advances at the maximum daily increase ``RRI`` whenever
the crop has emerged and the soil is not severely water-stressed.
Growth is capped at the maximum rooting depth ``RDM`` and gated by
an ``INSW``-style water-stress step (zero growth when
``TRANRF < 0.01``).

Equations
---------
Root depth growth:

    ``RR = min(RRI · INSW(TRANRF − 0.01, 0, 1), RDM − RD)``

Root senescence:

    ``DRRT = WRT · RDRRT``  when ``DVS ≥ DVSDR``, else ``0``

where ``RDRRT = RDRRTB(DVS) · scale_factor_rdr_roots`` is the DVS-
indexed relative root death rate.

Net living-root biomass change is ``RWRT = GRT · FRT − DRRT``; the
``GRT · FRT`` term arrives pre-computed as ``g_root`` from
`Partitioning`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.states.model_state import ModelState


class RootDynamics(nn.Module):
    """Root depth, root biomass and dead-root rates."""

    def forward(
        self,
        state: ModelState,
        g_root: torch.Tensor,
        tranrf: torch.Tensor,
        params: CropParameters,
        emerg: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute root depth and root-biomass rates for one day.

        Args:
            state: Current state; uses ``state.rootd``, ``state.dvs``,
                ``state.wrt`` and (for the default emergence mask)
                ``state.tsump``.
            g_root: Gross root biomass growth from partitioning
                [g DM m⁻² d⁻¹], shape ``[B]`` — the ``GRT · FRT``
                term.
            tranrf: Water-stress factor in ``[0, 1]``, shape ``[B]``.
            params: Crop parameters; uses ``rri`` (maximum daily root
                depth increase), ``rdmcr`` (crop-specific maximum
                rooting depth), ``rdrrtb`` (relative root death rate vs
                DVS), ``scale_factor_rdr_roots`` and ``dvsdr`` (DVS
                threshold above which root death starts).
            emerg: Optional emergence mask in ``{0, 1}`` (broadcast
                to ``[B]``). When ``0``, the root front does not
                advance (pre-emergence) and dead-root accumulation is
                also suppressed. Defaults to
                ``state.tsump >= params.tsumem``.

        Returns:
            Dict of ``[B]`` tensors grouped as follows.

            Rate variables (consumed by the engine for state update):

                * ``rootd_rate`` [m d⁻¹] — Daily increment of rooting
                  depth, ``min(RRI · INSW(TRANRF − 0.01, 0, 1),
                  RDM − RD)``, gated to zero pre-emergence.
                * ``wrt_rate`` [g DM m⁻² d⁻¹] — Net daily change in
                  living root biomass (``= g_root − drrt``).
                * ``wrtd_rate`` [g DM m⁻² d⁻¹] — Daily senesced root
                  mass transferred to the dead-root pool (``= drrt``).

            Diagnostics:

                * ``drrt`` [g DM m⁻² d⁻¹] — Root death rate
                  ``DRRT = WRT · RDRRT · 𝟙[DVS ≥ DVSDR]``.
                * ``rdrrt`` [d⁻¹] — Effective DVS-indexed relative
                  root death rate after the scale factor.
        """
        rootd = state.rootd
        dvs = state.dvs
        wrt = state.wrt

        dtype = rootd.dtype
        zero = torch.zeros_like(rootd)
        one = torch.ones_like(rootd)

        if emerg is None:
            emerg = (state.tsump >= params.tsumem).to(dtype)
        else:
            emerg = emerg.to(dtype)

        # ----- Root depth growth -----
        # Hard step at TRANRF = 0.01: full root-front velocity above
        # the threshold, zero below — not a linear scaling.
        insw_water = torch.where(tranrf - 0.01 < 0.0, zero, one)
        rr_pot = params.rri * insw_water

        # Avoid overshoot beyond the crop-specific maximum rooting
        # depth (``rdmcr``).
        headroom = torch.clamp(params.rdmcr - rootd, min=0.0)
        rootd_rate = torch.minimum(rr_pot, headroom)

        # Root growth only after emergence.
        rootd_rate = rootd_rate * emerg

        # ----- Root senescence (DRRT) -----
        # DRRT = WRT * RDRRT when DVS >= DVSDR, else 0.
        rdrrt = interpolate(params.rdrrtb, dvs) * params.scale_factor_rdr_roots
        death_mask = (dvs >= params.dvsdr).to(dtype)
        drrt = wrt * rdrrt * death_mask * emerg

        # ----- Net living-root biomass change -----
        # RWRT = GRT * FRT - DRRT; g_root already carries the
        # GRT * FRT contribution from `Partitioning`.
        wrt_rate = g_root - drrt

        return {
            "rootd_rate": rootd_rate,
            "wrt_rate": wrt_rate,
            "wrtd_rate": drrt,
            "drrt": drrt,
            "rdrrt": rdrrt,
        }
