"""Stem biomass net growth and stem senescence.

Stem senescence shares the ``DVS ≥ DVSDR`` gate with root senescence
but uses its own DVS-indexed relative death rate table ``RDRSTB``.

Equations
---------
Stem senescence:

    ``DRST = WST · RDRST``   when ``DVS ≥ DVSDR``, else ``0``

where ``RDRST = RDRSTB(DVS) · scale_factor_rdr_stems`` is the DVS-
indexed relative stem death rate.

Net living-stem biomass change is ``RWST = AGRT · FST − DRST``; the
``AGRT · FST`` term arrives pre-computed as ``g_st`` from
`Partitioning`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.states.model_state import ModelState


class StemDynamics(nn.Module):
    """Stem biomass net rate and dead-stem accumulation rate."""

    def forward(
        self,
        state: ModelState,
        g_st: torch.Tensor,
        params: CropParameters,
        emerg: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute stem biomass rates for one day.

        Args:
            state: Current state; uses ``state.dvs``, ``state.wst`` and
                (for the default emergence mask) ``state.tsump``.
            g_st: Gross stem biomass growth from partitioning
                [g DM m⁻² d⁻¹], shape ``[B]`` — the ``AGRT · FST``
                term.
            params: Crop parameters; uses ``rdrstb`` (relative stem
                death rate vs DVS), ``scale_factor_rdr_stems`` and
                ``dvsdr`` (DVS threshold above which stem death
                starts).
            emerg: Optional emergence mask in ``{0, 1}`` (broadcast
                to ``[B]``). When ``0``, dead stems do not
                accumulate. Defaults to
                ``state.tsump >= params.tsumem``.

        Returns:
            Dict of ``[B]`` tensors grouped as follows.

            Rate variables (consumed by the engine for state update):

            * ``wst_rate`` [g DM m⁻² d⁻¹] — Net daily change in
                living stem biomass (``= g_st − drst``).
            * ``wstd_rate`` [g DM m⁻² d⁻¹] — Daily senesced stem
                mass transferred to the dead-stem pool (``= drst``).

            Diagnostics:

            * ``drst`` [g DM m⁻² d⁻¹] — Stem death rate
                ``DRST = WST · RDRST · 𝟙[DVS ≥ DVSDR]``.
            * ``rdrst`` [d⁻¹] — Effective DVS-indexed relative
                stem death rate after the scale factor.
        """
        dvs = state.dvs
        wst = state.wst
        dtype = wst.dtype

        if emerg is None:
            emerg = (state.tsump >= params.tsumem).to(dtype)
        else:
            emerg = emerg.to(dtype)

        # ----- Stem senescence (DRST) -----
        # DRST = WST * RDRST when DVS >= DVSDR, else 0.
        rdrst = interpolate(params.rdrstb, dvs) * params.scale_factor_rdr_stems
        death_mask = (dvs >= params.dvsdr).to(dtype)
        drst = wst * rdrst * death_mask * emerg

        # ----- Net living-stem biomass change -----
        # RWST = AGRT * FST - DRST.
        wst_rate = g_st - drst

        return {
            "wst_rate": wst_rate,
            "wstd_rate": drst,
            "drst": drst,
            "rdrst": rdrst,
        }
