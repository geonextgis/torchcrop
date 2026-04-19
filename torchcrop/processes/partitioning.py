"""Biomass allocation to plant organs.

References:
    Biomass partitioning block of ``Lintul5.java``.

Partitioning fractions depend on DVS via the parameters ``FLTB``, ``FSTB``,
``FRTB``, ``FOTB`` (leaves, stems, roots, storage). The above-ground fraction
(leaves+stems+storage) plus the below-ground fraction (roots) sum to ~1.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.states.model_state import ModelState


class Partitioning(nn.Module):
    """Allocate daily gross biomass production to organ-specific pools."""

    def forward(
        self,
        state: ModelState,
        gtotal: torch.Tensor,
        params: CropParameters,
    ) -> dict[str, torch.Tensor]:
        """Split gross daily biomass production across organs.

        Args:
            state: Current state (uses ``state.dvs`` for table lookups).
            gtotal: Gross daily biomass production [g DM m⁻² d⁻¹], shape
                ``[B]``.
            params: Crop parameters; uses the partitioning tables ``frtb``,
                ``fltb``, ``fstb``, ``fotb``.

        Returns:
            Dict of ``[B]`` tensors grouped as follows.

            Rate variables (per-organ growth, fed to leaf/root/state updates
            and nutrient demand):

                * ``g_root`` [g DM m⁻² d⁻¹] — Root biomass growth rate
                  (``= gtotal * fr``); becomes ``wrt_rate``.
                * ``g_lv`` [g DM m⁻² d⁻¹] — Leaf growth before senescence;
                  `LeafDynamics` converts it into ``wlv_rate`` and
                  ``lai_rate``.
                * ``g_st`` [g DM m⁻² d⁻¹] — Stem growth rate; becomes
                  ``wst_rate`` directly.
                * ``g_so`` [g DM m⁻² d⁻¹] — Storage organ growth rate;
                  becomes ``wso_rate`` and drives final yield.

            Diagnostics (normalised partitioning fractions):

                * ``fr`` [-] — Below-ground (root) fraction, clamped to
                  ``[0, 0.95]``.
                * ``fl``, ``fs``, ``fo`` [-] — Above-ground fractions to
                  leaves, stems, storage organs. Re-normalised so that
                  ``fl + fs + fo == 1``.
        """
        dvs = state.dvs
        fr = interpolate(params.frtb, dvs)
        fl = interpolate(params.fltb, dvs)
        fs = interpolate(params.fstb, dvs)
        fo = interpolate(params.fotb, dvs)

        # Normalise above-ground fractions so that fl + fs + fo = 1
        ag_sum = fl + fs + fo
        ag_sum = torch.where(ag_sum > 1e-8, ag_sum, torch.ones_like(ag_sum))
        fl_n = fl / ag_sum
        fs_n = fs / ag_sum
        fo_n = fo / ag_sum

        # fr controls the root:shoot split
        fr = torch.clamp(fr, 0.0, 0.95)
        g_root = gtotal * fr
        g_shoot = gtotal * (1.0 - fr)

        g_lv = g_shoot * fl_n
        g_st = g_shoot * fs_n
        g_so = g_shoot * fo_n

        return {
            "g_root": g_root,
            "g_lv": g_lv,
            "g_st": g_st,
            "g_so": g_so,
            "fr": fr,
            "fl": fl_n,
            "fs": fs_n,
            "fo": fo_n,
        }
