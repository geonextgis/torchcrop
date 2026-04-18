"""Biomass allocation to plant organs.

Reference
---------
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
