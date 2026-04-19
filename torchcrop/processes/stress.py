"""Composite stress factors.

A light wrapper module that bundles the water-stress factor ``TRANRF`` and
the nutrient-stress factor ``NSTRESS`` into a single multiplicative reducer.

This module is deliberately minimal: the substantive computations live in
``water_balance.py`` and ``nutrient_demand.py``. Use `StressFactors`
if you want to swap in a learned alternative via the hybrid API.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StressFactors(nn.Module):
    """Combine water and nutrient stress into a single growth reducer."""

    def forward(
        self,
        tranrf: torch.Tensor,
        nstress: torch.Tensor,
    ) -> torch.Tensor:
        """Combine water and nutrient stress into a single growth reducer.

        Args:
            tranrf: Water-stress factor in ``[0, 1]`` from
                `WaterBalance`, shape ``[B]``.
            nstress: Nutrient-stress factor in ``[0, 1]`` from
                `NutrientDemand`, shape ``[B]``.

        Returns:
            Combined stress reducer ``= tranrf * nstress`` in ``[0, 1]``,
            shape ``[B]``. This is *not* a rate — it is a multiplicative
            factor that scales the gross growth rate ``gtotal`` (and
            therefore propagates into every per-organ rate
            ``g_lv``/``g_st``/``g_root``/``g_so``).
        """
        return tranrf * nstress
