"""Composite stress factors.

A light wrapper module that bundles the water-stress factor
``TRANRF`` and the nutrient reduction factor ``NPKREF`` into a single
growth reducer via the **law of the minimum** — the more limiting of
the two factors governs growth, matching SIMPLACE Lintul5 ``GROWTH``.

This module is deliberately minimal — the substantive computations
live in `WaterBalance` and `NutrientDemand`. Use `StressFactors` if
you want to swap in a learned alternative via the hybrid API.
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
            nstress: Nutrient reduction factor ``NPKREF`` in ``[0, 1]``
                (the ``NLUE``-transformed NPK index), shape ``[B]``.

        Returns:
            Combined stress reducer ``= min(tranrf, nstress)`` in
            ``[0, 1]``, shape ``[B]`` — the **law of the minimum**, so
            the more limiting of water/nutrient stress governs growth
            (matches SIMPLACE ``GROWTH``: ``min(TRANRF, NPKREF)``).
            This is *not* a rate — it is a multiplicative factor that
            scales the gross growth rate ``gtotal`` and therefore
            propagates into every per-organ rate
            (``g_lv``/``g_st``/``g_root``/``g_so``).
        """
        return torch.minimum(tranrf, nstress)
