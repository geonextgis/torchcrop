"""Composite stress factors.

A light wrapper module that bundles the water-stress factor ``TRANRF`` and
the nutrient-stress factor ``NSTRESS`` into a single multiplicative reducer.

This module is deliberately minimal: the substantive computations live in
``water_balance.py`` and ``nutrient_demand.py``. Use :class:`StressFactors`
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
        return tranrf * nstress
