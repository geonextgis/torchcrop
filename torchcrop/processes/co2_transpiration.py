r"""CO₂ influence on crop transpiration.

Elevated atmospheric CO₂ reduces stomatal conductance, and
therefore crop transpiration. This module scales the potential
transpiration ``PTRAN`` by a linear reduction factor before it enters
the water balance, so the effect propagates into the water-stress
factor ``TRANRF``.

This is a *distinct* mechanism from the CO₂ correction
applied to reference ET in
`torchcrop.processes.evapotranspiration`: that one rescales the
reference evapotranspiration ``ET0`` via an interpolation table,
whereas this module rescales the *potential transpiration demand*
that the water balance compares against soil water supply.

Equations
---------
Linear reduction factor as a function of the atmospheric
CO₂ concentration:

$$
f(\text{CO}_2) = m \cdot \text{CO}_2 + b
$$

$$
\text{PTRAN}_{\text{reduced}} = \text{PTRAN} \cdot f(\text{CO}_2)
$$

With the defaults ``m = -0.0003`` ppm⁻¹ and ``b = 1.1``, the factor
is ``≈ 1`` at the ``≈ 350`` ppm reference and falls below ``1`` for
higher concentrations (less transpiration under elevated CO₂).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Co2Transpiration(nn.Module):
    r"""Reduce potential transpiration as a linear function of CO₂.

    Args:
        transpiration_m: Slope ``cTranspiration_m`` of the linear reduction
            factor [ppm⁻¹] (default ``-0.0003``).
        transpiration_b: Intercept ``cTranspiration_b`` of the linear
            reduction factor [-] (default ``1.1``).
        clamp_nonnegative: If ``True`` (default), clamp the reduction
            factor to ``≥ 0`` so an extreme CO₂ value can never
            flip the sign of transpiration.
    """

    def __init__(
        self,
        transpiration_m: float = -0.0003,
        transpiration_b: float = 1.1,
        clamp_nonnegative: bool = True,
    ) -> None:
        super().__init__()
        self.transpiration_m = transpiration_m
        self.transpiration_b = transpiration_b
        self.clamp_nonnegative = clamp_nonnegative

    def factor(self, co2: torch.Tensor) -> torch.Tensor:
        r"""Compute the CO₂ transpiration-reduction factor.

        Args:
            co2: Atmospheric CO₂ concentration [ppm], broadcastable
                to the transpiration tensor.

        Returns:
            Reduction factor ``f(CO₂) = m·CO₂ + b`` [-]; clamped to ``≥ 0``
            when ``clamp_nonnegative`` is set.
        """
        f = self.transpiration_m * co2 + self.transpiration_b
        if self.clamp_nonnegative:
            f = torch.clamp(f, min=0.0)
        return f

    def forward(
        self,
        ptran: torch.Tensor,
        co2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        r"""Apply the CO₂ reduction to potential transpiration.

        Args:
            ptran: Potential canopy transpiration [mm d⁻¹], shape ``[B]``.
            co2: Atmospheric CO₂ concentration [ppm], a scalar
                tensor or shape broadcastable to ``[B]``.

        Returns:
            Dict of ``[B]`` tensors:

            * ``ptran`` [mm d⁻¹] — CO₂-reduced potential
              transpiration ``PTRAN · f(CO₂)``.
            * ``co2_factor`` [-] — The applied reduction factor ``f(CO₂)``.
        """
        co2_factor = self.factor(co2)
        return {
            "ptran": ptran * co2_factor,
            "co2_factor": torch.broadcast_to(co2_factor, ptran.shape),
        }
