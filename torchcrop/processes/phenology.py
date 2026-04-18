"""Phenological development — DVS accumulation, thermal time, vernalisation.

Reference
---------
Wolf (2012), Section 3.2; SIMPLACE ``Phenology.java``.

Equations
---------
Effective daily temperature:

.. math::
    \\text{DTSU} = \\text{AFGEN}(\\text{DTSMTB}, T_\\text{avg})

Development rate (pre-emergence, vegetative, generative):

.. math::
    \\text{DVR} =
    \\begin{cases}
        \\text{DTSU} \\cdot \\text{PHOTFAC} \\cdot \\text{VERNFAC} / \\text{TSUM1}, & 0 \\le DVS < 1 \\\\
        \\text{DTSU} / \\text{TSUM2}, & 1 \\le DVS < 2
    \\end{cases}
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.states.model_state import ModelState


class Phenology(nn.Module):
    """Development-stage rate calculation."""

    def __init__(self, smooth: bool = False, k_sharp: float = 50.0) -> None:
        super().__init__()
        self.smooth = smooth
        self.k_sharp = k_sharp

    def forward(
        self,
        state: ModelState,
        davtmp: torch.Tensor,
        ddlp: torch.Tensor,
        params: CropParameters,
    ) -> dict[str, torch.Tensor]:
        """Compute phenology rates for one day.

        Parameters
        ----------
        state : ModelState
        davtmp : torch.Tensor
            Mean daily air temperature [°C], shape ``[B]``.
        ddlp : torch.Tensor
            Photoperiodic daylength [h], shape ``[B]``.
        params : CropParameters

        Returns
        -------
        dict
            ``dvs_rate``, ``tsum_rate``, ``tsump_rate``, ``vern_rate``,
            ``photofac``, ``vernfac``.
        """
        # Effective thermal time
        dtsu = torch.clamp(interpolate(params.dtsmtb, davtmp), min=0.0)

        # Thermal sum since sowing — simple base-temp accumulation capped at TEFFMX
        tbasem = params.tbasem
        teffmx = params.teffmx
        tsump_rate = torch.clamp(davtmp - tbasem, min=0.0)
        tsump_rate = torch.minimum(tsump_rate, teffmx - tbasem)

        # Photoperiod factor (reduction of dev rate pre-anthesis)
        photofac = interpolate(params.phottb, ddlp)

        # Vernalisation factor (1.0 here — extended model applies interp table)
        vernfac = torch.ones_like(davtmp)
        vern_rate = torch.zeros_like(davtmp)

        # Pre-emergence: DVS stays 0 until thermal sum reaches TSUMEM
        # We encode emergence by letting DVS start growing once TSUM >= TSUMEM.
        emerged = state.tsump >= params.tsumem
        if self.smooth:
            emerged_f = torch.sigmoid(self.k_sharp * (state.tsump - params.tsumem))
        else:
            emerged_f = emerged.to(davtmp.dtype)

        tsum_rate = dtsu * emerged_f

        # DVS rate: piecewise on DVS
        dvr_veg = dtsu * photofac * vernfac / _safe(params.tsum1)
        dvr_gen = dtsu / _safe(params.tsum2)

        if self.smooth:
            alpha = torch.sigmoid(self.k_sharp * (state.dvs - 1.0))
            dvs_rate = (1.0 - alpha) * dvr_veg + alpha * dvr_gen
        else:
            dvs_rate = torch.where(state.dvs < 1.0, dvr_veg, dvr_gen)

        # Turn off DVS progression before emergence and after maturity
        if self.smooth:
            pre_mat = torch.sigmoid(self.k_sharp * (2.0 - state.dvs))
        else:
            pre_mat = (state.dvs < 2.0).to(davtmp.dtype)
        dvs_rate = dvs_rate * emerged_f * pre_mat

        return {
            "dvs_rate": dvs_rate,
            "tsum_rate": tsum_rate,
            "tsump_rate": tsump_rate,
            "vern_rate": vern_rate,
            "photofac": photofac,
            "vernfac": vernfac,
            "emerged": emerged_f,
            "dtsu": dtsu,
        }


def _safe(t: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Guard a denominator against zero."""
    return torch.where(t.abs() > eps, t, torch.ones_like(t))
