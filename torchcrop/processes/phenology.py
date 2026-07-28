"""Phenological development: thermal time, daylength, vernalisation,
and development stage (DVS).

Equations
---------
Effective daily thermal time:

$$
\\text{DTSU} = \\text{AFGEN}(\\text{DTSMTB}, T_\\text{avg})
$$

Daylength- and vernalisation-modified thermal time, which drives
development up to anthesis:

$$
\\text{DTSUML} = \\text{DTSU} \\cdot \\text{PHOTFAC} \\cdot \\text{VERNFAC}
$$

Development rate (vegetative, generative). The vegetative phase spans
$\\text{TSUM1}$ degree-days and the generative phase $\\text{TSUM2}$, so
the two phases convert thermal time into DVS at different rates. On the
day the crop passes anthesis, only the fraction of $\\text{DTSUML}$ that
still fits inside $\\text{TSUM1}$ may be charged at the vegetative rate;
the surplus $\\text{OVERLAP}$ belongs to the generative phase and is
charged at $1/\\text{TSUM2}$. Splitting the day this way conserves
thermal time across the phase boundary — without it a whole day's worth
of degree-days would be converted at the wrong rate and the crop would
carry a permanent DVS offset into grain filling.

$$
\\text{OVERLAP} = \\max\\!\\left(0,\\;
    \\text{DTSUML} - (1 - \\text{DVS}) \\cdot \\text{TSUM1}\\right)
$$

$$
\\text{DVR} =
\\begin{cases}
    \\dfrac{\\text{DTSUML} - \\text{OVERLAP}}{\\text{TSUM1}} +
        \\dfrac{\\text{OVERLAP}}{\\text{TSUM2}},
        & 0 \\le \\text{DVS} < 1 \\\\[2ex]
    \\dfrac{\\text{DTSU}}{\\text{TSUM2}},
        & 1 \\le \\text{DVS} < 2
\\end{cases}
$$

$\\text{OVERLAP}$ is zero on every day except the one that crosses
anthesis, so the vegetative branch reduces to
$\\text{DTSUML}/\\text{TSUM1}$ for the rest of the phase.

Photoperiod factor (active when ``IDSL >= 1``):

$$
\\text{PHOTFAC} =
\\begin{cases}
    \\text{AFGEN}(\\text{PHOTTB}, \\text{DDLP}),
        & \\text{IDSL} \\ge 1 \\\\
    1, & \\text{otherwise}
\\end{cases}
$$

Vernalisation (active when ``IDSL = 2``). With
$V_\\text{dvs}$ = ``vernalisation_devstage``,
$V_\\text{base}$ = ``VBASE``, $V_\\text{sat}$ = ``VERSAT``,
$V_\\text{min}$ = ``minimal_vernalisation_factor``:

$$
\\text{VERNR} = \\text{AFGEN}(\\text{VERNRT}, T_\\text{avg})
$$

$$
\\text{VERN}_\\text{rate} =
\\begin{cases}
    \\text{VERNR}, & \\text{DVS} < V_\\text{dvs} \\\\
    0,            & \\text{otherwise}
\\end{cases}
$$

$$
\\text{VERNFAC} =
\\begin{cases}
    \\text{LIMIT}\\!\\left(V_\\text{min},\\, 1,\\,
        \\dfrac{\\text{VERN} - V_\\text{base}}
              {V_\\text{sat} - V_\\text{base}}\\right),
        & \\text{DVS} < V_\\text{dvs},\\
          V_\\text{sat} \\neq V_\\text{base} \\\\
    1, & \\text{otherwise}
\\end{cases}
$$
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.states.model_state import ModelState


class Phenology(nn.Module):
    """Development-stage rate calculation.

    Args:
        smooth: If ``True``, replace hard stage branches with sigmoid
            blends of sharpness ``k_sharp`` for smoother gradients.
        k_sharp: Sigmoid sharpness; ignored when ``smooth=False``.
    """

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
        sown: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute phenology rates for one day.

        Args:
            state: Current state; uses ``state.dvs`` and ``state.tsump``.
            davtmp: Mean daily air temperature [°C], shape ``[B]``.
            ddlp: Photoperiodic daylength [h], shape ``[B]``.
            params: Crop parameters; uses ``dtsmtb``, ``phottb``,
                ``tsum1``, ``tsum2``, ``tsumem``, ``tbasem``, ``teffmx``,
                and the vernalisation block (``idsl``, ``vernrt``,
                ``vbase``, ``versat``, ``vernalisation_devstage``,
                ``minimal_vernalisation_factor``).
            sown: Optional sowing-gate mask in ``{0, 1}``, shape ``[B]``.
                Multiplies the emergence (``tsump_rate``) and
                vernalisation (``vern_rate``) clocks so that nothing
                crop-related accumulates before sowing. ``None`` (the
                default) is treated as "sown everywhere" (all ones),
                reproducing the original always-sown-at-``t=0`` behaviour.

        Returns:
            Dict of ``[B]`` tensors.

            Rates (integrated by the engine):

            * ``dvs_rate`` [d⁻¹] — DVS increment; gated to zero
              pre-emergence and post-maturity.
            * ``tsum_rate`` [°C d d⁻¹] — ``dtsu`` once emerged, else
              ``0``.
            * ``tsump_rate`` [°C d d⁻¹] — ``max(0, T_avg − tbasem)``,
              capped at ``teffmx − tbasem``.
            * ``vern_rate`` [d d⁻¹] — ``AFGEN(vernrt, T_avg)``, zero
              when ``idsl < 2`` or
              ``dvs >= vernalisation_devstage``.

            Diagnostics:

            * ``photofac`` [-] — equals ``1`` when ``idsl < 1``.
            * ``vernfac`` [-] — in
              ``[minimal_vernalisation_factor, 1]``; equals ``1``
              when ``idsl < 2`` or
              ``dvs >= vernalisation_devstage``.
            * ``emerged`` [-] — hard (or smooth) mask of
              ``tsump >= tsumem``.
            * ``dtsu`` [°C d d⁻¹] — ``AFGEN(dtsmtb, T_avg)``.
        """
        # Sowing gate — 1 from the sowing day onward, 0 during any
        # pre-sowing spin-up. ``None`` means "always sown" (legacy
        # behaviour where t=0 is the sowing day).
        if sown is None:
            sown = torch.ones_like(davtmp)

        # Effective thermal time.
        dtsu = torch.clamp(interpolate(params.dtsmtb, davtmp), min=0.0)

        # Thermal sum since sowing — base-temperature accumulation
        # capped at TEFFMX. Gated by ``sown`` so the emergence clock
        # only starts once the seed is in the ground.
        tbasem = params.tbasem
        teffmx = params.teffmx
        tsump_rate = torch.clamp(davtmp - tbasem, min=0.0)
        tsump_rate = torch.minimum(tsump_rate, teffmx - tbasem) * sown

        # IDSL controls which factors modulate pre-anthesis development:
        #   0 -> temperature only        (photofac = 1, vernfac = 1)
        #   1 -> temperature + daylength (photofac from PHOTTB, vernfac = 1)
        #   2 -> temperature + daylength + vernalisation (both active)
        idsl_photo_active = (params.idsl >= 1.0).to(davtmp.dtype)
        idsl_active = (params.idsl >= 2.0).to(davtmp.dtype)

        # Photoperiod factor (reduction of dev rate pre-anthesis), gated by IDSL.
        photofac_raw = interpolate(params.phottb, ddlp)
        ones_photo = torch.ones_like(photofac_raw)
        photofac = photofac_raw * idsl_photo_active + ones_photo * (1.0 - idsl_photo_active)

        # Vernal-day rate from the temperature-driven interpolation table.
        vernr = interpolate(params.vernrt, davtmp)

        # Mask: vernalisation is only applied while DVS < vernalisation_devstage.
        vern_devstage = params.vernalisation_devstage
        if self.smooth:
            pre_vern_dvs = torch.sigmoid(self.k_sharp * (vern_devstage - state.dvs))
        else:
            pre_vern_dvs = (state.dvs < vern_devstage).to(davtmp.dtype)

        # Gated by ``sown``: a winter-wheat seed does not vernalise
        # before it is sown.
        vern_rate = vernr * pre_vern_dvs * idsl_active * sown

        # VERNF = LIMIT(MinVF, 1, (VERN - VBASE) / (VERSAT - VBASE)).
        # When VERSAT == VBASE, fall back to VERNF = 1 to avoid a
        # division by zero.
        versat_minus_vbase = params.versat - params.vbase
        vernf_raw = (state.vern - params.vbase) / _safe(versat_minus_vbase)
        ones = torch.ones_like(vernf_raw)
        vernf_clamped = torch.maximum(
            params.minimal_vernalisation_factor.to(vernf_raw.dtype),
            torch.minimum(vernf_raw, ones),
        )
        denom_zero = (versat_minus_vbase.abs() <= 1e-10).to(vernf_raw.dtype)
        vernf = vernf_clamped * (1.0 - denom_zero) + ones * denom_zero

        # Apply VERNF only while DVS < VernDvs.
        vernf_pheno_gated = vernf * pre_vern_dvs + ones * (1.0 - pre_vern_dvs)

        # Gate the entire vernalisation effect by IDSL
        # (vernfac = 1 when off).
        vernfac = vernf_pheno_gated * idsl_active + ones * (1.0 - idsl_active)

        # Pre-emergence: DVS stays at 0 until the thermal sum reaches
        # TSUMEM. DVS starts to increase once TSUM ≥ TSUMEM.
        emerged = state.tsump >= params.tsumem
        if self.smooth:
            emerged_f = torch.sigmoid(self.k_sharp * (state.tsump - params.tsumem))
        else:
            emerged_f = emerged.to(davtmp.dtype)

        tsum_rate = dtsu * emerged_f

        # DVS rate — piecewise on DVS.
        # Split thermal time on the anthesis-crossing day between the vegetative and
        # generative phases. The overlap is zero on all other days and can be derived from DVS.
        dtsuml = dtsu * photofac * vernfac
        overlap = torch.clamp(
            dtsuml - (1.0 - state.dvs) * params.tsum1, min=0.0
        )
        dvr_veg = (dtsuml - overlap) / _safe(params.tsum1) + overlap / _safe(
            params.tsum2
        )
        dvr_gen = dtsu / _safe(params.tsum2)

        if self.smooth:
            alpha = torch.sigmoid(self.k_sharp * (state.dvs - 1.0))
            dvs_rate = (1.0 - alpha) * dvr_veg + alpha * dvr_gen
        else:
            dvs_rate = torch.where(state.dvs < 1.0, dvr_veg, dvr_gen)

        # Turn off DVS progression before emergence and after maturity.
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
    """Replace near-zero entries of ``t`` with ``1`` to guard division.

    Args:
        t: Denominator tensor.
        eps: Magnitude threshold for the replacement.
    """
    return torch.where(t.abs() > eps, t, torch.ones_like(t))
