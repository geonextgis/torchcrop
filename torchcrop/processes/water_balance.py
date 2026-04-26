"""Soil water balance — two-zone port of SIMPLACE ``WaterBalance.java``.

References:
    SIMPLACE ``WaterBalance.java`` (the ``WATBALS`` routine) and
    ``LintulFunctions.SWEAF``.

Design:
    The implementation faithfully ports the SIMPLACE two-zone bucket model.

    **Zones.** The soil column is split into (i) a rooted zone of depth
    ``rootd`` storing ``wa`` [mm] and (ii) a lower zone of depth
    ``rdm - rootd`` storing ``wa_lower`` [mm], where ``rdm = min(rdmso,
    rdmcr)`` is the soil-/crop-limited maximum rooting depth. Root-front
    advance ``rr`` moves water with content ``SMACTL`` from the lower zone
    into the rooted zone via the ``WDR``/``WDRA`` fluxes.

    **Percolation cascade.** Net infiltration (after evaporation and
    transpiration) flows into the rooted zone (``PERC1``); excess above
    field capacity trickles to the lower zone (``PERC2``); excess below the
    lower zone's field capacity leaves the profile as deep drainage
    (``PERC3``). All three are rate-limited by ``KSUB`` and by the free
    pore space of the receiving layer, exactly as in SIMPLACE.

    **Stress factors.** Transpiration is reduced multiplicatively by a
    drought factor ``RDRY`` and (for non-rice crops) an oxygen-shortage
    factor ``RWET``. ``RWET`` ramps with ``DSOS`` (days of oxygen shortage,
    persistent across days, clipped to ``[0, 4]``).

    **Soil evaporation.** Stroosnijder model: if infiltration ≥ 5 mm d⁻¹
    the evaporation is reset to the potential rate and ``DSLR`` to 1,
    otherwise ``DSLR`` increments and evaporation follows the
    ``sqrt(DSLR) - sqrt(DSLR-1)`` decay, capped by air-dry capacity.

    **Irrigation.** Supports the three SIMPLACE modes controlled by
    ``params.irri``: 0 = none, 1 = automatic refill when ``SMACT`` falls
    below ``SMCR + 0.02`` and rain < 10 mm, 2 = table look-up of
    ``irrtab`` scaled by ``scale_factor_irr``.

Equations:
    Actual soil-moisture contents [m³ m⁻³]:

    $$
    \\theta = \\frac{W_a}{1000 \\cdot D_\\text{root}},
    \\qquad
    \\theta_\\ell = \\frac{W_{a,\\ell}}{1000 \\cdot (D_\\text{rdm} -
    D_\\text{root})}
    $$

    Easily-available fraction (SIMPLACE ``SWEAF``) and critical content:

    $$
    f_\\text{eaw} = \\mathrm{clip}\\!\\left(\\frac{1}{A + B\\,\\text{ETC}_
    \\text{cm}} - (5-\\text{DEPNR})\\cdot 0.10,\\ 0.10,\\ 0.95\\right),
    $$
    $$
    \\theta_\\text{crit} = (1 - f_\\text{eaw})(\\theta_\\text{fc} -
    \\theta_\\text{wp}) + \\theta_\\text{wp}.
    $$

    Drought and oxygen reductions:

    $$
    R_\\text{dry} = \\mathrm{clip}\\!\\left(
    \\frac{\\theta - \\theta_\\text{wp}}{\\theta_\\text{crit} -
    \\theta_\\text{wp}},\\ 0,\\ 1\\right),
    $$
    $$
    R_\\text{wet,max} = \\mathrm{clip}\\!\\left(
    \\frac{\\theta_\\text{sat} - \\theta}{\\theta_\\text{sat} -
    \\theta_\\text{air}},\\ 0,\\ 1\\right),
    \\quad
    R_\\text{wet} = R_\\text{wet,max} + \\left(1 - \\tfrac{\\text{DSOS}}{4}
    \\right)(1 - R_\\text{wet,max}).
    $$

    Root-front water transfer:

    $$
    \\text{WDR} = 1000 \\cdot r_r \\cdot \\theta_\\ell,
    \\qquad
    \\text{WDRA} = 1000 \\cdot r_r \\cdot (\\theta_\\ell -
    \\theta_\\text{wp})_+.
    $$

    Percolation cascade (subscripts ``0`` denote saturation-capacity
    headroom and unlabelled ones field-capacity headroom):

    $$
    \\begin{aligned}
    \\text{PERC} &= (1 - \\text{RUNFR})\\cdot \\text{RAIN} + \\text{RIRR},\\\\
    \\text{PERC1P} &= \\text{PERC} - E_a - T_a,\\\\
    \\text{PERC1} &= \\min(\\text{KSUB} + \\text{CAP}_0,\\ \\text{PERC1P}),\\\\
    \\text{RUNOFF} &= \\text{RUNFR}\\cdot \\text{RAIN} + (\\text{PERC1P}
    - \\text{PERC1})_+,\\\\
    \\text{PERC2} &= \\mathbb{1}_{\\text{CAP}\\le \\text{PERC1}}\\cdot
    \\min(\\text{KSUB} + \\text{CAP}_{\\ell,0},\\ (\\text{PERC1} -
    \\text{CAP})_+),\\\\
    \\text{PERC3} &= \\mathbb{1}_{\\text{CAP}_\\ell\\le \\text{PERC2}}\\cdot
    \\min(\\text{KSUB},\\ (\\text{PERC2} - \\text{CAP}_\\ell)_+).
    \\end{aligned}
    $$

    Rates (integrated with forward Euler, ``dt = 1`` d):

    $$
    \\dot W_a = \\text{PERC1} - \\text{PERC2} + \\text{WDR},
    \\qquad
    \\dot W_{a,\\ell} = \\text{PERC2} - \\text{PERC3} - \\text{WDR}.
    $$
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions.fst_functions import limit, notnul
from torchcrop.functions.interpolation import interpolate
from torchcrop.parameters.soil_params import SoilParameters
from torchcrop.states.model_state import ModelState

# ---------------------------------------------------------------------------- #
# SIMPLACE helper: fraction of easily-available soil water (SWEAF)
# ---------------------------------------------------------------------------- #


def _sweaf(etc: torch.Tensor, depnr: torch.Tensor) -> torch.Tensor:
    """Port of ``LintulFunctions.SWEAF`` (Doorenbos & Kassam).

    Args:
        etc: CO2-corrected potential canopy evapotranspiration [mm d⁻¹],
            shape ``[B]``.
        depnr: Crop-group depletion number ``cDEPNR`` in ``[1, 5]``,
            broadcastable to ``[B]``. (from 1 (=drought-sensitive) to 5 (=drought-resistent)).

    Returns:
        Easily-available fraction of soil water in ``[0.10, 0.95]``.
    """
    a = 0.76
    b = 1.5
    etc_cm = etc / 10.0  # mm d⁻¹ → cm d⁻¹
    tsweaf = 1.0 / notnul(a + b * etc_cm) - (5.0 - depnr) * 0.10
    correction = (etc_cm - 0.6) / notnul(depnr * (depnr + 3.0))
    tsweaf = torch.where(depnr < 3.0, tsweaf + correction, tsweaf)
    return limit(0.10, 0.95, tsweaf)


# ---------------------------------------------------------------------------- #
# Irrigation demand — supports SIMPLACE IRRI modes 0, 1, 2
# ---------------------------------------------------------------------------- #


def _irrigation_demand(
    params: SoilParameters,
    smact: torch.Tensor,
    smcr: torch.Tensor,
    wavt: torch.Tensor,
    rootd: torch.Tensor,
    rain: torch.Tensor,
    doy: torch.Tensor | None,
    external: torch.Tensor | None,
) -> torch.Tensor:
    """Daily effective irrigation [mm d⁻¹].

    Args:
        params: Soil parameters (uses ``irri``, ``irrtab``,
            ``scale_factor_irr``, ``wcfc``, ``wcwp``).
        smact: Actual soil-moisture content of the rooted zone [m³ m⁻³].
        smcr: Critical soil-moisture content [m³ m⁻³].
        wavt: Available water in the rooted zone [mm].
        rootd: Rooting depth [m].
        rain: Daily precipitation [mm d⁻¹].
        doy: Day-of-year, required for ``IRRI = 2`` table look-up; may be
            ``None`` (the table branch then resolves to zero).
        external: Externally supplied irrigation override [mm d⁻¹]; when
            provided it short-circuits the ``IRRI`` logic.

    Returns:
        Daily irrigation amount [mm d⁻¹], shape ``[B]``.
    """
    if external is not None:
        return external

    irri = params.irri
    wavfc = 1000.0 * rootd * (params.wcfc - params.wcwp)
    # --- Mode 1: automatic refill ---
    auto_condition = (smact <= (smcr + 0.02)) & (rain < 10.0)
    auto_dose = limit(0.0, 10.0, 0.7 * (wavfc - wavt))
    dirr_auto = torch.where(auto_condition, auto_dose, torch.zeros_like(auto_dose))
    # --- Mode 2: table look-up scaled by scale_factor_irr ---
    if params.irrtab is not None and doy is not None:
        table_value = interpolate(params.irrtab, doy.to(wavfc.dtype))
        dirr_table = torch.clamp(
            params.scale_factor_irr * table_value, min=0.0, max=10.0
        )
    else:
        dirr_table = torch.zeros_like(wavfc)

    zero = torch.zeros_like(wavfc)
    # Select by IRRI value in a differentiable manner
    return torch.where(
        torch.isclose(irri, torch.ones_like(irri)),
        dirr_auto,
        torch.where(
            torch.isclose(irri, torch.full_like(irri, 2.0)),
            dirr_table,
            zero,
        ),
    )


# ---------------------------------------------------------------------------- #
# WaterBalance module
# ---------------------------------------------------------------------------- #


class WaterBalance(nn.Module):
    """Two-zone port of SIMPLACE Lintul5 ``WATBALS``.

    Computes daily water fluxes (transpiration, evaporation, runoff,
    drainage), stress factors (``TRANRF``), and rate variables for the
    rooted / lower / Stroosnijder / oxygen states, in a fully batched
    and autograd-safe manner.
    """

    def forward(
        self,
        state: ModelState,
        rain: torch.Tensor,
        pevap: torch.Tensor,
        ptran: torch.Tensor,
        params: SoilParameters,
        rdm: torch.Tensor,
        etc: torch.Tensor | None = None,
        rr: torch.Tensor | None = None,
        irrigation: torch.Tensor | None = None,
        doy: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute one day of water-balance rates and fluxes.

        Args:
            state: Current `ModelState`; uses ``wa``, ``wa_lower``,
                ``rootd``, ``dslr``, ``dsos``.
            rain: Daily precipitation [mm d⁻¹], shape ``[B]``.
            pevap: Potential soil evaporation [mm d⁻¹], shape ``[B]``.
            ptran: Potential transpiration [mm d⁻¹], shape ``[B]``.
            params: Soil parameter container.
            rdm: Soil-/crop-limited maximum rooting depth
                ``min(rdmso, rdmcr)`` [m], shape ``[B]``.
            etc: CO2-corrected reference canopy ET [mm d⁻¹], shape ``[B]``;
                falls back to ``ptran`` when ``None``.
            rr: Root-front velocity [m d⁻¹], shape ``[B]``; ``None`` → 0
                (no root-front water transfer).
            irrigation: Externally supplied irrigation [mm d⁻¹] that
                overrides ``params.irri`` mode.
            doy: Day-of-year tensor needed by the ``IRRI = 2`` table
                look-up; shape ``[B]``.

        Returns:
            Dict of ``[B]`` tensors.

            **Rate variables** (consumed by the engine):

                * ``wa_rate``       — ``perc1 - perc2 + wdr`` [mm d⁻¹].
                * ``wa_lower_rate`` — ``perc2 - perc3 - wdr`` [mm d⁻¹].
                * ``dslr_rate``     — ``dslr_new - dslr`` [d d⁻¹].
                * ``dsos_rate``     — ``dsos_new - dsos`` [d d⁻¹].

            **Fluxes / diagnostics**:

                * ``tran``   — actual transpiration [mm d⁻¹].
                * ``evap``   — actual soil evaporation [mm d⁻¹].
                * ``runoff`` — surface runoff (preliminary + rejected
                  infiltration) [mm d⁻¹].
                * ``drain``  — deep drainage = ``perc3`` [mm d⁻¹].
                * ``perc1`` / ``perc2`` / ``perc3`` — cascade fluxes
                  [mm d⁻¹].
                * ``wdr`` / ``wdra`` — root-front water transfer to the
                  rooted zone (total / available) [mm d⁻¹].
                * ``rirr``   — effective irrigation [mm d⁻¹].
                * ``tranrf`` — water-stress factor in ``[0, 1]``.
                * ``smact`` / ``smactl`` — soil-moisture contents
                  [m³ m⁻³].
                * ``smcr``   — critical soil-moisture content [m³ m⁻³].
                * ``rdry`` / ``rwet`` — drought / oxygen reduction
                  factors in ``[0, 1]``.
                * ``wbal``   — rooted-zone mass-balance residual [mm]
                  (should be ≈ 0).
        """
        factor = 1000.0  # root [m] · volumetric water-content → water [mm]
        rootd = torch.clamp(state.rootd, min=1e-4)
        rdm_eff = torch.clamp(rdm, min=rootd + 1e-4)
        rd_lower = torch.clamp(rdm_eff - rootd, min=1e-4)

        # ---------------------------------------------------------------- #
        # 1. Actual volumetric soil-moisture contents [m³ m⁻³]
        # ---------------------------------------------------------------- #
        smact = state.wa / notnul(factor * rootd)
        smactl = state.wa_lower / notnul(factor * rd_lower)

        # ---------------------------------------------------------------- #
        # 2. Critical moisture content and drought reduction factor
        # ---------------------------------------------------------------- #
        etc_eff = ptran if etc is None else etc
        sweaf = _sweaf(etc_eff, params.depnr)
        smcr = (1.0 - sweaf) * (params.wcfc - params.wcwp) + params.wcwp
        rdry = limit(0.0, 1.0, (smact - params.wcwp) / notnul(smcr - params.wcwp))

        # ---------------------------------------------------------------- #
        # 3. Oxygen-shortage factor with DSOS accumulator
        # ---------------------------------------------------------------- #
        smair = params.wcst - params.crairc
        rwetmx = limit(0.0, 1.0, (params.wcst - smact) / notnul(params.wcst - smair))
        dsos_new = torch.where(
            smact >= smair,
            torch.clamp(state.dsos + 1.0, max=4.0),
            torch.zeros_like(state.dsos),
        )
        rwet_nonrice = rwetmx + (1.0 - dsos_new / 4.0) * (1.0 - rwetmx)
        is_aquatic = (params.iairdu > 0.5).to(rwet_nonrice.dtype)
        rwet = is_aquatic + (1.0 - is_aquatic) * rwet_nonrice
        rwet = limit(0.0, 1.0, rwet)
        
        # ---------------------------------------------------------------- #
        # 4. Actual transpiration and water-stress factor
        # ---------------------------------------------------------------- #
        wwp = factor * params.wcwp * rootd
        wavt = torch.clamp(state.wa - wwp, min=0.0)
        tran = torch.clamp(torch.minimum(wavt, rdry * rwet * ptran), min=0.0)
        tranrf = tran / notnul(ptran)

        # ---------------------------------------------------------------- #
        # 5. Irrigation demand
        # ---------------------------------------------------------------- #
        rirr = _irrigation_demand(
            params=params,
            smact=smact,
            smcr=smcr,
            wavt=wavt,
            rootd=rootd,
            rain=rain,
            doy=doy,
            external=irrigation,
        )

        # ---------------------------------------------------------------- #
        # 6. Stroosnijder soil evaporation with DSLR accumulator
        # ---------------------------------------------------------------- #
        perc = (1.0 - params.runfr) * rain + rirr
        runofp = params.runfr * rain

        wet_day = (perc >= 5.0).to(state.dslr.dtype)
        dslr_new = wet_day * torch.ones_like(state.dslr) + (1.0 - wet_day) * (
            state.dslr + 1.0
        )
        # Evaporation on dry days — Stroosnijder (1987): sqrt(t) - sqrt(t-1)
        dslr_prev = torch.clamp(dslr_new - 1.0, min=0.0)
        decay = torch.sqrt(torch.clamp(dslr_new, min=1e-8)) - torch.sqrt(dslr_prev)
        evmaxt = pevap * limit(0.0, 1.0, decay * params.cfev)
        # Cap by air-dry topsoil water capacity (SIMPLACE: 100·(SMACT - SMDRY))
        evap_cap = 100.0 * torch.clamp(smact - params.wcad, min=0.0)
        evap_dry = torch.clamp(
            torch.minimum(pevap, torch.minimum(evmaxt + perc, evap_cap)),
            min=0.0,
        )
        evap_wet = pevap
        evap = wet_day * evap_wet + (1.0 - wet_day) * evap_dry
        # Final safety clamp by available water above air-dry content
        wad_mm = factor * params.wcad * rootd
        evap = torch.minimum(evap, torch.clamp(state.wa - wad_mm, min=0.0))

        # ---------------------------------------------------------------- #
        # 7. Root-front water transfer (WDR / WDRA)
        # ---------------------------------------------------------------- #
        rr_eff = rr if rr is not None else torch.zeros_like(rain)
        wdr = factor * rr_eff * smactl
        wdra = factor * rr_eff * torch.clamp(smactl - params.wcwp, min=0.0)

        # ---------------------------------------------------------------- #
        # 8. Percolation cascade PERC1 → PERC2 → PERC3
        # ---------------------------------------------------------------- #
        cap = torch.clamp(params.wcfc - smact, min=0.0) * factor * rootd
        cap0 = torch.clamp(params.wcst - smact, min=0.0) * factor * rootd
        capl = torch.clamp(params.wcfc - smactl, min=0.0) * factor * rd_lower
        capl0 = torch.clamp(params.wcst - smactl, min=0.0) * factor * rd_lower

        perc1p = perc - evap - tran
        perc1 = torch.minimum(params.ksub + cap0, perc1p)
        extra_runoff = torch.clamp(perc1p - perc1, min=0.0)
        runoff = runofp + extra_runoff

        perc2_candidate = torch.minimum(
            params.ksub + capl0, torch.clamp(perc1 - cap, min=0.0)
        )
        perc2 = torch.where(cap <= perc1, perc2_candidate, torch.zeros_like(perc1))

        perc3_candidate = torch.minimum(
            params.ksub + torch.zeros_like(perc2),
            torch.clamp(perc2 - capl, min=0.0),
        )
        perc3 = torch.where(capl <= perc2, perc3_candidate, torch.zeros_like(perc2))

        # ---------------------------------------------------------------- #
        # 9. Rate variables
        # ---------------------------------------------------------------- #
        wa_rate = perc1 - perc2 + wdr
        wa_lower_rate = perc2 - perc3 - wdr
        dslr_rate = dslr_new - state.dslr
        dsos_rate = dsos_new - state.dsos

        # ---------------------------------------------------------------- #
        # 10. Mass-balance residual for diagnostics
        # ---------------------------------------------------------------- #
        wbal = rain + rirr - runoff - evap - tran - perc3 - (wa_rate + wa_lower_rate)

        return {
            # rates
            "wa_rate": wa_rate,
            "wa_lower_rate": wa_lower_rate,
            "dslr_rate": dslr_rate,
            "dsos_rate": dsos_rate,
            # fluxes
            "tran": tran,
            "evap": evap,
            "runoff": runoff,
            "drain": perc3,
            "perc1": perc1,
            "perc2": perc2,
            "perc3": perc3,
            "wdr": wdr,
            "wdra": wdra,
            "rirr": rirr,
            # stress factors / diagnostics
            "tranrf": tranrf,
            "smact": smact,
            "smactl": smactl,
            "smcr": smcr,
            "rdry": rdry,
            "rwet": rwet,
            "wbal": wbal,
        }
