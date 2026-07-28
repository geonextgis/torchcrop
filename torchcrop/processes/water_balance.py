"""Two-zone soil water balance.

Tracks water storage in a rooted zone and a lower zone, runs a
percolation cascade to deep drainage, and produces the water-stress
factor ``TRANRF`` that gates crop growth.

Design
------
**Zones.** The soil column is split into a rooted zone of depth
``rootd`` storing ``wa`` [mm] and a lower zone of depth
``rdm − rootd`` storing ``wa_lower`` [mm], where
``rdm = min(rdmso, rdmcr)`` is the soil-/crop-limited maximum rooting
depth. Root-front advance ``rr`` moves water at content ``SMACTL``
from the lower zone into the rooted zone via the ``WDR``/``WDRA``
fluxes.

**Percolation cascade.** Net infiltration (after evaporation and
transpiration) enters the rooted zone as ``PERC1``; excess above
field capacity descends to the lower zone as ``PERC2``; excess below
the lower zone's field capacity leaves the profile as deep drainage
``PERC3``. Each step is rate-limited by ``KSUB`` and by the free pore
space of the receiving layer.

**Stress factors.** Transpiration is reduced multiplicatively by a
drought factor ``RDRY`` and (for non-rice crops) an oxygen factor
``RWET`` that ramps with ``DSOS`` — days of oxygen shortage,
persistent across days and clipped to ``[0, 4]``. Under **potential
production** (``iopt = 1``) none of this applies: the crop transpires
at the full potential rate, unconstrained by the available water, so
``TRANRF`` is identically ``1`` and the root zone may be drawn below
wilting point. The soil water balance itself still runs normally, so
the profile dries down exactly as in a rain-fed run.

**Root-front velocity.** Derived from ``RRI``, the maximum daily
rooting-depth increase: root growth is switched off under severe
drought (``TRANRF < 0.01``), capped by the headroom to the
soil-/crop-limited maximum depth ``D_\\text{rdm}``, and gated by crop
emergence. Rooting depth integrates this *same-day* velocity, while
the ``WDR``/``WDRA`` transfer uses the *previous* day's velocity
(``rr_lag``) — SIMPLACE runs ``WaterBalance`` before ``Biomass`` and
feeds it ``Biomass.rRR``, so the transfer consumes the prior step's
``RR``.

**Soil evaporation.** Stroosnijder model: when infiltration
≥ 5 mm d⁻¹, evaporation is reset to the potential rate and ``DSLR``
to ``1``; otherwise ``DSLR`` increments and evaporation follows the
``sqrt(DSLR) − sqrt(DSLR − 1)`` decay, capped by air-dry capacity.

**Irrigation.** Three modes selected by ``params.irri``: ``0`` = none;
``1`` = automatic refill when ``SMACT`` falls below ``SMCR + 0.02``
and rain ``< 10`` mm; ``2`` = table look-up of ``irrtab`` scaled by
``scale_factor_irr``. Mode 1 is demand-driven and therefore only runs
while a crop stands in the field (the ``crop_present`` gate): its dose
closes a deficit defined from the rooting depth, which is meaningless
on a bare seedbed or a harvested field. Mode 2 is an explicit
management schedule and is applied whenever the table says so, since
pre-sowing applications are a real practice. Delivery is limited to
``10 mm d⁻¹``; a larger
scheduled application is spread over consecutive days rather than
truncated, with the undelivered depth held in ``ModelState.dirro``:

$$
\\text{DIRR}_1 = \\text{IRRTAB}(\\text{DOY}) \\cdot
    f_\\text{irr} + \\text{DIRRO}_{t-1}
$$

$$
\\text{DIRR} = \\min(10, \\text{DIRR}_1), \\qquad
\\text{DIRRO}_t = \\text{DIRR}_1 - \\text{DIRR}
$$

Externally supplied irrigation (the ``irrigation`` argument of
`WaterBalance.forward`) bypasses both the mode logic and the delivery
limit — it is applied exactly as given.

Equations
---------
Volumetric soil-moisture contents [m³ m⁻³]:

$$
\\theta = \\frac{W_a}{1000 \\cdot D_\\text{root}},
\\qquad
\\theta_\\ell = \\frac{W_{a,\\ell}}{1000 \\cdot (D_\\text{rdm} -
D_\\text{root})}
$$

Easily-available fraction (``SWEAF``) and the critical moisture
content below which transpiration starts to decline:

$$
f_\\text{eaw} = \\mathrm{clip}\\!\\left(\\frac{1}{A + B\\,\\text{ETC}_
\\text{cm}} - (5-\\text{DEPNR})\\cdot 0.10,\\ 0.10,\\ 0.95\\right),
\\qquad
\\theta_\\text{crit} = (1 - f_\\text{eaw})(\\theta_\\text{fc} -
\\theta_\\text{wp}) + \\theta_\\text{wp}.
$$

Actual transpiration and the water-stress factor, where the potential
branch (``IOPT = 1``) bypasses both reduction factors and the
available-water ceiling:

$$
T_a =
\\begin{cases}
T_p & \\text{if IOPT} = 1 \\\\
\\max\\!\\left(0,\\ \\min\\!\\left(W_\\text{avt},\\
R_\\text{dry} R_\\text{wet} T_p\\right)\\right) & \\text{otherwise}
\\end{cases}
\\qquad
\\text{TRANRF} = \\frac{T_a}{T_p}.
$$

Drought and oxygen reduction factors:

$$
R_\\text{dry} = \\mathrm{clip}\\!\\left(
\\frac{\\theta - \\theta_\\text{wp}}{\\theta_\\text{crit} -
\\theta_\\text{wp}},\\ 0,\\ 1\\right),
\\quad
R_\\text{wet,max} = \\mathrm{clip}\\!\\left(
\\frac{\\theta_\\text{sat} - \\theta}{\\theta_\\text{sat} -
\\theta_\\text{air}},\\ 0,\\ 1\\right),
\\quad
R_\\text{wet} = R_\\text{wet,max} + \\left(1 - \\tfrac{\\text{DSOS}}{4}
\\right)(1 - R_\\text{wet,max}).
$$

Root-front velocity:

$$
r_r = \\min\\!\\left(\\text{RRI}\\cdot
\\mathbb{1}_{\\text{TRANRF}\\ge 0.01},\\
(D_\\text{rdm} - D_\\text{root})_+\\right)\\cdot
\\mathbb{1}_\\text{emerged}.
$$

Root-front water transfer (total / above wilting point):

$$
\\text{WDR} = 1000 \\cdot r_r \\cdot \\theta_\\ell,
\\qquad
\\text{WDRA} = 1000 \\cdot r_r \\cdot (\\theta_\\ell -
\\theta_\\text{wp})_+.
$$

Percolation cascade (subscript ``0`` = saturation-capacity headroom;
unlabelled = field-capacity headroom). The headroom terms are signed:
a zone holding more than field capacity has $\\text{CAP} < 0$, which
makes ``PERC2`` exceed ``PERC1`` and drains the stored excess downward
until the zone relaxes to field capacity:

$$
\\begin{aligned}
\\text{PERC} &= (1 - \\text{RUNFR})\\cdot \\text{RAIN} + \\text{RIRR},\\\\
\\text{PERC1P} &= \\text{PERC} - E_a - T_a,\\\\
\\text{PERC1} &= \\min(\\text{KSUB} + \\text{CAP}_0,\\ \\text{PERC1P}),\\\\
\\text{RUNOFF} &= \\text{RUNFR}\\cdot \\text{RAIN} + (\\text{PERC1P}
- \\text{PERC1})_+,\\\\
\\text{PERC2} &= \\mathbb{1}_{\\text{CAP}\\le \\text{PERC1}}\\cdot
\\min(\\text{KSUB} + \\text{CAP}_{\\ell,0},\\ \\text{PERC1} -
\\text{CAP}),\\\\
\\text{PERC3} &= \\mathbb{1}_{\\text{CAP}_\\ell\\le \\text{PERC2}}\\cdot
\\min(\\text{KSUB},\\ \\text{PERC2} - \\text{CAP}_\\ell).
\\end{aligned}
$$

State rates (forward Euler, ``dt = 1`` d):

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
# Helper: fraction of easily-available soil water (SWEAF)
# ---------------------------------------------------------------------------- #


def _sweaf(etc: torch.Tensor, depnr: torch.Tensor) -> torch.Tensor:
    """Easily-available soil-water fraction (Doorenbos & Kassam).

    Args:
        etc: CO₂-corrected potential canopy evapotranspiration
            [mm d⁻¹], shape ``[B]``.
        depnr: Crop-group depletion number ``cDEPNR`` in ``[1, 5]``,
            broadcastable to ``[B]``. ``1`` = drought-sensitive,
            ``5`` = drought-resistant.

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
# Irrigation demand — supports IRRI modes 0, 1, 2
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
    dirro: torch.Tensor | None = None,
    crop_present: torch.Tensor | float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Daily effective irrigation [mm d⁻¹] and the surplus carried over.

    The irrigation system can only deliver ``10 mm d⁻¹``. A scheduled
    application larger than that is not lost: the excess is held over
    (``DIRRO``) and added to the following day's application, so a 20 mm
    entry is delivered as 10 mm on two consecutive days and the full
    planned depth still reaches the soil.

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
            provided it short-circuits the ``IRRI`` logic and is applied
            as given, without the delivery cap or carry-over.
        dirro: Surplus held over from the previous day [mm], shape
            ``[B]``. ``None`` is treated as zero.
        crop_present: Gate in ``{0, 1}`` marking the days a crop stands in
            the field, broadcastable to ``[B]``. Suppresses the automatic
            mode outside that window; defaults to ``1`` (always irrigate).

    Returns:
        Tuple ``(dirr, dirro_next)`` of ``[B]`` tensors: the depth applied
        today [mm d⁻¹] and the surplus to carry into tomorrow [mm].
    """
    if external is not None:
        return external, torch.zeros_like(external)

    irri = params.irri
    wavfc = 1000.0 * rootd * (params.wcfc - params.wcwp)
    zero = torch.zeros_like(wavfc)
    held_over = zero if dirro is None else dirro
    # --- Mode 1: automatic refill ---
    # Demand-driven: it exists to keep a growing crop out of water stress,
    # so it only runs while a crop stands in the field. A bare seedbed or a
    # harvested field is not watered to keep it near field capacity, and
    # ``WAVFC`` — the deficit the dose aims to close — is defined from the
    # rooting depth, which has no meaning without a crop. The dose never
    # exceeds the 10 mm delivery limit by construction, so nothing is ever
    # held over.
    auto_condition = (smact <= (smcr + 0.02)) & (rain < 10.0)
    auto_dose = limit(0.0, 10.0, 0.7 * (wavfc - wavt))
    dirr_auto = (
        torch.where(auto_condition, auto_dose, torch.zeros_like(auto_dose))
        * crop_present
    )
    # --- Mode 2: table look-up scaled by scale_factor_irr ---
    # Today's scheduled depth plus yesterday's surplus, capped at the
    # 10 mm delivery limit; whatever does not fit rolls forward.
    if params.irrtab is not None and doy is not None:
        table_value = interpolate(params.irrtab, doy.to(wavfc.dtype))
        scheduled = torch.clamp(params.scale_factor_irr * table_value, min=0.0)
    else:
        scheduled = zero
    dirr1 = scheduled + held_over
    dirr_table = torch.clamp(dirr1, max=10.0)
    dirro_table = dirr1 - dirr_table

    # Select by IRRI value in a differentiable manner. Modes 0 and 1 carry
    # nothing forward, so any surplus left by an earlier mode-2 schedule is
    # dropped when the mode changes.
    is_auto = torch.isclose(irri, torch.ones_like(irri))
    is_table = torch.isclose(irri, torch.full_like(irri, 2.0))
    dirr = torch.where(is_auto, dirr_auto, torch.where(is_table, dirr_table, zero))
    dirro_next = torch.where(is_table, dirro_table, zero)
    return dirr, dirro_next


# ---------------------------------------------------------------------------- #
# WaterBalance module
# ---------------------------------------------------------------------------- #


class WaterBalance(nn.Module):
    """Two-zone soil water balance.

    Computes daily water fluxes (transpiration, evaporation, runoff,
    drainage), the stress factor ``TRANRF``, and rate variables for
    the rooted / lower / Stroosnijder / oxygen states, in a fully
    batched and autograd-safe manner.
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
        rr_lag: torch.Tensor | None = None,
        rri: torch.Tensor | float | None = None,
        emerg: torch.Tensor | float = 1.0,
        irrigation: torch.Tensor | None = None,
        doy: torch.Tensor | None = None,
        crop_present: torch.Tensor | float = 1.0,
        depnr: torch.Tensor | float = 4.5,
        iairdu: torch.Tensor | float = 0.0,
        iopt: torch.Tensor | float = 2.0,
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
            etc: CO₂-corrected reference canopy ET [mm d⁻¹], shape
                ``[B]``; falls back to ``ptran`` when ``None``.
            rr_lag: Root-front velocity from the *previous* day [m d⁻¹],
                shape ``[B]``. Used for the ``WDR``/``WDRA`` subsoil-water
                transfer instead of the same-day ``rr``. When ``None`` the
                transfer falls back to the same-day ``rr`` (no lag),
                preserving standalone-call behaviour.
            rri: Maximum daily rooting-depth increase ``cRRI`` [m d⁻¹],
                a crop trait supplied from `CropParameters.rri`. The
                same-day ``rr`` (returned and used for rooting-depth
                growth) is derived from it; when ``None`` it falls back to
                ``0`` (no root-front water transfer).
            emerg: Crop-emergence gate in ``{0, 1}`` (or a float mask),
                broadcastable to ``[B]``.
            irrigation: Externally supplied irrigation [mm d⁻¹] that
                overrides the ``params.irri`` mode.
            doy: Day-of-year tensor needed by the ``IRRI = 2`` table
                look-up; shape ``[B]``.
            crop_present: Gate in ``{0, 1}`` marking the days a crop stands
                in the field (sown and not yet mature), broadcastable to
                ``[B]``. Automatic irrigation (``IRRI = 1``) is suppressed
                outside that window. Defaults to ``1``, i.e. irrigate on
                every day, which is what a standalone call wants.
            depnr: Crop-group depletion number ``cDEPNR`` [-], a crop
                trait supplied from `CropParameters.depnr` by
                `Lintul5Model`; defaults to the Lintul5 value ``4.5``.
            iairdu: Root air-ducts flag ``cIAIRDU`` (0/1), a crop trait
                supplied from `CropParameters.iairdu` by `Lintul5Model`;
                defaults to ``0`` (non-aquatic).
            iopt: Production-mode switch ``cIOPT`` ∈ ``{1, 2, 3, 4}``,
                supplied from `CropParameters.iopt` by `Lintul5Model`.
                Only ``1`` (potential production) is special-cased here:
                transpiration then runs at the unreduced potential rate
                and ``TRANRF`` is identically ``1``. Defaults to ``2``
                (water-limited), i.e. the reducing behaviour, so
                standalone calls are unaffected.

        Returns:
            Dict of ``[B]`` tensors.

            **Rate variables** (consumed by the engine):

            * ``wa_rate``       — ``perc1 − perc2 + wdr`` [mm d⁻¹].
            * ``wa_lower_rate`` — ``perc2 − perc3 − wdr`` [mm d⁻¹].
            * ``dslr_rate``     — ``dslr_new − dslr`` [d d⁻¹].
            * ``dsos_rate``     — ``dsos_new − dsos`` [d d⁻¹].

            **Fluxes / diagnostics**:

            * ``tran``   — actual transpiration [mm d⁻¹].
            * ``evap``   — actual soil evaporation [mm d⁻¹].
            * ``runoff`` — surface runoff (preliminary + rejected
              infiltration) [mm d⁻¹].
            * ``drain``  — deep drainage (``= perc3``) [mm d⁻¹].
            * ``perc1`` / ``perc2`` / ``perc3`` — cascade fluxes
              [mm d⁻¹].
            * ``wdr`` / ``wdra`` — root-front water transfer to the
              rooted zone (total / available) [mm d⁻¹].
            * ``rr``     — same-day root-front velocity [m d⁻¹].
            * ``rirr``   — effective irrigation actually delivered
              [mm d⁻¹], i.e. after the ``10 mm d⁻¹`` limit.
            * ``dirro_next`` — irrigation depth held over to the next day
              because it exceeded that limit [mm]; latched into
              `ModelState.dirro` by `Lintul5Model`.
            * ``tranrf`` — water-stress factor in ``[0, 1]``.
            * ``smact`` / ``smactl`` — soil-moisture contents
              [m³ m⁻³].
            * ``smcr``   — critical soil-moisture content [m³ m⁻³].
            * ``rdry`` / ``rwet`` — drought / oxygen reduction
              factors in ``[0, 1]``.
            * ``wbal``   — rooted-zone mass-balance residual [mm]
              (should be ≈ 0).
        """
        # ``depnr`` / ``iairdu`` are crop traits (see `CropParameters`),
        # passed in from the crop config by `Lintul5Model`. Coerce to tensors
        # so a scalar default still flows through the tensor maths below (and
        # broadcasts over the batch).
        depnr = torch.as_tensor(depnr, dtype=ptran.dtype, device=ptran.device)
        iairdu = torch.as_tensor(iairdu, dtype=ptran.dtype, device=ptran.device)

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
        sweaf = _sweaf(etc_eff, depnr)
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
        is_aquatic = (iairdu > 0.5).to(rwet_nonrice.dtype)
        rwet = is_aquatic + (1.0 - is_aquatic) * rwet_nonrice
        rwet = limit(0.0, 1.0, rwet)

        # ---------------------------------------------------------------- #
        # 4. Actual transpiration and water-stress factor
        # ---------------------------------------------------------------- #
        wwp = factor * params.wcwp * rootd
        wavt = torch.clamp(state.wa - wwp, min=0.0)
        tran_limited = torch.clamp(
            torch.minimum(wavt, rdry * rwet * ptran), min=0.0
        )
        # Potential production (IOPT = 1) transpires at the *unreduced*
        # potential rate — neither the drought/oxygen factors nor the
        # available-water ceiling apply, so the root zone may be drawn
        # below wilting point and TRANRF is identically 1
        iopt_t = torch.as_tensor(iopt, dtype=ptran.dtype, device=ptran.device)
        is_potential = torch.isclose(iopt_t, torch.ones_like(iopt_t))
        tran = torch.where(is_potential, ptran, tran_limited)
        tranrf = tran / notnul(ptran)

        # ---------------------------------------------------------------- #
        # 5. Irrigation demand
        # ---------------------------------------------------------------- #
        rirr, dirro_next = _irrigation_demand(
            params=params,
            smact=smact,
            smcr=smcr,
            wavt=wavt,
            rootd=rootd,
            rain=rain,
            doy=doy,
            external=irrigation,
            dirro=getattr(state, "dirro", None),
            crop_present=crop_present,
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
        # Evaporation on dry days — Stroosnijder (1987):
        #   sqrt(t) − sqrt(t − 1).
        dslr_prev = torch.clamp(dslr_new - 1.0, min=0.0)
        decay = torch.sqrt(torch.clamp(dslr_new, min=1e-8)) - torch.sqrt(dslr_prev)
        evmaxt = pevap * limit(0.0, 1.0, decay * params.cfev)
        # Cap by air-dry topsoil water capacity:
        #   100 · (SMACT − SMDRY).
        evap_cap = 100.0 * torch.clamp(smact - params.wcad, min=0.0)
        evap_dry = torch.clamp(
            torch.minimum(pevap, torch.minimum(evmaxt + perc, evap_cap)),
            min=0.0,
        )
        evap_wet = pevap
        evap = wet_day * evap_wet + (1.0 - wet_day) * evap_dry
        # Final safety clamp by available water above air-dry content.
        wad_mm = factor * params.wcad * rootd
        evap = torch.minimum(evap, torch.clamp(state.wa - wad_mm, min=0.0))

        # 6b. Same-day root-front velocity — drives rooting-depth growth.
        if rri is not None:
            rri_t = torch.as_tensor(rri, dtype=ptran.dtype, device=ptran.device)
            emerg_t = torch.as_tensor(emerg, dtype=ptran.dtype, device=ptran.device)
            insw_water = torch.where(
                tranrf - 0.01 < 0.0, torch.zeros_like(tranrf), torch.ones_like(tranrf)
            )
            headroom = torch.clamp(rdm_eff - rootd, min=0.0)
            rr_eff = torch.minimum(rri_t * insw_water, headroom) * emerg_t
        else:
            rr_eff = torch.zeros_like(rain)
            
        # ---------------------------------------------------------------- #
        # 7. Root-front water transfer — uses the lagged velocity (prior
        # step's RR) when supplied, else the same-day rr_eff.
        # ---------------------------------------------------------------- #
        rr_wdr = rr_lag if rr_lag is not None else rr_eff
        wdr = factor * rr_wdr * smactl
        wdra = factor * rr_wdr * torch.clamp(smactl - params.wcwp, min=0.0)

        # ---------------------------------------------------------------- #
        # 8. Percolation cascade PERC1 → PERC2 → PERC3
        # ---------------------------------------------------------------- #
        # Storage headroom of each zone, up to field capacity (``CAP``,
        # ``CAPL``) and up to saturation (``CAP0``, ``CAPL0``). These are
        # *signed*: a zone that already holds more than field capacity has
        # negative headroom, and that is what drives it back down. With
        # ``CAP < 0`` the root-zone balance becomes
        # ``DWOT = PERC1 - PERC2 + WDR = CAP + WDR``, i.e. the stored
        # excess is released to the lower zone (subject to the
        # ``KSUB + CAPL0`` conductivity limit) until the zone relaxes to
        # field capacity. Clamping the headroom at zero would strand that
        # excess in the root zone indefinitely.
        cap = (params.wcfc - smact) * factor * rootd
        cap0 = (params.wcst - smact) * factor * rootd
        capl = (params.wcfc - smactl) * factor * rd_lower
        capl0 = (params.wcst - smactl) * factor * rd_lower

        perc1p = perc - evap - tran
        perc1 = torch.minimum(params.ksub + cap0, perc1p)
        extra_runoff = torch.clamp(perc1p - perc1, min=0.0)
        runoff = runofp + extra_runoff

        perc2_candidate = torch.minimum(params.ksub + capl0, perc1 - cap)
        perc2 = torch.where(cap <= perc1, perc2_candidate, torch.zeros_like(perc1))

        perc3_candidate = torch.minimum(
            params.ksub + torch.zeros_like(perc2), perc2 - capl
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
            "rr": rr_eff,
            "rirr": rirr,
            "dirro_next": dirro_next,
            # stress factors / diagnostics
            "tranrf": tranrf,
            "smact": smact,
            "smactl": smactl,
            "smcr": smcr,
            "rdry": rdry,
            "rwet": rwet,
            "wbal": wbal,
        }
