"""Top-level Lintul5 model.

Wires all process sub-modules together and provides both a high-level and a
low-level API.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn

from torchcrop.drivers.weather import WeatherDriver
from torchcrop.engine import SimulationEngine, euler_update
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.parameters.site_params import SiteParameters
from torchcrop.parameters.soil_params import SoilParameters
from torchcrop.processes import (
    Astro,
    Co2Transpiration,
    HeatStressOnGrain,
    HeatStressOnLeafSenescence,
    Irradiation,
    LeafDynamics,
    NutrientDemand,
    Partitioning,
    Phenology,
    Photosynthesis,
    PotentialEvapoTranspiration,
    RootDynamics,
    SoilNutrients,
    StemDynamics,
    StressFactors,
    WaterBalance,
)
from torchcrop.states.model_state import DiagnosticState, ModelState


@dataclass
class ModelOutput:
    """Container for a full simulation run.

    Attributes:
        states: Per-day state snapshots (length ``T + 1``; the first entry is
            the initial condition).
        rates: Per-day rate dicts (length ``T``).
        diagnostics: Per-day `DiagnosticState` snapshots (length ``T``).
            Holds non-integrated stress factors, growth drivers, light
            interception, phenology modifiers and per-day water/nutrient
            fluxes computed from the same intermediates as ``rates`` —
            see `DiagnosticState` for the full field list.
        yield_: Final storage-organ dry weight ``WSO`` at the last step
            [g m-2], *before* the heat-stress adjustment.
        lai: LAI trajectory of shape ``[B, T + 1]``.
        dvs: DVS trajectory of shape ``[B, T + 1]``.
        biomass: Above-ground biomass trajectory of shape ``[B, T + 1]``.
        heat_stress_factor: Around-anthesis heat-stress factor ``HSF`` in
            ``[0, 1]`` from `HeatStressOnGrain`, shape ``[B]``.
        adjusted_yield: Heat-stress-adjusted yield
            ``(1 − HSF) · yield_`` [g m-2], shape ``[B]``.
    """

    states: list[ModelState]
    rates: list[dict[str, torch.Tensor]]
    diagnostics: list[DiagnosticState]
    yield_: torch.Tensor
    lai: torch.Tensor
    dvs: torch.Tensor
    biomass: torch.Tensor
    heat_stress_factor: torch.Tensor
    adjusted_yield: torch.Tensor


class Lintul5Model(nn.Module):
    """Differentiable reimplementation of the Lintul5 crop growth model.

    Args:
        crop_params: Crop parameter container (see `torchcrop.parameters`).
        soil_params: Soil parameter container.
        site_params: Site parameter container (e.g. latitude, altitude).
        smooth: If ``True``, use smooth (sigmoid-blend) replacements for
            stage-based branching.
        stress_module: Optional replacement for the default
            `StressFactors` combiner.
        residual_modules: Optional neural residual corrections keyed by
            process name (``"photosynthesis"`` adds to ``gtotal``;
            ``"partitioning"`` adds to the four allocation fractions;
            ``"leaf_dynamics"`` adds to ``lai_rate``).
    """

    def __init__(
        self,
        crop_params: CropParameters | None = None,
        soil_params: SoilParameters | None = None,
        site_params: SiteParameters | None = None,
        smooth: bool = False,
        stress_module: nn.Module | None = None,
        residual_modules: dict[str, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.crop_params = crop_params or CropParameters()
        self.soil_params = soil_params or SoilParameters()
        self.site_params = site_params or SiteParameters()
        self.smooth = smooth

        self.astro = Astro()
        self.phenology = Phenology(smooth=smooth)
        self.irradiation = Irradiation()
        self.evapotranspiration = PotentialEvapoTranspiration()
        self.co2_transpiration = Co2Transpiration()
        self.water_balance = WaterBalance()
        self.photosynthesis = Photosynthesis()
        self.partitioning = Partitioning()
        self.leaf_dynamics = LeafDynamics()
        self.leaf_heat_stress = HeatStressOnLeafSenescence(smooth=smooth)
        self.root_dynamics = RootDynamics()
        self.stem_dynamics = StemDynamics()
        self.nutrient_demand = NutrientDemand()
        self.soil_nutrients = SoilNutrients()
        self.heat_stress_grain = HeatStressOnGrain()
        self.stress = stress_module or StressFactors()

        self.residual_modules = nn.ModuleDict(residual_modules or {})

        self.engine = SimulationEngine(
            compute_rates=self._compute_rates_dispatch,
            update_state=euler_update,
            dt=1.0,
        )

    # ------------------------------------------------------------------ #
    # High-level API
    # ------------------------------------------------------------------ #

    def initialize(
        self,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> ModelState:
        """Build an initial state for a batch, using ``dvsi`` from crop params.

        Args:
            batch_size: Number of parallel simulation instances ``B``.
            dtype: Tensor dtype.
            device: Torch device (e.g. ``"cpu"``, ``"cuda"``).

        Returns:
            A fresh `ModelState` with initial DVS, root depth, soil
            water at field capacity, and a seeded leaf mass so that LAI
            growth has a substrate post-emergence.
        """
        dvsi = float(self.crop_params.dvsi.detach().cpu().item())
        rootdi = float(self.crop_params.rootdi.detach().cpu().item())
        # Initialise at field capacity × initial rooting depth (mm)
        wfc = float(self.soil_params.wcfc.detach().cpu().item())
        wai = 1000.0 * wfc * rootdi
        # Lower-zone initial water — SIMPLACE WTOTL = factor·(RDM − RDI)·SMLOWI
        rdmso = float(self.soil_params.rdmso.detach().cpu().item())
        rdmcr = float(self.crop_params.rdmcr.detach().cpu().item())
        rdm_val = min(rdmso, rdmcr)
        wci_lower = float(self.soil_params.wci_lower.detach().cpu().item())
        wa_lower_i = 1000.0 * max(rdm_val - rootdi, 1e-4) * wci_lower
        # Seed the soil mineral pools from soil_params. The organic
        # pools (NMIN/PMIN/KMIN) start at NMINI/PMINI/KMINI; the
        # directly available inorganic pools (NMINT/PMINT/KMINT) start
        # at NMINTI/PMINTI/KMINTI (Lintul5 default 0).
        nmini = float(self.soil_params.nmini.detach().cpu().item())
        pmini = float(self.soil_params.pmini.detach().cpu().item())
        kmini = float(self.soil_params.kmini.detach().cpu().item())
        nminti = float(self.soil_params.nminti.detach().cpu().item())
        pminti = float(self.soil_params.pminti.detach().cpu().item())
        kminti = float(self.soil_params.kminti.detach().cpu().item())
        state = ModelState.initial(
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            dvsi=dvsi,
            wai=wai,
            rootdi=rootdi,
            wa_lower_i=wa_lower_i,
            dslri=3.0,
            dsosi=0.0,
            nmini=nmini,
            pmini=pmini,
            kmini=kmini,
            nminti=nminti,
            pminti=pminti,
            kminti=kminti,
        )
        # Sowing-day state: bare soil, no canopy. The Lintul5.java
        # initValues block (lines 793–810) seeds WLVGI/WSTI/WRTI/WSOI/LAII,
        # but in SIMPLACE it runs on iDoSow and is implicitly *the*
        # emergence event (Lintul5.java:1047). torchcrop simulates the
        # sowing→emergence interval explicitly via tsump/tsumem in
        # Phenology, so we leave biomass pools at zero here and inject
        # the Java initValues deltas as a one-shot rate in
        # `_compute_rates_dispatch` on the day tsump first crosses
        # tsumem (LAI is bootstrapped separately by the GLA emergence
        # branch in LeafDynamics).
        return state

    def forward(
        self,
        weather: WeatherDriver | torch.Tensor,
        start_doy: int = 1,
        initial_state: ModelState | None = None,
    ) -> ModelOutput:
        """Run a full simulation and return trajectories plus final yield.

        Args:
            weather: `WeatherDriver` or a raw ``[B, T, C]`` tensor of
                daily weather forcing.
            start_doy: Day-of-year of the first simulated day.
            initial_state: Optional pre-built `ModelState`. When
                omitted, `initialize` is called automatically.

        Returns:
            A `ModelOutput` containing the full state/rate trajectories
            and summary variables (``lai``, ``dvs``, ``biomass``, ``yield_``).
        """
        if isinstance(weather, torch.Tensor):
            weather = WeatherDriver(weather)
        batch_size = weather.batch_size
        if initial_state is None:
            state = self.initialize(
                batch_size=batch_size,
                dtype=weather.data.dtype,
                device=weather.data.device,
            )
        else:
            state = initial_state

        states, rates, diagnostics = self.engine.run(
            state=state,
            weather=weather,
            start_doy=start_doy,
            crop_params=self.crop_params,
            soil_params=self.soil_params,
            site_params=self.site_params,
        )

        lai = torch.stack([s.lai for s in states], dim=1)  # [B, T+1]
        dvs = torch.stack([s.dvs for s in states], dim=1)
        biomass = torch.stack([s.wlv + s.wst + s.wso for s in states], dim=1)
        yield_ = states[-1].wso

        # Heat-stress penalty on grain yield around anthesis. The DVS
        # entering each weather day is ``dvs[:, :-1]`` — the trajectory
        # carries a leading initial-condition entry, so it has length
        # ``T + 1`` against the ``T`` weather days.
        hsg = self.heat_stress_grain(
            tmin=weather.channel("tmin"),
            tmax=weather.channel("tmax"),
            dvs=dvs[:, :-1],
            params=self.crop_params,
            yield_=yield_,
        )

        return ModelOutput(
            states=states,
            rates=rates,
            diagnostics=diagnostics,
            yield_=yield_,
            lai=lai,
            dvs=dvs,
            biomass=biomass,
            heat_stress_factor=hsg["heat_stress_factor"],
            adjusted_yield=hsg["adjusted_yield"],
        )

    # ------------------------------------------------------------------ #
    # Low-level API — single-step rate + state update
    # ------------------------------------------------------------------ #

    def compute_rates(
        self,
        state: ModelState,
        weather_day: dict[str, torch.Tensor],
        doy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the rate vector for a single day (low-level API).

        Args:
            state: Current `ModelState`.
            weather_day: Dict of named weather channels for the current day
                (see `WEATHER_CHANNELS`), each of shape ``[B]``.
            doy: Day-of-year tensor of shape ``[B]``.

        Returns:
            Dict of rate tensors keyed by ``"<field>_rate"`` plus diagnostics
            (``tranrf``, ``nstress``, ``gtotal``). The companion
            `DiagnosticState` is discarded here — use the full model
            `forward` to access the diagnostic trajectory.
        """
        rates, _ = self._compute_rates_dispatch(
            state=state,
            weather_day=weather_day,
            doy=doy,
            crop_params=self.crop_params,
            soil_params=self.soil_params,
            site_params=self.site_params,
        )
        return rates

    def update_state(
        self,
        state: ModelState,
        rates: dict[str, torch.Tensor],
        dt: float = 1.0,
    ) -> ModelState:
        """Apply a forward-Euler step to advance the state by ``dt`` days.

        Args:
            state: Current `ModelState`.
            rates: Dict of rate tensors produced by `compute_rates`.
            dt: Integration step in days.

        Returns:
            A new `ModelState` advanced by one step.
        """
        return euler_update(state, rates, dt)

    # ------------------------------------------------------------------ #
    # Internal: one-day rate computation in the SIMPLACE execution order
    # ------------------------------------------------------------------ #

    def _compute_rates_dispatch(
        self,
        state: ModelState,
        weather_day: dict[str, torch.Tensor],
        doy: torch.Tensor,
        crop_params: CropParameters,
        soil_params: SoilParameters,
        site_params: SiteParameters,
    ) -> tuple[dict[str, torch.Tensor], DiagnosticState]:
        # Extract weather variables (SIMPLACE order)
        davtmp = weather_day["davtmp"]
        tmin = weather_day["tmin"]
        tmax = weather_day["tmax"]
        dtr = weather_day["irrad"]
        rain = weather_day["rain"]
        vap = weather_day["vp"]  # [kPa] from weather
        wind = weather_day["wind"]

        # 1. Astro — solar declination, daylength
        lat_b = (
            site_params.latitude.expand_as(doy)
            if site_params.latitude.dim() > 0
            else site_params.latitude
        )
        astro = self.astro(doy=doy, latitude=lat_b)
        dayl = astro["daylength"]
        sinld = astro["sinld"]
        cosld = astro["cosld"]
        ddlp = astro["ddlp"]

        # 2. Irradiation — daily total irradiation and PAR interception
        irrad_out = self.irradiation(
            state=state,
            doy=doy.float(),
            dayl=dayl,
            sinld=sinld,
            cosld=cosld,
            dtr=dtr,
            params=crop_params,
        )
        avrad = irrad_out["avrad"]
        atmtr = irrad_out["atmtr"]
        frac_int = irrad_out["frac_intercepted"]

        # 3. Phenology
        pheno = self.phenology(state, davtmp, ddlp, crop_params)

        # 4. Evapotranspiration — PENMAN formula. CO2 is a site/scenario
        #    property: SiteParameters.co2 is the single source of truth
        #    feeding the ET0, RUE and transpiration CO2 corrections.
        et = self.evapotranspiration(
            tmin=tmin,
            tmax=tmax,
            wind=wind,
            vap=vap,
            avrad=avrad,
            atmtr=atmtr,
            frac_int=frac_int,
            co2=site_params.co2,
        )

        # 4b. CO2 influence on transpiration — scale the potential
        #     transpiration *demand* by the linear CO2 reduction factor
        #     before it enters the water balance, so elevated CO2
        #     propagates into the water-stress factor TRANRF (distinct
        #     from the CO2 correction already applied to reference ET).
        co2_trans = self.co2_transpiration(et["ptran"], site_params.co2)
        ptran_eff = co2_trans["ptran"]
        co2_factor = co2_trans["co2_factor"]

        # 5. Water balance (two-zone, with SIMPLACE percolation cascade)
        rdm = torch.minimum(
            soil_params.rdmso + torch.zeros_like(state.rootd),
            crop_params.rdmcr + torch.zeros_like(state.rootd),
        )
        water = self.water_balance(
            state=state,
            rain=rain,
            pevap=et["pevap"],
            ptran=ptran_eff,
            params=soil_params,
            rdm=rdm,
            etc=et["etc"],
            doy=doy,
        )
        tranrf = water["tranrf"]

        # 6+7. Nutrient preliminary step — we first estimate partitioning
        #      using a "no nutrient stress" GTOTAL to compute demand, then
        #      finalise with the resulting nstress.
        photo_pre = self.photosynthesis(
            tmax=tmax,
            tmin=tmin,
            dvs=state.dvs,
            params=crop_params,
            co2=site_params.co2,
        )
        # GTOTAL = RUE * RTMCO * PARINT * TRANRF * NSTRESS
        # (LintulFunctions.GROWTH; NSTRESS=1 here for the pre-step).
        # PARINT is in J m-2 d-1; SIMPLACE GROWTH expects MJ PAR m-2 d-1
        # (LintulFunctions.java:852), so we convert here.
        parint_mj = irrad_out["parint"] * 1e-6
        gtotal_pre = (
            photo_pre["rue"]
            * photo_pre["rtmco"]
            * parint_mj
            * tranrf
        )
        # Pre-step: NNI is not yet known, so assume no N stress (nni=1).
        # TRANRF is already known and feeds the water-stress branch of SUBPAR.
        part_pre = self.partitioning(
            state=state,
            gtotal=gtotal_pre,
            params=crop_params,
            tranrf=tranrf,
            nni=torch.ones_like(tranrf),
        )
        nut = self.nutrient_demand(
            state=state,
            crop_params=crop_params,
            soil_params=soil_params,
            tranrf=tranrf,
        )
        nstress = nut["nstress"]

        # Soil mineral-pool balance (SoilNutrientRates in Lintul5.java).
        # Sits *after* NutrientDemand because it consumes the day's
        # NUPTR/PUPTR/KUPTR (depletes NMINT/PMINT/KMINT) and the same
        # NLIMIT/EMERG gates used by uptake.
        soil_nut = self.soil_nutrients(
            state=state,
            nuptr=nut["nuptr"],
            puptr=nut["puptr"],
            kuptr=nut["kuptr"],
            nlimit=nut["nlimit"],
            emerg=nut["emerg"],
            doy=doy,
            crop_params=crop_params,
            soil_params=soil_params,
        )

        # 8. Photosynthesis (final) with nutrient + water stress
        photo = self.photosynthesis(
            tmax=tmax,
            tmin=tmin,
            dvs=state.dvs,
            params=crop_params,
            co2=site_params.co2,
        )
        # GTOTAL = RUE * RTMCO * PARINT * min(TRANRF, NPKREF)
        # The combined stress is produced by self.stress; divide-cancel keeps
        # the existing semantics of stress() returning TRANRF*combined.
        combined_stress = self.stress(tranrf, nstress) / torch.clamp(tranrf, min=1e-6)
        gtotal = (
            photo["rue"]
            * photo["rtmco"]
            * parint_mj
            * tranrf
            * combined_stress
        )

        # Residual correction on gtotal
        if "photosynthesis" in self.residual_modules:
            ctx = torch.stack(
                [state.lai, state.dvs, davtmp, dtr, tranrf, nstress, state.wa, doy],
                dim=-1,
            )
            gtotal = gtotal + self.residual_modules["photosynthesis"](ctx).squeeze(-1)
            gtotal = torch.clamp(gtotal, min=0.0)

        # 9. Partitioning (final, with water and N stress fed into SUBPAR;
        # nstress here is the NPK index used as a proxy for NNI).
        part = self.partitioning(
            state=state,
            gtotal=gtotal,
            params=crop_params,
            tranrf=tranrf,
            nni=nstress,
        )

        # 10. Leaf dynamics — with heat-stress acceleration of senescence.
        # HeatStressOnLeafSenescence returns a multiplier on RDR that is
        # 1.0 outside the heat-stress regime (Tmax < Tc or DVS < DVS_c),
        # so this is a no-op under non-stress conditions.
        leaf_heat = self.leaf_heat_stress(tmax=tmax, dvs=state.dvs, params=crop_params)
        leaf = self.leaf_dynamics(
            state=state,
            g_lv=part["g_lv"],
            dtsu=pheno["dtsu"],
            davtmp=davtmp,
            tranrf=tranrf,
            nstress=nstress,
            params=crop_params,
            heat_stress=leaf_heat,
        )

        # 11. Root dynamics
        root = self.root_dynamics(
            state=state,
            g_root=part["g_root"],
            tranrf=tranrf,
            params=crop_params,
        )

        # 12. Stem dynamics
        stem = self.stem_dynamics(
            state=state,
            g_st=part["g_st"],
            params=crop_params,
        )

        # Gate all growth/senescence rates post-maturity
        active = (state.dvs < 2.0).to(davtmp.dtype)
        gate = lambda x: x * active  # noqa: E731

        rates: dict[str, torch.Tensor] = {
            "dvs_rate": pheno["dvs_rate"],
            "tsum_rate": pheno["tsum_rate"],
            "tsump_rate": pheno["tsump_rate"],
            "vern_rate": pheno["vern_rate"],
            "wlv_rate": gate(leaf["wlv_rate"]),
            "wlvd_rate": gate(leaf["wlvd_rate"]),
            "wst_rate": gate(stem["wst_rate"]),
            "wstd_rate": gate(stem["wstd_rate"]),
            "wrt_rate": gate(root["wrt_rate"]),
            "wrtd_rate": gate(root["wrtd_rate"]),
            "wso_rate": gate(part["g_so"]),
            "lai_rate": gate(leaf["lai_rate"]),
            "rootd_rate": root["rootd_rate"],
            "wa_rate": water["wa_rate"],
            "wa_lower_rate": water["wa_lower_rate"],
            "dslr_rate": water["dslr_rate"],
            "dsos_rate": water["dsos_rate"],
            "anlv_rate": gate(nut["n_lv_rate"]),
            "anst_rate": gate(nut["n_st_rate"]),
            "anrt_rate": gate(nut["n_rt_rate"]),
            "anso_rate": gate(nut["n_so_rate"]),
            "aplv_rate": gate(nut["p_lv_rate"]),
            "apst_rate": gate(nut["p_st_rate"]),
            "aprt_rate": gate(nut["p_rt_rate"]),
            "apso_rate": gate(nut["p_so_rate"]),
            "aklv_rate": gate(nut["k_lv_rate"]),
            "akst_rate": gate(nut["k_st_rate"]),
            "akrt_rate": gate(nut["k_rt_rate"]),
            "akso_rate": gate(nut["k_so_rate"]),
            # Soil pool dynamics: organic pool depletion (negative
            # rates) and inorganic pool balance (fertiliser +
            # mineralisation − uptake). Mineralisation is already gated
            # internally by EMERG/NLIMIT; we deliberately do *not*
            # multiply by ``active`` so that mineralisation and
            # fertiliser additions to NMINT continue post-maturity
            # (matches SIMPLACE — the soil keeps running even when the
            # crop has died).
            "nmin_rate": soil_nut["nmin_rate"],
            "pmin_rate": soil_nut["pmin_rate"],
            "kmin_rate": soil_nut["kmin_rate"],
            "nmint_rate": soil_nut["nmint_rate"],
            "pmint_rate": soil_nut["pmint_rate"],
            "kmint_rate": soil_nut["kmint_rate"],
            "tran_cum_rate": water["tran"],
            "evap_cum_rate": water["evap"],
            # Cumulative water / nutrient / growth accumulators —
            # integrated by the engine via the same `_rate` mechanism as
            # `tran_cum`/`evap_cum`. The daily flux IS the increment
            # because the engine uses dt = 1 day.
            "rain_cum_rate": rain,
            "irrig_cum_rate": water["rirr"],
            "runoff_cum_rate": water["runoff"],
            "drain_cum_rate": water["drain"],
            "nuptr_cum_rate": nut["nuptr"],
            "puptr_cum_rate": nut["puptr"],
            "kuptr_cum_rate": nut["kuptr"],
            "nfixtr_cum_rate": nut["nfixtr"],
            # PAR stored in MJ m-2 (parint_mj is the same conversion
            # used for the GTOTAL calculation upstream).
            "parint_cum_rate": parint_mj,
            "gtotal_cum_rate": gtotal,
            # Diagnostics (not integrated)
            "tranrf": tranrf,
            "nstress": nstress,
            "gtotal": gtotal,
        }

        # ---- Emergence-day bootstrap (Java Lintul5.java initValues,
        # lines 793–810). Fires once per batch element on the step where
        # tsump first crosses tsumem; injects WLVGI/WSTI/WRTI/WSOI and
        # LAII = WLVGI · scale_factor_sla · SLATB(DVSI) as one-shot
        # deltas, matching the SIMPLACE convention that the emergence
        # event (iDoSow in Java) instantly mobilises seed reserves into
        # a juvenile-canopy state. The resulting one-day LAI jump is
        # part of the Lintul5 abstraction; if you need a gradient
        # sowing-to-canopy ramp instead, lower `crop_params.tdwi` (less
        # seed reserve) or raise `crop_params.rgrl` (faster juvenile
        # expansion).
        from torchcrop.functions import interpolate
        dvsi = crop_params.dvsi
        tdwi = crop_params.tdwi
        x = dvsi.reshape(1) if dvsi.dim() == 0 else dvsi
        frtb_d = interpolate(crop_params.frtb, x).reshape(())
        fltb_d = interpolate(crop_params.fltb, x).reshape(())
        fstb_d = interpolate(crop_params.fstb, x).reshape(())
        fotb_d = interpolate(crop_params.fotb, x).reshape(())
        sla_d = interpolate(crop_params.slatb, x).reshape(())
        wrti = frtb_d * tdwi
        tagb = tdwi - wrti
        wlvgi = fltb_d * tagb
        wsti = fstb_d * tagb
        wsoi = fotb_d * tagb
        laii_dyn = wlvgi * crop_params.scale_factor_sla * sla_d

        tsump_next = state.tsump + rates["tsump_rate"] * self.engine.dt
        emerg_now = (
            (state.tsump < crop_params.tsumem) & (tsump_next >= crop_params.tsumem)
        ).to(davtmp.dtype)
        rates["wlv_rate"] = rates["wlv_rate"] + emerg_now * wlvgi / self.engine.dt
        rates["wst_rate"] = rates["wst_rate"] + emerg_now * wsti / self.engine.dt
        rates["wrt_rate"] = rates["wrt_rate"] + emerg_now * wrti / self.engine.dt
        rates["wso_rate"] = rates["wso_rate"] + emerg_now * wsoi / self.engine.dt
        rates["lai_rate"] = rates["lai_rate"] + emerg_now * laii_dyn / self.engine.dt

        # Per-day DiagnosticState snapshot — built from the same
        # intermediate tensors used to assemble `rates`. Pure read-only:
        # does not enter the Euler update, does not touch any state
        # field, and adds no autograd ops beyond the broadcasting of
        # scalars (`combined_stress`, `co2_factor`) to ``[B]``.
        b_shape = tranrf.shape
        diagnostic = DiagnosticState(
            tranrf=tranrf,
            rdry=water["rdry"],
            rwet=water["rwet"],
            nstress=nstress,
            nni=nut["nni"],
            pni=nut["pni"],
            kni=nut["kni"],
            leaf_heat_factor=torch.broadcast_to(leaf_heat, b_shape),
            combined_stress=torch.broadcast_to(combined_stress, b_shape),
            co2_factor=torch.broadcast_to(co2_factor, b_shape),
            gtotal=gtotal,
            rue=photo["rue"],
            rtmco=photo["rtmco"],
            frac_intercepted=frac_int,
            parint=irrad_out["parint"],
            dtsu=pheno["dtsu"],
            photofac=pheno["photofac"],
            vernfac=pheno["vernfac"],
            tran=water["tran"],
            evap=water["evap"],
            runoff=water["runoff"],
            drain=water["drain"],
            rirr=water["rirr"],
            smact=water["smact"],
            smactl=water["smactl"],
            nuptr=nut["nuptr"],
            puptr=nut["puptr"],
            kuptr=nut["kuptr"],
            nfixtr=nut["nfixtr"],
            n_demand=nut["n_demand"],
            p_demand=nut["p_demand"],
            k_demand=nut["k_demand"],
            fr=part["fr"],
            fl=part["fl"],
            fs=part["fs"],
            fo=part["fo"],
        )

        return rates, diagnostic

    # ------------------------------------------------------------------ #
    # Convenience: flatten all learnable parameters across dataclasses
    # ------------------------------------------------------------------ #

    def learnable_parameter_groups(self) -> dict[str, Any]:
        """Return a dict of named `nn.Parameter` tensors.

        Walks the ``crop``/``soil``/``site`` parameter containers and
        collects every field that is an `nn.Parameter`.

        Returns:
            Dict keyed by ``"<container>.<field>"`` mapping to the
            corresponding `nn.Parameter`.
        """
        out: dict[str, Any] = {}
        for name, params in (
            ("crop", self.crop_params),
            ("soil", self.soil_params),
            ("site", self.site_params),
        ):
            for f in fields(params):
                v = getattr(params, f.name)
                if isinstance(v, nn.Parameter):
                    out[f"{name}.{f.name}"] = v
        return out
