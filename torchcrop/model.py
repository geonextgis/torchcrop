"""Top-level Lintul5 model.

Wires the biophysical process sub-modules into a single differentiable
crop simulation and exposes both a high-level (``forward``) and a
low-level (``compute_rates`` / ``update_state``) API.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn

from torchcrop.drivers.weather import WeatherDriver
from torchcrop.engine import SimulationEngine, euler_update
from torchcrop.functions import interpolate
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
    """Container for the results of a full simulation run.

    Attributes:
        states: Per-day `ModelState` snapshots of length ``T + 1``. The
            first entry is the initial condition.
        rates: Per-day rate dicts of length ``T``.
        diagnostics: Per-day `DiagnosticState` snapshots of length ``T``,
            holding non-integrated stress factors, growth drivers,
            light interception, phenology modifiers and per-day
            water/nutrient fluxes. See `DiagnosticState` for the full
            field list.
        yield_: Final storage-organ dry weight ``WSO`` at the last step
            [g m⁻²], shape ``[B]``, before the heat-stress adjustment.
        lai: Leaf area index trajectory of shape ``[B, T + 1]``.
        dvs: Development-stage trajectory of shape ``[B, T + 1]``.
        biomass: Above-ground biomass trajectory of shape ``[B, T + 1]``.
        heat_stress_factor: Window-averaged around-anthesis heat-stress
            factor ``HSF`` in ``[0, 1]`` from `HeatStressOnGrain`, shape
            ``[B]``. Computed once over the full trajectory, not a
            per-day series.
        adjusted_yield: Heat-stress-adjusted yield
            ``period_ended · (1 − HSF) · yield_`` [g m⁻²], shape ``[B]``.
            Equals ``0`` until the DVS trajectory exits the anthesis
            window (``max(DVS) > grain_heat_end_devstage``), matching
            SIMPLACE ``pPeriodEnded`` gating.
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

    The model assembles all biophysical process modules (phenology,
    photosynthesis, partitioning, water and nutrient balances, etc.)
    and integrates them with a forward-Euler scheme on a daily time
    step. The full simulation is autograd-friendly, so gradients flow
    end-to-end from any parameter to the simulated yield.

    Args:
        crop_params: Crop parameter container (see `torchcrop.parameters`).
        soil_params: Soil parameter container.
        site_params: Site parameter container (e.g. latitude, altitude).
        smooth: If ``True``, use smooth (sigmoid-blend) replacements for
            stage-based branching.
        stress_module: Optional replacement for the default
            `StressFactors` combiner.
        residual_modules: Optional neural residual corrections keyed by
            process name:

            * ``"photosynthesis"`` adds to ``gtotal``.
            * ``"partitioning"`` adds to the four allocation fractions.
            * ``"leaf_dynamics"`` adds to ``lai_rate``.
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
        """Build a sowing-day initial state for a batch.

        Args:
            batch_size: Number of parallel simulation instances ``B``.
            dtype: Tensor dtype for all state fields.
            device: Torch device (e.g. ``"cpu"``, ``"cuda"``).

        Returns:
            A fresh `ModelState` representing a bare-soil, pre-emergence
            condition: initial DVS taken from ``crop_params.dvsi``,
            rooting depth ``rootdi``, root- and lower-zone water from
            ``wci`` / ``wci_lower`` clipped to the plant-available range
            ``[wcwp, wcfc]`` (matching SIMPLACE
            ``SMACT = max(SMW, min(SMI, SMFC))``), and the soil organic
            (``nmin``/``pmin``/``kmin``) and inorganic
            (``nmint``/``pmint``/``kmint``) mineral pools seeded from
            soil parameters. Biomass pools, per-organ NPK pools and LAI
            start at zero; their initial values are injected as a
            one-shot rate on the emergence day inside
            `_compute_rates_dispatch`. The dead-tissue NPK loss
            accumulators (``nlossl``/``nlossr``/``nlosss`` and the P, K
            analogues) also start at zero and grow as senescence
            proceeds.
        """
        dvsi = float(self.crop_params.dvsi.detach().cpu().item())
        rootdi = float(self.crop_params.rootdi.detach().cpu().item())
        # Root-zone water from the user-specified initial volumetric content
        # ``wci``, clipped to the plant-available range ``[wcwp, wcfc]``
        wcwp = float(self.soil_params.wcwp.detach().cpu().item())
        wcfc = float(self.soil_params.wcfc.detach().cpu().item())
        wci = float(self.soil_params.wci.detach().cpu().item())
        smact_i = max(wcwp, min(wci, wcfc))
        wai = 1000.0 * smact_i * rootdi
        # Lower-zone water spans the unrooted profile between ``rootdi``
        # and the soil-/crop-limited maximum rooting depth ``rdm``. The
        # lower-zone initial content is clipped to ``[wcwp, wcfc]``
        rdmso = float(self.soil_params.rdmso.detach().cpu().item())
        rdmcr = float(self.crop_params.rdmcr.detach().cpu().item())
        rdm_val = min(rdmso, rdmcr)
        wci_lower = float(self.soil_params.wci_lower.detach().cpu().item())
        smactl_i = max(wcwp, min(wci_lower, wcfc))
        wa_lower_i = 1000.0 * max(rdm_val - rootdi, 1e-4) * smactl_i
        # Soil mineral pools: the organic (mineralisable) pools start at
        # ``nmini``/``pmini``/``kmini`` and the directly available
        # inorganic pools start at ``nminti``/``pminti``/``kminti``.
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
        # The sowing → emergence interval is simulated explicitly via
        # the ``tsump``/``tsumem`` thermal-sum logic in Phenology, so
        # all biomass pools remain zero until emergence. The initial
        # seed-reserve allocation is then injected as a one-shot rate
        # on the day ``tsump`` first crosses ``tsumem``; see the
        # emergence bootstrap block in `_compute_rates_dispatch`.
        return state

    def forward(
        self,
        weather: WeatherDriver | torch.Tensor,
        start_doy: int = 1,
        initial_state: ModelState | None = None,
    ) -> ModelOutput:
        """Run a full simulation and return trajectories plus final yield.

        Args:
            weather: A `WeatherDriver` or a raw ``[B, T, C]`` tensor of
                daily weather forcing.
            start_doy: Day-of-year of the first simulated day.
            initial_state: Optional pre-built `ModelState`. When
                omitted, `initialize` is called automatically with the
                weather batch size, dtype, and device. When supplied,
                the user is responsible for any
                ``wci``/``wci_lower``/``rdm`` clipping — this path
                bypasses the SIMPLACE-parity setup in `initialize`.

        Returns:
            A `ModelOutput` containing the full state, rate and
            diagnostic trajectories together with summary variables
            (``lai``, ``dvs``, ``biomass``, ``yield_`` and the
            heat-stress-adjusted yield).

        Notes:
            * ``HeatStressOnGrain`` is fed ``dvs[:, 1:]`` — the
              *post-integration* DVS for each weather day — mirroring
              SIMPLACE's component order (Phenology runs before HSG
              within a day).
            * ``adjusted_yield`` is gated by a ``period_ended`` mask
              that fires only once ``max(DVS) > grain_heat_end_devstage``,
              so a truncated run that never exits the anthesis window
              returns ``0`` (matches SIMPLACE ``pPeriodEnded``).
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

        # Heat-stress penalty on grain yield around anthesis. SIMPLACE
        # runs Phenology before HeatStressOnGrain within a day, so HSG
        # sees the *post-integration* DVS for that day → ``dvs[:, 1:]``
        # (the trajectory carries a leading initial-condition entry, so
        # it has length ``T + 1`` against the ``T`` weather days).
        hsg = self.heat_stress_grain(
            tmin=weather.channel("tmin"),
            tmax=weather.channel("tmax"),
            dvs=dvs[:, 1:],
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
        """Compute the daily rate vector for a single day (low-level API).

        Args:
            state: Current `ModelState`.
            weather_day: Dict of named weather channels for the current
                day (see `WEATHER_CHANNELS`), each of shape ``[B]``.
            doy: Day-of-year tensor of shape ``[B]``.

        Returns:
            Dict of rate tensors keyed by ``"<field>_rate"`` plus a
            handful of scalar diagnostics (``tranrf``, ``nstress``,
            ``gtotal``). The companion `DiagnosticState` is discarded
            here; use the high-level `forward` to access the full
            diagnostic trajectory.
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
            dt: Integration step size in days.

        Returns:
            A new `ModelState` advanced by one Euler step.
        """
        return euler_update(state, rates, dt)

    # ------------------------------------------------------------------ #
    # Internal: one-day rate computation
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
        # Unpack the day's weather forcing.
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

        # 3. Evapotranspiration — Penman formulation. CO₂ is a
        #    site/scenario property: ``SiteParameters.co2`` is the
        #    single source of truth feeding the ET0, RUE and
        #    transpiration CO₂ corrections.
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

        # 3b. CO₂ influence on transpiration — scale the potential
        #     transpiration *demand* by the linear CO₂ reduction
        #     factor before it enters the water balance, so elevated
        #     CO₂ propagates into the water-stress factor ``TRANRF``.
        #     This is distinct from the CO₂ correction already applied
        #     to reference ET.
        co2_trans = self.co2_transpiration(et["ptran"], site_params.co2)
        ptran_eff = co2_trans["ptran"]
        co2_factor = co2_trans["co2_factor"]

        # 4. Two-zone water balance with a percolation cascade.
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

        # 5. Phenology. Reads only ``state.dvs``, ``state.tsump``,
        #    ``davtmp`` and ``ddlp``, so its position in the sequence
        #    is independent of the water / ET branch.
        pheno = self.phenology(state, davtmp, ddlp, crop_params)

        # 6. Nutrient demand — independent of partitioning; produces the
        #    NPK stress index that multiplies ``GTOTAL`` downstream.
        #    ``PARINT`` is converted from J m⁻² d⁻¹ to MJ PAR m⁻² d⁻¹
        #    here because the growth formula expects MJ.
        parint_mj = irrad_out["parint"] * 1e-6
        nut = self.nutrient_demand(
            state=state,
            crop_params=crop_params,
            soil_params=soil_params,
            tranrf=tranrf,
        )
        nstress = nut["nstress"]

        # 7. Soil mineral-pool balance. Runs after NutrientDemand
        #    because it consumes the day's NUPTR/PUPTR/KUPTR (which
        #    deplete NMINT/PMINT/KMINT) and reuses the same
        #    NLIMIT/EMERG gates that drive uptake.
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

        # 8. Photosynthesis with nutrient and water stress applied.
        #
        # ``GTOTAL = RUE · RTMCO · PARINT · TRANRF · combined_stress``,
        # where ``combined_stress`` is the nutrient component obtained
        # by dividing out ``TRANRF`` from ``self.stress(tranrf, nstress)``.
        # This factoring keeps ``self.stress`` swappable (e.g. with a
        # learned stress module) while preserving the explicit
        # ``TRANRF`` multiplier.
        photo = self.photosynthesis(
            tmax=tmax,
            tmin=tmin,
            dvs=state.dvs,
            params=crop_params,
            co2=site_params.co2,
        )
        combined_stress = self.stress(tranrf, nstress) / torch.clamp(tranrf, min=1e-6)
        gtotal = (
            photo["rue"]
            * photo["rtmco"]
            * parint_mj
            * tranrf
            * combined_stress
        )

        # Optional neural residual correction on ``gtotal``.
        if "photosynthesis" in self.residual_modules:
            ctx = torch.stack(
                [state.lai, state.dvs, davtmp, dtr, tranrf, nstress, state.wa, doy],
                dim=-1,
            )
            gtotal = gtotal + self.residual_modules["photosynthesis"](ctx).squeeze(-1)
            gtotal = torch.clamp(gtotal, min=0.0)

        # 9. Partitioning. Water and N stress are fed into ``SUBPAR``;
        #    ``nstress`` is used as the NPK proxy for ``NNI``.
        part = self.partitioning(
            state=state,
            gtotal=gtotal,
            params=crop_params,
            tranrf=tranrf,
            nni=nstress,
        )

        # 10. Leaf dynamics with heat-stress acceleration of senescence.
        #     `HeatStressOnLeafSenescence` returns a multiplier on
        #     ``RDR`` that equals ``1.0`` outside the heat-stress regime
        #     (``Tmax < Tc`` or ``DVS < DVS_c``), so the heat term is a
        #     no-op under non-stress conditions.
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

        # Mask used to switch off crop growth and senescence after
        # maturity (``DVS ≥ 2``).
        active = (state.dvs < 2.0).to(davtmp.dtype)
        gate = lambda x: x * active  # noqa: E731

        # Per-organ NPK losses to dead tissue
        # (``RNLDLV = rnflv · DLV``, etc.). The leaf/stem/root dynamics
        # modules emit the daily senescence biomass fluxes
        # ``wlvd_rate``/``wstd_rate``/``wrtd_rate`` (= ``dlv``/``drst``/
        # ``drrt``); multiplying by the residual concentrations gives
        # the corresponding N/P/K losses. These are subtracted from the
        # living-organ NPK net rates and integrated into the
        # ``nlossl``/``nlossr``/``nlosss`` (and P, K) accumulators so
        # that the SIMPLACE balance ``NUPTT + NFIXTT + initial =
        # ANLV+... + NLOSSL+...`` holds.
        dlv = leaf["wlvd_rate"]
        drst = stem["wstd_rate"]
        drrt = root["wrtd_rate"]
        rnldlv = crop_params.rnflv * dlv
        rnldst = crop_params.rnfst * drst
        rnldrt = crop_params.rnfrt * drrt
        rpldlv = crop_params.rpflv * dlv
        rpldst = crop_params.rpfst * drst
        rpldrt = crop_params.rpfrt * drrt
        rkldlv = crop_params.rkflv * dlv
        rkldst = crop_params.rkfst * drst
        rkldrt = crop_params.rkfrt * drrt

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
            "anlv_rate": gate(nut["n_lv_rate"] - rnldlv),
            "anst_rate": gate(nut["n_st_rate"] - rnldst),
            "anrt_rate": gate(nut["n_rt_rate"] - rnldrt),
            "anso_rate": gate(nut["n_so_rate"]),
            "aplv_rate": gate(nut["p_lv_rate"] - rpldlv),
            "apst_rate": gate(nut["p_st_rate"] - rpldst),
            "aprt_rate": gate(nut["p_rt_rate"] - rpldrt),
            "apso_rate": gate(nut["p_so_rate"]),
            "aklv_rate": gate(nut["k_lv_rate"] - rkldlv),
            "akst_rate": gate(nut["k_st_rate"] - rkldst),
            "akrt_rate": gate(nut["k_rt_rate"] - rkldrt),
            "akso_rate": gate(nut["k_so_rate"]),
            # Dead-tissue NPK accumulators (SIMPLACE NLOSSL/NLOSSR/NLOSSS
            # and P, K analogues). Gated by ``active`` so accumulation
            # stops after maturity, mirroring the per-organ death rates.
            "nlossl_rate": gate(rnldlv),
            "nlossr_rate": gate(rnldrt),
            "nlosss_rate": gate(rnldst),
            "plossl_rate": gate(rpldlv),
            "plossr_rate": gate(rpldrt),
            "plosss_rate": gate(rpldst),
            "klossl_rate": gate(rkldlv),
            "klossr_rate": gate(rkldrt),
            "klosss_rate": gate(rkldst),
            # Soil pool dynamics. Organic pools deplete (negative
            # rates); inorganic pools follow ``fertiliser +
            # mineralisation − uptake``. Mineralisation is already
            # gated internally by EMERG/NLIMIT, and the soil pools are
            # *not* multiplied by ``active`` so that mineralisation
            # and fertiliser additions continue after crop maturity.
            "nmin_rate": soil_nut["nmin_rate"],
            "pmin_rate": soil_nut["pmin_rate"],
            "kmin_rate": soil_nut["kmin_rate"],
            "nmint_rate": soil_nut["nmint_rate"],
            "pmint_rate": soil_nut["pmint_rate"],
            "kmint_rate": soil_nut["kmint_rate"],
            "tran_cum_rate": water["tran"],
            "evap_cum_rate": water["evap"],
            # Cumulative water, nutrient and growth accumulators are
            # integrated by the engine via the same ``_rate`` mechanism
            # as ``tran_cum`` / ``evap_cum``. With ``dt = 1`` day, the
            # daily flux is itself the increment.
            "rain_cum_rate": rain,
            "irrig_cum_rate": water["rirr"],
            "runoff_cum_rate": water["runoff"],
            "drain_cum_rate": water["drain"],
            "nuptr_cum_rate": nut["nuptr"],
            "puptr_cum_rate": nut["puptr"],
            "kuptr_cum_rate": nut["kuptr"],
            "nfixtr_cum_rate": nut["nfixtr"],
            # PAR accumulated in MJ m⁻² (same conversion used in the
            # GTOTAL calculation above).
            "parint_cum_rate": parint_mj,
            "gtotal_cum_rate": gtotal,
            # Scalar diagnostics — not integrated by the engine.
            "tranrf": tranrf,
            "nstress": nstress,
            "gtotal": gtotal,
        }

        # ---- Emergence-day bootstrap.
        # Fires once per batch element on the step where ``tsump``
        # first crosses ``tsumem``. The seed-reserve mass ``tdwi`` is
        # partitioned at ``DVSI`` to give ``WRTI``/``WLVGI``/``WSTI``/
        # ``WSOI`` and ``LAII = WLVGI · scale_factor_sla · SLATB(DVSI)``,
        # which are injected as one-shot rate deltas so that emergence
        # instantly mobilises seed reserves into a juvenile-canopy
        # state. The one-day LAI jump is intrinsic to the Lintul5
        # abstraction; for a gradient sowing-to-canopy ramp, lower
        # ``crop_params.tdwi`` (smaller seed reserve) or raise
        # ``crop_params.rgrl`` (faster juvenile expansion).
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

        # Seed NPK content per organ — ``ANLVI = NMAXLV(DVSI) · WLVGI``
        # etc. in SIMPLACE ``Lintul5.java`` initValues(). Stems and roots
        # use the leaf max-concentration scaled by ``LSNR``/``LRNR`` (and
        # the P/K analogues). Storage organs start empty (``ANSOI = 0``).
        nmaxlv_i = interpolate(crop_params.nmxlv, x).reshape(())
        pmaxlv_i = interpolate(crop_params.pmxlv, x).reshape(())
        kmaxlv_i = interpolate(crop_params.kmxlv, x).reshape(())
        anlvi = nmaxlv_i * wlvgi
        ansti = crop_params.lsnr * nmaxlv_i * wsti
        anrti = crop_params.lrnr * nmaxlv_i * wrti
        aplvi = pmaxlv_i * wlvgi
        apsti = crop_params.lspr * pmaxlv_i * wsti
        aprti = crop_params.lrpr * pmaxlv_i * wrti
        aklvi = kmaxlv_i * wlvgi
        aksti = crop_params.lskr * kmaxlv_i * wsti
        akrti = crop_params.lrkr * kmaxlv_i * wrti

        tsump_next = state.tsump + rates["tsump_rate"] * self.engine.dt
        emerg_now = (
            (state.tsump < crop_params.tsumem) & (tsump_next >= crop_params.tsumem)
        ).to(davtmp.dtype)
        rates["wlv_rate"] = rates["wlv_rate"] + emerg_now * wlvgi / self.engine.dt
        rates["wst_rate"] = rates["wst_rate"] + emerg_now * wsti / self.engine.dt
        rates["wrt_rate"] = rates["wrt_rate"] + emerg_now * wrti / self.engine.dt
        rates["wso_rate"] = rates["wso_rate"] + emerg_now * wsoi / self.engine.dt
        rates["lai_rate"] = rates["lai_rate"] + emerg_now * laii_dyn / self.engine.dt
        rates["anlv_rate"] = rates["anlv_rate"] + emerg_now * anlvi / self.engine.dt
        rates["anst_rate"] = rates["anst_rate"] + emerg_now * ansti / self.engine.dt
        rates["anrt_rate"] = rates["anrt_rate"] + emerg_now * anrti / self.engine.dt
        rates["aplv_rate"] = rates["aplv_rate"] + emerg_now * aplvi / self.engine.dt
        rates["apst_rate"] = rates["apst_rate"] + emerg_now * apsti / self.engine.dt
        rates["aprt_rate"] = rates["aprt_rate"] + emerg_now * aprti / self.engine.dt
        rates["aklv_rate"] = rates["aklv_rate"] + emerg_now * aklvi / self.engine.dt
        rates["akst_rate"] = rates["akst_rate"] + emerg_now * aksti / self.engine.dt
        rates["akrt_rate"] = rates["akrt_rate"] + emerg_now * akrti / self.engine.dt

        # Per-day `DiagnosticState` snapshot — built from the same
        # intermediate tensors used to assemble ``rates``. It is
        # read-only: it neither enters the Euler update nor touches
        # any state field, and it adds no autograd ops beyond
        # broadcasting scalar factors (``combined_stress``,
        # ``co2_factor``) to ``[B]``.
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
        """Collect every learnable parameter across the parameter containers.

        Walks the ``crop``, ``soil`` and ``site`` parameter containers
        and returns every field that is an `nn.Parameter` (i.e. has
        been marked as learnable by the user).

        Returns:
            Dict keyed by ``"<container>.<field>"`` (for example,
            ``"crop.rue"``) mapping to the corresponding `nn.Parameter`.
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
