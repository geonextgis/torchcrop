"""Top-level Lintul5 model.

Wires all process sub-modules together and provides both a high-level and a
low-level API (see ``CLAUDE.md``, §7).
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
    Irradiation,
    LeafDynamics,
    NutrientDemand,
    Partitioning,
    Phenology,
    Photosynthesis,
    PotentialEvapoTranspiration,
    RootDynamics,
    StressFactors,
    WaterBalance,
)
from torchcrop.states.model_state import ModelState


@dataclass
class ModelOutput:
    """Container for a full simulation run.

    Attributes
    ----------
    states : list[ModelState]
        Per-day state snapshots (length ``T + 1``; the first entry is the
        initial condition).
    rates : list[dict[str, torch.Tensor]]
        Per-day rate dicts (length ``T``).
    yield_ : torch.Tensor
        Final storage-organ dry weight ``WSO`` at the last step [g m-2].
    lai : torch.Tensor
        LAI trajectory of shape ``[B, T + 1]``.
    dvs : torch.Tensor
        DVS trajectory of shape ``[B, T + 1]``.
    biomass : torch.Tensor
        Above-ground biomass trajectory of shape ``[B, T + 1]``.
    """

    states: list[ModelState]
    rates: list[dict[str, torch.Tensor]]
    yield_: torch.Tensor
    lai: torch.Tensor
    dvs: torch.Tensor
    biomass: torch.Tensor


class Lintul5Model(nn.Module):
    """Differentiable reimplementation of the Lintul5 crop growth model.

    Parameters
    ----------
    crop_params, soil_params, site_params :
        Parameter containers (see :mod:`torchcrop.parameters`).
    smooth : bool
        Use smooth (sigmoid-blend) replacements for stage-based branching.
    stress_module : nn.Module, optional
        Replacement for the default :class:`StressFactors` combiner.
    residual_modules : dict[str, nn.Module], optional
        Optional neural residual corrections keyed by process name
        (``"photosynthesis"`` adds to ``gtotal``; ``"partitioning"`` adds to
        the four allocation fractions; ``"leaf_dynamics"`` adds to ``lai_rate``).
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
        self.water_balance = WaterBalance()
        self.photosynthesis = Photosynthesis()
        self.partitioning = Partitioning()
        self.leaf_dynamics = LeafDynamics()
        self.root_dynamics = RootDynamics()
        self.nutrient_demand = NutrientDemand()
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
        """Build an initial state for a batch, using ``dvsi`` from crop params."""
        dvsi = float(self.crop_params.dvsi.detach().cpu().item())
        rootdi = float(self.crop_params.rootdi.detach().cpu().item())
        # Initialise at field capacity × initial rooting depth (mm)
        wfc = float(self.soil_params.wcfc.detach().cpu().item())
        wai = 1000.0 * wfc * rootdi
        state = ModelState.initial(
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            dvsi=dvsi,
            wai=wai,
            rootdi=rootdi,
        )
        # Seed leaf mass so that LAI growth has a substrate post-emergence
        laii = float(self.crop_params.laii.detach().cpu().item())
        sla = float(self.crop_params.sla.detach().cpu().item())
        wlv0 = torch.full_like(state.wlv, laii / max(sla, 1e-6))
        lai0 = torch.full_like(state.lai, laii)
        return state.replace(wlv=wlv0, lai=lai0)

    def forward(
        self,
        weather: WeatherDriver | torch.Tensor,
        start_doy: int = 1,
        initial_state: ModelState | None = None,
    ) -> ModelOutput:
        """Run a full simulation and return trajectories plus final yield."""
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

        states, rates = self.engine.run(
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

        return ModelOutput(
            states=states,
            rates=rates,
            yield_=yield_,
            lai=lai,
            dvs=dvs,
            biomass=biomass,
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
        """Compute the rate vector for a single day (low-level API)."""
        return self._compute_rates_dispatch(
            state=state,
            weather_day=weather_day,
            doy=doy,
            crop_params=self.crop_params,
            soil_params=self.soil_params,
            site_params=self.site_params,
        )

    def update_state(
        self,
        state: ModelState,
        rates: dict[str, torch.Tensor],
        dt: float = 1.0,
    ) -> ModelState:
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
    ) -> dict[str, torch.Tensor]:
        davtmp = weather_day["davtmp"]
        irrad = weather_day["irrad"]
        rain = weather_day["rain"]

        # 1. Astro
        lat_b = site_params.latitude.expand_as(doy) if site_params.latitude.dim() > 0 else site_params.latitude
        astro = self.astro(doy=doy, latitude=lat_b)
        ddlp = astro["ddlp"]

        # 2. Phenology
        pheno = self.phenology(state, davtmp, ddlp, crop_params)

        # 3. Evapotranspiration
        et = self.evapotranspiration(
            davtmp=davtmp,
            irrad=irrad,
            lai=state.lai,
            k_ext=crop_params.k,
        )

        # 4. Water balance (uses current LAI and root depth)
        water = self.water_balance(
            state=state,
            rain=rain,
            pevap=et["pevap"],
            ptran=et["ptran"],
            params=soil_params,
        )
        tranrf = water["tranrf"]

        # 5. Irradiation — PAR intercepted
        irrad_out = self.irradiation(state=state, irrad=irrad, params=crop_params)

        # 6+7. Nutrient preliminary step — we first estimate partitioning
        #      using a "no nutrient stress" GTOTAL to compute demand, then
        #      finalise with the resulting nstress.
        photo_pre = self.photosynthesis(
            parint=irrad_out["parint"],
            davtmp=davtmp,
            tranrf=tranrf,
            nstress=torch.ones_like(davtmp),
            params=crop_params,
        )
        part_pre = self.partitioning(state=state, gtotal=photo_pre["gtotal"], params=crop_params)
        nut = self.nutrient_demand(
            state=state,
            g_lv=part_pre["g_lv"],
            g_st=part_pre["g_st"],
            g_rt=part_pre["g_root"],
            g_so=part_pre["g_so"],
            crop_params=crop_params,
            soil_params=soil_params,
        )
        nstress = nut["nstress"]

        # 8. Photosynthesis (final) with nutrient + water stress
        photo = self.photosynthesis(
            parint=irrad_out["parint"],
            davtmp=davtmp,
            tranrf=tranrf,
            nstress=self.stress(tranrf, nstress) / torch.clamp(tranrf, min=1e-6),
            params=crop_params,
        )
        gtotal = photo["gtotal"]

        # Residual correction on gtotal
        if "photosynthesis" in self.residual_modules:
            ctx = torch.stack(
                [state.lai, state.dvs, davtmp, irrad, tranrf, nstress, state.wa, doy],
                dim=-1,
            )
            gtotal = gtotal + self.residual_modules["photosynthesis"](ctx).squeeze(-1)
            gtotal = torch.clamp(gtotal, min=0.0)

        # 9. Partitioning
        part = self.partitioning(state=state, gtotal=gtotal, params=crop_params)

        # 10. Leaf dynamics
        leaf = self.leaf_dynamics(
            state=state,
            g_lv=part["g_lv"],
            dtsu=pheno["dtsu"],
            tranrf=tranrf,
            nstress=nstress,
            params=crop_params,
        )

        # 11. Root dynamics
        root = self.root_dynamics(
            state=state,
            g_root=part["g_root"],
            tranrf=tranrf,
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
            "wst_rate": gate(part["g_st"]),
            "wrt_rate": gate(root["wrt_rate"]),
            "wso_rate": gate(part["g_so"]),
            "lai_rate": gate(leaf["lai_rate"]),
            "rootd_rate": root["rootd_rate"],
            "wa_rate": water["wa_rate"],
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
            "tran_cum_rate": water["tran"],
            "evap_cum_rate": water["evap"],
            # Diagnostics (not integrated)
            "tranrf": tranrf,
            "nstress": nstress,
            "gtotal": gtotal,
        }
        return rates

    # ------------------------------------------------------------------ #
    # Convenience: flatten all learnable parameters across dataclasses
    # ------------------------------------------------------------------ #

    def learnable_parameter_groups(self) -> dict[str, Any]:
        """Return a dict of named nn.Parameters across all parameter containers."""
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
