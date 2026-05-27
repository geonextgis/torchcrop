"""Tensor containers for Lintul5 simulation state and per-day diagnostics.

This module defines two dataclasses:

* `ModelState` — the **integrated** state vector advanced by the engine
  each day (phenology, biomass pools, canopy, water, NPK pools, soil
  minerals, and a set of cumulative accumulators).
* `DiagnosticState` — a **non-integrated** per-day snapshot of stress
  factors, growth drivers, light interception, phenology modifiers and
  daily fluxes, collected alongside `ModelState` for interpretation and
  ML loss design.

Every field has shape ``[B]`` for scalar-per-batch quantities. All
containers are plain ``@dataclass`` instances; updates are *functional*
(``replace`` returns a new instance), so the autograd computation graph
is preserved across the explicit-Euler step.

Terminology:
    * **State variables** — quantities that persist across time steps
      and define the system at an instant (e.g. ``dvs``, ``lai``,
      ``wlv``, ``wa``). Integrated over time via explicit Euler.
    * **Rate variables** — time derivatives in units of state-unit per
      day (e.g. ``dvs_rate``, ``wlv_rate``, ``lai_rate``). Produced by
      process ``forward()`` calls and consumed by the state update
      ``s_{t+1} = s_t + r_t · dt``. Rates are *not* stored on state.
    * **Output variables** — per-step snapshots collected by the engine
      and returned by ``Lintul5Model`` as a ``ModelOutput`` (e.g.
      trajectories of ``dvs``, ``lai``, biomass, and final
      ``yield_ = wso`` at maturity).

Reference:
    Wolf, J. (2012). *User guide for LINTUL5*. Wageningen UR.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any

import torch


@dataclass
class ModelState:
    """Full Lintul5 state vector.

    Holds every persistent quantity advanced by the simulation engine.
    All fields are ``torch.Tensor`` of shape ``[B]`` (one scalar per
    batch element) with units following Wolf (2012). The container is
    purely a *state* snapshot — daily *rates* are produced by the
    process modules and the *outputs* (per-step trajectories, final
    yield, etc.) are assembled by the engine.

    Attributes are grouped below by physical role.

    ## Phenology:
    - `dvs`: Development stage in ``[0, 2]`` (0 = emergence,
            1 = anthesis, 2 = maturity) [-].
    - `tsum`: Thermal time accumulated since emergence [°C d].
    - `tsump`: Thermal time accumulated since sowing [°C d].
    - `vern`: Vernalisation days accumulated [d].

    ## Biomass pools [g DM m⁻²]:
    - `wlv`, `wst`, `wrt`, `wso`: Living dry weight of leaves, stems, roots,
        and storage organs (``wso`` drives final yield).
    - `wlvd`, `wstd`, `wrtd`: Dead (senesced) dry weight of leaves, stems,
        and roots. Each accumulates the corresponding daily
        senescence flux so the N/P/K carried by dead tissue is
        conserved rather than discarded.

    ## Canopy and roots:
    - `lai`: Leaf area index [m² m⁻²].
    - `rootd`: Rooting depth [m].

    ## Soil water (two-zone bucket):
    - `wa`: Water stored in the rooted zone (full column, not "above
        wilting point") [mm].
    - `wa_lower`: Water stored in the lower zone, between ``rootd`` and
        the maximum rooting depth ``rdm`` [mm].
    - `dslr`: Days since the last (infiltrating) rain event [d];
        drives the Stroosnijder soil-evaporation model. Maintained
        as a 1-based counter (minimum value 1).
    - `dsos`: Days of oxygen shortage [d], clipped to ``[0, 4]``;
        drives the time-dependent ``RWET`` waterlogging factor.

    ## Per-organ nutrient pools [g X m⁻²]:
    - `anlv`, `anst`, `anrt`, `anso`: Nitrogen in leaves, stems, roots,
        storage organs.
    - `aplv`, `apst`, `aprt`, `apso`: Phosphorus in the same four organs.
    - `aklv`, `akst`, `akrt`, `akso`: Potassium in the same four organs.

    ## Soil mineral pools [g X m⁻²]:
    - `nmin`, `pmin`, `kmin`: Mineralisable organic N/P/K pools. Depleted
        each day by the mineralisation flux (handled with a
        negative-rate sign convention).
    - `nmint`, `pmint`, `kmint`: Directly available inorganic N/P/K pools.
        Replenished daily by fertiliser (after the recovery
        fraction) plus mineralisation from
        ``nmin``/``pmin``/``kmin``, and drawn down by crop uptake.
        This is the pool that caps soil-limited uptake under the
        higher ``IOPT`` modes.

    ## Cumulative water accumulators [mm]:
    - `tran_cum`: Actual transpiration.
    - `evap_cum`: Soil evaporation.
    - `rain_cum`: Precipitation.
    - `irrig_cum`: Effective irrigation.
    - `runoff_cum`: Surface runoff.
    - `drain_cum`: Deep drainage (cascade flux below the lower zone).

    ## Cumulative nutrient accumulators [g X m⁻²]:
    - `nuptr_cum`, `puptr_cum`, `kuptr_cum`: Crop NPK uptake from the soil
        (excludes biological N fixation).
    - `nfixtr_cum`: Biological N₂ fixation.

    ## Cumulative growth accumulators:
    - `parint_cum`: Canopy-intercepted PAR [MJ PAR m⁻²].
    - `gtotal_cum`: Gross daily assimilate from photosynthesis,
            pre-partitioning [g DM m⁻²].

    ## Note:
    Updates are functional — `replace` returns a *new* ``ModelState``
    rather than mutating in place — so the autograd graph is
    preserved across the explicit-Euler step
    ``s_{t+1} = s_t + r_t · dt``. All cumulative accumulators are
    integrated by the same ``_rate`` mechanism as ``tran_cum`` /
    ``evap_cum``: a daily flux routed into ``<field>_rate`` and
    added by the engine with ``dt = 1`` day.
    """

    # Phenology
    dvs: torch.Tensor  # [B] development stage [0, 2]
    tsum: torch.Tensor  # [B] °C d — thermal time since emergence
    tsump: torch.Tensor  # [B] °C d — thermal time since sowing/planting
    vern: torch.Tensor  # [B] d — vernalisation days

    # Biomass pools [g m-2]
    wlv: torch.Tensor  # green leaves
    wlvd: torch.Tensor  # dead leaves
    wst: torch.Tensor  # stems
    wstd: torch.Tensor  # dead stems
    wrt: torch.Tensor  # roots
    wrtd: torch.Tensor  # dead roots
    wso: torch.Tensor  # storage organs

    # Canopy
    lai: torch.Tensor  # [B] m2 m-2

    # Roots
    rootd: torch.Tensor  # [B] m

    # Water — two-zone bucket
    wa: torch.Tensor  # [B] mm — total water in rooted zone
    wa_lower: torch.Tensor  # [B] mm — total water in lower zone (between rootd and rdm)
    dslr: torch.Tensor  # [B] d — days since last rain (Stroosnijder evap model)
    dsos: torch.Tensor  # [B] d — days of oxygen shortage (0–4, RWET model)

    # Nitrogen pools [g N m-2]
    anlv: torch.Tensor
    anst: torch.Tensor
    anrt: torch.Tensor
    anso: torch.Tensor

    # Phosphorus pools [g P m-2]
    aplv: torch.Tensor
    apst: torch.Tensor
    aprt: torch.Tensor
    apso: torch.Tensor

    # Potassium pools [g K m-2]
    aklv: torch.Tensor
    akst: torch.Tensor
    akrt: torch.Tensor
    akso: torch.Tensor

    # Soil mineral pools [g X m-2]
    nmin: torch.Tensor  # mineralisable organic N pool
    pmin: torch.Tensor  # mineralisable organic P pool
    kmin: torch.Tensor  # mineralisable organic K pool
    nmint: torch.Tensor  # directly available inorganic N pool
    pmint: torch.Tensor  # directly available inorganic P pool
    kmint: torch.Tensor  # directly available inorganic K pool

    # Cumulative per-organ NPK losses to dead tissue [g X m-2]
    # (SIMPLACE NLOSSL / NLOSSR / NLOSSS and P, K analogues).
    # Each is integrated from the daily residual-concentration flux
    # ``rXfORG · dORG`` (e.g. ``RNLDLV = rnflv · dlv``) so that NPK
    # carried by senesced biomass is conserved rather than discarded.
    nlossl: torch.Tensor = field(default=None)
    nlossr: torch.Tensor = field(default=None)
    nlosss: torch.Tensor = field(default=None)
    plossl: torch.Tensor = field(default=None)
    plossr: torch.Tensor = field(default=None)
    plosss: torch.Tensor = field(default=None)
    klossl: torch.Tensor = field(default=None)
    klossr: torch.Tensor = field(default=None)
    klosss: torch.Tensor = field(default=None)

    # Optional bookkeeping — cumulative water and growth accumulators.
    # Each is integrated by the standard `_rate` mechanism: a daily flux
    # is routed into ``<field>_rate`` in `_compute_rates_dispatch` and
    # the engine applies forward-Euler exactly as for any other state
    # field.
    tran_cum: torch.Tensor = field(default=None)  # cumulative transpiration [mm]
    evap_cum: torch.Tensor = field(default=None)  # cumulative evaporation [mm]
    rain_cum: torch.Tensor = field(default=None)  # cumulative precipitation [mm]
    irrig_cum: torch.Tensor = field(default=None)  # cumulative irrigation [mm]
    runoff_cum: torch.Tensor = field(default=None)  # cumulative surface runoff [mm]
    drain_cum: torch.Tensor = field(default=None)  # cumulative deep drainage [mm]
    nuptr_cum: torch.Tensor = field(default=None)  # cumulative soil N uptake [g N m-2]
    puptr_cum: torch.Tensor = field(default=None)  # cumulative soil P uptake [g P m-2]
    kuptr_cum: torch.Tensor = field(default=None)  # cumulative soil K uptake [g K m-2]
    nfixtr_cum: torch.Tensor = field(default=None)  # cumulative N fixation [g N m-2]
    parint_cum: torch.Tensor = field(default=None)  # cumulative intercepted PAR [MJ m-2]
    gtotal_cum: torch.Tensor = field(default=None)  # cumulative gross assimilate [g DM m-2]

    @classmethod
    def initial(
        cls,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        dvsi: float = 0.0,
        wai: float = 60.0,
        rootdi: float = 0.10,
        wa_lower_i: float = 400.0,
        dslri: float = 3.0,
        dsosi: float = 0.0,
        nmini: float = 0.0,
        pmini: float = 0.0,
        kmini: float = 0.0,
        nminti: float = 0.0,
        pminti: float = 0.0,
        kminti: float = 0.0,
    ) -> "ModelState":
        """Construct a zeroed initial state for a batch.

        All biomass, per-organ nutrient, cumulative-accumulator, and
        thermal-time fields are initialised to zero. The remaining
        scalars are taken from the keyword arguments.

        Args:
            batch_size: Number of parallel simulation instances ``B``.
            dtype: Tensor dtype.
            device: Torch device.
            dvsi: Initial development stage.
            wai: Initial soil water in the root zone [mm].
            rootdi: Initial rooting depth [m].
            wa_lower_i: Initial soil water in the lower zone [mm].
            dslri: Initial days since last rain [d].
            dsosi: Initial days of oxygen shortage [d].
            nmini, pmini, kmini: Initial mineralisable organic NPK
                pools [g X m⁻²].
            nminti, pminti, kminti: Initial directly available
                inorganic NPK pools [g X m⁻²] (default 0).

        Returns:
            A fresh `ModelState` with all biomass, nutrient, and
            cumulative pools at zero and the specified initial values
            for ``dvs``, ``rootd``, the two soil-water stores, the
            counters ``dslr``/``dsos``, and the soil mineral pools.
        """
        zeros = torch.zeros(batch_size, dtype=dtype, device=device)
        full = lambda v: torch.full((batch_size,), float(v), dtype=dtype, device=device)
        return cls(
            dvs=full(dvsi),
            tsum=zeros.clone(),
            tsump=zeros.clone(),
            vern=zeros.clone(),
            wlv=zeros.clone(),
            wlvd=zeros.clone(),
            wst=zeros.clone(),
            wstd=zeros.clone(),
            wrt=zeros.clone(),
            wrtd=zeros.clone(),
            wso=zeros.clone(),
            lai=zeros.clone(),
            rootd=full(rootdi),
            wa=full(wai),
            wa_lower=full(wa_lower_i),
            dslr=full(dslri),
            dsos=full(dsosi),
            anlv=zeros.clone(),
            anst=zeros.clone(),
            anrt=zeros.clone(),
            anso=zeros.clone(),
            aplv=zeros.clone(),
            apst=zeros.clone(),
            aprt=zeros.clone(),
            apso=zeros.clone(),
            aklv=zeros.clone(),
            akst=zeros.clone(),
            akrt=zeros.clone(),
            akso=zeros.clone(),
            nmin=full(nmini),
            pmin=full(pmini),
            kmin=full(kmini),
            nmint=full(nminti),
            pmint=full(pminti),
            kmint=full(kminti),
            tran_cum=zeros.clone(),
            evap_cum=zeros.clone(),
            rain_cum=zeros.clone(),
            irrig_cum=zeros.clone(),
            runoff_cum=zeros.clone(),
            drain_cum=zeros.clone(),
            nuptr_cum=zeros.clone(),
            puptr_cum=zeros.clone(),
            kuptr_cum=zeros.clone(),
            nfixtr_cum=zeros.clone(),
            parint_cum=zeros.clone(),
            gtotal_cum=zeros.clone(),
            nlossl=zeros.clone(),
            nlossr=zeros.clone(),
            nlosss=zeros.clone(),
            plossl=zeros.clone(),
            plossr=zeros.clone(),
            plosss=zeros.clone(),
            klossl=zeros.clone(),
            klossr=zeros.clone(),
            klosss=zeros.clone(),
        )

    def replace(self, **updates: Any) -> "ModelState":
        """Return a new `ModelState` with selected fields replaced.

        Args:
            **updates: Field name / tensor pairs to override. Fields not
                passed are copied through unchanged.

        Returns:
            A new `ModelState` instance with the updates applied.
        """
        return replace(self, **updates)

    def stack(self) -> torch.Tensor:
        """Stack all scalar-per-batch tensors into a single ``[B, C]`` tensor.

        Returns:
            A ``[B, C]`` tensor where ``C`` is the number of tensor fields,
            laid out in field-definition order.
        """
        tensors = [
            getattr(self, f.name)
            for f in fields(self)
            if isinstance(getattr(self, f.name), torch.Tensor)
        ]
        return torch.stack(tensors, dim=-1)

    @property
    def field_names(self) -> list[str]:
        """Names of all tensor fields on this `ModelState`."""
        return [
            f.name
            for f in fields(self)
            if isinstance(getattr(self, f.name), torch.Tensor)
        ]


@dataclass
class DiagnosticState:
    """Per-day diagnostic snapshot — companion to `ModelState`.

    Holds quantities that are **not** integrated by the engine but are
    valuable for interpretation, attribution, and ML loss design:
    stress factors, photosynthesis drivers, light interception,
    phenology modifiers, and per-day water and nutrient fluxes.

    Instances are constructed fresh each step from the same intermediate
    tensors used to build the rates dict. The engine collects one
    `DiagnosticState` per simulated day alongside the `ModelState`
    trajectory. Because these tensors never feed back into the state
    update, the diagnostic capture has zero impact on autograd or on
    the rate-update logic.

    All fields are ``torch.Tensor`` of shape ``[B]``. Attributes are
    grouped below by physical role.

    ## Stress factors:
    - `tranrf`: Water-stress (transpiration-reduction) factor in
        ``[0, 1]``.
    - `rdry`: Drought reduction component in ``[0, 1]`` (root-zone
        moisture vs. critical content).
    - `rwet`: Oxygen-shortage reduction component in ``[0, 1]``
        (waterlogging, ramped by the days-of-oxygen-shortage
        counter).
    - `nstress`: Combined NPK index ``min(nni, pni, kni)`` in
        ``[0, 1]`` — the nutrient multiplier on ``gtotal``.
    - `nni`, `pni`, `kni`: Per-nutrient concentration-based nutrition
        indices in ``[0, 1]``.
    - `leaf_heat_factor`: Per-day heat-stress acceleration of the
        relative leaf death rate (``≥ 1`` in regime, ``1``
        otherwise).
    - `combined_stress`: Final multiplicative growth reducer applied
        to ``gtotal`` (water × nutrient via the configured stress
        combiner).
    - `co2_factor`: CO₂ transpiration-reduction factor applied to
        potential transpiration before the water balance.

    ## Photosynthesis and growth drivers:
    - `gtotal`: Gross daily assimilate, pre-partitioning
        [g DM m⁻² d⁻¹].
    - `rue`: Radiation use efficiency [g MJ⁻¹].
    - `rtmco`: Combined temperature × CO₂ correction factor on RUE.

    ## Light and canopy:
    - `frac_intercepted`: Beer–Lambert canopy interception fraction.
    - `parint`: Canopy-intercepted PAR [J m⁻² d⁻¹].

    ## Phenology drivers:
    - `dtsu`: Effective daily thermal time [°C d d⁻¹].
    - `photofac`: Photoperiod factor (``1`` when daylength response is
        disabled).
    - `vernfac`: Vernalisation factor (``1`` when vernalisation is
        disabled or no longer active).

    ## Water fluxes [mm d⁻¹] and contents:
    - `tran`, `evap`: Actual transpiration / soil evaporation.
    - `runoff`: Surface runoff (preliminary + rejected infiltration).
    - `drain`: Deep drainage below the lower zone.
    - `rirr`: Effective irrigation.
    - `smact`, `smactl`: Volumetric soil-moisture content in the rooted /
        lower zone [m³ m⁻³].

    ## Nutrient fluxes [g X m⁻² d⁻¹]:
    - `nuptr`, `puptr`, `kuptr`: Daily soil NPK uptake (no fixation).
    - `nfixtr`: Daily biological N₂ fixation.
    - `n_demand`, `p_demand`, `k_demand`: Daily vegetative NPK demand.

    ## Partitioning fractions [-]:
    - `fr`, `fl`, `fs`, `fo`: Stress-modified fractions to root, leaf,
        stem and storage organ. Root + above-ground sum to 1.
    """

    # Stress factors
    tranrf: torch.Tensor
    rdry: torch.Tensor
    rwet: torch.Tensor
    nstress: torch.Tensor
    nni: torch.Tensor
    pni: torch.Tensor
    kni: torch.Tensor
    leaf_heat_factor: torch.Tensor
    combined_stress: torch.Tensor
    co2_factor: torch.Tensor

    # Photosynthesis / growth drivers
    gtotal: torch.Tensor
    rue: torch.Tensor
    rtmco: torch.Tensor

    # Light / canopy
    frac_intercepted: torch.Tensor
    parint: torch.Tensor

    # Phenology drivers
    dtsu: torch.Tensor
    photofac: torch.Tensor
    vernfac: torch.Tensor

    # Water fluxes (per-day, not cumulative)
    tran: torch.Tensor
    evap: torch.Tensor
    runoff: torch.Tensor
    drain: torch.Tensor
    rirr: torch.Tensor
    smact: torch.Tensor
    smactl: torch.Tensor

    # Nutrient fluxes (per-day)
    nuptr: torch.Tensor
    puptr: torch.Tensor
    kuptr: torch.Tensor
    nfixtr: torch.Tensor
    n_demand: torch.Tensor
    p_demand: torch.Tensor
    k_demand: torch.Tensor

    # Partitioning fractions
    fr: torch.Tensor
    fl: torch.Tensor
    fs: torch.Tensor
    fo: torch.Tensor

    def stack(self) -> torch.Tensor:
        """Stack all tensor fields into a single ``[B, C]`` tensor.

        Returns:
            A ``[B, C]`` tensor laid out in field-definition order.
        """
        tensors = [
            getattr(self, f.name)
            for f in fields(self)
            if isinstance(getattr(self, f.name), torch.Tensor)
        ]
        return torch.stack(tensors, dim=-1)

    @property
    def field_names(self) -> list[str]:
        """Names of all tensor fields on this `DiagnosticState`."""
        return [
            f.name
            for f in fields(self)
            if isinstance(getattr(self, f.name), torch.Tensor)
        ]
