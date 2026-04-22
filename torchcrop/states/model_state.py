"""Tensor container for all Lintul5 state variables.

Every field has shape ``[B]`` for scalar-per-batch quantities. The container
is a simple ``@dataclass``; all operations that "update" state return a new
``ModelState`` instance so the computation graph is preserved for autograd.

Terminology:
    * **State variables**: quantities that persist across time steps and
      define the system at a given instant (e.g., ``dvs``, ``lai``, ``wlv``,
      ``wa``). They are integrated over time via explicit Euler updates.
    * **Rate variables**: time derivatives of states (units of state-unit per
      day) computed by the process modules each step (e.g., ``dvs_rate``,
      ``wlv_rate``, ``lai_rate``). Rates are *not* stored on ``ModelState``;
      they are produced by process ``forward()`` calls and consumed by the
      state update step ``s_{t+1} = s_t + r_t * dt``.
    * **Output variables**: per-step snapshots of selected states /
      diagnostics collected by the engine and returned by ``Lintul5Model``
      as a ``ModelOutput`` (e.g., trajectories of ``dvs``, ``lai``, total
      biomass, and final ``yield_ = wso`` at maturity).

References:
    * Wolf, J. (2012). *User guide for LINTUL5*. Wageningen UR.
    * SIMPLACE reference: ``simplace/sim/components/models/lintul5/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any

import torch


@dataclass
class ModelState:
    """Full Lintul5 state vector.

    Holds every persistent quantity advanced by the simulation engine. All
    fields are ``torch.Tensor`` of shape ``[B]`` (one scalar per batch
    element) and units follow Wolf (2012). The container is purely a *state*
    snapshot — daily *rates* are produced by the process modules and the
    *outputs* (per-step trajectories, final yield, etc.) are assembled by the
    engine; see the module docstring for the full catalogue.

    Attributes:
        dvs: Development stage in ``[0, 2]`` (0=emergence, 1=anthesis,
            2=maturity), ``[B]``, dimensionless.
        tsum: Thermal time accumulated since emergence, ``[B]`` [°C d].
        tsump: Thermal time accumulated since sowing, ``[B]`` [°C d].
        vern: Vernalisation days accumulated, ``[B]`` [d].
        wlv: Green leaf dry weight, ``[B]`` [g m⁻²].
        wlvd: Dead leaf dry weight (senesced), ``[B]`` [g m⁻²].
        wst: Stem dry weight, ``[B]`` [g m⁻²].
        wrt: Root dry weight, ``[B]`` [g m⁻²].
        wso: Storage organ dry weight (drives final yield), ``[B]``
            [g m⁻²].
        lai: Leaf area index, ``[B]`` [m² m⁻²].
        rootd: Rooting depth, ``[B]`` [m].
        wa: Total soil water in the rooted zone, ``[B]`` [mm].
        wa_lower: Total soil water in the lower zone (between ``rootd`` and
            the maximum rooting depth ``rdm``), ``[B]`` [mm].
        dslr: Days since last (infiltrating) rain event, ``[B]`` [d];
            drives the Stroosnijder soil-evaporation model.
        dsos: Days of oxygen shortage, ``[B]`` [d], clipped to ``[0, 4]``;
            drives the time-dependent ``RWET`` waterlogging factor.
        anlv, anst, anrt, anso: Nitrogen pools in leaves, stems, roots,
            storage organs, each ``[B]`` [g N m⁻²].
        aplv, apst, aprt, apso: Phosphorus pools, each ``[B]`` [g P m⁻²].
        aklv, akst, akrt, akso: Potassium pools, each ``[B]`` [g K m⁻²].
        tran_cum: Cumulative actual transpiration, ``[B]`` [mm].
        evap_cum: Cumulative soil evaporation, ``[B]`` [mm].

    Note:
        Updates are functional: `replace` returns a *new*
        ``ModelState`` rather than mutating in place, so the autograd graph
        is preserved across the explicit-Euler step
        ``s_{t+1} = s_t + r_t * dt``.
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
    wrt: torch.Tensor  # roots
    wso: torch.Tensor  # storage organs

    # Canopy
    lai: torch.Tensor  # [B] m2 m-2

    # Roots
    rootd: torch.Tensor  # [B] m

    # Water — two-zone bucket (SIMPLACE WATBALS)
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

    # Optional bookkeeping
    tran_cum: torch.Tensor = field(default=None)  # cumulative transpiration [mm]
    evap_cum: torch.Tensor = field(default=None)  # cumulative evaporation [mm]

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
    ) -> "ModelState":
        """Construct a zeroed initial state for a batch.

        Args:
            batch_size: Number of parallel simulation instances ``B``.
            dtype: Tensor dtype.
            device: Torch device.
            dvsi: Initial development stage ``DVSI``.
            wai: Initial soil water content in the root zone [mm].
            rootdi: Initial rooting depth [m].

        Returns:
            A fresh `ModelState` with all biomass / nutrient pools at
            zero and the specified initial values for ``dvs``, ``rootd`` and
            ``wa``.
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
            wrt=zeros.clone(),
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
            tran_cum=zeros.clone(),
            evap_cum=zeros.clone(),
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
