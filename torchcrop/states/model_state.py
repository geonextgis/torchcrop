"""Tensor container for all Lintul5 state variables.

Every field has shape ``[B]`` for scalar-per-batch quantities. The container
is a simple ``@dataclass``; all operations that "update" state return a new
``ModelState`` instance so the computation graph is preserved for autograd.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any

import torch


@dataclass
class ModelState:
    """Full Lintul5 state vector.

    Units follow Wolf (2012).
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

    # Water
    wa: torch.Tensor  # [B] mm — actual soil water in root zone

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
    ) -> "ModelState":
        """Construct a zeroed initial state for a batch."""
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
        """Functional update — returns a new ``ModelState`` with fields replaced."""
        return replace(self, **updates)

    def stack(self) -> torch.Tensor:
        """Stack all scalar-per-batch tensors into ``[B, C]`` for logging."""
        tensors = [
            getattr(self, f.name)
            for f in fields(self)
            if isinstance(getattr(self, f.name), torch.Tensor)
        ]
        return torch.stack(tensors, dim=-1)

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)]
