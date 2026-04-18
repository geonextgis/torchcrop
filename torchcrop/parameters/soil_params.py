"""Soil-specific parameters for the Lintul5 water balance.

Shapes
------
Scalar per batch: ``[B]`` or ``[]`` (broadcastable).
Multi-layer (optional): ``[B, N_layers]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import torch


def _t(x: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype)


@dataclass
class SoilParameters:
    """Soil hydraulic parameters.

    Notes
    -----
    Only a single-layer bucket model is implemented in the initial scaffold;
    the containers accept either ``[B]`` (single-layer) or ``[B, N_layers]``
    tensors for future multi-layer extensions.
    """

    # Volumetric soil water contents [m3 m-3]
    wcwp: torch.Tensor = field(default_factory=lambda: _t(0.12))
    wcfc: torch.Tensor = field(default_factory=lambda: _t(0.30))
    wcst: torch.Tensor = field(default_factory=lambda: _t(0.40))
    wcad: torch.Tensor = field(default_factory=lambda: _t(0.05))

    # Initial water content [mm] — optional; if None, initialised to field capacity
    wci: torch.Tensor | None = None

    # Drainage / runoff
    drate: torch.Tensor = field(default_factory=lambda: _t(50.0))
    runfr: torch.Tensor = field(default_factory=lambda: _t(0.0))

    # Soil mineral N supply [g N m-2 d-1] (simple background supply)
    nmins: torch.Tensor = field(default_factory=lambda: _t(0.50))
    pmins: torch.Tensor = field(default_factory=lambda: _t(0.05))
    kmins: torch.Tensor = field(default_factory=lambda: _t(0.30))

    def to(self, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> "SoilParameters":
        kwargs: dict[str, Any] = {}
        for f in fields(self):
            t = getattr(self, f.name)
            if isinstance(t, torch.Tensor):
                kwargs[f.name] = t.to(dtype=dtype, device=device)
            else:
                kwargs[f.name] = t
        return SoilParameters(**kwargs)


def default_loam_params(dtype: torch.dtype = torch.float32) -> SoilParameters:
    """Return a plausible loam default parameterisation."""
    return SoilParameters().to(dtype=dtype)
