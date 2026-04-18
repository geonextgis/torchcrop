"""Site-level parameters: latitude, altitude, initial conditions."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import torch


def _t(x: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype)


@dataclass
class SiteParameters:
    """Geographical site parameters.

    Parameters
    ----------
    latitude : torch.Tensor
        Decimal latitude in degrees, shape ``[B]`` or ``[]``.
    altitude : torch.Tensor
        Elevation in metres.
    angstrom_a, angstrom_b : torch.Tensor
        Coefficients for the Ångström–Prescott radiation formula
        (used when solar radiation is derived from sunshine hours).
    """

    latitude: torch.Tensor = field(default_factory=lambda: _t(52.0))
    altitude: torch.Tensor = field(default_factory=lambda: _t(10.0))
    angstrom_a: torch.Tensor = field(default_factory=lambda: _t(0.25))
    angstrom_b: torch.Tensor = field(default_factory=lambda: _t(0.50))

    def to(self, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> "SiteParameters":
        kwargs: dict[str, Any] = {}
        for f in fields(self):
            t = getattr(self, f.name)
            if isinstance(t, torch.Tensor):
                kwargs[f.name] = t.to(dtype=dtype, device=device)
            else:
                kwargs[f.name] = t
        return SiteParameters(**kwargs)
