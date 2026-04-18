"""Crop-specific parameters for Lintul5.

All scalar parameters are stored as :class:`torch.Tensor` so they can be made
learnable via :class:`torch.nn.Parameter`. Table parameters (``fltb``,
``fstb``, ``frtb``, ``fotb``, ``sla_tb``, ``rdrtb``, ``phottb``, ``dtsmtb``)
are tensors of shape ``[N, 2]`` or ``[B, N, 2]`` interpreted by
:func:`torchcrop.functions.interpolate`.

Parameter names follow the original Lintul5 (Wolf, 2012) lowercased.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import torch


def _t(x: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a scalar tensor with the requested dtype."""
    return torch.tensor(x, dtype=dtype)


def _table(rows: list[tuple[float, float]], dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(rows, dtype=dtype)


@dataclass
class CropParameters:
    """Species-specific Lintul5 parameters.

    Scalar fields have shape ``[]`` or ``[B]`` (broadcastable against the
    batch dimension). Table fields have shape ``[N, 2]`` or ``[B, N, 2]``.
    """

    # Phenology
    tbasem: torch.Tensor = field(default_factory=lambda: _t(0.0))
    teffmx: torch.Tensor = field(default_factory=lambda: _t(30.0))
    tsumem: torch.Tensor = field(default_factory=lambda: _t(110.0))
    tsum1: torch.Tensor = field(default_factory=lambda: _t(900.0))
    tsum2: torch.Tensor = field(default_factory=lambda: _t(700.0))
    dvsi: torch.Tensor = field(default_factory=lambda: _t(0.0))
    dtsmtb: torch.Tensor = field(
        default_factory=lambda: _table([(-5.0, 0.0), (0.0, 0.0), (30.0, 30.0), (45.0, 30.0)])
    )
    phottb: torch.Tensor = field(
        default_factory=lambda: _table([(0.0, 0.0), (8.0, 1.0), (12.0, 1.0), (18.0, 1.0)])
    )
    vernrt: torch.Tensor = field(
        default_factory=lambda: _table([(0.0, 1.0), (1.0, 1.0)])
    )

    # Photosynthesis / radiation
    k: torch.Tensor = field(default_factory=lambda: _t(0.60))
    rue: torch.Tensor = field(default_factory=lambda: _t(3.0))
    co2: torch.Tensor = field(default_factory=lambda: _t(360.0))

    # Leaf dynamics
    sla: torch.Tensor = field(default_factory=lambda: _t(0.022))
    laii: torch.Tensor = field(default_factory=lambda: _t(0.012))
    rgrl: torch.Tensor = field(default_factory=lambda: _t(0.009))
    rdrshm: torch.Tensor = field(default_factory=lambda: _t(0.03))
    laicr: torch.Tensor = field(default_factory=lambda: _t(4.0))
    rdrtb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(0.0, 0.0), (1.0, 0.0), (1.5, 0.02), (2.0, 0.05)]
        )
    )

    # Partitioning tables (fraction of growth to leaves/stems/roots/storage vs DVS)
    fltb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(0.0, 0.60), (0.5, 0.40), (1.0, 0.00), (2.0, 0.00)]
        )
    )
    fstb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(0.0, 0.30), (0.5, 0.50), (1.0, 0.25), (2.0, 0.00)]
        )
    )
    frtb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(0.0, 0.50), (0.5, 0.30), (1.0, 0.00), (2.0, 0.00)]
        )
    )
    fotb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(0.0, 0.00), (0.5, 0.00), (1.0, 0.75), (2.0, 1.00)]
        )
    )

    # Root dynamics
    rrdmax: torch.Tensor = field(default_factory=lambda: _t(0.012))
    rootdi: torch.Tensor = field(default_factory=lambda: _t(0.10))
    rootdm: torch.Tensor = field(default_factory=lambda: _t(1.20))

    # Nutrient (NPK) maxima and minima (fractions in dry matter) per organ
    nmaxlv: torch.Tensor = field(default_factory=lambda: _t(0.050))
    nmaxst: torch.Tensor = field(default_factory=lambda: _t(0.020))
    nmaxrt: torch.Tensor = field(default_factory=lambda: _t(0.015))
    nmaxso: torch.Tensor = field(default_factory=lambda: _t(0.025))
    pmaxlv: torch.Tensor = field(default_factory=lambda: _t(0.008))
    pmaxst: torch.Tensor = field(default_factory=lambda: _t(0.004))
    pmaxrt: torch.Tensor = field(default_factory=lambda: _t(0.003))
    pmaxso: torch.Tensor = field(default_factory=lambda: _t(0.005))
    kmaxlv: torch.Tensor = field(default_factory=lambda: _t(0.040))
    kmaxst: torch.Tensor = field(default_factory=lambda: _t(0.025))
    kmaxrt: torch.Tensor = field(default_factory=lambda: _t(0.015))
    kmaxso: torch.Tensor = field(default_factory=lambda: _t(0.005))

    nresid: torch.Tensor = field(default_factory=lambda: _t(0.004))
    presid: torch.Tensor = field(default_factory=lambda: _t(0.001))
    kresid: torch.Tensor = field(default_factory=lambda: _t(0.005))

    # Temperature response for RUE (Q10-like)
    tmpf_tb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(-10.0, 0.0), (0.0, 0.0), (10.0, 0.6), (20.0, 1.0), (30.0, 1.0), (40.0, 0.0)]
        )
    )

    def to(self, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> "CropParameters":
        """Return a new ``CropParameters`` with tensors moved/cast."""
        kwargs: dict[str, Any] = {}
        for f in fields(self):
            t = getattr(self, f.name)
            if isinstance(t, torch.Tensor):
                kwargs[f.name] = t.to(dtype=dtype, device=device)
            else:
                kwargs[f.name] = t
        return CropParameters(**kwargs)


def default_wheat_params(dtype: torch.dtype = torch.float32) -> CropParameters:
    """Return a plausible wheat-like default parameterisation."""
    p = CropParameters()
    return p.to(dtype=dtype)
