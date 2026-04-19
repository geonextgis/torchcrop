"""Soil-specific parameters for the Lintul5 water balance.

Ports the **constant** soil-level inputs of the SIMPLACE
``WaterBalance.java`` component to a PyTorch-friendly dataclass.

Shapes:
    * Scalar per batch: ``[B]`` or ``[]`` (broadcastable).
    * Multi-layer (optional, for future extensions): ``[B, N_layers]``.

The naming convention follows Lintul5: ``cSM*`` (volumetric soil moisture
contents) drop the ``cSM`` prefix and become ``wc*`` (water content); other
constants drop the ``c`` prefix.

References:
    * Wolf, J. (2012). *User guide for LINTUL5*. Wageningen UR.
    * SIMPLACE source:
      ``simplace/sim/components/models/lintul5/WaterBalance.java``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import torch


def _t(x: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a scalar tensor with the requested dtype.

    Args:
        x: Python scalar value.
        dtype: Target tensor dtype.

    Returns:
        A 0-dimensional `torch.Tensor` holding ``x``.
    """
    return torch.tensor(x, dtype=dtype)


@dataclass
class SoilParameters:
    """Soil hydraulic and water-balance parameters.

    Note:
        Only a single-layer bucket model is implemented in the initial
        scaffold; the containers accept either ``[B]`` (single-layer) or
        ``[B, N_layers]`` tensors for future multi-layer extensions.
    """

    # ------------------------------------------------------------------ #
    # 1. Volumetric soil-moisture contents [m³ m⁻³]
    # ------------------------------------------------------------------ #

    wcad: torch.Tensor = field(default_factory=lambda: _t(0.150))
    """``cSMDRY``. Volumetric soil-moisture content at air-dry
    [m³ m⁻³] (≈ pF 6.0). Lower bound of plant-available water."""

    wcwp: torch.Tensor = field(default_factory=lambda: _t(0.200))
    """``cSMW``. Volumetric soil-moisture content at the permanent
    wilting point [m³ m⁻³] (pF ≈ 4.2)."""

    wcfc: torch.Tensor = field(default_factory=lambda: _t(0.350))
    """``cSMFC``. Volumetric soil-moisture content at field capacity
    [m³ m⁻³] (pF ≈ 2.3) — drainage threshold."""

    wcst: torch.Tensor = field(default_factory=lambda: _t(0.450))
    """``cSM0``. Volumetric soil-moisture content at saturation
    [m³ m⁻³] — upper limit of pore-space water content."""

    wci: torch.Tensor = field(default_factory=lambda: _t(0.300))
    """``cSMI``. Initial volumetric soil-moisture content in the rooted
    zone [m³ m⁻³] at planting / emergence."""

    wci_lower: torch.Tensor = field(default_factory=lambda: _t(0.300))
    """``cSMLOWI``. Initial volumetric soil-moisture content in the
    **lower** zone (below the rooted zone) [m³ m⁻³]."""

    crairc: torch.Tensor = field(default_factory=lambda: _t(0.07))
    """``cCRAIRC``. Critical soil-air content [m³ m⁻³] for aeration —
    below this air-filled porosity the crop suffers waterlogging stress."""

    # ------------------------------------------------------------------ #
    # 2. Drainage, runoff, and percolation
    # ------------------------------------------------------------------ #

    ksub: torch.Tensor = field(default_factory=lambda: _t(10.0))
    """``cKSUB``. Maximum percolation rate [mm d⁻¹] from the lower zone
    to deeper soil layers (sub-surface drainage)."""

    runfr: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cRUNFR``. Average fraction [0–1] of incoming precipitation lost
    to surface runoff."""

    drate: torch.Tensor = field(default_factory=lambda: _t(50.0))
    """Maximum drainage rate [mm d⁻¹] out of the rooted zone (used by the
    simplified bucket implementation; replaces SIMPLACE's WTRLOS)."""

    cfev: torch.Tensor = field(default_factory=lambda: _t(2.0))
    """``cCFEV``. Correction factor [-] for the time course of soil
    evaporation under drying conditions (Stroosnijder; range 1 – 4)."""

    # ------------------------------------------------------------------ #
    # 3. Rooting depth (soil-side)
    # ------------------------------------------------------------------ #

    rdmso: torch.Tensor = field(default_factory=lambda: _t(1.50))
    """``cRDMSO``. Maximum rooting depth [m] as constrained by the
    soil profile (e.g. by an impermeable layer). The actual maximum
    rooting depth is ``min(cRDMCR, cRDMSO)``."""

    # ------------------------------------------------------------------ #
    # 4. Crop water-uptake response
    # ------------------------------------------------------------------ #

    depnr: torch.Tensor = field(default_factory=lambda: _t(4.5))
    """``cDEPNR``. Crop group number [-] for soil-water depletion
    (Doorenbos & Kassam). Used to compute the critical soil-moisture
    content above which transpiration is unrestricted."""

    iairdu: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cIAIRDU``. Boolean flag (0/1) indicating whether the crop has
    air ducts in its roots (=1, e.g. rice → tolerates waterlogging) or
    not (=0). Stored as float for batch broadcasting."""

    # ------------------------------------------------------------------ #
    # 5. Irrigation
    # ------------------------------------------------------------------ #

    irri: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cIRRI``. Irrigation mode: ``0`` → no irrigation,
    ``1`` → automatic (refill to field capacity),
    ``2`` → use day-resolved `irrtab`."""

    irrtab: torch.Tensor | None = None
    """``cIRRTAB``. Optional table of effective irrigation applications
    [mm d⁻¹] as a function of day number; shape ``[N, 2]``."""

    scale_factor_irr: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cScaleFactorIRR``. Scale factor on `irrtab` y-values for
    sensitivity analysis / calibration."""

    # ------------------------------------------------------------------ #
    # 6. Soil mineral nutrient supply (background mineralisation)
    # ------------------------------------------------------------------ #

    nmins: torch.Tensor = field(default_factory=lambda: _t(0.50))
    """Background daily mineralisation supply of plant-available
    soil N [g N · m⁻² · d⁻¹]. (Simplified surrogate of the SIMPLACE
    ``cRTNMINS`` × ``sNMIN`` flux.)"""

    pmins: torch.Tensor = field(default_factory=lambda: _t(0.05))
    """Background daily supply of plant-available soil P
    [g P · m⁻² · d⁻¹]."""

    kmins: torch.Tensor = field(default_factory=lambda: _t(0.30))
    """Background daily supply of plant-available soil K
    [g K · m⁻² · d⁻¹]."""

    rtnmins: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cRTNMINS``. Fraction [d⁻¹] of the soil organic-N pool that
    becomes mineralised and plant-available per day."""

    rtpmins: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cRTPMINS``. Fraction [d⁻¹] of soil P becoming available per
    day."""

    rtkmins: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cRTKMINS``. Fraction [d⁻¹] of soil K becoming available per
    day."""

    nmini: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cNMINI``. Initial amount [g N m⁻²] (at crop emergence) of
    potentially available soil organic N."""

    pmini: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cPMINI``. Initial amount [g P m⁻²] of potentially available
    soil P."""

    kmini: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cKMINI``. Initial amount [g K m⁻²] of potentially available
    soil K."""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def to(
        self,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "SoilParameters":
        """Cast and/or move all tensor fields to a new dtype/device.

        Args:
            dtype: Target tensor dtype, or ``None`` to leave unchanged.
            device: Target torch device, or ``None`` to leave unchanged.

        Returns:
            A new `SoilParameters` with every tensor field moved/cast.
        """
        kwargs: dict[str, Any] = {}
        for f in fields(self):
            t = getattr(self, f.name)
            if isinstance(t, torch.Tensor):
                kwargs[f.name] = t.to(dtype=dtype, device=device)
            else:
                kwargs[f.name] = t
        return SoilParameters(**kwargs)


def default_loam_params(dtype: torch.dtype = torch.float32) -> SoilParameters:
    """Return the SIMPLACE Lintul5 default loam-like soil parameter set.

    Args:
        dtype: Target tensor dtype for all scalar fields.

    Returns:
        A fresh `SoilParameters` with the Lintul5 loam defaults
        cast to ``dtype``.
    """
    return SoilParameters().to(dtype=dtype)
