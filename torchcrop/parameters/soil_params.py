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
    # 4. Irrigation
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
    # 5. Soil mineral nutrient supply (mineralisation kinetics)
    # ------------------------------------------------------------------ #
    # The full Lintul5 NPK chain integrates two state pools per nutrient
    # (organic ``NMIN`` and inorganic ``NMINT``) updated daily by
    # ``SoilNutrients``. The constants below specify the *kinetics* of
    # that integration — the *current* pool levels live on `ModelState`.

    rtnmins: torch.Tensor = field(default_factory=lambda: _t(0.01))
    """``cRTNMINS``. Fraction [d⁻¹] of the soil organic-N pool that
    becomes mineralised and plant-available per day. Multiplied by the
    *initial* organic pool ``nmini`` (Lintul5 convention) and capped by
    the current ``NMIN``. Default 0.01 gives a steady-state daily
    mineralisation of ``0.01 × nmini`` g N m⁻² d⁻¹."""

    rtpmins: torch.Tensor = field(default_factory=lambda: _t(0.01))
    """``cRTPMINS``. Fraction [d⁻¹] of soil P becoming available per
    day. Default 0.01."""

    rtkmins: torch.Tensor = field(default_factory=lambda: _t(0.01))
    """``cRTKMINS``. Fraction [d⁻¹] of soil K becoming available per
    day. Default 0.01."""

    nmini: torch.Tensor = field(default_factory=lambda: _t(50.0))
    """``cNMINI``. Initial amount [g N m⁻²] of mineralisable soil
    organic N at crop emergence; this seeds `ModelState.nmin`. Default
    50 g/m² (~500 kg N/ha) combined with ``rtnmins=0.01`` produces a
    steady-state mineralisation flux of 0.5 g N m⁻² d⁻¹, matching the
    previous scalar surrogate."""

    pmini: torch.Tensor = field(default_factory=lambda: _t(5.0))
    """``cPMINI``. Initial mineralisable organic P pool [g P m⁻²].
    Default 5 g/m² × ``rtpmins=0.01`` ⇒ 0.05 g P m⁻² d⁻¹ steady-state
    supply."""

    kmini: torch.Tensor = field(default_factory=lambda: _t(30.0))
    """``cKMINI``. Initial mineralisable organic K pool [g K m⁻²].
    Default 30 g/m² × ``rtkmins=0.01`` ⇒ 0.30 g K m⁻² d⁻¹ steady-state
    supply."""

    nminti: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """Initial directly available inorganic N pool ``NMINT`` [g N m⁻²]
    at sowing/emergence. SIMPLACE default is 0; raise this to represent
    residual mineral N at planting."""

    pminti: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """Initial directly available inorganic P pool."""

    kminti: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """Initial directly available inorganic K pool."""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Validate discrete/categorical soil fields.

        Checks that the irrigation mode ``irri`` holds only the
        supported discrete values ``{0, 1, 2}`` (per element when
        batched). The water-balance selection logic matches these modes
        by exact equality, so an off-grid value (e.g. ``1.5`` or ``3``)
        would silently fall through to "no irrigation"; this guard turns
        that into an explicit error.

        Raises:
            ValueError: If any ``irri`` element is not one of
                ``{0, 1, 2}``.
        """
        valid = torch.zeros_like(self.irri, dtype=torch.bool)
        for mode in (0.0, 1.0, 2.0):
            valid = valid | torch.isclose(
                self.irri, torch.full_like(self.irri, mode)
            )
        if not bool(valid.all()):
            raise ValueError(
                "soil_params.irri must be one of {0, 1, 2} (0=none, "
                f"1=auto refill, 2=irrtab); got {self.irri.tolist()}"
            )

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
