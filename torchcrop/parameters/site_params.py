"""Site-level parameters: latitude, altitude, atmosphere, sowing calendar.

Ports the **constant** site-level inputs of the SIMPLACE Lintul5 component
family — primarily ``PotentialEvapoTranspiration.java`` (atmosphere) and
the sowing/emergence-calendar constants of ``Phenology.java`` — to a
PyTorch-friendly dataclass.

Note:
    Latitude is normally consumed by the
    `AstronomicParametersTransformer` upstream of Lintul5; we keep it
    here so torchcrop can compute astronomy and irradiation internally.
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


def _table(
    rows: list[tuple[float, float]],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an ``[N, 2]`` interpolation table tensor.

    Args:
        rows: List of ``(x, y)`` breakpoints.
        dtype: Target tensor dtype.

    Returns:
        A 2-D tensor of shape ``[N, 2]``.
    """
    return torch.tensor(rows, dtype=dtype)


def _check_discrete(
    name: str, value: torch.Tensor, allowed: tuple[float, ...]
) -> None:
    """Assert that every element of ``value`` is one of ``allowed``.

    Args:
        name: Field name, used in the error message.
        value: Tensor of switch values (scalar or batched ``[B]``).
        allowed: The permitted discrete values.

    Raises:
        ValueError: If any element of ``value`` is not in ``allowed``.
    """
    valid = torch.zeros_like(value, dtype=torch.bool)
    for a in allowed:
        valid = valid | torch.isclose(value, torch.full_like(value, a))
    if not bool(valid.all()):
        allowed_str = ", ".join(
            str(int(a)) if float(a).is_integer() else str(a) for a in allowed
        )
        raise ValueError(
            f"{name} must be one of {{{allowed_str}}}; got {value.tolist()}"
        )


@dataclass
class SiteParameters:
    """Geographical, atmospheric, and calendar parameters for a site.

    Scalar fields have shape ``[]`` or ``[B]`` (broadcastable). Defaults
    correspond to a temperate Western-European site (≈ Wageningen).
    """

    # ------------------------------------------------------------------ #
    # 1. Geography
    # ------------------------------------------------------------------ #

    latitude: torch.Tensor = field(default_factory=lambda: _t(52.0))
    """Decimal latitude [°] (positive northward). Drives the
    astronomical day-length and solar-declination calculations."""

    altitude: torch.Tensor = field(default_factory=lambda: _t(10.0))
    """``cALTI``. Site / weather-station altitude [m a.s.l.]. Used in the
    psychrometric-constant calculation of the Penman ET₀ routine."""

    # ------------------------------------------------------------------ #
    # 2. Atmosphere
    # ------------------------------------------------------------------ #

    co2: torch.Tensor = field(default_factory=lambda: _t(370.0))
    """``cCO``. Atmospheric CO₂ concentration [ppm]. **Single source of
    truth** for CO₂ across the model — it feeds the Penman ET₀ correction
    (`fpenmtb`), the RUE CO₂ correction (`CropParameters.cotb`), and the
    `Co2Transpiration` reduction of potential transpiration. Modelled as a
    site/scenario property rather than a crop trait; wrap it in an
    `nn.Parameter` to calibrate or run elevated-CO₂ experiments. The
    Lintul5 ET routine uses 370 ppm as its reference."""

    cfet: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cCFET``. Empirical correction factor [-] applied to the Penman
    transpiration rate (canopy-resistance proxy)."""

    fpenmtb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(40.0, 1.05), (360.0, 1.00), (720.0, 0.95), (1000.0, 0.92), (2000.0, 0.92)]
        )
    )
    """``cFPENMTB`` (= ``cET0CorrectionTableCo2`` × ``cET0CorrectionTableFactor``).
    Multiplicative correction factor [-] for the Penman ET₀ as a function
    of atmospheric CO₂ concentration [ppm] (C3 crops)."""

    # ------------------------------------------------------------------ #
    # 3. Solar-radiation reconstruction (when only sunshine duration is
    #    available — Ångström–Prescott formula).
    # ------------------------------------------------------------------ #

    angstrom_a: torch.Tensor = field(default_factory=lambda: _t(0.25))
    """Ångström–Prescott coefficient ``a`` [-]: fraction of extra-
    terrestrial radiation reaching the surface on overcast days."""

    angstrom_b: torch.Tensor = field(default_factory=lambda: _t(0.50))
    """Ångström–Prescott coefficient ``b`` [-]: additional fraction
    reaching the surface on cloudless days, multiplied by relative
    sunshine duration."""

    # ------------------------------------------------------------------ #
    # 4. Sowing / emergence calendar (from Phenology.java)
    # ------------------------------------------------------------------ #

    plant_at_sowing: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cPL``. Boolean flag (0/1) — if 1, the simulation starts at
    **planting** (uses `idpl`); if 0, it starts at **emergence**
    (uses `idem`)."""

    idpl: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cIDPL``. Day of planting [day-of-year, 0–366]. Used when
    `plant_at_sowing` = 1."""

    idem: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cIDEM``. Day of emergence [day-of-year, 0–366]. Used when
    `plant_at_sowing` = 0 to bypass the temperature-sum-driven
    emergence calculation."""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        """Validate discrete/categorical site fields.

        Checks the ``plant_at_sowing`` start-mode flag against its
        boolean domain ``{0, 1}`` (``1`` → start at planting,
        ``0`` → start at emergence). The companion calendar selectors
        ``idpl`` / ``idem`` are day-of-year integers spanning a wide
        range and are left unconstrained here.

        Raises:
            ValueError: If ``plant_at_sowing`` is not ``0`` or ``1``.
        """
        _check_discrete(
            "site_params.plant_at_sowing", self.plant_at_sowing, (0.0, 1.0)
        )

    def to(
        self,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "SiteParameters":
        """Cast and/or move all tensor fields to a new dtype/device.

        Args:
            dtype: Target tensor dtype, or ``None`` to leave unchanged.
            device: Target torch device, or ``None`` to leave unchanged.

        Returns:
            A new `SiteParameters` with every tensor field moved/cast.
        """
        kwargs: dict[str, Any] = {}
        for f in fields(self):
            t = getattr(self, f.name)
            if isinstance(t, torch.Tensor):
                kwargs[f.name] = t.to(dtype=dtype, device=device)
            else:
                kwargs[f.name] = t
        return SiteParameters(**kwargs)
