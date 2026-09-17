"""What crosses a harvest boundary, and what is wiped.

Over many seasons this split decides whether the simulation represents
*one field across 30 years* or *30 independent seasons sharing a weather
file*. Both are legitimate questions, so `CarryOverPolicy` makes the
choice explicit.

The default represents a real field: soil water and soil mineral pools run
continuously across the boundary, and everything belonging to the crop is
cleared.

## Clearing the crop

Harvest returns the field to a **bare, pre-sowing condition**: the
phenology clocks (``tsum``, ``tsump``, ``vern``), the biomass and canopy
pools, the per-organ nutrients and the rooting depth all go back to their
sowing-day values. With ``tsump = 0`` the emergence gates in
`Irradiation`, `LeafDynamics` and `NutrientDemand` are shut, so the field
intercepts no light, grows nothing and takes up no nutrients during the
fallow — only the soil water and mineral balances evolve, until the next
sowing puts a fresh seed reserve in the ground.

## Conserving water through the root-zone reset

`ModelState.wa` is the water held *in the rooted zone*, whose depth is
`ModelState.rootd`. Shrinking ``rootd`` from a mature 1.2 m back to the
seedling ``rdi`` therefore has to say where the water below the new root
front goes. `reset_at_harvest` redistributes it into the lower zone:

$$
\\theta = \\frac{W_a}{1000 \\cdot D_{\\text{root}}}, \\qquad
W_a' = 1000 \\cdot D_{\\text{rdi}} \\cdot \\theta, \\qquad
W_{a,\\text{lower}}' = W_a + W_{a,\\text{lower}} - W_a'
$$

The volumetric content of the new shallow zone is unchanged and total
profile water is conserved exactly. This is the inverse of the ``WDR``
transfer `WaterBalance` applies as the root front advances. Should the
receiving lower zone end up above field capacity, the percolation cascade
drains it over the following days — its headroom terms are signed so that
stored excess is released.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from torchcrop.functions.interpolation import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.parameters.soil_params import SoilParameters
from torchcrop.states.model_state import ModelState

CarryMode = Literal["carry", "reset"]

#: Crop state cleared at every harvest, with the value each field takes.
#: ``dvs`` and ``rootd`` are handled separately (they come from crop
#: parameters), as are the water stores.
_CROP_ZERO_FIELDS: tuple[str, ...] = (
    # Phenology clocks — the crop's development and emergence timers.
    "tsum",
    "tsump",
    "vern",
    "sown",
    # Biomass, living and dead.
    "wlv",
    "wlvd",
    "wst",
    "wstd",
    "wrt",
    "wrtd",
    "wso",
    "lai",
    # Per-organ N, P and K.
    "anlv",
    "anst",
    "anrt",
    "anso",
    "aplv",
    "apst",
    "aprt",
    "apso",
    "aklv",
    "akst",
    "akrt",
    "akso",
    # Dead-tissue nutrient loss accumulators.
    "nlossl",
    "nlossr",
    "nlosss",
    "plossl",
    "plossr",
    "plosss",
    "klossl",
    "klossr",
    "klosss",
    # One-step lags and irrigation carry-over.
    "rr_prev",
    "dirro",
)

#: Lagged nutrition indices restart at "no stress".
_CROP_ONE_FIELDS: tuple[str, ...] = ("nni_prev", "npki_prev")

#: Cumulative water / nutrient / growth accumulators.
_ACCUMULATOR_FIELDS: tuple[str, ...] = (
    "tran_cum",
    "evap_cum",
    "rain_cum",
    "irrig_cum",
    "runoff_cum",
    "drain_cum",
    "nuptr_cum",
    "puptr_cum",
    "kuptr_cum",
    "nfixtr_cum",
    "parint_cum",
    "gtotal_cum",
)

#: Soil mineral pools — organic (mineralisable) and inorganic (available).
_SOIL_MINERAL_FIELDS: tuple[str, ...] = (
    "nmin",
    "pmin",
    "kmin",
    "nmint",
    "pmint",
    "kmint",
)

#: Which `SoilParameters` field re-seeds each mineral pool under
#: ``soil_minerals="reset"``.
_MINERAL_INITIALS: dict[str, str] = {
    "nmin": "nmini",
    "pmin": "pmini",
    "kmin": "kmini",
    "nmint": "nminti",
    "pmint": "pminti",
    "kmint": "kminti",
}


@dataclass
class CarryOverPolicy:
    """What survives a harvest boundary in a long-term run.

    Crop state — phenology clocks, biomass, canopy, per-organ nutrients
    and rooting depth — is always cleared, and soil water is always
    continuous. The remaining choices are configured here.

    Attributes:
        soil_minerals: ``"carry"`` (default) carries the six soil mineral
            pools across the boundary, so nutrient supply is continuous
            from year to year. ``"reset"`` re-seeds them from
            ``nmini``/``nminti`` (and the P, K analogues) at every
            harvest, which turns the run into a sequence of independent
            seasons sharing one weather and water stream — the right
            choice for reproducing per-season SIMPLACE runs, or for
            isolating weather variability from soil-fertility drift.
        accumulators: ``"reset"`` (default) zeroes the cumulative water,
            nutrient and growth accumulators at each harvest, so every
            `torchcrop.longterm.SeasonRecord` carries a self-contained
            per-season budget. ``"carry"`` keeps them running for a
            whole-run total.
        residue_fraction: Fraction in ``[0, 1]`` of the harvest residue
            N/P/K returned to the **mineralisable organic** soil pools
            (``nmin``/``pmin``/``kmin``). Residue is all dead tissue plus
            the living leaves, stems and roots left in the field —
            storage organs are removed as yield. Defaults to ``0.0``,
            i.e. residue is removed from the field.

            Enabling it matters over long horizons. Lintul5 mineralises at
            ``rtnmins · nmini`` capped by the *current* pool and has no
            replenishment pathway, so under ``soil_minerals="carry"`` a
            continuous multi-decade run gradually draws the organic pool
            down and the crop comes to depend on fertiliser alone. Residue
            return closes that loop, at the cost of introducing soil
            chemistry beyond what the SIMPLACE reference covers.
        residue_removes_stems: If ``True``, stems (living and dead) are
            treated as removed from the field — baled straw — and only
            leaf and root residue is returned. Defaults to ``False``
            (stems stay). Only has an effect when ``residue_fraction``
            is non-zero.
    """

    soil_minerals: CarryMode = "carry"
    accumulators: CarryMode = "reset"
    residue_fraction: float = 0.0
    residue_removes_stems: bool = False

    def __post_init__(self) -> None:
        for name in ("soil_minerals", "accumulators"):
            value = getattr(self, name)
            if value not in ("carry", "reset"):
                raise ValueError(
                    f"{name} must be 'carry' or 'reset'; got {value!r}"
                )
        if not 0.0 <= self.residue_fraction <= 1.0:
            raise ValueError(
                "residue_fraction must lie in [0, 1]; got "
                f"{self.residue_fraction}"
            )


def _residue_nutrients(
    state: ModelState, policy: CarryOverPolicy
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """N, P and K returned to the soil as crop residue at harvest.

    Residue is every nutrient pool left in the field: the standing
    living leaves, stems and roots, plus whatever senesced tissue has
    accumulated. Storage-organ nutrients leave with the yield.

    The senesced pools ``nlossl``/``nlossr``/``nlosss`` (and the P, K
    analogues) already hold the nutrients carried out of the living
    organs by senescence, so summing them with the living pools counts
    each atom exactly once.

    Args:
        state: Pre-reset `ModelState` at the moment of harvest.
        policy: Carry-over policy; supplies ``residue_fraction`` and
            ``residue_removes_stems``.

    Returns:
        Tuple of ``[B]`` tensors ``(n, p, k)`` [g X m⁻²] to add to the
        mineralisable organic pools.
    """
    f = policy.residue_fraction
    keep_stems = 0.0 if policy.residue_removes_stems else 1.0
    n = f * (
        state.anlv
        + state.anrt
        + state.nlossl
        + state.nlossr
        + keep_stems * (state.anst + state.nlosss)
    )
    p = f * (
        state.aplv
        + state.aprt
        + state.plossl
        + state.plossr
        + keep_stems * (state.apst + state.plosss)
    )
    k = f * (
        state.aklv
        + state.akrt
        + state.klossl
        + state.klossr
        + keep_stems * (state.akst + state.klosss)
    )
    return n, p, k


def reset_at_harvest(
    state: ModelState,
    harvest: torch.Tensor,
    crop_params: CropParameters,
    soil_params: SoilParameters,
    policy: CarryOverPolicy | None = None,
) -> ModelState:
    """Clear the crop from the field for the batch elements harvesting today.

    Every field is merged with ``torch.where(harvest, reset, current)``,
    so batch elements that are not harvesting on this day pass through
    completely untouched. Nothing is modified in place and no branch runs
    on a tensor value, so the operation is batch-safe and keeps the
    autograd graph intact.

    The resulting state is a bare, pre-sowing field: with ``tsump = 0``
    the emergence gates in `Irradiation`, `LeafDynamics` and
    `NutrientDemand` are shut, so it neither intercepts light nor grows
    nor takes up nutrients until the next sowing.

    Args:
        state: `ModelState` **after** the day's Euler update.
        harvest: ``[B]`` mask in ``{0, 1}`` (bool or float) marking the
            batch elements whose crop is coming out of the field today.
        crop_params: Crop parameters; supplies ``dvsi`` (the development
            stage a new crop starts at) and ``rdi`` (initial rooting
            depth).
        soil_params: Soil parameters; supplies the initial mineral pools
            used by ``soil_minerals="reset"``.
        policy: Carry-over policy. Defaults to `CarryOverPolicy()`.

    Returns:
        A new `ModelState` with the crop cleared on the harvesting
        elements.
    """
    policy = policy or CarryOverPolicy()
    ref = state.dvs
    mask = harvest.to(dtype=ref.dtype, device=ref.device)
    bool_mask = mask > 0.5

    def _merge(current: torch.Tensor, reset: torch.Tensor) -> torch.Tensor:
        """Select ``reset`` where harvesting, ``current`` elsewhere."""
        return torch.where(bool_mask, torch.broadcast_to(reset, current.shape), current)

    def _param(value: torch.Tensor) -> torch.Tensor:
        """Read a parameter as a detached tensor in the state's dtype."""
        return value.detach().to(dtype=ref.dtype, device=ref.device)

    updates: dict[str, torch.Tensor] = {}
    zero = torch.zeros_like(ref)
    one = torch.ones_like(ref)

    for name in _CROP_ZERO_FIELDS:
        updates[name] = _merge(getattr(state, name), zero)
    for name in _CROP_ONE_FIELDS:
        updates[name] = _merge(getattr(state, name), one)

    # A new crop starts at the parameterised initial development stage,
    # matching `Lintul5Model.initialize`.
    updates["dvs"] = _merge(state.dvs, _param(crop_params.dvsi))

    # ---------------------------------------------------------------- #
    # Root zone: shrink back to the seedling depth, moving the water
    # that is no longer rooted into the lower zone so the profile total
    # is conserved (see the module docstring).
    # ---------------------------------------------------------------- #
    rdi = _param(crop_params.rdi)
    rootd_old = torch.clamp(state.rootd, min=1e-4)
    smact = state.wa / (1000.0 * rootd_old)
    wa_new = 1000.0 * rdi * smact
    wa_lower_new = state.wa + state.wa_lower - wa_new
    updates["rootd"] = _merge(state.rootd, torch.broadcast_to(rdi, ref.shape))
    updates["wa"] = _merge(state.wa, wa_new)
    updates["wa_lower"] = _merge(state.wa_lower, wa_lower_new)

    # ---------------------------------------------------------------- #
    # Soil mineral pools.
    # ---------------------------------------------------------------- #
    res_n, res_p, res_k = _residue_nutrients(state, policy)
    residue = {"nmin": res_n, "pmin": res_p, "kmin": res_k}
    for name in _SOIL_MINERAL_FIELDS:
        current = getattr(state, name)
        if policy.soil_minerals == "reset":
            target = torch.broadcast_to(
                _param(getattr(soil_params, _MINERAL_INITIALS[name])), current.shape
            )
        else:
            target = current
        # Residue returns to the mineralisable organic pools only. Under
        # "reset" it is added on top of the re-seeded pool, so enabling
        # residue return is meaningful in either mode.
        if policy.residue_fraction > 0.0 and name in residue:
            target = target + residue[name]
        if target is not current:
            updates[name] = _merge(current, target)

    # ---------------------------------------------------------------- #
    # Cumulative accumulators.
    # ---------------------------------------------------------------- #
    if policy.accumulators == "reset":
        for name in _ACCUMULATOR_FIELDS:
            updates[name] = _merge(getattr(state, name), zero)

    return state.replace(**updates)


def seed_reserve_biomass(crop_params: CropParameters) -> torch.Tensor:
    """Total dry weight a fresh sowing puts into the field [g m⁻²].

    Mirrors the seed-reserve bootstrap in `Lintul5Model._compute_rates_dispatch`
    (``TDWI`` split over the organs at ``DVSI``). Provided so per-season
    biomass accounting can subtract the sown seed from harvested biomass
    when a strict net-production figure is wanted.

    Args:
        crop_params: Crop parameters; uses ``tdwi``.

    Returns:
        Scalar tensor holding ``TDWI``.
    """
    return crop_params.tdwi.detach()


def initial_lai(crop_params: CropParameters) -> torch.Tensor:
    """Leaf area index a fresh sowing starts with [m² m⁻²].

    Reproduces ``LAII = WLVGI · scale_factor_sla · SLATB(DVSI)`` from the
    sowing-day bootstrap, using the same differentiable interpolation the
    model uses.

    Args:
        crop_params: Crop parameters; uses ``dvsi``, ``tdwi``, ``frtb``,
            ``fltb``, ``slatb`` and ``scale_factor_sla``.

    Returns:
        Scalar tensor holding the initial LAI.
    """
    dvsi = crop_params.dvsi
    x = dvsi.reshape(1) if dvsi.dim() == 0 else dvsi
    frtb_d = interpolate(crop_params.frtb, x).reshape(())
    fltb_d = interpolate(crop_params.fltb, x).reshape(())
    sla_d = interpolate(crop_params.slatb, x).reshape(())
    wlvgi = fltb_d * (crop_params.tdwi - frtb_d * crop_params.tdwi)
    return (wlvgi * crop_params.scale_factor_sla * sla_d).detach()
