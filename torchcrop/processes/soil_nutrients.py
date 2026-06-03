"""Soil mineral N/P/K balance: mineralisation, fertiliser, and
uptake bookkeeping.

State variables advanced here:

* ``nmin`` / ``pmin`` / ``kmin``: mineralisable organic N/P/K pool.
  Daily rate ``RNMINS`` is **negative or zero** (sign convention), so
  adding it to the state depletes the pool.
* ``nmint`` / ``pmint`` / ``kmint``: directly available inorganic
  pool. Daily rate ``RNMINT = FERTNS − NUPTR − RNMINS`` (subtracting
  the negative ``RNMINS`` adds the mineralisation flux **into**
  ``NMINT``).

The module is a stateless `nn.Module` that returns one rate dict per
day; the engine integrates the pools via the standard explicit-Euler
update.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.parameters.soil_params import SoilParameters
from torchcrop.states.model_state import ModelState


def _lookup_or_scalar(
    table: torch.Tensor | None,
    scalar: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Interpolate ``table`` at ``x``; fall back to broadcasting ``scalar``.

    Args:
        table: Optional ``[N, 2]`` breakpoint table. If ``None``, the
            scalar fallback is broadcast to the shape of ``x``.
        scalar: 0-d / 1-d tensor used when ``table is None``.
        x: Query points of shape ``[B]``.

    Returns:
        A tensor of shape ``[B]`` (broadcasting ``scalar`` if needed).
    """
    if table is None:
        return scalar + torch.zeros_like(x)
    return interpolate(table, x)


class SoilNutrients(nn.Module):
    """Daily soil mineral-pool balance for the NPK chain.

    Computes — entirely from current-day soil/crop inputs — the rate
    variables that advance the six state pools (``nmin``, ``pmin``,
    ``kmin``, ``nmint``, ``pmint``, ``kmint``) in the next
    explicit-Euler step.

    All operations are vectorised over the batch dimension ``[B]`` so
    the module composes with the rest of the differentiable
    simulation pipeline without breaking autograd or batching.
    """

    def forward(
        self,
        state: ModelState,
        nuptr: torch.Tensor,
        puptr: torch.Tensor,
        kuptr: torch.Tensor,
        nlimit: torch.Tensor,
        emerg: torch.Tensor,
        doy: torch.Tensor,
        crop_params: CropParameters,
        soil_params: SoilParameters,
        external: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute mineralisation + fertiliser + pool-balance rates.

        Args:
            state: Current `ModelState`. Reads the mineralisable
                organic pools ``nmin``/``pmin``/``kmin`` to cap the
                mineralisation flux.
            nuptr: Daily **soil** N uptake by the crop [g N m⁻² d⁻¹],
                shape ``[B]``. **Excludes** biological N fixation
                (which does not draw from the soil pool). Comes from
                `NutrientDemand`.
            puptr: Daily soil P uptake, shape ``[B]``.
            kuptr: Daily soil K uptake, shape ``[B]``.
            nlimit: Nutrient-limit gate in ``{0, 1}``, shape ``[B]``;
                ``1`` within the uptake window (``DVS < DVSNLT`` and
                ``TRANRF ≥ 0.01``), else ``0``. Reuses the gate used
                by `NutrientDemand` so mineralisation pauses when
                uptake pauses.
            emerg: Emergence mask in ``{0, 1}``, shape ``[B]``; ``0``
                before crop emergence (mineralisation is suppressed).
            doy: Day-of-year tensor, shape ``[B]``. Used to interpolate
                the fertiliser application tables
                ``ferntab``/``ferptab``/``ferktab`` and the recovery
                tables ``nrftab``/``prftab``/``krftab``.
            crop_params: Fertiliser tables and scale factors, plus the
                scalar default recovery fractions
                ``nrf``/``prf``/``krf``.
            external: Optional externally supplied fertiliser application
                of shape ``[B, 3]`` with the last axis ordered
                ``(N, P, K)`` [g X m⁻² d⁻¹]. When provided it overrides
                the ``ferntab``/``ferptab``/``ferktab`` table look-ups
                (the *raw* applied amount); the scale factors
                ``scale_factor_fer*`` and the recovery fractions
                ``nrf``/``prf``/``krf`` are still applied downstream, so
                the soil-chemistry logic is unchanged. ``None`` leaves
                the internal table-driven application in control.
            soil_params: Mineralisation kinetics
                ``rtnmins``/``rtpmins``/``rtkmins`` and the initial
                organic pools ``nmini``/``pmini``/``kmini``. The
                latter are used by the mineralisation formula
                ``RTNMINS · NMINI`` (not the current pool), capped by
                the current ``NMIN`` to prevent over-depletion.

        Returns:
            Dict of ``[B]`` tensors keyed by ``"<field>_rate"`` for the
            six soil pools, plus diagnostics:

            * ``nmin_rate``, ``pmin_rate``, ``kmin_rate``
                [g X m⁻² d⁻¹] — Rate of change of the **organic**
                pools (= ``RNMINS``, negative when mineralising).
            * ``nmint_rate``, ``pmint_rate``, ``kmint_rate``
                [g X m⁻² d⁻¹] — Rate of change of the **inorganic**
                pools (= ``RNMINT`` etc. = fertiliser + mineralisation
                − crop uptake).
            * ``fertns``, ``fertps``, ``fertks`` [g X m⁻² d⁻¹] —
                Effective fertiliser supply after recovery.
            * ``mineralisation_n``, ``mineralisation_p``,
                ``mineralisation_k`` [g X m⁻² d⁻¹] — Positive
                mineralisation flux into ``NMINT`` (= ``−RNMINS``).
        """
        cp = crop_params
        sp = soil_params

        # Look up fertiliser applications and recovery fractions.
        # Missing tables ⇒ no application / use scalar recovery
        # fraction default. An externally supplied ``[B, 3]`` driver
        # short-circuits the table look-up at the *raw application*
        # level (last axis ordered N, P, K); the scale factors and
        # recovery fractions below are applied to it unchanged.
        if external is not None:
            fertn_raw = external[..., 0]
            fertp_raw = external[..., 1]
            fertk_raw = external[..., 2]
        else:
            fertn_raw = _lookup_or_scalar(cp.ferntab, torch.zeros_like(doy), doy)
            fertp_raw = _lookup_or_scalar(cp.ferptab, torch.zeros_like(doy), doy)
            fertk_raw = _lookup_or_scalar(cp.ferktab, torch.zeros_like(doy), doy)
        nrf = _lookup_or_scalar(cp.nrftab, cp.nrf, doy)
        prf = _lookup_or_scalar(cp.prftab, cp.prf, doy)
        krf = _lookup_or_scalar(cp.krftab, cp.krf, doy)

        fertn = cp.scale_factor_fern * fertn_raw
        fertp = cp.scale_factor_ferp * fertp_raw
        fertk = cp.scale_factor_ferk * fertk_raw

        fertns = fertn * nrf
        fertps = fertp * prf
        fertks = fertk * krf

        # Mineralisation flux from the depleting organic pool.
        # Sign convention: ``RNMINS`` is *negative* (depletes
        # ``NMIN``); the flux into ``NMINT`` is therefore ``−RNMINS``
        # (positive).
        rtnmins_flux = sp.rtnmins * sp.nmini * nlimit
        rtpmins_flux = sp.rtpmins * sp.pmini * nlimit
        rtkmins_flux = sp.rtkmins * sp.kmini * nlimit

        # Cap by what is left in the pool (cannot mineralise more
        # than is currently present). Multiplied by ``emerg`` so
        # mineralisation only runs after emergence.
        mineralisation_n = torch.clamp(
            torch.minimum(rtnmins_flux, state.nmin), min=0.0
        ) * emerg
        mineralisation_p = torch.clamp(
            torch.minimum(rtpmins_flux, state.pmin), min=0.0
        ) * emerg
        mineralisation_k = torch.clamp(
            torch.minimum(rtkmins_flux, state.kmin), min=0.0
        ) * emerg

        rnmins = -mineralisation_n
        rpmins = -mineralisation_p
        rkmins = -mineralisation_k

        # Inorganic pool balance:
        #   RNMINT = FERTNS − NUPTR − RNMINS
        # (subtracting the negative RNMINS adds the mineralisation
        # flux into NMINT).
        rnmint = fertns - nuptr - rnmins
        rpmint = fertps - puptr - rpmins
        rkmint = fertks - kuptr - rkmins

        return {
            "nmin_rate": rnmins,
            "pmin_rate": rpmins,
            "kmin_rate": rkmins,
            "nmint_rate": rnmint,
            "pmint_rate": rpmint,
            "kmint_rate": rkmint,
            "fertns": fertns,
            "fertps": fertps,
            "fertks": fertks,
            "mineralisation_n": mineralisation_n,
            "mineralisation_p": mineralisation_p,
            "mineralisation_k": mineralisation_k,
        }
