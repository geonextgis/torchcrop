"""Simulation engine — orchestrates the daily time-stepping loop.

The engine owns no trainable parameters of its own; it wires together the
process sub-modules supplied by :class:`~torchcrop.model.Lintul5Model`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable

import torch
import torch.nn as nn

from torchcrop.drivers.weather import WeatherDriver
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.parameters.site_params import SiteParameters
from torchcrop.parameters.soil_params import SoilParameters
from torchcrop.states.model_state import ModelState


@dataclass
class StepResult:
    """Outputs of a single simulation step."""

    state: ModelState
    rates: dict[str, torch.Tensor]


class SimulationEngine(nn.Module):
    """Daily time-stepping loop.

    The engine expects a callable ``compute_rates(state, weather_day, doy)``
    returning a dict of rate tensors indexed by state-field name, plus a
    callable ``update_state(state, rates, dt)``.
    """

    def __init__(
        self,
        compute_rates: Callable[..., dict[str, torch.Tensor]],
        update_state: Callable[..., ModelState],
        dt: float = 1.0,
    ) -> None:
        super().__init__()
        self._compute_rates = compute_rates
        self._update_state = update_state
        self.dt = dt

    def step(
        self,
        state: ModelState,
        weather_day: dict[str, torch.Tensor],
        doy: torch.Tensor,
        crop_params: CropParameters,
        soil_params: SoilParameters,
        site_params: SiteParameters,
    ) -> StepResult:
        rates = self._compute_rates(
            state=state,
            weather_day=weather_day,
            doy=doy,
            crop_params=crop_params,
            soil_params=soil_params,
            site_params=site_params,
        )
        new_state = self._update_state(state, rates, self.dt)
        return StepResult(state=new_state, rates=rates)

    def run(
        self,
        state: ModelState,
        weather: WeatherDriver,
        start_doy: int,
        crop_params: CropParameters,
        soil_params: SoilParameters,
        site_params: SiteParameters,
    ) -> tuple[list[ModelState], list[dict[str, torch.Tensor]]]:
        """Run the full trajectory, returning per-day state and rate lists."""
        states: list[ModelState] = [state]
        rates_all: list[dict[str, torch.Tensor]] = []

        n_days = weather.n_days
        for t in range(n_days):
            weather_day = weather.day(t)
            doy_t = torch.full_like(
                state.dvs,
                float(((start_doy - 1 + t) % 365) + 1),
            )
            result = self.step(
                state=states[-1],
                weather_day=weather_day,
                doy=doy_t,
                crop_params=crop_params,
                soil_params=soil_params,
                site_params=site_params,
            )
            states.append(result.state)
            rates_all.append(result.rates)
        return states, rates_all


def euler_update(state: ModelState, rates: dict[str, torch.Tensor], dt: float) -> ModelState:
    """Forward-Euler update of a :class:`ModelState` from a dict of rate tensors.

    Rate keys must match state field names with a ``_rate`` suffix (e.g.
    ``dvs_rate`` updates ``dvs``). Fields without a matching rate are left
    unchanged.
    """
    updates: dict[str, torch.Tensor] = {}
    for f in fields(state):
        rate_key = f"{f.name}_rate"
        current = getattr(state, f.name)
        if rate_key in rates and isinstance(current, torch.Tensor):
            new_val = current + dt * rates[rate_key]
            # Non-negative clamping where it makes physical sense
            if f.name in {
                "tsum",
                "tsump",
                "vern",
                "wlv",
                "wlvd",
                "wst",
                "wrt",
                "wso",
                "lai",
                "rootd",
                "wa",
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
                "tran_cum",
                "evap_cum",
            }:
                new_val = torch.clamp(new_val, min=0.0)
            if f.name == "dvs":
                new_val = torch.clamp(new_val, min=0.0, max=2.0)
            updates[f.name] = new_val
    return state.replace(**updates)
