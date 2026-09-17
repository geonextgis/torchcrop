"""Multi-year simulation driver built around `Lintul5Model`.

`LongTermSimulator` runs the Lintul5 day step over a weather series
spanning decades, turning it into a sequence of complete cropping seasons:

1. **Sowing.** A `CropCalendar` supplies a per-day sowing signal, so the
   seed-reserve bootstrap fires once for every season in the run.
2. **Harvest.** When a season ends, the crop is cleared from the field
   between two Euler steps (`torchcrop.longterm.reset_at_harvest`),
   leaving a bare seedbed for the following crop while soil water and
   nutrients continue uninterrupted.
3. **Bounded output.** Full per-day `ModelState` objects are never
   retained — only the variables named in ``collect`` — and the
   dependency chain is cut at season boundaries by default.

Everything between sowing and harvest is the ordinary torchcrop day step,
called through the same `SimulationEngine` the single-season path uses, so
a one-season run through this class reproduces `Lintul5Model.forward`
exactly.

## Gradients and memory over long horizons

``truncate_bptt=True`` (the default) detaches the state at each sowing
day, so season ``s``'s results depend only on season ``s``. Each season's
yield still carries a full gradient path back to its own sowing, which is
what fitting a parameter against 30 years of observed yields requires.
``truncate_bptt=False`` keeps one graph across the whole run so gradients
also cross season boundaries through the carried soil state — for example
through nitrogen that one crop leaves behind for the next.

Truncation bounds the *dependency chain*, not by itself the memory. The
per-season summaries are differentiable, and a live tensor keeps its own
season's graph alive, so holding 50 differentiable yields costs roughly
50 season-graphs whether or not they are chained.

The simulator itself is therefore flat in the horizon — the growth is
entirely the retained autograd graphs. Two ways to keep a long run cheap:

* **Scenario runs need no gradients.** Wrap the call in
  ``torch.no_grad()``; memory then stops growing with the horizon
  (a 50-year run costs about what a 10-year one does).
* **Calibration should run in chunks.** Simulate a decade, build the loss
  from its season records, call ``backward()``, then continue with
  ``initial_state=output.final_state.detach()`` over the next slice of
  weather. Chunked and single-call runs agree, so this costs nothing
  but bounds memory to one chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Sequence

import torch
import torch.nn as nn

from torchcrop.drivers.weather import WeatherDriver
from torchcrop.longterm.calendar import NO_SEASON, CropCalendar
from torchcrop.longterm.carryover import CarryOverPolicy, reset_at_harvest
from torchcrop.longterm.management import ManagementSchedule
from torchcrop.model import Lintul5Model
from torchcrop.states.model_state import DiagnosticState, ModelState


@dataclass
class SeasonRecord:
    """Per-season summaries of a long-term run.

    Every field is a ``[B, S]`` tensor — batch element by season slot —
    aligned with `CropCalendar.sow_days`. Slots that never ran (padded
    seasons, or seasons past the end of the weather series) hold ``0``,
    and `valid` marks the ones that did.

    Attributes:
        valid: ``1`` where the season actually ran to a harvest.
        sow_day: Day index the crop was sown on, `NO_SEASON` if unused.
        harvest_day: Day index the crop was taken off, `NO_SEASON` if
            unused.
        duration: ``harvest_day − sow_day`` [d].
        reached_maturity: ``1`` where the crop reached ``DVS = 2`` before
            being harvested, ``0`` where it was cut at a fixed date or
            forced off by the next sowing. A run of zeros here means the
            season length or thermal regime never completed the crop.
        yield_: Storage-organ dry weight ``WSO`` at harvest [g m⁻²].
        adjusted_yield: Heat-stress-adjusted yield
            ``(1 − HSF) · yield_`` [g m⁻²] from `HeatStressOnGrain`,
            evaluated over that season's anthesis window only.
        heat_stress_factor: Window-averaged around-anthesis heat-stress
            factor ``HSF`` in ``[0, 1]``.
        tagb: Above-ground living biomass at harvest,
            ``WLV + WST + WSO`` [g m⁻²].
        max_lai: Peak leaf area index reached during the season
            [m² m⁻²].
        tran_cum, evap_cum, rain_cum, irrig_cum, runoff_cum, drain_cum:
            Season water budget [mm]. These are per-season totals under
            the default ``accumulators="reset"`` policy and running
            run-to-date totals under ``"carry"``.
        nuptr_cum, puptr_cum, kuptr_cum, nfixtr_cum: Season nutrient
            uptake and biological N fixation [g X m⁻²].
        parint_cum: Season intercepted PAR [MJ m⁻²].
        gtotal_cum: Season gross assimilate [g DM m⁻²].

    Note:
        ``HSF`` is a trajectory-level quantity computed from the collected
        DVS series, which `LongTermSimulator` detaches under
        ``truncate_bptt=True``. In that mode ``heat_stress_factor`` carries
        no gradient and ``adjusted_yield`` is differentiable only through
        its ``yield_`` factor. Run with ``truncate_bptt=False`` to
        differentiate through the heat-stress term as well.
    """

    valid: torch.Tensor
    sow_day: torch.Tensor
    harvest_day: torch.Tensor
    duration: torch.Tensor
    reached_maturity: torch.Tensor
    yield_: torch.Tensor
    adjusted_yield: torch.Tensor
    heat_stress_factor: torch.Tensor
    tagb: torch.Tensor
    max_lai: torch.Tensor
    tran_cum: torch.Tensor
    evap_cum: torch.Tensor
    rain_cum: torch.Tensor
    irrig_cum: torch.Tensor
    runoff_cum: torch.Tensor
    drain_cum: torch.Tensor
    nuptr_cum: torch.Tensor
    puptr_cum: torch.Tensor
    kuptr_cum: torch.Tensor
    nfixtr_cum: torch.Tensor
    parint_cum: torch.Tensor
    gtotal_cum: torch.Tensor

    def to_dataframe(self, calendar: CropCalendar | None = None) -> Any:
        """Flatten the records into a tidy pandas DataFrame.

        One row per (batch element, season), which is the convenient shape
        for plotting a multi-decade yield series or joining against
        observations.

        Args:
            calendar: Optional calendar carrying a ``start_date``; when
                given, ``sow_date`` and ``harvest_date`` columns are added
                alongside the day indices.

        Returns:
            A ``pandas.DataFrame`` with columns ``batch``, ``season`` and
            one column per record field, restricted to valid seasons.

        Raises:
            ImportError: If pandas is not installed (``pip install
                "torchcrop[extra]"``).
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "SeasonRecord.to_dataframe requires pandas; install it with "
                'pip install pandas'
            ) from exc

        b, s = self.valid.shape
        batch_idx, season_idx = torch.meshgrid(
            torch.arange(b), torch.arange(s), indexing="ij"
        )
        data: dict[str, Any] = {
            "batch": batch_idx.reshape(-1).tolist(),
            "season": season_idx.reshape(-1).tolist(),
        }
        for f in fields(self):
            data[f.name] = (
                getattr(self, f.name).detach().cpu().reshape(-1).tolist()
            )
        frame = pd.DataFrame(data)
        if calendar is not None and calendar.start_date is not None:
            for name in ("sow_day", "harvest_day"):
                dates = calendar.season_dates(getattr(self, name).to(torch.long))
                frame[name.replace("_day", "_date")] = [
                    d for row in dates for d in row
                ]
        return frame[frame["valid"] > 0].reset_index(drop=True)


@dataclass
class LongTermOutput:
    """Results of a long-term run.

    Attributes:
        seasons: Per-season summaries, `[B, S]` per field.
        daily: Requested daily trajectories, keyed by variable name.
            `ModelState` variables have shape ``[B, T + 1]`` (the leading
            entry is the initial condition, matching
            `torchcrop.ModelOutput`); `DiagnosticState` variables have
            shape ``[B, T]``. Empty when ``collect`` was not given.
        final_state: `ModelState` at the end of the run — pass it as
            ``initial_state`` to continue the simulation over a further
            slice of weather.
        calendar: The calendar the run was driven by, for reporting.
        n_days: Number of simulated days ``T``.
    """

    seasons: SeasonRecord
    daily: dict[str, torch.Tensor]
    final_state: ModelState
    calendar: CropCalendar
    n_days: int

    @property
    def yield_(self) -> torch.Tensor:
        """Per-season storage-organ yield ``[B, S]`` [g m⁻²]."""
        return self.seasons.yield_


class LongTermSimulator(nn.Module):
    """Run `Lintul5Model` over many consecutive seasons.

    Args:
        model: The `Lintul5Model` to drive. Its parameters, hybrid
            residual slots and calibration hooks are used unchanged, so
            everything that works for a single season works here.
        carryover: What crosses each harvest boundary. Defaults to
            `CarryOverPolicy()` — continuous soil water and soil mineral
            pools, crop state cleared, per-season budgets.
        truncate_bptt: If ``True`` (the default) the state is detached at
            every sowing day, so each season's results depend only on that
            season. Set to ``False`` to keep one autograd graph across the
            whole run, letting gradients propagate between seasons through
            the carried soil state. Either way the simulated values are
            identical — see the module docstring for what this does and
            does not do for memory.

    Example:
        >>> model = torchcrop.Lintul5Model(crop, soil, site)
        >>> calendar = CropCalendar.annual(
        ...     sow_doy=285, n_years=30, start_date="1990-01-01",
        ...     n_days=weather.n_days,
        ... )
        >>> sim = LongTermSimulator(model)
        >>> out = sim.run(weather, calendar, collect=("dvs", "lai"))
        >>> out.seasons.yield_.shape
        torch.Size([1, 30])
    """

    def __init__(
        self,
        model: Lintul5Model,
        carryover: CarryOverPolicy | None = None,
        truncate_bptt: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.carryover = carryover or CarryOverPolicy()
        self.truncate_bptt = truncate_bptt

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(
        self,
        weather: WeatherDriver | torch.Tensor,
        calendar: CropCalendar,
        management: ManagementSchedule | None = None,
        irrigation: torch.Tensor | None = None,
        fertilizer: torch.Tensor | None = None,
        initial_state: ModelState | None = None,
        collect: Sequence[str] | None = None,
        start_doy: int = 1,
        use_weather_doy: bool = True,
    ) -> LongTermOutput:
        """Simulate every season in ``calendar`` over one weather series.

        Args:
            weather: `WeatherDriver` or raw ``[B, T, C]`` tensor covering
                the whole horizon — all years end to end, not one season.
            calendar: Sowing and harvest schedule. Its ``n_days`` and
                ``batch_size`` must match the weather.
            management: Optional `ManagementSchedule` expanded into daily
                irrigation and fertiliser arrays. Mutually exclusive with
                the explicit ``irrigation`` / ``fertilizer`` arguments.
            irrigation: Optional daily irrigation ``[B, T]`` [mm d⁻¹],
                as accepted by `Lintul5Model.forward`.
            fertilizer: Optional daily fertiliser ``[B, T, 3]``
                [g X m⁻² d⁻¹] ordered ``(N, P, K)``.
            initial_state: Optional starting `ModelState`; defaults to a
                fresh bare-soil state from `Lintul5Model.initialize`.
                Pass a previous run's ``final_state`` to continue a
                simulation across weather files.
            collect: Names of daily variables to retain, resolved against
                `ModelState` and `DiagnosticState` fields (for example
                ``("dvs", "lai", "wso", "tranrf")``). ``None`` keeps no
                daily output — only the per-season summaries — which is
                what keeps a multi-decade run cheap.
            start_doy: Day-of-year of the first simulated day. Used only
                when ``use_weather_doy`` is ``False``.
            use_weather_doy: If ``True`` (the default) the true calendar
                day-of-year is taken from weather channel ``0``. This
                matters over decades: the internal
                ``((start_doy - 1 + t) % 365) + 1`` fallback drifts by a
                day per leap year, shifting day length and solar geometry
                against the real calendar.

        Returns:
            A `LongTermOutput` with per-season summaries, any requested
            daily trajectories, and the final state.

        Raises:
            ValueError: If the weather, calendar and management shapes
                disagree, if both ``management`` and explicit driver
                arrays are given, or if ``collect`` names an unknown
                variable.
        """
        if isinstance(weather, torch.Tensor):
            weather = WeatherDriver(weather)
        dtype, device = weather.data.dtype, weather.data.device
        batch_size, n_days = weather.batch_size, weather.n_days

        self._validate(weather, calendar, management, irrigation, fertilizer)
        calendar = calendar.to(device)

        if management is not None:
            irrigation, fertilizer = management.expand(calendar, dtype, device)

        model = self.model
        model.crop_params.validate()
        model.soil_params.validate()
        model.site_params.validate()

        state = (
            initial_state
            if initial_state is not None
            else model.initialize(batch_size, dtype=dtype, device=device)
        )
        model.hybrid.reset_penalty()

        # Per-day event signals. Sowing is a single pulse: the model's
        # latch is ``sown = max(state.sown, sowing)``, so one pulse puts
        # the crop in the ground and it stays there until harvest clears
        # ``state.sown`` again.
        sow_signal = calendar.sow_signal(dtype, device)
        fixed_signal = calendar.fixed_harvest_signal(dtype, device)
        deadline_signal = calendar.deadline_signal(dtype, device)
        doy_all = weather.channel("doy") if use_weather_doy else None

        state_names, diag_names = self._resolve_collect(collect)
        daily_state: dict[str, list[torch.Tensor]] = {
            name: [getattr(state, name)] for name in state_names
        }
        daily_diag: dict[str, list[torch.Tensor]] = {name: [] for name in diag_names}
        dvs_days: list[torch.Tensor] = []

        n_seasons = calendar.n_seasons
        zeros_b = torch.zeros(batch_size, dtype=dtype, device=device)
        acc: dict[str, torch.Tensor] = {
            name: torch.zeros((batch_size, n_seasons), dtype=dtype, device=device)
            for name in self._SEASON_SUMS
        }
        season_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        days_mature = torch.zeros(batch_size, dtype=dtype, device=device)
        season_max_lai = zeros_b.clone()

        use_maturity = calendar.harvest_rule == "maturity"
        ripening = float(calendar.after_maturity_days)

        for t in range(n_days):
            sow_t = sow_signal[:, t]
            # Cut the graph while the field is bare, between two seasons.
            if self.truncate_bptt and bool((sow_t > 0.5).any()):
                state = state.detach()

            step = model.engine.step(
                state=state,
                weather_day=weather.day(t),
                doy=(
                    torch.full_like(state.dvs, float(((start_doy - 1 + t) % 365) + 1))
                    if doy_all is None
                    else doy_all[:, t]
                ),
                crop_params=model.crop_params,
                soil_params=model.soil_params,
                site_params=model.site_params,
                irrigation=None if irrigation is None else irrigation[:, t],
                fertilizer=None if fertilizer is None else fertilizer[:, t, :],
                sowing=sow_t,
            )
            state = step.state

            # ---------------------------------------------------------- #
            # Season bookkeeping — all masks, no Python branching on
            # tensor values, so batch elements may harvest on different
            # days of different years.
            # ---------------------------------------------------------- #
            in_field = state.sown > 0.5
            mature = state.dvs >= 2.0
            season_max_lai = torch.maximum(
                season_max_lai * (1.0 - sow_t), state.lai
            )
            days_mature = torch.where(
                mature, days_mature + 1.0, torch.zeros_like(days_mature)
            )

            due = fixed_signal[:, t] + deadline_signal[:, t] > 0.5
            if use_maturity:
                due = due | (days_mature >= ripening + 1.0)
            harvest = in_field & due
            harvest_f = harvest.to(dtype)

            # The daily record shows the field as it stood at the close of
            # day ``t`` — with the ripe crop still on it. Harvest is a
            # boundary event applied *between* days, so its effect first
            # appears in day ``t + 1``. Recording before the reset also
            # makes a one-season run here byte-identical to
            # ``Lintul5Model.forward`` on every day, final day included.
            dvs_days.append(self._keep(state.dvs))
            for name in state_names:
                daily_state[name].append(self._keep(getattr(state, name)))
            if diag_names and step.diagnostic is not None:
                for name in diag_names:
                    daily_diag[name].append(
                        self._keep(getattr(step.diagnostic, name))
                    )

            if bool(harvest.any()):
                self._record(
                    acc,
                    season_idx,
                    harvest_f,
                    state,
                    mature.to(dtype),
                    season_max_lai,
                    float(t),
                    n_seasons,
                )
                state = reset_at_harvest(
                    state,
                    harvest_f,
                    model.crop_params,
                    model.soil_params,
                    self.carryover,
                )
                days_mature = days_mature * (1.0 - harvest_f)
                season_max_lai = season_max_lai * (1.0 - harvest_f)
                season_idx = torch.clamp(
                    season_idx + harvest.long(), max=max(n_seasons - 1, 0)
                )

        daily = {
            name: torch.stack(values, dim=1)
            for name, values in (*daily_state.items(), *daily_diag.items())
        }
        seasons = self._finalise(
            acc, calendar, weather, torch.stack(dvs_days, dim=1), n_days
        )
        return LongTermOutput(
            seasons=seasons,
            daily=daily,
            final_state=state,
            calendar=calendar,
            n_days=n_days,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    #: Season-summary fields accumulated during the loop. The
    #: ``*_cum`` entries are read straight off the corresponding
    #: `ModelState` accumulator at the moment of harvest.
    _STATE_SUMS: tuple[str, ...] = (
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
    _SEASON_SUMS: tuple[str, ...] = (
        "valid",
        "harvest_day",
        "reached_maturity",
        "yield_",
        "tagb",
        "max_lai",
        *_STATE_SUMS,
    )

    def _keep(self, tensor: torch.Tensor) -> torch.Tensor:
        """Store a daily value, dropping its graph when truncating.

        Retaining collected trajectories with their graphs attached would
        defeat ``truncate_bptt`` — the released season graph would stay
        alive through the output — so collected values are detached in
        that mode.

        Args:
            tensor: The per-day value to store.

        Returns:
            The tensor, detached when ``truncate_bptt`` is set.
        """
        return tensor.detach() if self.truncate_bptt else tensor

    def _record(
        self,
        acc: dict[str, torch.Tensor],
        season_idx: torch.Tensor,
        harvest: torch.Tensor,
        state: ModelState,
        mature: torch.Tensor,
        max_lai: torch.Tensor,
        day: float,
        n_seasons: int,
    ) -> None:
        """Write today's harvest into each element's current season slot.

        Uses a one-hot mask times the value rather than an indexed
        assignment, so the recorded yields keep their gradient path back
        through the season.

        Args:
            acc: Accumulator dict of ``[B, S]`` tensors, updated in place
                (rebinding entries, never mutating tensors).
            season_idx: ``[B]`` index of the season each element is in.
            harvest: ``[B]`` float mask of elements harvesting today.
            state: Post-update, pre-reset `ModelState`.
            mature: ``[B]`` float mask of elements that reached maturity.
            max_lai: ``[B]`` peak LAI of the season being closed.
            day: The current day index.
            n_seasons: Number of season slots ``S``.
        """
        one_hot = torch.nn.functional.one_hot(
            torch.clamp(season_idx, max=max(n_seasons - 1, 0)), n_seasons
        ).to(harvest.dtype) * harvest.unsqueeze(1)

        values: dict[str, torch.Tensor] = {
            "valid": torch.ones_like(harvest),
            "harvest_day": torch.full_like(harvest, day),
            "reached_maturity": mature,
            "yield_": state.wso,
            "tagb": state.wlv + state.wst + state.wso,
            "max_lai": max_lai,
            **{name: getattr(state, name) for name in self._STATE_SUMS},
        }
        for name, value in values.items():
            acc[name] = acc[name] + one_hot * value.unsqueeze(1)

    def _finalise(
        self,
        acc: dict[str, torch.Tensor],
        calendar: CropCalendar,
        weather: WeatherDriver,
        dvs_days: torch.Tensor,
        n_days: int,
    ) -> SeasonRecord:
        """Assemble the `SeasonRecord`, including per-season heat stress.

        The around-anthesis heat penalty is a *trajectory-level* quantity,
        so it is evaluated season by season: the DVS trajectory is masked
        to one season's days before being handed to
        `HeatStressOnGrain`. Days outside the season carry ``DVS = 0``,
        which falls below the anthesis window and is therefore ignored by
        the module's own window mask.

        Args:
            acc: Accumulated ``[B, S]`` season summaries.
            calendar: The run's calendar.
            weather: The weather driver, for ``tmin``/``tmax``.
            dvs_days: ``[B, T]`` post-update DVS trajectory.
            n_days: Number of simulated days ``T``.

        Returns:
            The completed `SeasonRecord`.
        """
        valid = acc["valid"]
        sow_day = calendar.sow_days.to(valid.dtype)
        harvest_day = acc["harvest_day"]
        unused = torch.full_like(sow_day, float(NO_SEASON))

        tmin, tmax = weather.channel("tmin"), weather.channel("tmax")
        day_index = torch.arange(n_days, device=valid.device).reshape(1, -1)
        hsf_cols, adj_cols = [], []
        for s in range(calendar.n_seasons):
            within = (day_index >= calendar.sow_days[:, s : s + 1]) & (
                day_index <= harvest_day[:, s : s + 1].to(day_index.dtype)
            )
            season_mask = within & (calendar.season_mask[:, s : s + 1])
            hsg = self.model.heat_stress_grain(
                tmin=tmin,
                tmax=tmax,
                dvs=dvs_days * season_mask.to(dvs_days.dtype),
                params=self.model.crop_params,
                yield_=acc["yield_"][:, s],
            )
            hsf_cols.append(hsg["heat_stress_factor"])
            adj_cols.append(hsg["adjusted_yield"])

        hsf = torch.stack(hsf_cols, dim=1) * valid
        adjusted = torch.stack(adj_cols, dim=1) * valid

        return SeasonRecord(
            valid=valid,
            sow_day=torch.where(valid > 0.5, sow_day, unused),
            harvest_day=torch.where(valid > 0.5, harvest_day, unused),
            duration=torch.where(
                valid > 0.5, harvest_day - sow_day, torch.zeros_like(sow_day)
            ),
            reached_maturity=acc["reached_maturity"],
            yield_=acc["yield_"],
            adjusted_yield=adjusted,
            heat_stress_factor=hsf,
            tagb=acc["tagb"],
            max_lai=acc["max_lai"],
            **{name: acc[name] for name in self._STATE_SUMS},
        )

    @staticmethod
    def _resolve_collect(
        collect: Sequence[str] | None,
    ) -> tuple[list[str], list[str]]:
        """Split requested variable names into state and diagnostic sets.

        Args:
            collect: Requested daily variable names, or ``None``.

        Returns:
            Tuple ``(state_names, diagnostic_names)``.

        Raises:
            ValueError: If a name matches neither container.
        """
        if not collect:
            return [], []
        state_fields = {f.name for f in fields(ModelState)}
        diag_fields = {f.name for f in fields(DiagnosticState)}
        state_names, diag_names, unknown = [], [], []
        for name in collect:
            if name in state_fields:
                state_names.append(name)
            elif name in diag_fields:
                diag_names.append(name)
            else:
                unknown.append(name)
        if unknown:
            known = sorted(state_fields | diag_fields)
            raise ValueError(
                f"unknown collect variable(s): {unknown}. Choose from "
                f"ModelState or DiagnosticState fields: {known}"
            )
        return state_names, diag_names

    @staticmethod
    def _validate(
        weather: WeatherDriver,
        calendar: CropCalendar,
        management: ManagementSchedule | None,
        irrigation: torch.Tensor | None,
        fertilizer: torch.Tensor | None,
    ) -> None:
        """Check that weather, calendar and driver shapes agree.

        Args:
            weather: The weather driver.
            calendar: The calendar to be run.
            management: Optional management schedule.
            irrigation: Optional explicit irrigation array.
            fertilizer: Optional explicit fertiliser array.

        Raises:
            ValueError: On any shape mismatch, or when a management
                schedule is combined with explicit driver arrays.
        """
        if calendar.n_days != weather.n_days:
            raise ValueError(
                f"calendar.n_days ({calendar.n_days}) must match the weather "
                f"series length T ({weather.n_days})"
            )
        if calendar.batch_size not in (1, weather.batch_size):
            raise ValueError(
                f"calendar batch size ({calendar.batch_size}) must match the "
                f"weather batch size ({weather.batch_size})"
            )
        if calendar.batch_size != weather.batch_size:
            raise ValueError(
                "calendar has a single row but the weather is batched; "
                f"expand the calendar to B = {weather.batch_size} rows"
            )
        if management is not None and (
            irrigation is not None or fertilizer is not None
        ):
            raise ValueError(
                "pass either a ManagementSchedule or explicit "
                "irrigation/fertilizer arrays, not both"
            )
        for name, tensor, expected in (
            ("irrigation", irrigation, (weather.batch_size, weather.n_days)),
            ("fertilizer", fertilizer, (weather.batch_size, weather.n_days, 3)),
        ):
            if tensor is not None and tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}; got {tuple(tensor.shape)}"
                )
