"""Time-varying irrigation and fertiliser schedules for long-term runs.

`ManagementSchedule` is a comfortable way to *write down* half a century
of field operations: describe each event once — on a fixed date, or
relative to whichever day the crop is sown that year — and expand it
against a `CropCalendar` into the daily irrigation ``[B, T]`` and
fertiliser ``[B, T, 3]`` arrays `Lintul5Model.forward` consumes.

Sowing-relative anchoring is what lets a schedule follow a sowing date
that moves between years. The ``ferntab``/``ferptab``/``ferktab`` and
``irrtab`` tables are keyed by day-of-year, so a table-driven schedule
repeats identically every year and stays fixed to the calendar.

The expanded tensors are ordinary tensors, so a schedule built from
`torch.nn.Parameter` amounts is differentiable end to end and can be
optimised through the simulation — the basis for gradient-based
fertiliser- or irrigation-strategy studies.

Note:
    Fertiliser amounts are **g m⁻² d⁻¹** of elemental N, P and K, matching
    the model's units. Field recommendations are usually kg ha⁻¹:
    divide by 10 to convert (``60 kg N ha⁻¹`` → ``6.0 g N m⁻²``).
    `ManagementSchedule.fertilizer` applies the whole amount on the named
    day; the model's ``scale_factor_fer*`` factors and recovery fractions
    ``nrf``/``prf``/``krf`` are applied downstream as usual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Sequence

import torch

from torchcrop.longterm.calendar import NO_SEASON, CropCalendar, _to_day_index


@dataclass
class ManagementEvent:
    """A single dated field operation.

    Exactly one of ``day``, ``date`` or ``days_after_sowing`` must be
    given. ``days_after_sowing`` is the one that makes a schedule follow a
    moving sowing date: it re-anchors to each season's own sowing day, so
    "60 kg N at 20 days after sowing, every year" survives sowing dates
    that shift by weeks between years.

    Attributes:
        amount: Quantity applied. For irrigation, mm d⁻¹. For fertiliser,
            g m⁻² d⁻¹ of the element named by ``nutrient``. May be a
            Python float or a scalar/``[B]`` tensor — a
            `torch.nn.Parameter` here stays differentiable through the
            expansion.
        day: Absolute day index into the weather series.
        date: Calendar date; requires the calendar to carry a
            ``start_date``.
        days_after_sowing: Offset in days from each season's sowing day.
            Negative values place the operation before sowing (e.g. a
            pre-sowing base dressing).
        seasons: Optional season indices this event applies to when
            anchored by ``days_after_sowing``. ``None`` (the default)
            applies it to every season.
        nutrient: For fertiliser events, which element the amount refers
            to — ``"n"``, ``"p"`` or ``"k"``. Ignored by irrigation
            events.
    """

    amount: float | torch.Tensor
    day: int | None = None
    date: date | datetime | str | None = None
    days_after_sowing: int | None = None
    seasons: Sequence[int] | None = None
    nutrient: str = "n"

    def __post_init__(self) -> None:
        anchors = [
            self.day is not None,
            self.date is not None,
            self.days_after_sowing is not None,
        ]
        if sum(anchors) != 1:
            raise ValueError(
                "a ManagementEvent needs exactly one of day, date or "
                "days_after_sowing"
            )
        if self.nutrient.lower() not in ("n", "p", "k"):
            raise ValueError(
                f"nutrient must be 'n', 'p' or 'k'; got {self.nutrient!r}"
            )

    def resolve_days(self, calendar: CropCalendar) -> torch.Tensor:
        """Day indices this event fires on, per batch element and season.

        Args:
            calendar: Calendar supplying the sowing days (and
                ``start_date`` for date-anchored events).

        Returns:
            A ``[B, S]`` long tensor of day indices, `NO_SEASON` where the
            event does not apply. Fixed-day and date-anchored events
            occupy the first column only, since they fire once for the
            whole run.

        Raises:
            ValueError: If a date-anchored event is used with a calendar
                that has no ``start_date``.
        """
        b, s = calendar.batch_size, calendar.n_seasons
        out = torch.full((b, s), NO_SEASON, dtype=torch.long)

        if self.days_after_sowing is not None:
            days = calendar.sow_days + self.days_after_sowing
            applies = calendar.season_mask.clone()
            if self.seasons is not None:
                keep = torch.zeros(s, dtype=torch.bool)
                keep[list(self.seasons)] = True
                applies = applies & keep.unsqueeze(0)
            # An offset can push an event off either end of the series;
            # such applications simply do not happen.
            applies = applies & (days >= 0) & (days < calendar.n_days)
            return torch.where(applies, days, out)

        if self.day is not None:
            day_index = int(self.day)
        else:
            if calendar.start_date is None:
                raise ValueError(
                    "date-anchored ManagementEvents need a calendar with "
                    "start_date set"
                )
            day_index = _to_day_index(
                self.date, calendar.start_date  # type: ignore[arg-type]
            )
        if not 0 <= day_index < calendar.n_days:
            raise ValueError(
                f"event day {day_index} falls outside the "
                f"{calendar.n_days}-day weather series"
            )
        out[:, 0] = day_index
        return out


@dataclass
class ManagementSchedule:
    """Irrigation and fertiliser events, expandable to daily driver arrays.

    Attributes:
        irrigation: Irrigation events [mm d⁻¹].
        fertilizer: Fertiliser events [g X m⁻² d⁻¹]; each carries the
            element it applies to in `ManagementEvent.nutrient`. Several
            events on the same day accumulate, so a compound NPK dressing
            is written as three events sharing an anchor.

    Example:
        >>> schedule = ManagementSchedule(
        ...     fertilizer=[
        ...         ManagementEvent(amount=4.0, days_after_sowing=0),
        ...         ManagementEvent(amount=6.0, days_after_sowing=120),
        ...     ],
        ...     irrigation=[ManagementEvent(amount=25.0, days_after_sowing=150)],
        ... )
        >>> irrig, fert = schedule.expand(calendar)
        >>> output = simulator.run(weather, calendar, management=schedule)
    """

    irrigation: list[ManagementEvent] = field(default_factory=list)
    fertilizer: list[ManagementEvent] = field(default_factory=list)

    _NUTRIENT_AXIS: dict[str, int] = field(
        default_factory=lambda: {"n": 0, "p": 1, "k": 2}, repr=False
    )

    def expand(
        self,
        calendar: CropCalendar,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Expand the events into the model's daily driver arrays.

        Args:
            calendar: Calendar the sowing-relative events anchor to, and
                the source of ``B`` and ``T``.
            dtype: Floating dtype of the returned tensors.
            device: Device of the returned tensors.

        Returns:
            Tuple ``(irrigation, fertilizer)`` where ``irrigation`` is
            ``[B, T]`` [mm d⁻¹] and ``fertilizer`` is ``[B, T, 3]``
            [g X m⁻² d⁻¹] ordered ``(N, P, K)``. Either is ``None`` when
            no events of that kind were given, which leaves the model's
            own ``irri`` mode or fertiliser tables in control.
        """
        b, t = calendar.batch_size, calendar.n_days

        irrigation = None
        if self.irrigation:
            irrigation = torch.zeros((b, t), dtype=dtype, device=device)
            for event in self.irrigation:
                irrigation = irrigation + self._scatter(
                    event, calendar, (b, t), dtype, device
                )

        fertilizer = None
        if self.fertilizer:
            layers = [torch.zeros((b, t), dtype=dtype, device=device) for _ in range(3)]
            for event in self.fertilizer:
                axis = self._NUTRIENT_AXIS[event.nutrient.lower()]
                layers[axis] = layers[axis] + self._scatter(
                    event, calendar, (b, t), dtype, device
                )
            fertilizer = torch.stack(layers, dim=-1)

        return irrigation, fertilizer

    def _scatter(
        self,
        event: ManagementEvent,
        calendar: CropCalendar,
        shape: tuple[int, int],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Place one event's amount on its days in a ``[B, T]`` grid.

        Built by multiplying a ``{0, 1}`` indicator by the amount rather
        than by index assignment, so a `torch.nn.Parameter` amount keeps
        its gradient path into the simulation.

        Args:
            event: The event to place.
            calendar: Calendar used to resolve the event's days.
            shape: Target ``(B, T)`` shape.
            dtype: Floating dtype of the result.
            device: Device of the result.

        Returns:
            A ``[B, T]`` tensor holding the event's amount on its days.
        """
        days = event.resolve_days(calendar)
        indicator = torch.zeros(shape, dtype=dtype, device=device)
        real = days != NO_SEASON
        rows, cols = torch.nonzero(real, as_tuple=True)
        if rows.numel():
            indicator[rows, days[rows, cols].to(device)] = 1.0
        amount = torch.as_tensor(event.amount, dtype=dtype, device=device)
        if amount.dim() == 0:
            return indicator * amount
        if amount.shape != (shape[0],):
            raise ValueError(
                f"event amount must be scalar or [B] = ({shape[0]},); "
                f"got {tuple(amount.shape)}"
            )
        return indicator * amount.unsqueeze(1)
