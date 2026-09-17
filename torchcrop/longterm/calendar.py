"""Season scheduling for long-term (multi-year) simulations.

A `CropCalendar` says *when* each crop goes into the ground and *when* it
comes out, for every element of the batch. It holds no physics — it is a
pure bookkeeping object that `LongTermSimulator` turns into two per-day
signals:

* a **sowing signal** ``[B, T]``, fed to `Lintul5Model.forward`'s
  ``sowing`` argument, which fires on each sowing day and puts a fresh
  seed reserve in the ground;
* a **deadline signal** ``[B, T]``, which forces a standing crop out of
  the field before the next crop is sown, so seasons can never overlap.

Sowing days are *absolute day indices* into the weather series
(``0`` = the first simulated day), which keeps the calendar independent of
any particular date convention. The `annual` and `from_dates`
constructors convert real dates into those indices for you.

Harvest may be **rule-driven** (the day the crop reaches maturity,
``DVS >= 2``, optionally plus a ripening/drying delay) or **fixed** (an
explicit day index per season), and the two can be mixed per season.
Because maturity is a state-dependent event that differs per batch
element and per year, the maturity part of the rule is evaluated day by
day inside the simulator; only the fixed and deadline parts can be
precomputed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Sequence

import torch

#: Value used to fill unused season slots in the ragged ``[B, S]`` day
#: tensors, allowing batch elements to contain different numbers of seasons.
NO_SEASON: int = -1


def _as_date(value: date | datetime | str) -> date:
    """Coerce a date-like value to a `datetime.date`.

    Args:
        value: A `datetime.date`, a `datetime.datetime`, or an ISO-8601
            ``"YYYY-MM-DD"`` string.

    Returns:
        The corresponding `datetime.date`.

    Raises:
        TypeError: If ``value`` is not one of the accepted types.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(
        f"expected a date, datetime or 'YYYY-MM-DD' string; got {type(value)!r}"
    )


def _to_day_index(value: date | datetime | str, start_date: date) -> int:
    """Convert a calendar date to a day index relative to ``start_date``.

    Args:
        value: The date to convert (see `_as_date` for accepted forms).
        start_date: Calendar date of simulated day ``0``.

    Returns:
        The integer offset ``(value - start_date).days``.
    """
    return (_as_date(value) - start_date).days


@dataclass
class CropCalendar:
    """Sowing and harvest schedule for a batch of long-term simulations.

    All day fields are **absolute day indices** into the weather series,
    where ``0`` is the first simulated day. Rows of the ``[B, S]`` tensors
    are per batch element and columns per season; unused trailing slots
    carry `NO_SEASON` (``-1``), so different batch elements may run
    different numbers of seasons in one call.

    Attributes:
        sow_days: ``[B, S]`` integer tensor of sowing day indices, strictly
            increasing along the season axis, with `NO_SEASON` padding in
            trailing slots only.
        n_days: Length ``T`` of the weather series the calendar is aligned
            to. Every finite day index must satisfy ``0 <= d < n_days``.
        harvest_days: Optional ``[B, S]`` integer tensor of fixed harvest
            day indices. `NO_SEASON` in a slot means "no fixed date for
            this season" — that season then ends on the rule set by
            ``harvest_rule``. ``None`` means no season has a fixed date.
        harvest_rule: What ends a season that has no fixed harvest date.
            ``"maturity"`` (the default) harvests on the first day the
            crop reaches ``DVS >= 2``, plus ``after_maturity_days``.
            ``"deadline"`` leaves the crop standing until the day before
            the next sowing (or the end of the run).
        after_maturity_days: Ripening/drying delay in days between
            reaching maturity and harvest, applied to the ``"maturity"``
            rule. Defaults to ``0``.
        start_date: Optional calendar date of simulated day ``0``. Purely
            for reporting — `season_dates` uses it to translate day
            indices back into dates.

    Note:
        Whatever the rule, `LongTermSimulator` always clears the field no
        later than the day before the next sowing. A crop that never
        reaches maturity (a failed year, or a run that ends mid-season) is
        therefore still harvested, and seasons can never overlap.
    """

    sow_days: torch.Tensor
    n_days: int
    harvest_days: torch.Tensor | None = None
    harvest_rule: str = "maturity"
    after_maturity_days: int = 0
    start_date: date | None = None

    _RULES = ("maturity", "deadline")

    def __post_init__(self) -> None:
        self.sow_days = self._validate_days(self.sow_days, "sow_days")
        if self.harvest_days is not None:
            self.harvest_days = self._validate_days(
                self.harvest_days, "harvest_days", monotone=False
            )
            if self.harvest_days.shape != self.sow_days.shape:
                raise ValueError(
                    "harvest_days must have the same shape as sow_days "
                    f"{tuple(self.sow_days.shape)}; got "
                    f"{tuple(self.harvest_days.shape)}"
                )
            valid = (self.harvest_days != NO_SEASON) & (
                self.sow_days != NO_SEASON
            )
            if bool((valid & (self.harvest_days <= self.sow_days)).any()):
                raise ValueError(
                    "every fixed harvest day must fall strictly after its "
                    "sowing day"
                )
        if self.harvest_rule not in self._RULES:
            raise ValueError(
                f"harvest_rule must be one of {self._RULES}; "
                f"got {self.harvest_rule!r}"
            )
        if self.after_maturity_days < 0:
            raise ValueError(
                "after_maturity_days must be >= 0; got "
                f"{self.after_maturity_days}"
            )
        if self.n_days <= 0:
            raise ValueError(f"n_days must be positive; got {self.n_days}")

    def _validate_days(
        self, days: torch.Tensor, name: str, monotone: bool = True
    ) -> torch.Tensor:
        """Coerce and validate a ``[B, S]`` day-index tensor.

        Args:
            days: Tensor (or tensor-like) of day indices.
            name: Field name, used in error messages.
            monotone: If ``True``, require strictly increasing indices
                along the season axis and `NO_SEASON` padding only in
                trailing slots.

        Returns:
            The tensor as a contiguous 2-D ``torch.long`` tensor.

        Raises:
            ValueError: If the shape, range, ordering, or padding is
                invalid.
        """
        days = torch.as_tensor(days, dtype=torch.long)
        if days.dim() != 2:
            raise ValueError(
                f"{name} must be 2-D [B, S]; got {tuple(days.shape)}"
            )
        real = days != NO_SEASON
        if bool((days[real] < 0).any()) or bool((days[real] >= self.n_days).any()):
            raise ValueError(
                f"{name} must lie in [0, n_days) = [0, {self.n_days}); got "
                f"min {int(days[real].min())}, max {int(days[real].max())}"
            )
        if monotone and days.shape[1] > 1:
            # Padding is trailing only: a real slot may never follow a
            # NO_SEASON slot in the same row.
            if bool((real[:, :-1] < real[:, 1:]).any()):
                raise ValueError(
                    f"{name} may only be padded with {NO_SEASON} in trailing "
                    "slots; found a real season after a padded one"
                )
            pair = real[:, :-1] & real[:, 1:]
            if bool((pair & (days[:, 1:] <= days[:, :-1])).any()):
                raise ValueError(
                    f"{name} must increase strictly along the season axis"
                )
        return days.contiguous()

    # ------------------------------------------------------------------ #
    # Shape helpers
    # ------------------------------------------------------------------ #

    @property
    def batch_size(self) -> int:
        """Number of parallel simulation instances ``B``."""
        return int(self.sow_days.shape[0])

    @property
    def n_seasons(self) -> int:
        """Number of season slots ``S`` (the maximum over batch elements)."""
        return int(self.sow_days.shape[1])

    @property
    def season_mask(self) -> torch.Tensor:
        """``[B, S]`` boolean mask of slots that hold a real season."""
        return self.sow_days != NO_SEASON

    # ------------------------------------------------------------------ #
    # Per-day signals consumed by the simulator
    # ------------------------------------------------------------------ #

    def _one_hot(
        self,
        days: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Scatter ``[B, S]`` day indices into a ``[B, T]`` indicator.

        Args:
            days: ``[B, S]`` day indices, `NO_SEASON` for empty slots.
            dtype: Floating dtype of the returned indicator.
            device: Device of the returned indicator.

        Returns:
            A ``[B, n_days]`` tensor holding ``1`` at every listed day and
            ``0`` elsewhere.
        """
        out = torch.zeros(
            (self.batch_size, self.n_days), dtype=dtype, device=device
        )
        real = days != NO_SEASON
        rows, cols = torch.nonzero(real, as_tuple=True)
        if rows.numel():
            out[rows, days[rows, cols].to(device)] = 1.0
        return out

    def sow_signal(
        self,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Per-day sowing indicator for `Lintul5Model.forward`.

        The signal fires on the sowing day **only**: the model's latch is
        ``sown = max(state.sown, sowing)``, so a single pulse is enough to
        put the crop in the ground and keep it there until the simulator
        clears the field at harvest.

        Args:
            dtype: Floating dtype of the returned signal.
            device: Device of the returned signal.

        Returns:
            A ``[B, T]`` tensor with ``1`` on each sowing day, else ``0``.
        """
        return self._one_hot(self.sow_days, dtype, device)

    def fixed_harvest_signal(
        self,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Per-day indicator of the explicitly scheduled harvest days.

        Args:
            dtype: Floating dtype of the returned signal.
            device: Device of the returned signal.

        Returns:
            A ``[B, T]`` tensor with ``1`` on each fixed harvest day, else
            ``0``. All zeros when no season has a fixed date.
        """
        if self.harvest_days is None:
            return torch.zeros(
                (self.batch_size, self.n_days), dtype=dtype, device=device
            )
        return self._one_hot(self.harvest_days, dtype, device)

    def deadline_signal(
        self,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Per-day indicator of the last possible day of each season.

        A season must end before the next one starts, so its deadline is
        the day before the following sowing. The final season's deadline
        is the last simulated day, which closes the books on a crop that
        is still standing when the weather runs out.

        Args:
            dtype: Floating dtype of the returned signal.
            device: Device of the returned signal.

        Returns:
            A ``[B, T]`` tensor with ``1`` on each deadline day, else
            ``0``.
        """
        deadlines = torch.full_like(self.sow_days, NO_SEASON)
        real = self.season_mask
        # Slot s ends the day before slot s+1 is sown ...
        if self.n_seasons > 1:
            nxt = real[:, 1:]
            deadlines[:, :-1] = torch.where(
                nxt, self.sow_days[:, 1:] - 1, deadlines[:, :-1]
            )
        # ... and the last real season of each row ends on the final day.
        last_real = real & ~torch.cat(
            [real[:, 1:], torch.zeros_like(real[:, :1])], dim=1
        )
        deadlines = torch.where(
            last_real, torch.full_like(deadlines, self.n_days - 1), deadlines
        )
        deadlines = torch.where(real, deadlines, torch.full_like(deadlines, NO_SEASON))
        return self._one_hot(deadlines, dtype, device)

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def season_dates(self, day_indices: torch.Tensor) -> list[list[date | None]]:
        """Translate day indices back into calendar dates.

        Args:
            day_indices: Any integer tensor of day indices, typically the
                ``sow_day`` / ``harvest_day`` fields of a
                `torchcrop.longterm.SeasonRecord`. `NO_SEASON` entries map
                to ``None``.

        Returns:
            A nested list matching the shape of ``day_indices``, holding
            `datetime.date` objects (or ``None`` for empty slots).

        Raises:
            ValueError: If the calendar has no ``start_date``.
        """
        if self.start_date is None:
            raise ValueError(
                "season_dates requires start_date to be set on the calendar"
            )
        idx = day_indices.detach().cpu().to(torch.long).tolist()
        return [
            [
                None if d == NO_SEASON else self.start_date + timedelta(days=int(d))
                for d in row
            ]
            for row in idx
        ]

    def to(self, device: torch.device | str) -> "CropCalendar":
        """Return a copy with the day tensors moved to ``device``.

        Args:
            device: Target torch device.

        Returns:
            A new `CropCalendar` on the requested device.
        """
        return CropCalendar(
            sow_days=self.sow_days.to(device),
            n_days=self.n_days,
            harvest_days=(
                None if self.harvest_days is None else self.harvest_days.to(device)
            ),
            harvest_rule=self.harvest_rule,
            after_maturity_days=self.after_maturity_days,
            start_date=self.start_date,
        )

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dates(
        cls,
        sow_dates: Sequence[date | datetime | str]
        | Sequence[Sequence[date | datetime | str]],
        start_date: date | datetime | str,
        n_days: int,
        harvest_dates: Sequence[date | datetime | str]
        | Sequence[Sequence[date | datetime | str | None]]
        | None = None,
        batch_size: int | None = None,
        **kwargs: object,
    ) -> "CropCalendar":
        """Build a calendar from real sowing dates.

        This is the constructor to reach for when sowing dates vary from
        year to year — pass the actual dates and they are converted to day
        indices against ``start_date``.

        Args:
            sow_dates: Either a flat sequence of dates shared by the whole
                batch, or a sequence of per-batch-element date sequences
                (which may have different lengths — short rows are padded
                with `NO_SEASON`). Dates may be `datetime.date`,
                `datetime.datetime`, or ``"YYYY-MM-DD"`` strings.
            start_date: Calendar date of simulated day ``0``, i.e. of
                ``weather[:, 0, :]``.
            n_days: Length ``T`` of the weather series.
            harvest_dates: Optional fixed harvest dates in the same layout
                as ``sow_dates``; ``None`` entries fall back to
                ``harvest_rule``.
            batch_size: Batch size ``B`` to broadcast a flat ``sow_dates``
                sequence to. Ignored when per-element sequences are given.
                Defaults to ``1``.
            **kwargs: Forwarded to `CropCalendar` (``harvest_rule``,
                ``after_maturity_days``).

        Returns:
            A validated `CropCalendar` carrying ``start_date`` for
            reporting.

        Example:
            >>> cal = CropCalendar.from_dates(
            ...     ["2020-10-12", "2021-10-05", "2022-09-28"],
            ...     start_date="2020-01-01",
            ...     n_days=1096,
            ... )
        """
        start = _as_date(start_date)

        def _rows(
            values: Sequence[object] | None, default_b: int
        ) -> list[list[int]] | None:
            if values is None:
                return None
            nested = bool(values) and isinstance(values[0], (list, tuple))
            seqs = (
                [list(v) for v in values]  # type: ignore[arg-type]
                if nested
                else [list(values)] * default_b
            )
            def _one(v: object) -> int:
                if v is None:
                    return NO_SEASON
                return _to_day_index(v, start)  # type: ignore[arg-type]

            return [[_one(v) for v in seq] for seq in seqs]

        nested_sow = bool(sow_dates) and isinstance(sow_dates[0], (list, tuple))
        b = len(sow_dates) if nested_sow else (batch_size or 1)
        sow_rows = _rows(sow_dates, b)
        assert sow_rows is not None
        harvest_rows = _rows(harvest_dates, b)

        width = max(len(r) for r in sow_rows)
        if harvest_rows is not None:
            if len(harvest_rows) != len(sow_rows):
                raise ValueError(
                    "harvest_dates must have one row per batch element "
                    f"({len(sow_rows)}); got {len(harvest_rows)}"
                )
            width = max(width, max(len(r) for r in harvest_rows))

        def _pad(rows: list[list[int]]) -> torch.Tensor:
            return torch.tensor(
                [r + [NO_SEASON] * (width - len(r)) for r in rows], dtype=torch.long
            )

        return cls(
            sow_days=_pad(sow_rows),
            n_days=n_days,
            harvest_days=None if harvest_rows is None else _pad(harvest_rows),
            start_date=start,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def annual(
        cls,
        sow_doy: int | Sequence[int],
        n_years: int,
        start_date: date | datetime | str,
        n_days: int,
        batch_size: int = 1,
        first_year_offset: int = 0,
        **kwargs: object,
    ) -> "CropCalendar":
        """Build a calendar that sows once per calendar year.

        Args:
            sow_doy: Day-of-year to sow on. A single ``int`` repeats every
                year; a sequence of length ``n_years`` gives a different
                day-of-year per year (the common case — sowing dates drift
                with the weather).
            n_years: Number of consecutive sowings.
            start_date: Calendar date of simulated day ``0``.
            n_days: Length ``T`` of the weather series.
            batch_size: Batch size ``B``; the same schedule is shared by
                every element.
            first_year_offset: Number of calendar years to skip before the
                first sowing — use ``1`` to leave a full spin-up year in
                which only the soil balances evolve.
            **kwargs: Forwarded to `CropCalendar` (``harvest_rule``,
                ``after_maturity_days``).

        Returns:
            A validated `CropCalendar`.

        Raises:
            ValueError: If ``sow_doy`` is a sequence of the wrong length,
                or if any sowing day falls outside the weather series.

        Example:
            >>> cal = CropCalendar.annual(
            ...     sow_doy=285, n_years=30, start_date="2000-01-01",
            ...     n_days=30 * 365,
            ... )
        """
        doys = (
            [int(sow_doy)] * n_years
            if isinstance(sow_doy, int)
            else [int(d) for d in sow_doy]
        )
        if len(doys) != n_years:
            raise ValueError(
                f"sow_doy sequence must have n_years = {n_years} entries; "
                f"got {len(doys)}"
            )
        start = _as_date(start_date)
        dates: list[date] = []
        for i, doy in enumerate(doys):
            year = start.year + first_year_offset + i
            dates.append(date(year, 1, 1) + timedelta(days=doy - 1))
        out_of_range = [d for d in dates if not 0 <= (d - start).days < n_days]
        if out_of_range:
            raise ValueError(
                "these sowing dates fall outside the "
                f"{n_days}-day weather series starting {start}: "
                f"{[d.isoformat() for d in out_of_range]}"
            )
        return cls.from_dates(
            dates, start_date=start, n_days=n_days, batch_size=batch_size, **kwargs
        )
