"""Long-term (multi-year, multi-season) simulation for torchcrop.

Wraps the single-season `torchcrop.Lintul5Model` in a season scheduler so
one call can simulate decades of a field: sow, grow, harvest, leave
fallow, sow again — with sowing dates and management that change from year
to year, and soil water and nutrient state carried continuously across the
boundaries.

The package operates on what happens *between* days: a per-day sowing
signal that arms the model's seed-reserve bootstrap each season, and a
harvest reset that clears the crop from the field. The day step itself is
the ordinary torchcrop one, so a one-season run through
`LongTermSimulator` reproduces `Lintul5Model.forward` exactly.

Example:
    >>> import torchcrop
    >>> from torchcrop.longterm import (
    ...     CropCalendar, LongTermSimulator, ManagementEvent, ManagementSchedule,
    ... )
    >>> model = torchcrop.Lintul5Model(crop, soil, site)
    >>> calendar = CropCalendar.annual(
    ...     sow_doy=285, n_years=30, start_date="1990-01-01",
    ...     n_days=weather.n_days,
    ... )
    >>> schedule = ManagementSchedule(
    ...     fertilizer=[ManagementEvent(amount=6.0, days_after_sowing=180)]
    ... )
    >>> out = LongTermSimulator(model).run(weather, calendar, schedule)
    >>> out.seasons.yield_          # [B, 30] storage-organ yield per season
"""

from torchcrop.longterm.calendar import NO_SEASON, CropCalendar
from torchcrop.longterm.carryover import (
    CarryOverPolicy,
    initial_lai,
    reset_at_harvest,
    seed_reserve_biomass,
)
from torchcrop.longterm.management import ManagementEvent, ManagementSchedule
from torchcrop.longterm.simulator import (
    LongTermOutput,
    LongTermSimulator,
    SeasonRecord,
)

__all__ = [
    "NO_SEASON",
    "CarryOverPolicy",
    "CropCalendar",
    "LongTermOutput",
    "LongTermSimulator",
    "ManagementEvent",
    "ManagementSchedule",
    "SeasonRecord",
    "initial_lai",
    "reset_at_harvest",
    "seed_reserve_biomass",
]
