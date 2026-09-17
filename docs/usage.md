# Usage

To use torchcrop in a project:

```python
import torchcrop
```

## Long-term simulations: many seasons in one run

`Lintul5Model` simulates **one season per call**. For multi-decade studies —
climate scenarios, rotations, soil-fertility trends — use
`torchcrop.longterm`, which wraps the same model in a season scheduler.

The layer works *between* days: a sowing pulse that arms the model's
seed-reserve bootstrap each season, and a harvest reset that clears the crop
from the field. The day step itself is the ordinary torchcrop one, so a
one-season run through `LongTermSimulator` reproduces `Lintul5Model.forward`
bit-for-bit.

### A 30-year run with varying sowing dates

```python
import torchcrop
from torchcrop.longterm import (
    CropCalendar, LongTermSimulator, ManagementEvent, ManagementSchedule,
)

model = torchcrop.Lintul5Model(crop_params, soil_params, site_params)

# Sowing dates that move from year to year — pass the real dates.
calendar = CropCalendar.from_dates(
    ["1990-10-12", "1991-10-05", "1992-09-28", ...],
    start_date="1990-01-01",
    n_days=weather.n_days,
)
# Or, for a fixed day-of-year every year:
# calendar = CropCalendar.annual(
#     sow_doy=285, n_years=30, start_date="1990-01-01", n_days=weather.n_days,
# )

# Management anchored to each season's own sowing day, so it follows a
# sowing date that shifts by weeks between years.
schedule = ManagementSchedule(
    fertilizer=[
        ManagementEvent(amount=4.0, days_after_sowing=0),    # g N m-2
        ManagementEvent(amount=6.0, days_after_sowing=180),
    ],
    irrigation=[ManagementEvent(amount=25.0, days_after_sowing=200)],
)

out = LongTermSimulator(model).run(
    weather, calendar, management=schedule, collect=("dvs", "lai"),
)
out.seasons.yield_          # [B, 30] storage-organ yield per season
out.seasons.to_dataframe(calendar)   # tidy per-season table (needs pandas)
```

### What happens at harvest

`reset_at_harvest` returns the field to a bare, **pre-sowing** condition: the
phenology clocks, the biomass and canopy pools, the per-organ nutrients and the
rooting depth all go back to their sowing-day values. With `tsump = 0` the
emergence gates shut, so during the fallow the field neither grows nor
intercepts light and only the soil balances evolve, until the next sowing puts
a fresh seed reserve in the ground.

`CarryOverPolicy` decides what crosses the boundary:

| Group | Default | Alternative |
| --- | --- | --- |
| Crop state (clocks, biomass, canopy, roots) | always cleared | — |
| Soil water `wa` / `wa_lower` | always continuous | — |
| Soil minerals `nmin…kmint` | `carry` — continuous fertility | `reset` — independent seasons |
| Cumulative accumulators | `reset` — per-season budgets | `carry` — whole-run totals |
| Harvest residue N/P/K | `residue_fraction=0.0` — removed | up to `1.0` returned to the organic pools |

Rooting depth is reset to `rdi` with a **water-conserving** redistribution:
the water that is no longer rooted moves into the lower zone rather than
disappearing, the exact inverse of the `WDR` transfer applied as roots grow.

!!! note "Organic-pool depletion over decades"
    Lintul5 mineralises at `rtnmins · nmini` capped by the *current* pool,
    with no replenishment pathway. Under `soil_minerals="carry"` a continuous
    multi-decade run therefore exhausts its organic N and the crop ends up
    living on fertiliser alone. Either enable `residue_fraction`, fertilise
    explicitly, or use `soil_minerals="reset"` — but be aware that only the
    last of the three stays inside strictly SIMPLACE-equivalent territory.

### Memory and gradients over decades

A 50-year, `B = 4` run is ~18 250 day-steps. Two settings decide what that
costs:

* **`collect`** names the daily variables worth keeping; full per-day
  `ModelState` objects are never retained. Per-season summaries always come
  back regardless.
* **`truncate_bptt=True`** (default) detaches the state at each sowing day, so
  season *s* depends only on season *s*. Each season's yield still carries a
  full gradient back to its own sowing — what fitting a parameter against 30
  years of observed yields needs. `truncate_bptt=False` keeps one graph across
  the whole run, so gradients also cross season boundaries through the carried
  soil state (for example, through nitrogen one crop leaves for the next).

Truncation bounds the *dependency chain*, not by itself the memory: the
per-season summaries are differentiable, and each live tensor keeps its own
season's graph alive. Measured peak memory at `B = 4`:

| Horizon | `no_grad` | gradients on |
| --- | --- | --- |
| 10 years | 442 MB | — |
| 25 years | 455 MB | — |
| 50 years | 478 MB | ~12 GB |

The simulator itself is flat in the horizon; the growth is entirely retained
autograd graphs. So:

```python
# Scenario runs need no gradients — memory stops growing with the horizon.
with torch.no_grad():
    out = LongTermSimulator(model).run(weather, calendar, collect=("lai",))

# Calibration: run in chunks and backward per chunk, bounding memory to one.
state = None
for chunk_weather, chunk_calendar, observed in decades:
    out = sim.run(chunk_weather, chunk_calendar, initial_state=state)
    loss_fn(out.seasons.yield_, observed).backward()
    state = out.final_state.detach()
```

Chunked and single-call runs give the same answer (asserted in
`tests/test_longterm/test_multi_season.py`).

Truncation is a memory setting, not a physics setting: the simulated values are
identical either way.

## Hybrid modeling: neural residual corrections

`torchcrop` lets you attach **optional** neural residual corrections to
individual mechanistic quantities through a constraint-aware framework
(`torchcrop.nn.hybrid`). Each correction target is described by a
`ResidualSpec` whose `constraint` selects a projection that keeps the learned
term inside the quantity's natural geometry, so corrections preserve sign,
the $[0, 1]$ range of stress factors, and the partition-of-unity of the
biomass split:

| Constraint | Target kind | Projection | Guarantee |
| --- | --- | --- | --- |
| `rate_factor` | non-negative scalar / flux / rate constant | $base \cdot e^{\delta}$ | stays $\ge 0$ |
| `transfer` | flux between two conserved pools | $base \cdot e^{\delta}$ (one number, both legs) | mass conserved |
| `unit_interval` | factor in $(0, 1)$ | $\sigma(\operatorname{logit}(base) + \delta)$ | stays in $(0, 1)$ |
| `simplex` | fractions summing to $1$ | $\operatorname{softmax}(\log base + \delta)$ | $\sum = 1$ exactly |

The raw correction is the bounded $\delta = \text{scale} \cdot \tanh(\text{MLP}(\text{context}))$,
and each head's final layer is **zero-initialised**, so a model built with
residual specs but untrained reproduces the pure mechanistic trajectory.

### Enabling corrections

```python
import torch.nn.functional as F
from torchcrop import Lintul5Model
from torchcrop.nn import ResidualSpec, default_slots

# Option A: hand-pick the slots tied to your observables (recommended).
model = Lintul5Model(residual_specs=[
    ResidualSpec(
        "photosynthesis.gtotal", "rate_factor",
        context=("lai", "dvs", "davtmp", "tranrf", "nstress"), scale=0.15,
    ),
])

# Option B: the full recommended catalogue (only if every pathway is observable).
model = Lintul5Model(residual_specs=default_slots())
```

All residual parameters live in `model.parameters()`, so training is standard
PyTorch. Add the residual-magnitude penalty to the loss to anchor corrections
toward zero:

```python
out = model(weather)
loss = F.mse_loss(out.yield_, observed_yield) + 1e-3 * model.hybrid.penalty()
loss.backward()
```

Supported slot names: `"photosynthesis.gtotal"`, `"water.tranrf"`,
`"partitioning.aboveground"`, `"partitioning.fr"`, `"leaf.rdr"`,
`"phenology.dvs_rate"`.

!!! warning "Choosing which slots to enable matters more than the list itself"
    Enable **only** the slots whose pathway is constrained by an observable in
    your calibration data — for example, `"photosynthesis.gtotal"` if you
    observe biomass or yield, `"water.tranrf"` if you observe soil moisture or
    transpiration. Turning on **all** slots at once invites *identifiability*
    and *compensation* problems: several residuals — and the mechanistic
    parameters they shadow, such as `gtotal` versus a learnable RUE — become
    degenerate, so the optimiser can fit the data while learning physically
    meaningless corrections.

    A robust workflow:

    1. Enable a **minimal** set of slots, each tied to an observable.
    2. Calibrate the **mechanistic** parameters first, then add residuals.
    3. Regularise with `model.hybrid.penalty()` to keep corrections small.
