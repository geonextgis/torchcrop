# Welcome to TorchCrop

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/geonextgis/torchcrop/blob/main)
[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geonextgis/torchcrop/main?labpath=notebooks%2Fintro.ipynb)
[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/geonextgis/torchcrop/blob/main/notebooks/intro.ipynb)
[![PyPI Version](https://img.shields.io/pypi/v/torchcrop.svg)](https://pypi.org/project/torchcrop)
[![Downloads](https://static.pepy.tech/badge/torchcrop)](https://pepy.tech/project/torchcrop)
[![Documentation Status](https://github.com/geonextgis/torchcrop/workflows/docs/badge.svg)](https://geonextgis.github.io/torchcrop)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <a href="https://geonextgis.github.io/torchcrop">
    <img src="https://raw.githubusercontent.com/geonextgis/torchcrop/main/docs/assets/logo.png" alt="logo" width="250"/>
  </a>
</div>

## Introduction

`torchcrop` is a fully differentiable reimplementation of the
[LINTUL-5](https://models.pps.wur.nl/lintul-5-crop-growth-simulation-model-potential-water-limited-n-limited-and-npk-limited-conditions) crop growth model (Wolf, 2012).
Every step of the simulation — from sowing to harvest — produces valid
`torch.autograd` gradients, so mechanistic crop processes can be combined
seamlessly with learnable components (neural residuals, learned stress
responses, parameter networks) and calibrated end-to-end with standard
`torch.optim` optimizers.

## Features

- **Differentiable Lintul5** — daily forward-Euler simulation of phenology,
  radiation interception, photosynthesis, partitioning, leaf/stem/root
  dynamics, water balance, and NPK demand, uptake, and soil availability,
  all as `torch.nn.Module`s. Supports potential (`IOPT=1`), water-limited
  (`IOPT=2`), and water-and-NPK-limited (`IOPT=3`/`4`) production modes,
  with optional automatic irrigation.
- **Batch-first** — every state, parameter and driver carries a leading
  batch dimension `[B, ...]` so that many sites, years, or parameter sets
  can be simulated in parallel on GPU.
- **23 bundled crop presets** — `torchcrop.available_crops()` lists species
  (wheat, maize, rice, soybean, potato, sugar beet, …); load one via
  `CropParameters(crop_name="wheat")`.
- **Gradient-based calibration** — `torchcrop.calibration` provides a
  constraint-aware (bounds, dtype, table-ordinate, ordering), transform-based
  `CalibrationManager` for fitting crop parameters to observations.
- **Hybrid modeling hooks** — a `HybridManager` wiring layer accepts
  declarative `ResidualSpec`s (see `default_slots()`) to inject
  `NeuralResidual` corrections at named points in the pipeline, plus
  drop-in `LearnedStressFactor` and `ParameterNet` modules.
- **External irrigation/fertiliser** — pass explicit `irrigation: [B, T]`
  and `fertilizer: [B, T, 3]` schedules to `model(...)`, overriding the
  internal table-driven application on a per-day basis.
- **Smooth options** — stage-based branching (`DVS < 1`, maturity, etc.) can
  be switched between hard `torch.where` and sigmoid blends for second-order
  smoothness.
- **Gradient-checked primitives** — differentiable AFGEN-style interpolation
  and soft FST helpers (`LIMIT`, `INSW`, `NOTNUL`) pass `torch.autograd.gradcheck`.

## Installation

```bash
pip install torchcrop
```

## Quickstart

```python
import torch
import torchcrop
from torchcrop.utils.io import make_constant_weather

weather = make_constant_weather(batch_size=2, n_days=150)
model = torchcrop.Lintul5Model()
output = model(weather, start_doy=60)

print(output.yield_)        # [B] final storage-organ biomass (g m-2)
print(output.lai.shape)     # [B, T+1] LAI trajectory
print(output.dvs.shape)     # [B, T+1] development stage trajectory
```

### Gradient-based parameter calibration

`torchcrop.calibration` turns bounded crop/soil/site parameters into
optimizable latents, keeping them inside their physical range by
construction:

```python
import torch
from torchcrop import CalibrationManager, Lintul5Model, ParameterSpec

model = Lintul5Model(crop_params=torchcrop.CropParameters().to(dtype=torch.float64)).double()
manager = CalibrationManager(
    model, specs=[ParameterSpec(name="crop.scale_factor_rue", bounds=(0.5, 1.5))]
)
optimizer = torch.optim.Adam(manager.parameters(), lr=1e-2)

for _ in range(50):
    optimizer.zero_grad()
    manager.materialize()  # write the current latents into model.crop_params
    out = model(weather.to(torch.float64), start_doy=60)
    loss = ((out.yield_ - observed_yield) ** 2).mean()
    loss.backward()
    optimizer.step()
```

See `docs/examples/04_calibration/` for a full worked example.

### Hybrid modeling

Inject a neural residual on top of a named point in the mechanistic pipeline
via a declarative `ResidualSpec`:

```python
from torchcrop.nn import ResidualSpec

model = torchcrop.Lintul5Model(
    residual_specs=[
        ResidualSpec(
            "photosynthesis.gtotal",
            "rate_factor",
            context=("lai", "dvs", "davtmp", "tranrf", "nstress"),
            scale=0.15,
        ),
    ],
)
```

`torchcrop.nn.default_slots()` returns the recommended catalogue of
observable-tied slots (photosynthesis, water stress, partitioning, leaf
senescence); pass a hand-picked subset rather than the whole list unless
every pathway is observable. All parameters — mechanistic and neural — are
surfaced by `model.parameters()` and can be optimized jointly.

## Package layout

```
torchcrop/
├── model.py                   # Lintul5Model(nn.Module)
├── engine.py                  # SimulationEngine time-stepping loop
├── config.py                  # RunConfig
├── parameters/                # CropParameters / SoilParameters / SiteParameters
│                              # + 23 bundled crop presets (crop_data/*.yaml)
├── drivers/weather.py         # WeatherDriver [B, T, C]
├── states/model_state.py      # ModelState / DiagnosticState tensor containers
├── processes/                 # Biophysical processes (astro, phenology,
│                              # irradiation, evapotranspiration,
│                              # co2_transpiration, water_balance,
│                              # photosynthesis, partitioning, leaf_dynamics,
│                              # stem_dynamics, root_dynamics, nutrient_demand,
│                              # soil_nutrients, heat_stress, stress)
├── functions/                 # Differentiable primitives (AFGEN, FST, smoothing)
├── nn/                        # NeuralResidual, LearnedStressFactor, ParameterNet,
│                              # HybridManager / ResidualSpec wiring layer
├── calibration/                # CalibrationManager, ParameterSpec,
│                              # ConstraintGroup, transforms
└── utils/                     # I/O, visualisation, validation helpers
```

## Examples

Worked notebooks under `docs/examples/` (rendered into the docs site):

- `01_potential/` — potential production (winter wheat)
- `02_water_limited/` — water-limited production (winter wheat)
- `03_water_and_nutrient_limited/` — water + N and water + NPK limited production
- `04_calibration/` — gradient-based parameter calibration
- `05_hybrid/` — hybrid ML residual corrections (reserved, notebook in progress)
- `06_daily_timestep/` — low-level, day-by-day API usage
- `others/data_prep.ipynb` — preparing the Brandenburg example dataset

## Development

```bash
pytest                    # run the test suite
flake8 torchcrop tests    # lint
black torchcrop tests     # format
pre-commit run --all-files
```

## References

- Wolf, J. (2012). _User guide for LINTUL5_. Wageningen UR.
  https://models.pps.wur.nl/lintul-5-crop-growth-simulation-model-potential-water-limited-n-limited-and-npk-limited-conditions
- WUR-AI. _diffWOFOST — Differentiable WOFOST crop model_.
  https://github.com/WUR-AI/diffWOFOST