# torchcrop

[![image](https://img.shields.io/pypi/v/torchcrop.svg)](https://pypi.python.org/pypi/torchcrop)
[![image](https://img.shields.io/conda/vn/conda-forge/torchcrop.svg)](https://anaconda.org/conda-forge/torchcrop)

**A PyTorch-native library for hybrid crop modeling.**

`torchcrop` is a fully differentiable reimplementation of the
[Lintul5](https://models.pps.wur.nl/node/450) crop growth model (Wolf, 2012).
Every step of the simulation — from sowing to harvest — produces valid
`torch.autograd` gradients, so mechanistic crop processes can be combined
seamlessly with learnable components (neural residuals, learned stress
responses, parameter networks) and calibrated end-to-end with standard
`torch.optim` optimizers.

- Free software: MIT License
- Documentation: https://geonextgis.github.io/torchcrop

## Features

- **Differentiable Lintul5** — daily forward-Euler simulation of phenology,
  radiation interception, photosynthesis, partitioning, leaf and root
  dynamics, water balance and NPK uptake, all as `torch.nn.Module`s.
- **Batch-first** — every state, parameter and driver carries a leading
  batch dimension `[B, ...]` so that many sites, years, or parameter sets
  can be simulated in parallel on GPU.
- **Hybrid modeling hooks** — drop-in `NeuralResidual`, `LearnedStressFactor`
  and `ParameterNet` modules that plug into the mechanistic pipeline.
- **Smooth options** — stage-based branching (`DVS < 1`, maturity, etc.) can
  be switched between hard `torch.where` and sigmoid blends for second-order
  smoothness.
- **Gradient-checked primitives** — differentiable AFGEN-style interpolation
  and soft FST helpers (`LIMIT`, `INSW`, `NOTNUL`) pass `torch.autograd.gradcheck`.

## Installation

```bash
pip install -e ".[dev]"
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

```python
import torch.nn as nn
from torchcrop import Lintul5Model, CropParameters

crop = CropParameters().to(dtype=torch.float64)
crop.rue = nn.Parameter(torch.tensor(3.0, dtype=torch.float64))

model = Lintul5Model(crop_params=crop).double()
optimizer = torch.optim.Adam([crop.rue], lr=1e-2)

for _ in range(50):
    optimizer.zero_grad()
    out = model(weather.to(torch.float64), start_doy=60)
    loss = ((out.yield_ - observed_yield) ** 2).mean()
    loss.backward()
    optimizer.step()
```

### Hybrid modeling

Inject a neural residual on top of the mechanistic photosynthesis output:

```python
from torchcrop.nn import NeuralResidual

residual = NeuralResidual(input_dim=8, output_dim=1, hidden_dim=32, scale=0.1)
hybrid = torchcrop.Lintul5Model(
    residual_modules={"photosynthesis": residual},
)
```

All parameters — mechanistic and neural — are surfaced by
`hybrid.parameters()` and can be optimized jointly.

## Package layout

```
torchcrop/
├── model.py                   # Lintul5Model(nn.Module)
├── engine.py                  # SimulationEngine time-stepping loop
├── config.py                  # RunConfig
├── parameters/                # CropParameters / SoilParameters / SiteParameters
├── drivers/weather.py         # WeatherDriver [B, T, C]
├── states/model_state.py      # ModelState tensor container
├── processes/                 # Biophysical processes (astro, phenology,
│                              # irradiation, evapotranspiration, water_balance,
│                              # photosynthesis, partitioning, leaf_dynamics,
│                              # root_dynamics, nutrient_demand, stress)
├── functions/                 # Differentiable primitives (AFGEN, FST, smoothing)
├── nn/                        # NeuralResidual, LearnedStressFactor, ParameterNet
└── utils/                     # I/O, visualisation, validation helpers
```

## Development

```bash
pytest                    # run the test suite
flake8 torchcrop tests    # lint
black torchcrop tests     # format
pre-commit run --all-files
```

## References

- Wolf, J. (2012). *User guide for LINTUL5*. Wageningen UR.
  https://models.pps.wur.nl/system/files/LINTUL5-report-vs1_0.zip
- WUR-AI. *diffWOFOST — Differentiable WOFOST crop model*.
  https://github.com/WUR-AI/diffWOFOST
- The SIMPLACE Java reference implementation of Lintul5 (read-only) lives
  under `simplace/sim/components/models/lintul5/` in this repository.
