# Changelog

## v1.1.0 - 2026-07-28

**New Features**:

-   Water-limited (`IOPT=2`), water-and-N-limited, and water-and-NPK-limited
    (`IOPT=3`/`4`) production modes, including automatic-irrigation triggers,
    validated against the SIMPLACE Lintul5 reference to within ~1-3% MAE.
-   `torchcrop.calibration` — a constraint-aware, transform-based parameter
    calibration framework (`CalibrationManager`, `ParameterSpec`,
    `ConstraintGroup`) supporting bounds, dtype, table-ordinate, and
    ordering constraints for gradient-based fitting of crop parameters.
-   `torchcrop.nn.hybrid` — a `HybridManager` wiring layer with
    `ResidualSpec`/`ResidualHead` and `default_slots()` for declaratively
    injecting neural residual corrections into named points of the
    mechanistic pipeline.
-   New process modules: `Co2Transpiration` (CO₂ effect on potential
    transpiration), `HeatStressOnGrain` / `HeatStressOnLeafSenescence`,
    `SoilNutrients` (soil NPK availability and mineralisation), and
    `StemDynamics` (stem biomass and reserves).
-   External irrigation (`irrigation: [B, T]`) and fertiliser
    (`fertilizer: [B, T, 3]`) inputs to `Lintul5Model.forward`, overriding
    the internal table-driven application on a per-day basis.
-   23 bundled crop presets (`torchcrop/parameters/crop_data/*.yaml`),
    listed via `torchcrop.available_crops()` and loaded with
    `CropParameters(crop_name=...)`.
-   Six worked example notebooks under `docs/examples/` covering potential,
    water-limited, water-and-nutrient-limited, calibration, hybrid, and
    daily-timestep low-level API workflows, plus a Brandenburg weather/soil
    dataset used across the examples.

**Improvements**:

-   Numerical parity fixes against the SIMPLACE reference: water-limited
    root-front (`WDR`) coupling, IOPT=3/4 NNI/NPKI component-lag alignment,
    and IOPT=1 sowing→emergence window handling.
-   Expanded `ModelState`/`DiagnosticState` coverage and `utils.io`,
    `utils.validation`, and `utils.vis` helpers.
-   MathJax-rendered equations and a refreshed docs theme/logo.

## v1.0.0 - 2026-04-19

**New Features**:

-   Full differentiable Lintul5 potential-production forward model:
    astronomical/daylength, phenology, irradiation, evapotranspiration,
    water balance, photosynthesis, partitioning, leaf/root dynamics, and
    NPK nutrient demand, each as an independent `nn.Module`.
-   `Lintul5Model`, `SimulationEngine`, `ModelState`/`DiagnosticState`, and
    `CropParameters`/`SoilParameters`/`SiteParameters` public API.
-   Optional hybrid-ML layer (`torchcrop.nn`): `NeuralResidual`,
    `LearnedStressFactor`, `ParameterNet`.
-   Differentiable primitives (`functions/`): piecewise-linear
    interpolation (AFGEN replacement), smoothing helpers, and FST function
    ports (`LIMIT`, `INSW`, `NOTNUL`).

## v0.0.1 - 2026-04-19

**New Features**:

-   Initial project scaffold: package layout, `SoilParameters`, and
    `ModelState` tensor containers.
