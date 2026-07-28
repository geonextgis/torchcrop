# SIMPLACE vs torchcrop — IOPT 1–4 comparison

DVS-axis comparison of the SIMPLACE Lintul5 reference implementation against
`torchcrop`, for all four production modes, over the 18-location Brandenburg
winter-wheat dataset (sowing 2020-09-26, maturity ≈ 2021-07-01).

Regenerate everything with:

```bash
python run_comparison.py
```

The script runs a stock `Lintul5Model` for all four modes through the public
`torchcrop` API — no wrappers or monkey-patching.

## Outputs (`figures/`)

| File | Content |
| --- | --- |
| `iopt{1,2,3,4}_ensemble_vs_dvs.png` | 12 key variables vs DVS; median + inter-quartile band across all 18 locations, SIMPLACE (solid) vs torchcrop (dashed), with per-panel MAE |
| `iopt{1,2,3,4}_locations_vs_dvs.png` | Same 12 variables for 3 representative sites spanning the water-holding range (colour = location, solid = SIMPLACE, dashed = torchcrop) |
| `summary_all_iopt.png` | Normalised-MAE heat map (variable × IOPT) and a 1:1 scatter of all dry-matter pools |
| `error_summary.csv` | Bias, MAE, RMSE and nMAE for every variable × IOPT |

All figures are written as 300 dpi PNG.

## Variable mapping

Verified against the SIMPLACE solution XML output rules
(`Lintul5SplitDemandSupply_WithCO2Transpiration.sol.xml`) and `Lintul5.java`.

| SIMPLACE column | SIMPLACE rule | torchcrop |
| --- | --- | --- |
| `DevStage` (x-axis) | `Phenology.sDVS` | `state.dvs` |
| `LAI` | `Biomass.sLAI` | `state.lai` |
| `AGBiomass` | `Biomass.sTAGB` | `wlv + wst + wso + wlvd + wstd` |
| `Yield` | `Biomass.sWSO` | `state.wso` |
| `WGLV` | `Biomass.sWLVG` | `state.wlv` |
| `WST` | `Biomass.sWST` | `state.wst` |
| `WRT` | `Biomass.sWRT` | `state.wrt` |
| `SMACT` | `WaterBalance.sSMACT` | `diag.smact` |
| `TRANRF` | `WaterBalance.TRANRF` | `diag.tranrf` |
| `NMINT` | `NPKSupply.sNMINT` | `state.nmint` |
| `NUPT` | `NPKDemand.sNUPTT` | `state.nuptr_cum` |
| `NNI` | `NPKDemand.NNI` | `diag.nni` |
| `NPKI` | `NPKDemand.NPKI` | `diag.nstress` |

`AGBiomass` is **not** the notebooks' `output.biomass` (that is TAGBG =
WLVG+WST+WSO); SIMPLACE's `sTAGB` additionally includes dead leaf and stem
material (`Lintul5.java:1097,1104`).

## Alignment conventions

Established empirically against the reference DVS series (max |ΔDVS| ≈ 3e-5
over the cycle):

* **Time index** — torchcrop `states[i]` ↔ SIMPLACE daily row `i`, so the
  *last* state entry is dropped (the trajectory has length `T+1`).
  `diagnostics[t]` maps directly onto row `t`.
* **First crop cycle only** — the SIMPLACE runs span two calendar years and
  contain a *second* winter wheat crop that starts after the first matures, so
  `DevStage` is not monotonic over the full file. Only sowing → first
  `DVS ≥ 2` is compared.
* **Growing window** — curves are drawn for `DVS > 0` (emergence onward). The
  pre-emergence period sits entirely at `DVS = 0` and cannot be resolved on a
  development-stage axis.
* **`Yield` in the daily CSV resets to 0** on the final rows (after the first
  crop is harvested); the cycle cut-off avoids that artefact.

## Management

SIMPLACE applies the `FERNTAB`/`FERPTAB`/`FERKTAB` tables from the shared
`mgm_test.xml` in **every** run, IOPT 1–4 alike (2 + 8 + 6 + 5 g N m⁻², plus
3 g P and 8 g K at sowing). The potential and water-limited example notebooks
omit fertiliser because nutrients cannot limit growth in those modes, but the
soil mineral-N pool still records the applications — so this comparison applies
the fertiliser in all four modes to keep `NMINT`/`NUPT` like-for-like. Doing so
leaves the IOPT 1/2 crop variables bit-for-bit unchanged, confirming nutrients
are genuinely non-limiting there.

Irrigation (`IRRTAB`, 20 mm spikes every 15 days) is only enabled in the
SIMPLACE IOPT 3/4 runs (`IRRI=2`); IOPT 1/2 run with `IRRI=0` and receive none.

## Potential mode (IOPT 1)

`WaterBalance.java` special-cases potential production:

```java
if (IOPT == 1)  ActualTranspiration = PotentialTranspiration;
else            ActualTranspiration = max(0, min(WAVT, RDRY*RWET*PTRAN));
TRANRF = ActualTranspiration / PotentialTranspiration;
```

The crop transpires at the *unreduced* potential rate — bypassing both the
drought/oxygen factors and the available-water ceiling — so `TRANRF` is
identically 1 and the root zone may be drawn below wilting point, while the
soil water balance itself still runs and the profile dries down rain-fed.

`torchcrop.processes.water_balance.WaterBalance` implements this branch
directly: `Lintul5Model` passes `crop_params.iopt` through, and the module
selects between the potential and water-limited transpiration with
`torch.where`, so the switch stays batched and differentiable. The comparison
therefore runs a stock `Lintul5Model` for every mode — no wrappers, and no
`irri = 1` workaround.

Measured effect on the IOPT 1 comparison:

| torchcrop config | LAI | Yield | SMACT | TRANRF |
| --- | --- | --- | --- | --- |
| `irri = 1`, no IOPT branch (old workaround) | 0.07 % | 0.01 % | 4.21 % | 0.00 % |
| `irri = 0`, no IOPT branch | 1.83 % | 3.40 % | 0.75 % | 17.39 % |
| `irri = 0` + **IOPT branch in the model** | 0.07 % | 0.01 % | **0.20 %** | 0.00 % |

## Results

All crop-growth variables agree closely across every production mode:

| | IOPT 1 | IOPT 2 | IOPT 3 | IOPT 4 |
| --- | --- | --- | --- | --- |
| LAI | 0.07 % | 0.09 % | 0.10 % | 0.10 % |
| Above-ground biomass | 0.01 % | 0.09 % | 0.22 % | 0.21 % |
| Storage organ (yield) | 0.01 % | 0.12 % | 0.35 % | 0.33 % |
| Root-zone soil moisture | 0.20 % | 0.16 % | 0.45 % | 0.45 % |
| Soil mineral N | 0.76 % | 0.71 % | 0.72 % | 0.72 % |
| N nutrition index | 0.00 % | 0.00 % | 2.30 % | 2.30 % |

nMAE = mean absolute error as a percentage of the SIMPLACE range.

The largest genuine residuals are the late grain-fill divergences in `NNI`/
`NPKI` (DVS > 1.5, IOPT 3/4) and the corresponding ~0.3 % yield gap — the
known senescence-N-loss and heat-stress coupling-lag artefacts, visible as the
small dashed/solid separation in the last panels of the IOPT 3 and 4 figures.
