"""Compare SIMPLACE Lintul5 reference output with torchcrop, for IOPT 1-4.

Runs the four production-mode configurations used by the worked example
notebooks (``01_potential``, ``02_water_limited``,
``03_water_and_nutrient_limited``) over the 18-location Brandenburg
dataset, reads the matching SIMPLACE ``*_Daily.csv`` reference files, and
writes DVS-axis comparison figures plus an error summary.

The script is read-only with respect to the model: it imports the public
torchcrop API and never mutates package code.

Run from this directory::

    python run_comparison.py
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from torchcrop import (
    CropParameters,
    Lintul5Model,
    SiteParameters,
    SoilParameters,
    WeatherDriver,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data" / "brandenburg" / "torchcrop"
SIMPLACE_DIR = HERE.parent / "PyTorch_Test"
OUT_DIR = HERE / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calendar start of the weather series (used to turn management dates into
# day indices, exactly as the example notebooks do).
SIM_START = pd.Timestamp("2020-01-01")


# --------------------------------------------------------------------- #
# Dataset (verbatim from the example notebooks)
# --------------------------------------------------------------------- #
class TorchCropDataset(Dataset):
    """Per-location weather / soil / site reader for torchcrop."""

    WEATHER_COLS = [
        "Date",
        "TempMean",
        "TempMin",
        "TempMax",
        "Radiation",
        "Precipitation",
        "VapPressure",
        "Windspeed",
    ]

    SOIL_MAP = {
        "SMDRY": "wcad",
        "SMW": "wcwp",
        "SMFC": "wcfc",
        "SMO": "wcst",
        "CRAIRC": "crairc",
        "SMI": "wci",
        "SMLOWI": "wci_lower",
        "RDMSO": "rdmso",
        "RUNFR": "runfr",
        "CFEV": "cfev",
        "KSUB": "ksub",
        "NMINS": "nmini",
        "PMINS": "pmini",
        "KMINS": "kmini",
        "RTNMINS": "rtnmins",
        "RTPMINS": "rtpmins",
        "RTKMINS": "rtkmins",
    }

    SITE_MAP = {
        "LATITUDE": "latitude",
        "ALTITUDE": "altitude",
        "IDPL": "idpl",
        "CO2": "co2",
    }

    def __init__(self, weather_dir, soil_dir, site_dir, dtype=torch.float32):
        super().__init__()
        self.weather_dir = Path(weather_dir)
        self.dtype = dtype
        self.soil_data = pd.read_csv(
            os.path.join(soil_dir, "soil.csv")
        ).set_index("location")
        self.site_data = pd.read_csv(
            os.path.join(site_dir, "site.csv")
        ).set_index("location")
        self.locations = self.soil_data.index.intersection(self.site_data.index)

    def __len__(self):
        return len(self.locations)

    def _load_weather(self, location):
        df = pd.read_csv(
            self.weather_dir / f"{location}.csv", parse_dates=["Date"]
        )
        df = df[self.WEATHER_COLS].copy()
        # SIMPLACE drives phenology with TMPA = (TMIN+TMAX)/2.
        df["TempMean"] = (df["TempMin"] + df["TempMax"]) / 2.0
        df["Radiation"] = df["Radiation"] / 1000.0  # kJ -> MJ m-2 d-1
        df["Date"] = df["Date"].dt.dayofyear
        return torch.as_tensor(df.values, dtype=self.dtype)

    def __getitem__(self, idx):
        location = self.locations[idx]
        weather = self._load_weather(location)
        srow = self.soil_data.loc[location]
        soil = {field: float(srow[col]) for col, field in self.SOIL_MAP.items()}
        trow = self.site_data.loc[location]
        site = {field: float(trow[col]) for col, field in self.SITE_MAP.items()}
        return {"weather": weather, "soil": soil, "site": site}


def collate_torchcrop(batch, dtype=torch.float32):
    """Collate samples into batched torchcrop driver/parameter objects."""
    weather = torch.stack([s["weather"] for s in batch], dim=0).to(dtype)
    weather = WeatherDriver(weather)

    soil_kwargs = {
        f: torch.tensor([s["soil"][f] for s in batch], dtype=dtype)
        for f in batch[0]["soil"]
    }
    site_kwargs = {
        f: torch.tensor([s["site"][f] for s in batch], dtype=dtype)
        for f in batch[0]["site"]
    }
    return weather, SoilParameters(**soil_kwargs), SiteParameters(**site_kwargs)


# --------------------------------------------------------------------- #
# Management schedules (verbatim from notebook 03)
# --------------------------------------------------------------------- #
def day_index(date):
    """0-based day index of a calendar date within the weather series."""
    return (pd.Timestamp(date) - SIM_START).days


def build_irrigation(B, T):
    """External irrigation ``[B, T]`` [mm d-1] mirroring ``IRRTAB``.

    20 mm spikes every 15 days through the 2021 season. Only the IOPT=3/4
    SIMPLACE runs enable irrigation (``IRRI=2``); IOPT=1/2 run with
    ``IRRI=0`` and receive none.
    """
    irrigation = torch.zeros(B, T)
    for date in pd.date_range("2021-04-01", "2021-07-30", freq="15D"):
        irrigation[:, day_index(date)] = 20.0
    return irrigation


def build_fertilizer(B, T):
    """External fertiliser ``[B, T, 3]`` [g m-2 d-1] as ``(N, P, K)``.

    Mirrors ``FERNTAB``/``FERPTAB``/``FERKTAB`` in the SIMPLACE
    ``mgm_test.xml``: N dressings of 2 + 8 + 6 + 5 g N m-2 and a single
    P/K dressing (3 g P, 8 g K) at sowing.

    These tables live in the shared management file and are therefore
    applied by **every** SIMPLACE run, IOPT 1-4 alike. The potential and
    water-limited example notebooks omit fertiliser because nutrients
    cannot limit growth in those modes, but the soil mineral-N pool still
    records the applications - so the comparison run applies them in all
    four modes to keep NMINT/NUPT like-for-like.
    """
    fertilizer = torch.zeros(B, T, 3)
    fert_plan = [
        ("2020-09-26", 2.0, 3.0, 8.0),
        ("2021-03-01", 8.0, 0.0, 0.0),
        ("2021-03-31", 6.0, 0.0, 0.0),
        ("2021-04-30", 5.0, 0.0, 0.0),
    ]
    for date, n, p, k in fert_plan:
        fertilizer[:, day_index(date), :] = torch.tensor([n, p, k])
    return fertilizer


# --------------------------------------------------------------------- #
# Run configurations
# --------------------------------------------------------------------- #
# ``irri``: torchcrop soil_params.irri (None = leave at the rain-fed
# default, matching the SIMPLACE runs, all of which use IRRI=0 or IRRI=2).
# Potential production needs no special handling: ``WaterBalance`` reads
# ``crop_params.iopt`` and applies the unreduced-transpiration branch itself.
RUNS = {
    1: dict(
        iopt=1.0,
        irri=None,
        irrigate=False,
        simplace="TorchCrop_Test_IOPT_1",
        title="IOPT 1 - Potential production",
    ),
    2: dict(
        iopt=2.0,
        irri=None,
        irrigate=False,
        simplace="TorchCrop_Test_IOPT_2",
        title="IOPT 2 - Water-limited production",
    ),
    3: dict(
        iopt=3.0,
        irri=None,
        irrigate=True,
        simplace="TorchCrop_Test_IOPT_3_NormalIrrigation",
        title="IOPT 3 - Water- and N-limited production",
    ),
    4: dict(
        iopt=4.0,
        irri=None,
        irrigate=True,
        simplace="TorchCrop_Test_IOPT_4_NormalIrrigation",
        title="IOPT 4 - Water- and NPK-limited production",
    ),
}


def run_torchcrop(dataset, cfg):
    """Run torchcrop for one IOPT configuration over all locations."""
    crop_params = CropParameters(crop_name="wheat")
    crop_params.iopt = torch.tensor(cfg["iopt"])

    samples = [dataset[i] for i in range(len(dataset))]
    weather, soil_params, site_params = collate_torchcrop(samples)
    B, T = weather.batch_size, weather.n_days

    if cfg["irri"] is not None:
        soil_params.irri = torch.tensor(cfg["irri"])

    # Fertiliser is applied in every SIMPLACE run; irrigation only when
    # that run enables it (IRRI=2 for IOPT 3/4).
    fertilizer = build_fertilizer(B, T).to(DEVICE)
    irrigation = build_irrigation(B, T).to(DEVICE) if cfg["irrigate"] else None

    crop_params = crop_params.to(device=DEVICE)
    soil_params = soil_params.to(device=DEVICE)
    site_params = site_params.to(device=DEVICE)
    weather = weather.to(device=DEVICE)

    model = Lintul5Model(crop_params, soil_params, site_params).eval().to(DEVICE)
    with torch.no_grad():
        out = model(
            weather,
            start_doy=1,
            irrigation=irrigation,
            fertilizer=fertilizer,
        )

    state_names = out.states[0].field_names
    diag_names = out.diagnostics[0].field_names
    states = torch.stack([s.stack() for s in out.states], dim=1).cpu().numpy()
    diags = torch.stack(
        [d.stack() for d in out.diagnostics], dim=1
    ).cpu().numpy()
    return {
        "state": states,  # [B, T+1, C]
        "diag": diags,  # [B, T, C]
        "state_names": list(state_names),
        "diag_names": list(diag_names),
    }


# --------------------------------------------------------------------- #
# Variable mapping: SIMPLACE column  <->  torchcrop field
# --------------------------------------------------------------------- #
# ``source`` is "state" or "diag"; ``expr`` is a callable on the named
# torchcrop arrays. Verified against the SIMPLACE solution XML
# (Lintul5SplitDemandSupply_WithCO2Transpiration.sol.xml) output rules.
VARIABLES = [
    dict(
        key="LAI",
        simplace="LAI",
        label="Leaf area index (LAI)",
        unit="m$^2$ m$^{-2}$",
        source="state",
        fields=["lai"],
        expr=lambda f: f["lai"],
        group="canopy",
    ),
    dict(
        key="AGBiomass",
        simplace="AGBiomass",
        label="Total above-ground biomass (TAGB)",
        unit="g DM m$^{-2}$",
        source="state",
        fields=["wlv", "wst", "wso", "wlvd", "wstd"],
        # SIMPLACE sTAGB = (WLVG + WST + WSO) + WLVD + WSTD  (Lintul5.java:1097,1104)
        expr=lambda f: f["wlv"] + f["wst"] + f["wso"] + f["wlvd"] + f["wstd"],
        group="biomass",
    ),
    dict(
        key="Yield",
        simplace="Yield",
        label="Storage organ dry weight (WSO)",
        unit="g DM m$^{-2}$",
        source="state",
        fields=["wso"],
        expr=lambda f: f["wso"],
        group="biomass",
    ),
    dict(
        key="WGLV",
        simplace="WGLV",
        label="Green leaf dry weight (WLVG)",
        unit="g DM m$^{-2}$",
        source="state",
        fields=["wlv"],
        expr=lambda f: f["wlv"],
        group="biomass",
    ),
    dict(
        key="WST",
        simplace="WST",
        label="Stem dry weight (WST)",
        unit="g DM m$^{-2}$",
        source="state",
        fields=["wst"],
        expr=lambda f: f["wst"],
        group="biomass",
    ),
    dict(
        key="WRT",
        simplace="WRT",
        label="Root dry weight (WRT)",
        unit="g DM m$^{-2}$",
        source="state",
        fields=["wrt"],
        expr=lambda f: f["wrt"],
        group="biomass",
    ),
    dict(
        key="SMACT",
        simplace="SMACT",
        label="Root-zone soil moisture (SMACT)",
        unit="m$^3$ m$^{-3}$",
        source="diag",
        fields=["smact"],
        expr=lambda f: f["smact"],
        group="water",
    ),
    dict(
        key="TRANRF",
        simplace="TRANRF",
        label="Water-stress factor (TRANRF)",
        unit="-",
        source="diag",
        fields=["tranrf"],
        expr=lambda f: f["tranrf"],
        group="water",
    ),
    dict(
        key="NMINT",
        simplace="NMINT",
        label="Soil mineral N pool (NMINT)",
        unit="g N m$^{-2}$",
        source="state",
        fields=["nmint"],
        expr=lambda f: f["nmint"],
        group="nitrogen",
    ),
    dict(
        key="NUPT",
        simplace="NUPT",
        label="Cumulative N uptake (NUPTT)",
        unit="g N m$^{-2}$",
        source="state",
        fields=["nuptr_cum"],
        expr=lambda f: f["nuptr_cum"],
        group="nitrogen",
    ),
    dict(
        key="NNI",
        simplace="NNI",
        label="N nutrition index (NNI)",
        unit="-",
        source="diag",
        fields=["nni"],
        expr=lambda f: f["nni"],
        group="nitrogen",
    ),
    dict(
        key="NPKI",
        simplace="NPKI",
        label="NPK nutrition index (NPKI)",
        unit="-",
        source="diag",
        fields=["nstress"],
        expr=lambda f: f["nstress"],
        group="nitrogen",
    ),
]


def torchcrop_series(tc, var):
    """Extract one mapped variable as a ``[B, T]`` array aligned to SIMPLACE.

    Alignment was established empirically against the reference DVS series
    (max |ΔDVS| ~3e-5 over the cycle):

    * ``state`` trajectories have length ``T + 1`` (a leading initial
      condition). Index ``i`` of that trajectory corresponds to SIMPLACE
      daily row ``i``, so the **last** entry is dropped, not the first.
    * ``diag`` trajectories already have length ``T`` and index ``t`` maps
      directly onto SIMPLACE row ``t``.
    """
    names = tc[f"{var['source']}_names"]
    arr = tc[var["source"]]
    fields = {f: arr[:, :, names.index(f)] for f in var["fields"]}
    series = var["expr"](fields)
    if var["source"] == "state":
        series = series[:, :-1]
    return series


def load_simplace(cfg, locations):
    """Load SIMPLACE daily output for every location as ``[B, T]`` arrays."""
    frames = {}
    for loc in locations:
        path = SIMPLACE_DIR / cfg["simplace"] / f"{loc}_{loc}.csv_Daily.csv"
        frames[loc] = pd.read_csv(path)
    return frames


def first_cycle_slice(dvs):
    """Index range ``[start, stop)`` of the first sowing->maturity cycle.

    The SIMPLACE runs span two calendar years and contain a *second*
    winter-wheat crop that starts once the first has matured, so DevStage
    is not monotonic over the full file. Growth begins at sowing (the day
    the seed reserve appears, while DVS is still 0) and the cycle closes
    on the first day DVS reaches 2.
    """
    mature = np.where(dvs >= 2.0)[0]
    stop = (mature[0] + 1) if len(mature) else len(dvs)
    return 0, stop


# --------------------------------------------------------------------- #
# Resampling onto a common DVS axis
# --------------------------------------------------------------------- #
DVS_GRID = np.linspace(0.0, 2.0, 201)


def to_dvs_grid(dvs, y):
    """Resample a single-location series onto `DVS_GRID`.

    Within one crop cycle DVS rises monotonically from 0 (emergence) to 2
    (maturity), so a plain linear interpolation is well defined. Points
    beyond the location's final DVS are left as NaN rather than being
    held flat, so the ensemble statistics never extrapolate.
    """
    out = np.full(DVS_GRID.shape, np.nan)
    inside = (DVS_GRID >= dvs[0]) & (DVS_GRID <= dvs[-1])
    out[inside] = np.interp(DVS_GRID[inside], dvs, y)
    return out


def collect(tc, sim, locations, var):
    """Build ``[B, len(DVS_GRID)]`` SIMPLACE / torchcrop arrays for one variable.

    Both models are restricted to the first crop cycle and to the growing
    window (DVS > 0); the pre-emergence period sits entirely at DVS = 0 and
    cannot be resolved on a development-stage axis.
    """
    tc_full = torchcrop_series(tc, var)
    sim_out, tc_out = [], []
    for b, loc in enumerate(locations):
        df = sim[loc]
        dvs_sim = df["DevStage"].values
        _, stop = first_cycle_slice(dvs_sim)
        d = dvs_sim[:stop]
        grow = d > 0.0
        d = d[grow]
        sim_out.append(to_dvs_grid(d, df[var["simplace"]].values[:stop][grow]))
        tc_out.append(to_dvs_grid(d, tc_full[b, :stop][grow]))
    return np.array(sim_out), np.array(tc_out)


def metrics(sim_arr, tc_arr):
    """Bias, MAE, RMSE and normalised MAE over the shared DVS window."""
    ok = np.isfinite(sim_arr) & np.isfinite(tc_arr)
    s, t = sim_arr[ok], tc_arr[ok]
    if s.size == 0:
        return dict(bias=np.nan, mae=np.nan, rmse=np.nan, nmae_pct=np.nan)
    diff = t - s
    scale = np.abs(s).max()
    return dict(
        bias=float(diff.mean()),
        mae=float(np.abs(diff).mean()),
        rmse=float(np.sqrt((diff**2).mean())),
        nmae_pct=float(100.0 * np.abs(diff).mean() / scale) if scale > 0 else 0.0,
    )


# --------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------- #
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

C_SIM = "#1f4e79"   # SIMPLACE  - dark blue
C_TC = "#c1272d"    # torchcrop - red
NCOLS = 3

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.8,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
    }
)


def _save(fig, stem):
    """Write one figure as a high-resolution PNG."""
    png = OUT_DIR / f"{stem}.png"
    fig.savefig(png, dpi=300)
    plt.close(fig)
    print(f"  wrote {png.name}")


def _decorate(ax, var, cfg):
    ax.set_title(f"{var['label']}\n[{var['unit']}]", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_xlim(0, 2.05)
    ax.margins(y=0.08)
    # Anthesis marker: DVS = 1 separates vegetative from reproductive growth.
    ax.axvline(1.0, color="0.55", lw=0.7, ls=":", zorder=0)


def plot_ensemble(iopt, cfg, data, locations):
    """Median + inter-quartile band across all locations, SIMPLACE vs torchcrop."""
    nrows = int(np.ceil(len(VARIABLES) / NCOLS))
    fig, axes = plt.subplots(
        nrows, NCOLS, figsize=(4.4 * NCOLS, 2.9 * nrows), sharex=True
    )
    axes = axes.ravel()

    for ax, var in zip(axes, VARIABLES):
        sim_arr, tc_arr = data[var["key"]]
        for arr, color, ls, lbl in (
            (sim_arr, C_SIM, "-", "SIMPLACE"),
            (tc_arr, C_TC, "--", "torchcrop"),
        ):
            # Grid points below the earliest emergence DVS are NaN for every
            # location; nanmedian warns on those all-NaN columns, and the
            # resulting NaNs are simply skipped by matplotlib.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(arr, axis=0)
                lo = np.nanpercentile(arr, 25, axis=0)
                hi = np.nanpercentile(arr, 75, axis=0)
            ax.fill_between(DVS_GRID, lo, hi, color=color, alpha=0.16, lw=0)
            ax.plot(DVS_GRID, med, color=color, ls=ls, lw=1.9, label=lbl)
        _decorate(ax, var, cfg)
        m = metrics(sim_arr, tc_arr)
        ax.text(
            0.03,
            0.95,
            f"MAE {m['mae']:.3g}\n({m['nmae_pct']:.2f}% of range)",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="0.25",
            bbox=dict(fc="white", ec="0.8", lw=0.5, alpha=0.85, pad=2.2),
        )

    for ax in axes[len(VARIABLES):]:
        ax.axis("off")
    for ax in axes[len(VARIABLES) - NCOLS: len(VARIABLES)]:
        ax.set_xlabel("Development stage, DVS [-]")

    handles = [
        Line2D([0], [0], color=C_SIM, lw=2.2, ls="-"),
        Line2D([0], [0], color=C_TC, lw=2.2, ls="--"),
        matplotlib.patches.Patch(fc="0.6", alpha=0.25, ec="none"),
        Line2D([0], [0], color="0.55", lw=0.9, ls=":"),
    ]
    labels = [
        "SIMPLACE Lintul5 (reference)",
        "torchcrop (PyTorch)",
        "inter-quartile range across 18 locations",
        "anthesis (DVS = 1)",
    ]
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    note = ""
    fig.suptitle(
        f"SIMPLACE vs torchcrop - {cfg['title']}\n"
        f"Winter wheat, 18 Brandenburg locations, median and IQR vs DVS{note}",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.985])
    _save(fig, f"iopt{iopt}_ensemble_vs_dvs")


def plot_locations(iopt, cfg, data, locations, picks):
    """Per-location detail for a few sites: SIMPLACE solid vs torchcrop dashed."""
    nrows = int(np.ceil(len(VARIABLES) / NCOLS))
    fig, axes = plt.subplots(
        nrows, NCOLS, figsize=(4.4 * NCOLS, 2.9 * nrows), sharex=True
    )
    axes = axes.ravel()
    palette = ["#1b6ca8", "#e07b39", "#2e7d32"]

    for ax, var in zip(axes, VARIABLES):
        sim_arr, tc_arr = data[var["key"]]
        for c, b in enumerate(picks):
            ax.plot(DVS_GRID, sim_arr[b], color=palette[c], ls="-", lw=1.7)
            ax.plot(DVS_GRID, tc_arr[b], color=palette[c], ls="--", lw=1.7)
        _decorate(ax, var, cfg)

    for ax in axes[len(VARIABLES):]:
        ax.axis("off")
    for ax in axes[len(VARIABLES) - NCOLS: len(VARIABLES)]:
        ax.set_xlabel("Development stage, DVS [-]")

    handles = [Line2D([0], [0], color=palette[c], lw=2.2) for c in range(len(picks))]
    handles += [
        Line2D([0], [0], color="0.2", lw=2.0, ls="-"),
        Line2D([0], [0], color="0.2", lw=2.0, ls="--"),
    ]
    labels = [f"location {locations[b]}" for b in picks]
    labels += ["SIMPLACE (solid)", "torchcrop (dashed)"]
    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels), frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle(
        f"SIMPLACE vs torchcrop - {cfg['title']}\n"
        f"Per-location detail for {len(picks)} representative sites, vs DVS",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.985])
    _save(fig, f"iopt{iopt}_locations_vs_dvs")


def plot_summary(all_data, all_cfg, err_df):
    """Cross-IOPT overview: agreement heat map + 1:1 scatter for key variables."""
    fig = plt.figure(figsize=(15.5, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.24)

    # ---- Left: normalised-MAE heat map (variable x IOPT).
    ax = fig.add_subplot(gs[0, 0])
    keys = [v["key"] for v in VARIABLES]
    mat = np.array(
        [[err_df.loc[(i, k), "nmae_pct"] for i in sorted(all_cfg)] for k in keys]
    )
    vmax = float(np.nanpercentile(mat, 95)) or 1.0
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=max(vmax, 1e-9), aspect="auto")
    ax.set_xticks(range(len(all_cfg)))
    ax.set_xticklabels([f"IOPT {i}" for i in sorted(all_cfg)])
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([v["label"].split(" (")[0] for v in VARIABLES])
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            val = mat[r, c]
            ax.text(
                c, r, f"{val:.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if val > 0.6 * vmax else "0.15",
            )
    ax.set_title(
        "Mean absolute error, % of the SIMPLACE range\n"
        "(lower = closer agreement)", fontsize=11
    )
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="nMAE [%]")

    # ---- Right: 1:1 scatter of the biomass/canopy variables.
    ax2 = fig.add_subplot(gs[0, 1])
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    cols = {1: "#1f4e79", 2: "#2e7d32", 3: "#e07b39", 4: "#c1272d"}
    lo = hi = None
    for iopt in sorted(all_cfg):
        xs, ys = [], []
        for key in ("AGBiomass", "Yield", "WGLV", "WST", "WRT"):
            s, t = all_data[iopt][key]
            ok = np.isfinite(s) & np.isfinite(t)
            xs.append(s[ok])
            ys.append(t[ok])
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        sub = slice(None, None, 37)  # thin for legibility
        ax2.scatter(
            xs[sub], ys[sub], s=9, alpha=0.45, marker=markers[iopt],
            color=cols[iopt], edgecolors="none", label=f"IOPT {iopt}",
        )
        lo = np.nanmin(xs) if lo is None else min(lo, np.nanmin(xs))
        hi = np.nanmax(xs) if hi is None else max(hi, np.nanmax(xs))
    ax2.plot([lo, hi], [lo, hi], color="0.35", lw=1.1, ls="--", label="1:1")
    ax2.set_xlabel("SIMPLACE [g DM m$^{-2}$]")
    ax2.set_ylabel("torchcrop [g DM m$^{-2}$]")
    ax2.set_title(
        "Dry-matter pools, all locations and development stages\n"
        "(TAGB, WSO, WLVG, WST, WRT)", fontsize=11
    )
    ax2.grid(alpha=0.25, lw=0.6)
    ax2.legend(frameon=False, loc="upper left", fontsize=9)

    fig.suptitle(
        "SIMPLACE Lintul5 vs torchcrop - agreement across production modes "
        "(winter wheat, 18 Brandenburg locations)",
        fontsize=13.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "summary_all_iopt")


def main():
    dataset = TorchCropDataset(
        DATA_DIR / "weather", DATA_DIR / "soil", DATA_DIR / "site"
    )
    locations = list(dataset.locations)
    print(f"Device: {DEVICE}; {len(locations)} locations")

    # Three representative sites spanning the soil-water-holding range.
    soil = dataset.soil_data.loc[locations]
    order = np.argsort(soil["SMFC"].values)
    picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]

    all_data, rows = {}, []
    for iopt, cfg in RUNS.items():
        print(f"\n=== IOPT {iopt}: {cfg['title']} ===")
        tc = run_torchcrop(dataset, cfg)
        sim = load_simplace(cfg, locations)
        data = {v["key"]: collect(tc, sim, locations, v) for v in VARIABLES}
        all_data[iopt] = data

        for v in VARIABLES:
            m = metrics(*data[v["key"]])
            rows.append(dict(iopt=iopt, variable=v["key"], **m))
            print(
                f"  {v['key']:>10s}  bias {m['bias']:+.4g}  MAE {m['mae']:.4g}"
                f"  RMSE {m['rmse']:.4g}  nMAE {m['nmae_pct']:.2f}%"
            )

        plot_ensemble(iopt, cfg, data, locations)
        plot_locations(iopt, cfg, data, locations, picks)

    err_df = pd.DataFrame(rows).set_index(["iopt", "variable"])
    err_df.to_csv(OUT_DIR / "error_summary.csv")
    print(f"\n  wrote error_summary.csv")
    plot_summary(all_data, RUNS, err_df)
    print("\nDone.")


if __name__ == "__main__":
    main()
