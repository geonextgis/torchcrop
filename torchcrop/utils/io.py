"""I/O helpers for weather data and parameter files."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from torchcrop.drivers.weather import WEATHER_CHANNELS, WeatherDriver


def make_constant_weather(
    batch_size: int,
    n_days: int,
    davtmp: float = 15.0,
    tmin: float = 10.0,
    tmax: float = 20.0,
    irrad: float = 15.0,
    rain: float = 2.0,
    vp: float = 1.2,
    wind: float = 2.0,
    start_doy: int = 1,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> WeatherDriver:
    """Synthetic constant weather for quick tests and demos."""
    doy = torch.arange(n_days, dtype=dtype, device=device) + float(start_doy)
    doy = ((doy - 1) % 365) + 1
    channels = {
        "doy": doy,
        "davtmp": torch.full((n_days,), davtmp, dtype=dtype, device=device),
        "tmin": torch.full((n_days,), tmin, dtype=dtype, device=device),
        "tmax": torch.full((n_days,), tmax, dtype=dtype, device=device),
        "irrad": torch.full((n_days,), irrad, dtype=dtype, device=device),
        "rain": torch.full((n_days,), rain, dtype=dtype, device=device),
        "vp": torch.full((n_days,), vp, dtype=dtype, device=device),
        "wind": torch.full((n_days,), wind, dtype=dtype, device=device),
    }
    stacked = torch.stack([channels[name] for name in WEATHER_CHANNELS], dim=-1)
    data = stacked.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    return WeatherDriver(data)


def load_weather_csv(
    path: str | Path,
    columns: Sequence[str] | None = None,
    dtype: torch.dtype = torch.float32,
) -> WeatherDriver:
    """Load a single-site daily weather CSV.

    The CSV must contain the columns listed in
    :data:`~torchcrop.drivers.weather.WEATHER_CHANNELS` (lowercase) or an
    explicit ``columns`` argument may be passed mapping file columns to the
    expected order.
    """
    path = Path(path)
    if columns is None:
        columns = WEATHER_CHANNELS
    if len(columns) != len(WEATHER_CHANNELS):
        raise ValueError(
            f"expected {len(WEATHER_CHANNELS)} columns, got {len(columns)}"
        )

    with path.open() as fh:
        header = fh.readline().strip().split(",")
    idx = [header.index(c) for c in columns]
    raw = np.loadtxt(path, delimiter=",", skiprows=1, usecols=idx)
    if raw.ndim == 1:
        raw = raw[None, :]
    data = torch.tensor(raw, dtype=dtype).unsqueeze(0)  # [1, T, C]
    return WeatherDriver(data)
