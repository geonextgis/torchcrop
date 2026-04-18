"""Weather driver tensor.

Shape ``[B, T, C_weather]`` with channels, in order:

    0 DOY     — day of year
    1 DAVTMP  — mean air temperature [°C]
    2 TMIN    — minimum air temperature [°C]
    3 TMAX    — maximum air temperature [°C]
    4 IRRAD   — daily global radiation [MJ m-2 d-1]
    5 RAIN    — precipitation [mm d-1]
    6 VP      — actual vapour pressure [kPa]
    7 WIND    — wind speed at 2 m [m s-1]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

WEATHER_CHANNELS: tuple[str, ...] = (
    "doy",
    "davtmp",
    "tmin",
    "tmax",
    "irrad",
    "rain",
    "vp",
    "wind",
)

N_WEATHER_CHANNELS = len(WEATHER_CHANNELS)


@dataclass
class WeatherDriver:
    """Thin wrapper around a ``[B, T, C]`` weather tensor.

    Provides named channel access and input validation. The underlying tensor
    is accessible as ``.data`` for direct use in process modules.
    """

    data: torch.Tensor

    def __post_init__(self) -> None:
        if self.data.dim() != 3:
            raise ValueError(
                f"weather tensor must be 3-D [B, T, C]; got {tuple(self.data.shape)}"
            )
        if self.data.shape[-1] != N_WEATHER_CHANNELS:
            raise ValueError(
                f"weather tensor must have {N_WEATHER_CHANNELS} channels; "
                f"got {self.data.shape[-1]}"
            )

    @property
    def batch_size(self) -> int:
        return self.data.shape[0]

    @property
    def n_days(self) -> int:
        return self.data.shape[1]

    def day(self, t: int) -> dict[str, torch.Tensor]:
        """Return a dict of named channels for day ``t``. Each value has shape ``[B]``."""
        slice_t = self.data[:, t, :]
        return {name: slice_t[:, i] for i, name in enumerate(WEATHER_CHANNELS)}

    def channel(self, name: str) -> torch.Tensor:
        """Return the full ``[B, T]`` trajectory of a named channel."""
        try:
            idx = WEATHER_CHANNELS.index(name)
        except ValueError as e:
            raise KeyError(f"unknown weather channel: {name!r}") from e
        return self.data[:, :, idx]

    def to(self, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> "WeatherDriver":
        return WeatherDriver(self.data.to(dtype=dtype, device=device))
