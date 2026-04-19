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

    Attributes:
        data: The underlying weather tensor of shape ``[B, T, C]``.

    Raises:
        ValueError: If ``data`` is not 3-D or does not carry exactly
            `N_WEATHER_CHANNELS` channels.
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
        """Leading batch dimension ``B``."""
        return self.data.shape[0]

    @property
    def n_days(self) -> int:
        """Number of daily time steps ``T``."""
        return self.data.shape[1]

    def day(self, t: int) -> dict[str, torch.Tensor]:
        """Return a dict of named channels for day ``t``.

        Args:
            t: Time index in ``[0, T)``.

        Returns:
            Dict mapping channel name to a ``[B]`` tensor.
        """
        slice_t = self.data[:, t, :]
        return {name: slice_t[:, i] for i, name in enumerate(WEATHER_CHANNELS)}

    def channel(self, name: str) -> torch.Tensor:
        """Return the full ``[B, T]`` trajectory of a named channel.

        Args:
            name: Channel name (see `WEATHER_CHANNELS`).

        Returns:
            A ``[B, T]`` tensor view of the requested channel.

        Raises:
            KeyError: If ``name`` is not a known weather channel.
        """
        try:
            idx = WEATHER_CHANNELS.index(name)
        except ValueError as e:
            raise KeyError(f"unknown weather channel: {name!r}") from e
        return self.data[:, :, idx]

    def to(self, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> "WeatherDriver":
        """Return a new `WeatherDriver` cast/moved to ``dtype``/``device``.

        Args:
            dtype: Target tensor dtype, or ``None`` to leave unchanged.
            device: Target torch device, or ``None`` to leave unchanged.

        Returns:
            A new `WeatherDriver` wrapping the cast tensor.
        """
        return WeatherDriver(self.data.to(dtype=dtype, device=device))
