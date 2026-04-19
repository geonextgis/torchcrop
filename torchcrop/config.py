"""Run-level configuration for torchcrop simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    """Simulation run configuration.

    Attributes:
        n_days: Number of daily time steps ``T`` in the simulation.
        start_doy: Day-of-year of the first simulated day. Used by the
            astronomical component to compute solar declination and daylength.
        dt: Integration step in days. Defaults to ``1.0`` — the Lintul5 model
            is formulated for daily forward-Euler integration.
        smooth: If ``True`` process modules use smooth (sigmoid-blend)
            replacements for stage-based branching; otherwise ``torch.where``
            is used.
        dtype: Tensor dtype for the simulation (``"float32"`` or
            ``"float64"``). Gradient checks require ``float64``.
        device: Device string (``"cpu"``, ``"cuda"``, ...).
    """

    n_days: int = 200
    start_doy: int = 1
    dt: float = 1.0
    smooth: bool = False
    dtype: str = "float32"
    device: str = "cpu"
