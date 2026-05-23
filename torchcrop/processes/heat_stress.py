"""Heat-stress effects on leaf senescence and on grain yield.

Two optional climate-change-related modules that are not part of the
core Lintul5 time loop but are routinely coupled to it:

* `HeatStressOnLeafSenescence` — a per-day module that accelerates
  leaf senescence when the daily maximum temperature exceeds a
  critical threshold. Its scalar output multiplies the relative
  death rate ``RDR`` in `torchcrop.processes.leaf_dynamics.LeafDynamics`.
* `HeatStressOnGrain` — a trajectory-level module that accumulates a
  daily heat-stress intensity over a development-stage window around
  anthesis and converts it into a multiplicative yield penalty.

Both modules are *stateless* with respect to crop constants: every
crop-specific value (critical temperatures, slopes, development-stage
thresholds) is read from `CropParameters` at call time. This keeps
the heat-stress configuration in the same per-crop parameter
container as the core Lintul5 parameters, so a single external crop
configuration file can drive a multi-crop simulation. The
``smooth``/``k_sharp`` constructor arguments are *behavioural* flags
(gradient smoothing), not crop constants, and therefore stay on the
module.

References:
    * Asseng, S., Foster, I., Turner, N. C. (2011). The impact of
      temperature variability on wheat yields. *Global Change
      Biology*.
    * Teixeira, E., Fischer, G., van Velthuizen, H., Walter, C.,
      Ewert, F. (2013). Global hot-spots of heat stress on
      agricultural crops due to climate change. *Agricultural and
      Forest Meteorology*, 170, 206–215.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.parameters.crop_params import CropParameters


class HeatStressOnLeafSenescence(nn.Module):
    r"""Heat-stress acceleration of leaf senescence.

    When the daily maximum temperature ``Tmax`` exceeds a critical
    threshold ``Tc`` *and* the crop has passed a critical development
    stage ``DVS_c``, the relative leaf death rate is multiplied by a
    factor that increases linearly with temperature:

    $$
    F_{\text{heat}}(T_{\max}, \text{DVS}) =
    \begin{cases}
        \max\!\left(0,\; m\,(T_{\max} - T_c) + F_{T_c}\right)
            & T_{\max} \ge T_c \;\wedge\; \text{DVS} \ge \text{DVS}_c \\
        1 & \text{otherwise}
    \end{cases}
    $$

    With the default values ``Tc = 34``, ``m = 0.5``, ``F_Tc = 3``,
    this is equivalent to the Asseng et al. (2011) expression
    ``F_heat = 4 − (1 − (Tmax − 34) / 2)``.

    The crop constants ``Tc``, ``F_Tc``, ``m`` and ``DVS_c`` are read
    from `CropParameters` — fields ``leaf_heat_temp_critical``,
    ``leaf_heat_factor_at_temp_critical``, ``leaf_heat_factor_slope``
    and ``leaf_heat_devstage_critical`` — so they can be loaded per
    crop from an external configuration file.

    Args:
        smooth: If ``True``, replace the hard ``Tmax``/``DVS``
            thresholds with sigmoid blends for smoother gradients.
        k_sharp: Sigmoid sharpness; ignored when ``smooth=False``.
    """

    def __init__(
        self,
        smooth: bool = False,
        k_sharp: float = 50.0,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.k_sharp = k_sharp

    def forward(
        self,
        tmax: torch.Tensor,
        dvs: torch.Tensor,
        params: CropParameters,
    ) -> torch.Tensor:
        """Compute the heat-stress leaf-senescence multiplier for one day.

        Args:
            tmax: Daily maximum air temperature [°C], shape ``[B]``.
            dvs: Development stage [-] (0–2), shape ``[B]``.
            params: Crop parameters; supplies ``leaf_heat_temp_critical``,
                ``leaf_heat_factor_at_temp_critical``,
                ``leaf_heat_factor_slope`` and
                ``leaf_heat_devstage_critical``.

        Returns:
            Leaf-senescence multiplier ``LeafSenescenceFactor`` [-], shape
            ``[B]``. Equals ``1`` outside the heat-stress regime and ``≥ 0``
            (typically ``> 1``) inside it. Intended to be passed as the
            ``heat_stress`` argument of `LeafDynamics`.
        """
        temp_critical = params.leaf_heat_temp_critical
        factor_at_temp_critical = params.leaf_heat_factor_at_temp_critical
        factor_slope = params.leaf_heat_factor_slope
        devstage_critical = params.leaf_heat_devstage_critical

        # Heat-stress factor when the regime is active.
        active_factor = torch.clamp(
            factor_slope * (tmax - temp_critical) + factor_at_temp_critical,
            min=0.0,
        )
        one = torch.ones_like(active_factor)

        if self.smooth:
            # Sigmoid blend: both thresholds must be satisfied, so the
            # gate is the product of the two soft step functions.
            gate = torch.sigmoid(
                self.k_sharp * (tmax - temp_critical)
            ) * torch.sigmoid(
                self.k_sharp * (dvs - devstage_critical)
            )
            return one + gate * (active_factor - one)

        # Hard branch.
        regime = (tmax >= temp_critical) & (dvs >= devstage_critical)
        return torch.where(regime, active_factor, one)


class HeatStressOnGrain(nn.Module):
    r"""Heat-stress penalty on grain yield around anthesis.

    Accumulates a daily heat-stress intensity over a development-stage
    window bracketing anthesis (``DVS`` roughly ``0.8``–``1.3``) and turns
    it into a multiplicative yield penalty. This module is *trajectory
    level*: unlike the per-day process modules it consumes the whole
    weather forcing and ``DVS`` trajectory at once, because the heat-stress
    factor is a window average.

    Daily day-time temperature and stress intensity (Teixeira et al. 2013):

    $$
    T_{\text{day},i} = T_{\max,i} - \tfrac{1}{4}\,(T_{\max,i} - T_{\min,i})
    $$

    $$
    s_i = \mathrm{clip}\!\left(
        \frac{T_{\text{day},i} - T_c}{T_{\text{limit}} - T_c},\; 0,\; 1
    \right)
    $$

    Window-averaged heat-stress factor and adjusted yield, over the days
    ``i`` whose development stage lies in
    ``[DVS_\text{begin}, DVS_\text{end}]``:

    $$
    \text{HSF} = \frac{\sum_i s_i \, \mathbb{1}[\text{DVS}_i \in
    [\text{DVS}_\text{begin}, \text{DVS}_\text{end}]]}
    {\sum_i \mathbb{1}[\text{DVS}_i \in
    [\text{DVS}_\text{begin}, \text{DVS}_\text{end}]]}
    $$

    $$
    \text{AdjustedYield} = (1 - \text{HSF}) \cdot \text{Yield}
    $$

    This implements the development-stage variant of the heat-stress
    model, which is the natural choice for a batched, differentiable
    setting where an explicit anthesis calendar date is not tracked.

    The crop constants ``T_c``, ``T_limit``, ``DVS_begin`` and
    ``DVS_end`` are read from `CropParameters` — fields
    ``grain_heat_temp_critical``, ``grain_heat_temp_limit``,
    ``grain_heat_begin_devstage`` and ``grain_heat_end_devstage`` —
    so they can be loaded per crop from an external configuration
    file.
    """

    def forward(
        self,
        tmin: torch.Tensor,
        tmax: torch.Tensor,
        dvs: torch.Tensor,
        params: CropParameters,
        yield_: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the heat-stress factor over the anthesis window.

        Args:
            tmin: Daily minimum air temperature [°C], shape ``[B, T]``.
            tmax: Daily maximum air temperature [°C], shape ``[B, T]``.
            dvs: Development-stage trajectory [-], shape ``[B, T]``, aligned
                day-for-day with ``tmin``/``tmax`` (the ``DVS`` value
                *entering* each simulated day).
            params: Crop parameters; supplies ``grain_heat_temp_critical``,
                ``grain_heat_temp_limit``, ``grain_heat_begin_devstage``
                and ``grain_heat_end_devstage``.
            yield_: Optional raw storage-organ yield ``WSO`` [g m⁻²], shape
                ``[B]``. When supplied, ``adjusted_yield`` is returned.

        Returns:
            Dict of tensors:

            * ``heat_stress_factor`` [-] — Window-averaged heat-stress
              factor ``HSF`` in ``[0, 1]``, shape ``[B]``. ``0`` when the
              window never opens (e.g. ``T`` too short) or never hot.
            * ``adjusted_yield`` [g m⁻²] — ``(1 − HSF) · yield_``, shape
              ``[B]``; only present when ``yield_`` is given.
            * ``daily_factor`` [-] — Daily stress intensity ``s_i``, shape
              ``[B, T]``.
            * ``window_mask`` [-] — ``1`` on days inside the
              development-stage window, else ``0``, shape ``[B, T]``.
            * ``window_days`` [-] — Number of days inside the window,
              shape ``[B]``.
        """
        temp_critical = params.grain_heat_temp_critical
        temp_limit = params.grain_heat_temp_limit
        begin_devstage = params.grain_heat_begin_devstage
        end_devstage = params.grain_heat_end_devstage

        # Day-time temperature and daily stress intensity.
        tday = tmax - 0.25 * (tmax - tmin)
        denom = temp_limit - temp_critical
        daily_factor = torch.clamp(
            (tday - temp_critical) / denom, min=0.0, max=1.0
        )

        # Development-stage window around anthesis.
        window_mask = (
            (dvs >= begin_devstage) & (dvs <= end_devstage)
        ).to(daily_factor.dtype)

        window_days = window_mask.sum(dim=1)  # [B]
        cumulated = (daily_factor * window_mask).sum(dim=1)  # [B]
        # Window-average the cumulated factor; guard against an empty
        # window (no anthesis reached) so HSF = 0 in that case.
        heat_stress_factor = cumulated / torch.clamp(window_days, min=1.0)

        out: dict[str, torch.Tensor] = {
            "heat_stress_factor": heat_stress_factor,
            "daily_factor": daily_factor,
            "window_mask": window_mask,
            "window_days": window_days,
        }
        if yield_ is not None:
            out["adjusted_yield"] = (1.0 - heat_stress_factor) * yield_
        return out
