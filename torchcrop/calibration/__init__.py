"""Constraint-aware parameter calibration for torchcrop.

A declarative, transform-based framework for calibrating Lintul5 parameters
under bounds, type (continuous / integer / categorical) and ordering
(ascending / descending / free) constraints — including the *ordinates* of
DVS-indexed interpolation tables — while keeping the whole pipeline
differentiable and optimizer-agnostic.

Typical use::

    from torchcrop.calibration import (
        CalibrationManager, ParameterSpec, ConstraintGroup,
    )

    cal = CalibrationManager(
        model,
        specs=[
            ParameterSpec("crop.tsum1", bounds=(1000, 2000)),
            ParameterSpec("crop.fltb@0.5", bounds=(0.30, 0.70)),
            ParameterSpec("crop.fltb@0.646", bounds=(0.10, 0.50)),
        ],
        groups=[
            ConstraintGroup(("crop.fltb@0.5", "crop.fltb@0.646"),
                            order="descending"),
        ],
    )
    opt = torch.optim.Adam(cal.parameters(), lr=1e-2)
    for _ in range(n_epochs):
        opt.zero_grad()
        cal.materialize()           # latents -> constrained params/tables
        loss = loss_fn(model(weather).yield_, observed)
        loss.backward()
        opt.step()
"""

from torchcrop.calibration.config import (
    load_calibration_config,
    specs_from_config,
)
from torchcrop.calibration.manager import CalibrationManager
from torchcrop.calibration.spec import ConstraintGroup, ParameterSpec
from torchcrop.calibration.transforms import (
    available_transforms,
    build_transform,
    register_transform,
    round_ste,
)

__all__ = [
    "CalibrationManager",
    "ConstraintGroup",
    "ParameterSpec",
    "available_transforms",
    "build_transform",
    "load_calibration_config",
    "register_transform",
    "round_ste",
    "specs_from_config",
]
