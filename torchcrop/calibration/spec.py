"""Declarative descriptions of a calibration problem.

A calibration problem is a *data* description — a list of
`ParameterSpec` (which parameters are free, their bounds and type) plus
a list of `ConstraintGroup` (ordering / cross-parameter relations).
These frozen dataclasses are deliberately behaviour-free, mirroring
`torchcrop.nn.hybrid.ResidualSpec`; all the machinery that turns them
into optimizable latents lives in
`torchcrop.calibration.manager.CalibrationManager`.

Target addressing (the ``name`` field) reuses the dotted
``"<container>.<field>"`` scheme already used by
``Lintul5Model.learnable_parameter_groups`` and extends it to table entries:

* ``"crop.tsum1"``      — a scalar crop parameter.
* ``"soil.wcfc"``       — a scalar soil parameter.
* ``"crop.fltb@0.5"``   — the **y-value** of the ``fltb`` table row at DVS 0.5
  (the abscissa stays fixed; only the ordinate is calibrated).
* ``"crop.fltb[3]"``    — the y-value of the 4th row of ``fltb`` (0-based),
  for tables with duplicate / ambiguous abscissae.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Kind = Literal["continuous", "integer", "categorical"]
Order = Literal["ascending", "descending", "free"]


@dataclass(frozen=True)
class ParameterSpec:
    """A single calibration target.

    Attributes:
        name: Dotted target path (see module docstring). Scalar fields are
            ``"<container>.<field>"``; table entries add ``"@<dvs>"`` (match by
            abscissa) or ``"[<i>]"`` (match by row index).
        bounds: ``(lo, hi)`` feasible range. Required for ``affine_sigmoid``
            and for any member of an ordered `ConstraintGroup`.
        kind: ``"continuous"`` (default), ``"integer"`` (rounded via a
            straight-through estimator), or ``"categorical"`` (excluded from
            the gradient path; driven by a gradient-free outer loop).
        transform: Registered transform name (see
            `torchcrop.calibration.transforms.available_transforms`).
            Defaults to ``"affine_sigmoid"`` when ``bounds`` are given,
            otherwise ``"identity"``.
        init: Initial physical value. Defaults to the parameter's current
            value in its container.
        categories: Allowed values for ``kind="categorical"``.
    """

    name: str
    bounds: tuple[float, float] | None = None
    kind: Kind = "continuous"
    transform: str | None = None
    init: float | None = None
    categories: tuple[float, ...] | None = None

    def resolved_transform(self) -> str:
        """Return the transform name, applying the bounds-based default."""
        if self.transform is not None:
            return self.transform
        return "affine_sigmoid" if self.bounds is not None else "identity"

    def __post_init__(self) -> None:
        if self.bounds is not None:
            lo, hi = self.bounds
            if not hi > lo:
                raise ValueError(
                    f"{self.name}: bounds must satisfy hi > lo; got {self.bounds}"
                )
            if self.init is not None and not (lo <= self.init <= hi):
                raise ValueError(
                    f"{self.name}: init {self.init} outside bounds {self.bounds}"
                )
        if self.kind == "categorical" and not self.categories:
            raise ValueError(
                f"{self.name}: kind='categorical' requires a non-empty "
                "`categories` tuple"
            )


@dataclass(frozen=True)
class ConstraintGroup:
    """An ordering / cross-parameter relation over several targets.

    The members are reconstructed jointly so the relation holds *by
    construction* on every iterate (no penalty term, no projection).

    Attributes:
        members: Spec names participating in the relation, **in the intended
            order** (for ``ascending`` / ``descending``).
        order: ``"ascending"`` → ``v0 <= v1 <= ...``; ``"descending"`` →
            ``v0 >= v1 >= ...``; ``"free"`` → no coupling (members optimized
            independently — useful as a grouping label only).
    """

    members: tuple[str, ...]
    order: Order = "ascending"

    def __post_init__(self) -> None:
        if self.order != "free" and len(self.members) < 2:
            raise ValueError(
                f"ordered group needs >= 2 members; got {self.members}"
            )
        if len(set(self.members)) != len(self.members):
            raise ValueError(f"duplicate members in group {self.members}")
