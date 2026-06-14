"""Differentiable bijections between unconstrained and constrained space.

Calibration is performed in an *unconstrained* latent space: the optimizer
holds raw real-valued variables ``z`` and a bijector maps them onto the
feasible region of the physical parameter on every forward pass. Because the
map is applied inside the autograd graph, gradients flow cleanly back to ``z``
and **every iterate is feasible by construction** — there is no need for
penalties or post-step projection.

This mirrors the constraint-aware projection idiom already used for neural
residuals in :mod:`torchcrop.nn.hybrid` (``exp`` for non-negative drivers,
``logit``/``sigmoid`` for unit-interval factors), applied here to *parameters*
rather than residual corrections.

Transforms are looked up through a small registry so new ones can be added
without touching :class:`~torchcrop.calibration.manager.CalibrationManager`
(open/closed principle).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

_EPS = 1e-6


@dataclass(frozen=True)
class Bijector:
    """A differentiable map between latent ``z`` and a constrained value.

    Attributes:
        forward: ``z -> value`` (unconstrained → constrained / physical).
        inverse: ``value -> z`` (constrained → unconstrained), used to seed
            the latent from an initial physical value.
    """

    forward: Callable[[torch.Tensor], torch.Tensor]
    inverse: Callable[[torch.Tensor], torch.Tensor]


# name -> builder(bounds) -> Bijector
_TRANSFORMS: dict[str, Callable[[tuple[float, float] | None], Bijector]] = {}


def register_transform(
    name: str,
) -> Callable[
    [Callable[[tuple[float, float] | None], Bijector]],
    Callable[[tuple[float, float] | None], Bijector],
]:
    """Register a transform builder under ``name``.

    Args:
        name: Key used in :class:`~torchcrop.calibration.spec.ParameterSpec`.

    Returns:
        A decorator registering the wrapped ``builder(bounds) -> Bijector``.
    """

    def _decorator(
        builder: Callable[[tuple[float, float] | None], Bijector],
    ) -> Callable[[tuple[float, float] | None], Bijector]:
        _TRANSFORMS[name] = builder
        return builder

    return _decorator


def build_transform(name: str, bounds: tuple[float, float] | None) -> Bijector:
    """Instantiate a registered transform.

    Args:
        name: Registered transform name (e.g. ``"affine_sigmoid"``).
        bounds: ``(lo, hi)`` bounds, interpreted per transform (some require
            both, some only the lower bound, some ignore them).

    Returns:
        The configured :class:`Bijector`.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    if name not in _TRANSFORMS:
        raise KeyError(
            f"unknown transform {name!r}; registered: {sorted(_TRANSFORMS)}"
        )
    return _TRANSFORMS[name](bounds)


def available_transforms() -> tuple[str, ...]:
    """Return the names of all registered transforms."""
    return tuple(sorted(_TRANSFORMS))


@register_transform("identity")
def _identity(bounds: tuple[float, float] | None) -> Bijector:
    """Unconstrained pass-through (``value = z``)."""
    return Bijector(forward=lambda z: z, inverse=lambda v: v)


@register_transform("affine_sigmoid")
def _affine_sigmoid(bounds: tuple[float, float] | None) -> Bijector:
    """Two-sided bound ``value = lo + (hi - lo) * sigmoid(z)``.

    The canonical transform for a box-bounded parameter (e.g. ``TSUM1`` in
    ``[1000, 2000]``). The result is confined to the open interval
    ``(lo, hi)`` for all real ``z``.
    """
    if bounds is None:
        raise ValueError("affine_sigmoid requires bounds=(lo, hi)")
    lo, hi = bounds
    if not hi > lo:
        raise ValueError(f"bounds must satisfy hi > lo; got {bounds}")
    span = hi - lo

    def forward(z: torch.Tensor) -> torch.Tensor:
        return lo + span * torch.sigmoid(z)

    def inverse(v: torch.Tensor) -> torch.Tensor:
        u = ((v - lo) / span).clamp(_EPS, 1.0 - _EPS)
        return torch.log(u) - torch.log1p(-u)

    return Bijector(forward=forward, inverse=inverse)


@register_transform("softplus")
def _softplus(bounds: tuple[float, float] | None) -> Bijector:
    """One-sided lower bound ``value = lo + softplus(z)`` (``lo`` default 0)."""
    lo = 0.0 if bounds is None else bounds[0]

    def forward(z: torch.Tensor) -> torch.Tensor:
        return lo + torch.nn.functional.softplus(z)

    def inverse(v: torch.Tensor) -> torch.Tensor:
        # inverse softplus: log(exp(y) - 1), numerically guarded for small y
        y = (v - lo).clamp(min=_EPS)
        return y + torch.log(-torch.expm1(-y))

    return Bijector(forward=forward, inverse=inverse)


@register_transform("log")
def _log(bounds: tuple[float, float] | None) -> Bijector:
    """Multiplicative / strictly-positive bound ``value = lo + exp(z)``."""
    lo = 0.0 if bounds is None else bounds[0]

    def forward(z: torch.Tensor) -> torch.Tensor:
        return lo + torch.exp(z)

    def inverse(v: torch.Tensor) -> torch.Tensor:
        return torch.log((v - lo).clamp(min=_EPS))

    return Bijector(forward=forward, inverse=inverse)


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through gradient estimator.

    Forward returns ``round(x)``; backward passes the gradient through
    unchanged (as if the op were the identity). This keeps an integer-valued
    parameter inside the gradient pipeline so a continuous surrogate can still
    be nudged by the optimizer while the value used in the forward pass stays
    integral.

    Args:
        x: Continuous surrogate value.

    Returns:
        ``round(x)`` in the forward pass, identity gradient in the backward.
    """
    return (torch.round(x) - x).detach() + x
