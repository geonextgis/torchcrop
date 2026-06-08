"""Generalized, constraint-aware neural residual framework for hybrid
LINTUL5.

The framework lets any mechanistic quantity receive an *optional* learned
correction without breaking the surrounding process behaviour, conservation
laws, gradients, or numerical stability. Each correction target is described
by a `ResidualSpec` whose ``constraint`` selects a projection that keeps the
correction inside the target's natural geometry:

* ``rate_factor`` — non-negative scalar driver / flux / rate constant.
  Correction is multiplicative, ``base · exp(δ)``, so it is sign-safe
  (a positive quantity stays positive) and the identity at ``δ = 0``.
* ``transfer`` — a single flux that moves mass between two conserved pools.
  Same math as ``rate_factor``; the *caller* wires both legs from the one
  corrected number, so mass is conserved by construction.
* ``unit_interval`` — a dimensionless factor in ``(0, 1)`` (e.g. ``TRANRF``).
  Correction is applied in logit space, ``σ(logit(base) + δ)``, so the
  result never leaves ``(0, 1)``.
* ``simplex`` — a vector of fractions summing to ``1`` (e.g. the
  above-ground partitioning split). Correction is additive in log space
  followed by ``softmax``, so ``Σ = 1`` holds exactly.

The raw correction is always the bounded ``δ = scale · tanh(MLP(context))``,
and the MLP's final layer is **zero-initialised**, so every freshly
constructed head is the identity map: a model built with residual specs but
untrained reproduces the pure mechanistic trajectory (exactly for the
multiplicative / logit projections, up to a tiny ``1e-8`` bias for the
softmax projection).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn

Constraint = Literal["rate_factor", "transfer", "unit_interval", "simplex"]


@dataclass(frozen=True)
class ResidualSpec:
    """Description of a single neural residual correction slot.

    Attributes:
        name: Unique slot identifier, e.g. ``"photosynthesis.gtotal"``.
        constraint: Geometry of the correction target — one of
            ``"rate_factor"``, ``"transfer"``, ``"unit_interval"`` or
            ``"simplex"`` (see module docstring).
        context: Ordered tuple of feature keys fed to the MLP. Keys must be
            resolvable from the per-day feature dict assembled by the model.
            Use dimensionless / physical descriptors only — never identity
            features such as day-of-year or a site index.
        output_dim: Dimensionality of the correction. ``1`` for scalar
            targets; the number of fractions for a ``simplex`` target.
        scale: Magnitude cap on ``δ`` (the bounded residual). Doubles as the
            anchor for output-magnitude regularization.
        hidden_dim: Hidden-layer width of the MLP.
        n_hidden: Number of hidden layers (each followed by ``Tanh``).
    """

    name: str
    constraint: Constraint
    context: tuple[str, ...]
    output_dim: int = 1
    scale: float = 0.1
    hidden_dim: int = 32
    n_hidden: int = 2


def _key(name: str) -> str:
    """Map a dotted slot name to a `nn.ModuleDict`-safe key.

    `nn.ModuleDict` forbids ``"."`` in keys, but slot names are dotted
    (``"photosynthesis.gtotal"``) for readability, so the dots are escaped.
    """
    return name.replace(".", "__")


def _zero_init_mlp(d_in: int, d_out: int, hidden: int, n_hidden: int) -> nn.Sequential:
    """Build an MLP whose final layer is zeroed so it starts as ``0``.

    Args:
        d_in: Input feature dimension.
        d_out: Output dimension.
        hidden: Hidden-layer width.
        n_hidden: Number of hidden layers.

    Returns:
        A `nn.Sequential` MLP that outputs all-zeros at initialisation (and
        is therefore the identity residual until trained).
    """
    layers: list[nn.Module] = []
    d = d_in
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    last = nn.Linear(d, d_out)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class ResidualHead(nn.Module):
    """One MLP plus the constraint-aware projection for a single slot.

    Args:
        spec: The `ResidualSpec` describing this slot.
    """

    def __init__(self, spec: ResidualSpec) -> None:
        super().__init__()
        self.spec = spec
        self.net = _zero_init_mlp(
            len(spec.context), spec.output_dim, spec.hidden_dim, spec.n_hidden
        )

    def forward(
        self, base: torch.Tensor, ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the bounded, constraint-projected correction to ``base``.

        Args:
            base: Mechanistic value to correct. Shape ``[B]`` for scalar
                targets, ``[B, output_dim]`` for a ``simplex`` target.
            ctx: Context feature tensor of shape ``[B, len(context)]``.

        Returns:
            Tuple ``(value, delta)`` where ``value`` is the corrected
            quantity (same shape as ``base``) and ``delta`` is the raw
            bounded residual ``scale · tanh(MLP)`` used for regularization.
        """
        delta = self.spec.scale * torch.tanh(self.net(ctx))
        c = self.spec.constraint
        if c in ("rate_factor", "transfer"):
            value = base * torch.exp(delta.squeeze(-1))
        elif c == "unit_interval":
            b = base.clamp(1e-6, 1.0 - 1e-6)
            value = torch.sigmoid(torch.logit(b) + delta.squeeze(-1))
        elif c == "simplex":
            value = torch.softmax(torch.log(base + 1e-8) + delta, dim=-1)
        else:  # pragma: no cover - guarded by the Constraint type
            raise ValueError(f"Unknown constraint: {c!r}")
        return value, delta


class HybridManager(nn.Module):
    """Registry and single dispatch point for neural residual corrections.

    Holds one `ResidualHead` per configured slot. Slots that are not
    configured are exact pass-throughs (``correct`` returns the input tensor
    object unchanged), so a model with no specs behaves identically to the
    pure mechanistic model with zero overhead.

    Note:
        Enable only the slots whose pathway is constrained by an observable
        in your calibration data. See `default_slots` for the recommended
        catalogue and the guidance on slot choice and regularization.

    Args:
        specs: Iterable of `ResidualSpec` slots to enable. ``None`` or empty
            disables all corrections.
    """

    def __init__(self, specs: Sequence[ResidualSpec] | None = None) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {_key(s.name): ResidualHead(s) for s in (specs or [])}
        )
        self._penalty: torch.Tensor | None = None

    def enabled(self, name: str) -> bool:
        """Return ``True`` if a residual head is registered for ``name``."""
        return _key(name) in self.heads

    def head(self, name: str) -> ResidualHead:
        """Return the `ResidualHead` registered under the slot ``name``."""
        return self.heads[_key(name)]

    def correct(
        self, name: str, base: torch.Tensor, features: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply the correction for slot ``name`` if registered.

        Args:
            name: Slot identifier.
            base: Mechanistic value to correct.
            features: Per-day feature dict. Must contain every key listed in
                the slot's ``context``.

        Returns:
            The corrected value, or ``base`` unchanged when the slot is not
            registered.

        Raises:
            KeyError: If a context feature required by the slot is missing
                from ``features``.
        """
        key = _key(name)
        if key not in self.heads:
            return base
        head = self.heads[key]
        try:
            ctx = torch.stack([features[k] for k in head.spec.context], dim=-1)
        except KeyError as exc:  # pragma: no cover - developer error
            raise KeyError(
                f"Residual slot {name!r} needs feature {exc.args[0]!r}, "
                f"which was not provided. Available: {sorted(features)}"
            ) from exc
        value, delta = head(base, ctx)
        if self.training:
            p = delta.pow(2).mean()
            self._penalty = p if self._penalty is None else self._penalty + p
        return value

    def reset_penalty(self) -> None:
        """Clear the accumulated residual-magnitude penalty.

        Called by the model at the start of each forward pass so the penalty
        reflects a single simulation.
        """
        self._penalty = None

    def penalty(self) -> torch.Tensor:
        """Return the accumulated mean-squared residual magnitude.

        Sum over every ``correct`` call since the last `reset_penalty` of
        ``mean(δ²)``. Add this to the training loss (scaled by a small
        coefficient) to anchor corrections toward zero — the main defence
        against compensation and overfitting. Returns a zero scalar when no
        corrections were applied (e.g. in ``eval`` mode or with no slots).
        """
        if self._penalty is None:
            return torch.zeros(())
        return self._penalty


def default_slots() -> list[ResidualSpec]:
    """Return the recommended initial catalogue of residual slots.

    These cover the highest-leverage, lowest-risk correction targets across
    photosynthesis, water stress, partitioning, leaf senescence and
    phenology.

    Note:
        Choosing *which* slots to enable matters more than the list itself.
        Enable only the slots whose pathway is constrained by an observable
        in your calibration data (e.g. enable ``"photosynthesis.gtotal"``
        only if you observe biomass/yield, ``"water.tranrf"`` only if you
        observe soil moisture or transpiration). Turning on **all** slots at
        once invites *identifiability* and *compensation* problems: several
        residuals — and the mechanistic parameters they shadow, such as
        ``gtotal`` vs. a learnable RUE — become degenerate, so the optimiser
        can fit the data while learning physically meaningless corrections.
        Prefer to (1) enable a minimal set tied to observables, (2) calibrate
        the mechanistic parameters first and then add residuals, and (3)
        regularise with `HybridManager.penalty` to anchor corrections toward
        zero.

    Returns:
        A list of `ResidualSpec` slots suitable for
        ``Lintul5Model(residual_specs=...)``. Pass a hand-picked subset
        rather than the whole list unless every pathway is observable.
    """
    return [
        ResidualSpec(
            "photosynthesis.gtotal",
            "rate_factor",
            context=("lai", "dvs", "davtmp", "tranrf", "nstress"),
            scale=0.15,
        ),
        ResidualSpec(
            "water.tranrf",
            "unit_interval",
            context=("lai", "smact", "rootd"),
            scale=0.10,
        ),
        ResidualSpec(
            "partitioning.aboveground",
            "simplex",
            context=("dvs", "tranrf", "nstress"),
            output_dim=3,
            scale=0.10,
        ),
        ResidualSpec(
            "partitioning.fr",
            "unit_interval",
            context=("dvs", "tranrf", "nstress"),
            scale=0.10,
        ),
        ResidualSpec(
            "leaf.rdr",
            "rate_factor",
            context=("lai", "dvs", "davtmp", "tranrf"),
            scale=0.10,
        ),
        # Phenology shifts everything downstream via the partition tables, so
        # this slot is the most identifiability-prone: keep its scale small.
        ResidualSpec(
            "phenology.dvs_rate",
            "rate_factor",
            context=("dvs", "davtmp", "ddlp"),
            scale=0.05,
        ),
    ]
