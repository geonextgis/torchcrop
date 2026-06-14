"""Constraint-aware parameter calibration for :class:`Lintul5Model`.

:class:`CalibrationManager` is the runtime counterpart of the declarative
:mod:`~torchcrop.calibration.spec` description, structurally parallel to
:class:`torchcrop.nn.hybrid.HybridManager`:

* it owns the **unconstrained latent** ``nn.Parameter`` set (what the optimizer
  sees via :meth:`parameters`);
* :meth:`materialize` maps the latents through their bijections — and through
  the joint reconstruction of ordered groups — into the *physical* crop / soil
  / site parameter tensors (rebuilding ``[N, 2]`` tables as needed) **before
  each forward pass**;
* :meth:`flatten` / :meth:`set_flat` expose the latent vector so the same
  problem definition can be driven by a gradient-free optimizer (CMA-ES,
  scipy, …) instead of autograd.

Because every latent is unconstrained and every map is feasible-by-construction,
each optimizer iterate yields a valid parameter set — no penalty terms, no
post-step projection.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

from torchcrop.calibration.paths import (
    Target,
    parse_path,
    rebuild_table,
    read_value,
    write_scalar,
)
from torchcrop.calibration.spec import ConstraintGroup, ParameterSpec
from torchcrop.calibration.transforms import build_transform, round_ste

_EPS = 1e-6


def _key(name: str) -> str:
    """Map a target path to an ``nn.ParameterDict``-safe key."""
    for ch in ".@[]":
        name = name.replace(ch, "__")
    return name


def _logit(u: torch.Tensor) -> torch.Tensor:
    u = u.clamp(_EPS, 1.0 - _EPS)
    return torch.log(u) - torch.log1p(-u)


class CalibrationManager(nn.Module):
    """Owns latent variables and writes constrained parameters into a model.

    Args:
        model: The :class:`~torchcrop.model.Lintul5Model` (or any object
            exposing ``crop_params`` / ``soil_params`` / ``site_params``) whose
            parameters are being calibrated.
        specs: The free parameters and their bounds / types.
        groups: Optional ordering / cross-parameter relations. Every member of
            an ``ascending`` / ``descending`` group must also appear in
            ``specs`` with ``bounds`` set.

    Raises:
        ValueError: On duplicate spec names, unknown group members, or bounds
            incompatible with a requested ordering.
    """

    def __init__(
        self,
        model: Any,
        specs: Sequence[ParameterSpec],
        groups: Sequence[ConstraintGroup] | None = None,
    ) -> None:
        super().__init__()
        # Hold the parameter containers WITHOUT registering ``model`` as a
        # submodule (it is a plain dict of dataclasses), so ``self.parameters()``
        # yields only the calibration latents — exactly what the optimizer wants.
        self._containers: dict[str, Any] = {
            "crop": model.crop_params,
            "soil": model.soil_params,
            "site": model.site_params,
        }

        self._specs: dict[str, ParameterSpec] = {}
        for s in specs:
            if s.name in self._specs:
                raise ValueError(f"duplicate spec for {s.name!r}")
            self._specs[s.name] = s

        self._targets: dict[str, Target] = {
            name: parse_path(name, self._containers) for name in self._specs
        }

        self._groups: list[ConstraintGroup] = list(groups or [])
        # member name -> owning ordered group index (free groups excluded)
        self._member_group: dict[str, int] = {}
        for gi, g in enumerate(self._groups):
            for m in g.members:
                if m not in self._specs:
                    raise ValueError(
                        f"group member {m!r} has no ParameterSpec"
                    )
            if g.order == "free":
                continue
            for m in g.members:
                if m in self._member_group:
                    raise ValueError(
                        f"{m!r} belongs to more than one ordered group"
                    )
                self._member_group[m] = gi
            self._validate_ordered_bounds(g)

        # Snapshot original tables BEFORE any materialize so frozen ordinates
        # and all abscissae stay pinned to the preset values.
        self._table_originals: dict[tuple[str, str], torch.Tensor] = {}
        for t in self._targets.values():
            if t.is_table and t.table_key not in self._table_originals:
                tbl = getattr(self._containers[t.container], t.field)
                self._table_originals[t.table_key] = tbl.detach().clone()

        self._latents = nn.ParameterDict()
        self._build_latents()

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #

    def _ref_tensor(self, name: str) -> torch.Tensor:
        """Return the live tensor backing ``name`` (scalar field or table y)."""
        t = self._targets[name]
        obj = self._containers[t.container]
        field = getattr(obj, t.field)
        return field[t.row, 1] if t.is_table else field

    def _init_value(self, name: str) -> torch.Tensor:
        """Initial physical value: explicit ``init`` or the current value."""
        spec = self._specs[name]
        ref = self._ref_tensor(name)
        v = spec.init if spec.init is not None else read_value(self._targets[name], self._containers)
        return torch.as_tensor(v, dtype=ref.dtype, device=ref.device)

    def _validate_ordered_bounds(self, g: ConstraintGroup) -> None:
        """Ensure member bounds are compatible with the requested ordering."""
        bnds = []
        for m in g.members:
            b = self._specs[m].bounds
            if b is None:
                raise ValueError(
                    f"ordered group member {m!r} requires bounds"
                )
            bnds.append(b)
        for (lo0, hi0), (lo1, hi1), m in zip(bnds, bnds[1:], g.members[1:]):
            if g.order == "ascending" and not (lo1 >= lo0 and hi1 >= hi0):
                raise ValueError(
                    f"ascending group: bounds of {m!r} {(lo1, hi1)} must be "
                    f">= previous member's bounds {(lo0, hi0)} (lo and hi)"
                )
            if g.order == "descending" and not (lo1 <= lo0 and hi1 <= hi0):
                raise ValueError(
                    f"descending group: bounds of {m!r} {(lo1, hi1)} must be "
                    f"<= previous member's bounds {(lo0, hi0)} (lo and hi)"
                )

    def _build_latents(self) -> None:
        """Create one latent per free / ordered-member spec (not categorical)."""
        # Free specs (not in any ordered group, not categorical): invert the
        # bijector at the init value to seed the latent.
        for name, spec in self._specs.items():
            if spec.kind == "categorical" or name in self._member_group:
                continue
            bij = build_transform(spec.resolved_transform(), spec.bounds)
            z = bij.inverse(self._init_value(name)).detach().clone()
            self._latents[_key(name)] = nn.Parameter(z)

        # Ordered-group members: seed latents from the incremental inverse.
        for g in self._groups:
            if g.order == "free":
                continue
            self._seed_ordered_latents(g)

    def _seed_ordered_latents(self, g: ConstraintGroup) -> None:
        """Seed per-member latents so materialize reproduces the init values."""
        inits = [self._init_value(m) for m in g.members]
        bnds = [self._specs[m].bounds for m in g.members]
        prev: torch.Tensor | None = None
        for m, v, (lo, hi) in zip(g.members, inits, bnds):
            lo_t = torch.as_tensor(lo, dtype=v.dtype, device=v.device)
            hi_t = torch.as_tensor(hi, dtype=v.dtype, device=v.device)
            if prev is None:
                lower, upper = lo_t, hi_t
            elif g.order == "ascending":
                lower, upper = torch.maximum(lo_t, prev), hi_t
            else:  # descending
                lower, upper = lo_t, torch.minimum(hi_t, prev)
            u = (v - lower) / (upper - lower).clamp(min=_EPS)
            self._latents[_key(m)] = nn.Parameter(_logit(u).detach().clone())
            prev = v

    # ------------------------------------------------------------------ #
    # Materialization
    # ------------------------------------------------------------------ #

    def _materialize_values(self) -> dict[str, torch.Tensor]:
        """Compute the physical value tensor for every spec."""
        values: dict[str, torch.Tensor] = {}

        for name, spec in self._specs.items():
            if spec.kind == "categorical" or name in self._member_group:
                continue
            bij = build_transform(spec.resolved_transform(), spec.bounds)
            v = bij.forward(self._latents[_key(name)])
            values[name] = round_ste(v) if spec.kind == "integer" else v

        for g in self._groups:
            if g.order == "free":
                continue
            for m, v in zip(g.members, self._reconstruct_ordered(g)):
                spec = self._specs[m]
                values[m] = round_ste(v) if spec.kind == "integer" else v

        for name, spec in self._specs.items():
            if spec.kind != "categorical":
                continue
            ref = self._ref_tensor(name)
            cur = read_value(self._targets[name], self._containers)
            nearest = min(spec.categories, key=lambda c: abs(c - cur))  # type: ignore[arg-type]
            values[name] = torch.as_tensor(
                nearest, dtype=ref.dtype, device=ref.device
            )
        return values

    def _reconstruct_ordered(self, g: ConstraintGroup) -> list[torch.Tensor]:
        """Map a group's latents to monotone values inside their boxes."""
        out: list[torch.Tensor] = []
        prev: torch.Tensor | None = None
        for m in g.members:
            lo, hi = self._specs[m].bounds  # type: ignore[misc]
            z = self._latents[_key(m)]
            lo_t = torch.as_tensor(lo, dtype=z.dtype, device=z.device)
            hi_t = torch.as_tensor(hi, dtype=z.dtype, device=z.device)
            if prev is None:
                lower, upper = lo_t, hi_t
            elif g.order == "ascending":
                lower, upper = torch.maximum(lo_t, prev), hi_t
            else:  # descending
                lower, upper = lo_t, torch.minimum(hi_t, prev)
            v = lower + (upper - lower) * torch.sigmoid(z)
            out.append(v)
            prev = v
        return out

    def materialize(self) -> dict[str, torch.Tensor]:
        """Write all constrained values into the parameter containers.

        Call once **before each forward pass**. Scalar targets overwrite their
        dataclass field; table targets are batched per owning table and the
        ``[N, 2]`` tensor is rebuilt once (non-in-place, so autograd is
        preserved).

        Returns:
            A mapping ``target name -> materialized value tensor`` for
            inspection / logging.
        """
        values = self._materialize_values()

        table_updates: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
        for name, value in values.items():
            target = self._targets[name]
            if target.is_table:
                table_updates.setdefault(target.table_key, {})[target.row] = value
            else:
                write_scalar(target, value, self._containers)

        for table_key, updates in table_updates.items():
            container, field = table_key
            new_table = rebuild_table(self._table_originals[table_key], updates)
            setattr(self._containers[container], field, new_table)

        return values

    forward = materialize  # so ``manager()`` materializes, like other nn.Modules

    # ------------------------------------------------------------------ #
    # Gradient-free / Bayesian bridge
    # ------------------------------------------------------------------ #

    def latent_names(self) -> tuple[str, ...]:
        """Stable, sorted order of latent target names."""
        return tuple(sorted(self._latents.keys()))

    def flatten(self) -> torch.Tensor:
        """Return the latent vector as a flat 1-D tensor (detached)."""
        return torch.stack(
            [self._latents[k].detach().reshape(()) for k in self.latent_names()]
        )

    @torch.no_grad()
    def set_flat(self, flat: torch.Tensor) -> None:
        """Overwrite the latent vector from a flat 1-D tensor.

        Args:
            flat: 1-D tensor matching :meth:`flatten`'s length and order.

        Raises:
            ValueError: If ``flat`` has the wrong length.
        """
        names = self.latent_names()
        if flat.numel() != len(names):
            raise ValueError(
                f"expected {len(names)} latents, got {flat.numel()}"
            )
        for k, v in zip(names, flat.reshape(-1)):
            self._latents[k].copy_(v.to(self._latents[k].dtype))

    def set_categorical(self, name: str, value: float) -> None:
        """Set a categorical target's current value (for an outer loop).

        Args:
            name: A spec name with ``kind="categorical"``.
            value: One of the spec's ``categories``.

        Raises:
            ValueError: If ``name`` is not categorical or ``value`` is not an
                allowed category.
        """
        spec = self._specs.get(name)
        if spec is None or spec.kind != "categorical":
            raise ValueError(f"{name!r} is not a categorical target")
        if value not in spec.categories:  # type: ignore[operator]
            raise ValueError(
                f"{value} not in categories {spec.categories} for {name!r}"
            )
        ref = self._ref_tensor(name)
        write_scalar(
            self._targets[name],
            torch.as_tensor(value, dtype=ref.dtype, device=ref.device),
            self._containers,
        )

    def named_values(self) -> dict[str, float]:
        """Materialize and return current physical values as Python floats."""
        return {k: float(v) for k, v in self.materialize().items()}
