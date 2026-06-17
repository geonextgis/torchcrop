"""Resolve dotted target paths to concrete parameter locations.

A path names *where* a calibration value is written. Two location kinds are
supported:

* **Scalar** — ``"<container>.<field>"`` resolves to a scalar tensor field on
  one of the ``crop`` / ``soil`` / ``site`` dataclasses.
* **Table entry** — ``"<container>.<field>@<dvs>"`` or ``"...[<i>]"`` resolves
  to a single ordinate (column 1) of an ``[N, 2]`` interpolation table; the
  abscissa (column 0) is held fixed.

The resolver returns a `Target` describing the container object, the
field name, and — for tables — the row index. The
`CalibrationManager` uses these to read
initial values and write materialized values each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

_ROW_TOL = 1e-6


@dataclass(frozen=True)
class Target:
    """A resolved write location for one calibration value.

    Attributes:
        container: Logical container name (``"crop"`` / ``"soil"`` / ``"site"``).
        field: Dataclass field name on that container.
        row: For a table entry, the 0-based row index whose ordinate is free;
            ``None`` for a scalar field.
    """

    container: str
    field: str
    row: int | None = None

    @property
    def is_table(self) -> bool:
        """Whether this target addresses a table ordinate."""
        return self.row is not None

    @property
    def table_key(self) -> tuple[str, str]:
        """``(container, field)`` identifying the owning table."""
        return (self.container, self.field)


def parse_path(
    name: str, containers: dict[str, Any]
) -> Target:
    """Resolve a dotted target path against the parameter containers.

    Args:
        name: Target path (see module docstring).
        containers: Mapping of container name → dataclass instance.

    Returns:
        The resolved `Target`.

    Raises:
        ValueError: If the path is malformed, the container/field is unknown,
            or a table abscissa cannot be matched unambiguously.
    """
    row: int | None = None
    selector: str | None = None

    if "@" in name:
        base, selector = name.split("@", 1)
        sel_kind = "key"
    elif name.endswith("]") and "[" in name:
        base, idx = name[:-1].split("[", 1)
        selector = idx
        sel_kind = "index"
    else:
        base, sel_kind = name, None

    if base.count(".") != 1:
        raise ValueError(
            f"target {name!r}: expected '<container>.<field>' form"
        )
    container, field = base.split(".")
    if container not in containers:
        raise ValueError(
            f"target {name!r}: unknown container {container!r} "
            f"(expected one of {sorted(containers)})"
        )
    obj = containers[container]
    if not hasattr(obj, field):
        raise ValueError(f"target {name!r}: {container} has no field {field!r}")

    if sel_kind is None:
        return Target(container=container, field=field, row=None)

    table = getattr(obj, field)
    if not isinstance(table, torch.Tensor) or table.dim() != 2 or table.shape[1] != 2:
        raise ValueError(
            f"target {name!r}: {container}.{field} is not an [N, 2] table"
        )

    if sel_kind == "index":
        try:
            row = int(selector)  # type: ignore[arg-type]
        except ValueError as exc:
            raise ValueError(f"target {name!r}: bad row index") from exc
        if not 0 <= row < table.shape[0]:
            raise ValueError(
                f"target {name!r}: row {row} out of range [0, {table.shape[0]})"
            )
        return Target(container=container, field=field, row=row)

    # selector by abscissa key
    key = float(selector)  # type: ignore[arg-type]
    xs = table[:, 0]
    matches = (xs - key).abs() < _ROW_TOL
    n = int(matches.sum().item())
    if n == 0:
        raise ValueError(
            f"target {name!r}: no table row at DVS={key}; "
            f"abscissae are {xs.tolist()}"
        )
    if n > 1:
        raise ValueError(
            f"target {name!r}: DVS={key} matches {n} rows; "
            "use the '[index]' form to disambiguate"
        )
    return Target(container=container, field=field, row=int(matches.nonzero()[0]))


def read_value(target: Target, containers: dict[str, Any]) -> float:
    """Read the current physical value at ``target``."""
    obj = containers[target.container]
    field = getattr(obj, target.field)
    if target.is_table:
        return float(field[target.row, 1])
    return float(field)


def write_scalar(
    target: Target, value: torch.Tensor, containers: dict[str, Any]
) -> None:
    """Write a scalar ``value`` into a non-table ``target`` field."""
    setattr(containers[target.container], target.field, value)


def rebuild_table(
    table: torch.Tensor, updates: dict[int, torch.Tensor]
) -> torch.Tensor:
    """Return a fresh ``[N, 2]`` table with selected ordinates replaced.

    The abscissa column and the frozen ordinates are taken (detached) from
    ``table``; the rows in ``updates`` are replaced by the supplied value
    tensors. The result is assembled with `torch.stack` (no in-place
    writes), so gradients flow from the loss back through the updated
    ordinates while keeping the interpolation table's sorted-abscissa contract
    intact.

    Args:
        table: Original ``[N, 2]`` table.
        updates: Mapping ``row_index -> scalar value tensor`` (shape ``[]``).

    Returns:
        A new ``[N, 2]`` table tensor.
    """
    xs = table[:, 0].detach()
    base_y = table[:, 1].detach()
    rows = [
        updates[i] if i in updates else base_y[i] for i in range(table.shape[0])
    ]
    ys = torch.stack(rows)
    return torch.stack([xs, ys], dim=1)
