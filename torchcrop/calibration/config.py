"""Declarative (YAML / dict) configuration of a calibration problem.

The schema keys on the same dotted ``"<container>.<field>"`` target paths used
throughout the calibration package, so a calibration setup can live next to a
crop preset as data:

.. code-block:: yaml

    parameters:
      crop.tsum1:       {bounds: [1000, 2000]}
      crop.tbasem:      {bounds: [-2, 5], kind: continuous}
      crop.fltb@0.5:    {bounds: [0.30, 0.70]}
      crop.fltb@0.646:  {bounds: [0.10, 0.50]}
    constraints:
      - members: [crop.fltb@0.5, crop.fltb@0.646]
        order: descending

The loader returns ``(specs, groups)`` ready to pass to
:class:`~torchcrop.calibration.manager.CalibrationManager`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torchcrop.calibration.spec import ConstraintGroup, ParameterSpec


def _import_yaml():
    """Import PyYAML lazily with a helpful error if it is missing."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Loading a calibration config from YAML requires PyYAML. "
            "Install it with `pip install pyyaml`."
        ) from exc
    return yaml


def _as_bounds(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    lo, hi = value
    return (float(lo), float(hi))


def specs_from_config(
    config: dict[str, Any],
) -> tuple[list[ParameterSpec], list[ConstraintGroup]]:
    """Build specs and groups from an already-parsed config mapping.

    Args:
        config: Mapping with optional ``parameters`` and ``constraints`` keys
            (see module docstring for the schema).

    Returns:
        ``(specs, groups)``.

    Raises:
        ValueError: If the mapping is malformed.
    """
    specs: list[ParameterSpec] = []
    for name, opts in (config.get("parameters") or {}).items():
        opts = opts or {}
        cats = opts.get("categories")
        specs.append(
            ParameterSpec(
                name=name,
                bounds=_as_bounds(opts.get("bounds")),
                kind=opts.get("kind", "continuous"),
                transform=opts.get("transform"),
                init=opts.get("init"),
                categories=tuple(cats) if cats is not None else None,
            )
        )

    groups: list[ConstraintGroup] = []
    for entry in config.get("constraints") or []:
        groups.append(
            ConstraintGroup(
                members=tuple(entry["members"]),
                order=entry.get("order", "ascending"),
            )
        )
    return specs, groups


def load_calibration_config(
    source: str | Path | dict[str, Any],
) -> tuple[list[ParameterSpec], list[ConstraintGroup]]:
    """Load a calibration problem from a YAML file, path, or dict.

    Args:
        source: Path to a YAML file, or an already-parsed config dict.

    Returns:
        ``(specs, groups)`` ready for
        :class:`~torchcrop.calibration.manager.CalibrationManager`.
    """
    if isinstance(source, dict):
        return specs_from_config(source)
    yaml = _import_yaml()
    with open(source) as fh:
        return specs_from_config(yaml.safe_load(fh) or {})
