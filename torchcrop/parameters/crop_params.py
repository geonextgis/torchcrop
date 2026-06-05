"""Crop-specific parameters for Lintul5.

This module ports the **constant** crop-level inputs of the SIMPLACE Lintul5
component family (``Lintul5.java``, ``Phenology.java``,
``RadiationUseEfficiency.java``) to a single PyTorch-friendly dataclass.

All scalar fields are stored as `torch.Tensor` so they can be made
learnable by wrapping them with `torch.nn.Parameter`. Table fields
(``..._tb`` / ``..._table``) carry shape ``[N, 2]`` (or ``[B, N, 2]`` if the
table itself is batch-varying), where column 0 is the abscissa (DVS or
temperature) and column 1 is the value. Tables are interpolated by
`torchcrop.functions.interpolate`.

Naming conventions:
    * Field names follow the original Lintul5 (Wolf, 2012) symbol,
      lowercased.
    * Constants prefixed with ``c`` in SIMPLACE drop the ``c`` prefix here.
    * Tables originally named ``cXxxTableY`` are concatenated into a single
      two-column tensor named ``xxx_tb`` (or ``xxx_table``).

References:
    Wolf, J. (2012). *User guide for LINTUL5*. Wageningen UR.
"""

from __future__ import annotations

import functools
from dataclasses import InitVar, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch

# Directory holding the per-crop YAML presets generated from the SIMPLACE
# reference XML files by ``scripts/convert_crop_xml_to_yaml.py``.
_CROP_DATA_DIR = Path(__file__).resolve().parent / "crop_data"


def _t(x: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Build a scalar tensor with the requested dtype.

    Args:
        x: Python scalar value.
        dtype: Target tensor dtype.

    Returns:
        A 0-dimensional `torch.Tensor` holding ``x``.
    """
    return torch.tensor(x, dtype=dtype)


def _table(
    rows: list[tuple[float, float]],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an ``[N, 2]`` interpolation table tensor.

    Args:
        rows: List of ``(x, y)`` breakpoints.
        dtype: Target tensor dtype.

    Returns:
        A 2-D tensor of shape ``[N, 2]`` with abscissa in column 0 and
        ordinate in column 1.
    """
    return torch.tensor(rows, dtype=dtype)


def _check_discrete(
    name: str, value: torch.Tensor, allowed: tuple[float, ...]
) -> None:
    """Assert that every element of ``value`` is one of ``allowed``.

    Discrete switch fields are matched by exact equality downstream, so
    an off-domain value would silently snap to the nearest mode. This
    helper turns that into an explicit error.

    Args:
        name: Field name, used in the error message.
        value: Tensor of switch values (scalar or batched ``[B]``).
        allowed: The permitted discrete values.

    Raises:
        ValueError: If any element of ``value`` is not in ``allowed``.
    """
    valid = torch.zeros_like(value, dtype=torch.bool)
    for a in allowed:
        valid = valid | torch.isclose(value, torch.full_like(value, a))
    if not bool(valid.all()):
        allowed_str = ", ".join(
            str(int(a)) if float(a).is_integer() else str(a) for a in allowed
        )
        raise ValueError(
            f"{name} must be one of {{{allowed_str}}}; got {value.tolist()}"
        )


def _import_yaml():
    """Import PyYAML lazily with a helpful error if it is missing.

    PyYAML is only needed to read the bundled crop presets; importing it
    lazily keeps ``import torchcrop`` working when the optional dependency
    is absent (until a crop preset is actually requested).

    Returns:
        The imported ``yaml`` module.

    Raises:
        ImportError: If PyYAML is not installed.
    """
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without PyYAML
        raise ImportError(
            "Loading crop presets by name requires PyYAML. "
            "Install it with `pip install pyyaml`."
        ) from exc
    return yaml


# Default crop name loaded by ``CropParameters()`` when neither ``crop_name``
# nor ``config_file`` is supplied.
_DEFAULT_CROP = "default"

# Sentinel for the ``crop_name`` init-var: distinguishes "argument omitted"
# (→ load the default preset) from an explicit ``crop_name=None`` (→ skip
# preset loading and keep the field values as passed, used internally by
# `CropParameters.to`).
_UNSET: Any = object()


def _slugify(name: str) -> str:
    """Normalise a crop name to its YAML file stem (lowercase, underscores).

    Spaces and hyphens collapse to single underscores, so ``"grain maize"``,
    ``"Grain-Maize"`` and ``"grain_maize"`` all resolve to ``grain_maize``.

    Args:
        name: User-supplied crop name.

    Returns:
        The normalised lookup key / file stem.
    """
    return "_".join(str(name).strip().lower().replace("-", " ").split())


@functools.lru_cache(maxsize=1)
def available_crops() -> tuple[str, ...]:
    """Return the built-in crop names loadable via ``crop_name``.

    Each name is the stem of a bundled YAML preset in
    ``parameters/crop_data/`` (e.g. ``"wheat"``, ``"maize"``, ``"soybean"``)
    and is also accepted, case- and whitespace-insensitively, by
    ``CropParameters(crop_name=...)``.

    Returns:
        A sorted tuple of available crop names.
    """
    return tuple(sorted(p.stem for p in _CROP_DATA_DIR.glob("*.yaml")))


def _builtin_crop_path(crop_name: str) -> Path:
    """Resolve a crop name to its bundled preset path.

    Args:
        crop_name: Crop name (see `available_crops`), case- and
            whitespace-insensitive.

    Returns:
        Path to the matching ``<crop>.yaml`` preset.

    Raises:
        ValueError: If no built-in preset matches ``crop_name``.
    """
    path = _CROP_DATA_DIR / f"{_slugify(crop_name)}.yaml"
    if not path.is_file():
        raise ValueError(
            f"Unknown crop {crop_name!r}. Available crops: "
            f"{', '.join(available_crops())}."
        )
    return path


def _iter_preset_groups(
    preset: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Collect ``(scalars, tables)`` groups from a preset, any layout.

    Yields the top-level flat ``scalars``/``tables`` (legacy layout) first,
    followed by the ``scalars``/``tables`` of every entry under a top-level
    ``sections`` mapping (current sectioned layout). Missing keys resolve to
    empty dicts, so both layouts — and a mix of the two — are handled
    uniformly by `CropParameters._apply_preset`.

    Args:
        preset: Parsed preset mapping.

    Returns:
        A list of ``(scalars, tables)`` dict pairs to apply in order.
    """
    groups: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (preset.get("scalars") or {}, preset.get("tables") or {})
    ]
    for section in (preset.get("sections") or {}).values():
        if isinstance(section, dict):
            groups.append(
                (section.get("scalars") or {}, section.get("tables") or {})
            )
    return groups


def _load_preset_file(path: Path) -> dict[str, Any]:
    """Load and parse a crop preset YAML file.

    Args:
        path: Path to a YAML file following the crop-preset schema
            (``scalars`` and ``tables`` mappings, plus optional metadata).

    Returns:
        The parsed preset dict.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Crop configuration file not found: {path}")
    yaml = _import_yaml()
    with open(path) as fh:
        return yaml.safe_load(fh)


@dataclass
class CropParameters:
    """Species-specific Lintul5 crop parameters.

    Scalar fields have shape ``[]`` or ``[B]`` (broadcastable against the
    batch dimension). Table fields have shape ``[N, 2]`` or ``[B, N, 2]``.

    Parameters are loaded from a YAML preset on construction:

        params = CropParameters()                       # built-in default.yaml
        params = CropParameters(crop_name="wheat")      # built-in by name
        params = CropParameters(crop_name="maize")
        params = CropParameters(config_file="my_crop.yaml")  # custom file

    ``crop_name`` selects a bundled preset from ``parameters/crop_data/`` and
    is matched case- and whitespace-insensitively (``"grain maize"`` →
    ``maize``); see `available_crops` for the full list. ``config_file``
    loads a user-provided YAML, making it easy to add custom
    parameterisations. With neither argument, the built-in ``default`` preset
    is loaded. ``crop_name`` and ``config_file`` are mutually exclusive.

    Two preset layouts are accepted (see `_apply_preset`): the **sectioned**
    layout the bundled presets use — a ``sections`` mapping of thematic
    groups (``Development``, ``Water Use``, ``Nutrient (N-P-K) Use``,
    ``Root Growth``, ``Stress Response``, …), each with its own ``scalars``
    and/or ``tables`` — and the legacy **flat** layout with top-level
    ``scalars`` and ``tables`` mappings. Both key on field names.

    A preset overwrites every field it defines; any field it omits keeps the
    class default (e.g. torchcrop-specific extensions not present in the
    SIMPLACE source). Customise individual values afterwards
    (``params.rue = ...``) or use `from_crop_name` for dtype control.
    """

    # ------------------------------------------------------------------ #
    # 1. Phenology (from Phenology.java)
    # ------------------------------------------------------------------ #

    idsl: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cIDSL``. Phenology mode selector:
    ``0`` → temperature only; ``1`` → temperature + day length;
    ``2`` → temperature + day length + vernalisation. Stored as float so it
    can be used in differentiable masking; cast to int for branching."""

    tbasem: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cTBASEM``. Lower threshold (base) temperature [°C] for
    accumulation of temperature sum **before** crop emergence."""

    teffmx: torch.Tensor = field(default_factory=lambda: _t(30.0))
    """``cTEFFMX``. Maximum effective temperature [°C] used when
    accumulating temperature sum **before** emergence (caps TEFF)."""

    tsumem: torch.Tensor = field(default_factory=lambda: _t(110.0))
    """``cTSUMEM``. Required temperature sum [°C·d] from sowing to
    emergence."""

    tsum1: torch.Tensor = field(default_factory=lambda: _t(900.0))
    """``cTSUM1``. Required temperature sum [°C·d] from emergence to
    flowering / anthesis (vegetative period, DVS 0 → 1)."""

    tsum2: torch.Tensor = field(default_factory=lambda: _t(700.0))
    """``cTSUM2``. Required temperature sum [°C·d] from anthesis to
    maturity (generative period, DVS 1 → 2)."""

    dvsi: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cDVSI``. Initial development stage of the crop at the start of
    the simulation (in the range ``0`` … ``2``)."""

    dtsmtb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(-5.0, 0.0), (0.0, 0.0), (30.0, 30.0), (45.0, 30.0)]
        )
    )
    """``cDTSMTB`` (= ``cTsumIncrementTableMeanTemp`` × ``cTsumIncrementTableRate``).
    Daily increment of temperature sum [°C·d] as a function of mean daily
    air temperature [°C]. Shape ``[N, 2]`` with x = T̄, y = ΔTsum/d."""

    phottb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(0.0, 0.0), (8.0, 1.0), (12.0, 1.0), (18.0, 1.0)]
        )
    )
    """``cPHOTTB`` (= ``cPhotoperiodTableHour`` × ``cPhotoperiodTableFactor``).
    Photoperiod reduction factor [-] of development rate (until anthesis)
    as a function of day length [h]."""

    vernrt: torch.Tensor = field(
        default_factory=lambda: _table([(0.0, 1.0), (1.0, 1.0)])
    )
    """``cVERNRT`` (= ``cVernalisationTableMeanTemp`` × ``cVernalisationTableRate``).
    Daily vernal-day rate [-] as a function of mean daily temperature [°C].
    Used only when `idsl` ≥ 2."""

    vbase: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cVBASE``. Vernalisation base [thermal day]: vernal days
    accumulated below this value contribute nothing to the vernalisation
    factor."""

    versat: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cVERSAT``. Vernalisation saturation [thermal day]: number of
    vernal days above which the vernalisation factor is fully released
    (=1)."""

    vernalisation_devstage: torch.Tensor = field(default_factory=lambda: _t(0.3))
    """``cVernalisationDevStage``. Maximum DVS [-] up to which the
    vernalisation factor is applied; beyond this stage VERNFAC = 1."""

    minimal_vernalisation_factor: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cMinimalVernalisationFactor``. Lower bound [0–1] on the
    vernalisation factor used to limit pheno-rate suppression."""

    # ------------------------------------------------------------------ #
    # 2. Radiation use efficiency / RUE (from RadiationUseEfficiency.java)
    # ------------------------------------------------------------------ #

    day_temp_factor: torch.Tensor = field(default_factory=lambda: _t(0.25))
    """``cDayTempFactor``. Weight ``f`` in
    ``T_day = TMAX − f·(TMAX − TMIN)`` used to derive a *daytime* mean
    temperature from min/max. ``f = 0.25`` → daytime mean (default);
    ``f = 0.5`` → 24-h mean."""

    rue: torch.Tensor = field(default_factory=lambda: _t(3.0))
    """``cRUETableRUE`` (scalar surrogate). Reference radiation use
    efficiency [g DM · MJ⁻¹ PAR] used when `ruetb` is omitted.
    Lintul5 wheat default is 3.0 g · MJ⁻¹."""

    ruetb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(0.0, 3.0), (1.0, 3.0), (1.3, 3.0), (2.0, 0.4)]
        )
    )
    """``cRUETB`` (= ``cRUETableDVS`` × ``cRUETableRUE``). Radiation use
    efficiency [g DM · MJ⁻¹ PAR] as a function of DVS (declines after
    grain filling)."""

    tmpftb: torch.Tensor = field(
        default_factory=lambda: _table(
            [
                (-1.0, 0.0),
                (0.0, 0.0),
                (10.0, 0.6),
                (15.0, 1.0),
                (30.0, 1.0),
                (35.0, 0.0),
                (40.0, 0.0),
            ]
        )
    )
    """``cTMPFTB``. RUE reduction factor [-] as a function of mean daytime
    temperature [°C] (high- and low-temperature stress)."""

    tmnftb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(-5.0, 0.0), (0.0, 0.0), (3.0, 1.0), (30.0, 1.0)]
        )
    )
    """``cTMNFTB``. RUE reduction factor [-] as a function of daily
    minimum temperature [°C] (cold-night stress)."""

    cotb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(40.0, 0.0), (360.0, 1.0), (720.0, 1.35), (1000.0, 1.50), (2000.0, 1.50)]
        )
    )
    """``cCOTB``. CO₂ correction factor [-] applied to RUE as a function of
    atmospheric CO₂ concentration [ppm]."""

    scale_factor_rue: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cScaleFactorRUE``. Multiplicative scale factor on the y-values of
    `ruetb` for sensitivity analysis / calibration."""

    # ------------------------------------------------------------------ #
    # 3. Light interception / canopy (from Lintul5.java)
    # ------------------------------------------------------------------ #

    kdiftb: torch.Tensor = field(
        default_factory=lambda: _table([(0.0, 0.6), (2.0, 0.6)])
    )
    """``cKDIFTB``. Extinction coefficient [-] for diffuse PAR as a
    function of DVS."""

    scale_factor_kdif: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cScaleFactorKDIF``. Scale factor on `kdiftb` y-values."""

    laicr: torch.Tensor = field(default_factory=lambda: _t(4.0))
    """``cLAICR``. Critical leaf area index [m² m⁻²] above which leaves
    suffer self-shading mortality."""

    # ------------------------------------------------------------------ #
    # 3b. Crop water-use response (from WaterBalance.java)
    # ------------------------------------------------------------------ #
    # ``cDEPNR`` and ``cIAIRDU`` are consumed by the water balance but are
    # *crop* traits in SIMPLACE — they live in the crop data file
    # (``simplace/crop_default/*.xml``, "Water Use" group) and vary by
    # species (e.g. rice has ``IAIRDU = 1`` and ``DEPNR = 3.5``), so they
    # belong here rather than on `SoilParameters`. The water balance reads
    # them through `Lintul5Model`, which forwards these crop fields.

    depnr: torch.Tensor = field(default_factory=lambda: _t(4.5))
    """``cDEPNR``. Crop group number [-] for soil-water depletion
    (Doorenbos & Kassam). Used to compute the critical soil-moisture
    content above which transpiration is unrestricted."""

    iairdu: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cIAIRDU``. Boolean flag (0/1) indicating whether the crop has
    air ducts in its roots (=1, e.g. rice → tolerates waterlogging) or
    not (=0). Stored as float for batch broadcasting."""

    # ------------------------------------------------------------------ #
    # 4. Initial biomass and rooting (from Lintul5.java)
    # ------------------------------------------------------------------ #

    tdwi: torch.Tensor = field(default_factory=lambda: _t(210.0))
    """``cTDWI``. Initial total crop dry weight [g DM m⁻²] at sowing /
    emergence; partitioned to organs via the partitioning tables."""

    rdi: torch.Tensor = field(default_factory=lambda: _t(0.10))
    """``cRDI``. Initial rooting depth [m] (also used by WaterBalance)."""

    rri: torch.Tensor = field(default_factory=lambda: _t(0.012))
    """``cRRI``. Maximum daily increase in rooting depth [m d⁻¹]."""

    rdmcr: torch.Tensor = field(default_factory=lambda: _t(1.20))
    """``cRDMCR``. Crop-specific maximum rooting depth [m]."""

    # ------------------------------------------------------------------ #
    # 5. Leaf dynamics & senescence (from Lintul5.java)
    # ------------------------------------------------------------------ #

    slatb: torch.Tensor = field(
        default_factory=lambda: _table([(0.0, 0.0212), (2.0, 0.0212)])
    )
    """``cSLATB``. Specific leaf area [m² g⁻¹] as a function of DVS."""

    scale_factor_sla: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cScaleFactorSLA``. Scale factor on `slatb` y-values."""

    laii: torch.Tensor = field(default_factory=lambda: _t(0.012))
    """``LAII`` initial leaf area index [m² m⁻²] at emergence (Lintul5
    output, persisted here as a parameter for re-init)."""

    rgrl: torch.Tensor = field(default_factory=lambda: _t(0.009))
    """``cRGRLAI``. Maximum relative increase in LAI [d⁻¹] during the
    juvenile (exponential) phase."""

    tbase: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cTBASE``. Lower threshold temperature [°C] for LAI increase."""

    rdrshm: torch.Tensor = field(default_factory=lambda: _t(0.03))
    """``cRDRSHM``. Maximum relative death rate of leaves [d⁻¹] caused by
    shading when LAI > `laicr`."""

    rdrl: torch.Tensor = field(default_factory=lambda: _t(0.05))
    """``cRDRL``. Maximum relative death rate of leaves [d⁻¹] due to
    water stress."""

    rdrns: torch.Tensor = field(default_factory=lambda: _t(0.05))
    """``cRDRNS``. Maximum relative death rate of leaves [d⁻¹] due to
    NPK (nutrient) stress."""

    rdrltb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(-10.0, 0.0), (10.0, 0.02), (15.0, 0.03), (30.0, 0.05), (50.0, 0.09)]
        )
    )
    """``cRDRLTB``. Relative death rate of leaves [d⁻¹] as a function of
    mean daily temperature [°C] (heat-stress curve)."""

    rdrrtb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(0.0, 0.0), (1.5, 0.0), (1.5001, 0.02), (2.0, 0.02)]
        )
    )
    """``cRDRRTB``. Relative death rate of roots [d⁻¹] as a function of
    DVS."""

    rdrstb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(0.0, 0.0), (1.5, 0.0), (1.5001, 0.02), (2.0, 0.02)]
        )
    )
    """``cRDRSTB``. Relative death rate of stems [d⁻¹] as a function of
    DVS."""

    scale_factor_rdr_leaves: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cScaleFactorRDRLeaves``. Scale factor on `rdrltb`."""

    scale_factor_rdr_stems: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cScaleFactorRDRStems``. Scale factor on `rdrstb`."""

    scale_factor_rdr_roots: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cScaleFactorRDRRoots``. Scale factor on `rdrrtb`."""

    # ------------------------------------------------------------------ #
    # 6. Biomass partitioning (from Lintul5.java)
    # ------------------------------------------------------------------ #

    frtb: torch.Tensor = field(
        default_factory=lambda: _table(
            [
                (0.0, 0.50),
                (0.1, 0.50),
                (0.2, 0.40),
                (0.35, 0.22),
                (0.4, 0.17),
                (0.5, 0.13),
                (0.7, 0.07),
                (0.9, 0.03),
                (1.2, 0.0),
                (2.0, 0.0),
            ]
        )
    )
    """``cFRTB``. Fraction of total daily DM growth allocated to **roots**
    as a function of DVS. The remainder (``1 − FR``) is split among leaves,
    stems and storage organs by ``FLTB``, ``FSTB``, ``FOTB``."""

    fltb: torch.Tensor = field(
        default_factory=lambda: _table(
            [
                (0.0, 0.65),
                (0.1, 0.65),
                (0.25, 0.70),
                (0.5, 0.50),
                (0.646, 0.30),
                (0.95, 0.0),
                (2.0, 0.0),
            ]
        )
    )
    """``cFLTB``. Fraction of above-ground DM allocated to **leaves** as a
    function of DVS."""

    fstb: torch.Tensor = field(
        default_factory=lambda: _table(
            [
                (0.0, 0.35),
                (0.1, 0.35),
                (0.25, 0.30),
                (0.5, 0.50),
                (0.646, 0.70),
                (0.95, 1.0),
                (1.0, 0.0),
                (2.0, 0.0),
            ]
        )
    )
    """``cFSTB``. Fraction of above-ground DM allocated to **stems** as a
    function of DVS."""

    fotb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(0.0, 0.0), (0.95, 0.0), (1.0, 1.0), (2.0, 1.0)]
        )
    )
    """``cFOTB``. Fraction of above-ground DM allocated to **storage
    organs** as a function of DVS."""

    # ------------------------------------------------------------------ #
    # 7. NPK demand and concentration limits (from Lintul5.java)
    # ------------------------------------------------------------------ #

    nmxlv: torch.Tensor = field(
        default_factory=lambda: _table(
            [(0.0, 0.06), (0.4, 0.04), (0.7, 0.03), (1.0, 0.02), (2.0, 0.014), (2.1, 0.017)]
        )
    )
    """``cNMXLV``. Maximum N concentration in leaves [g N · g⁻¹ DM] as a
    function of DVS. Drives N demand and the N nutrition index NNI."""

    pmxlv: torch.Tensor = field(
        default_factory=lambda: _table(
            [(0.0, 0.011), (0.4, 0.008), (0.7, 0.006), (1.0, 0.004), (2.0, 0.0027), (2.1, 0.0027)]
        )
    )
    """``cPMXLV``. Maximum P concentration in leaves [g P · g⁻¹ DM] vs
    DVS."""

    kmxlv: torch.Tensor = field(
        default_factory=lambda: _table(
            [(0.0, 0.12), (0.4, 0.08), (0.7, 0.06), (1.0, 0.04), (2.0, 0.028), (2.1, 0.028)]
        )
    )
    """``cKMXLV``. Maximum K concentration in leaves [g K · g⁻¹ DM] vs
    DVS."""

    nmaxso: torch.Tensor = field(default_factory=lambda: _t(0.0176))
    """``cNMAXSO``. Maximum N concentration [g N · g⁻¹ DM] in storage
    organs."""

    pmaxso: torch.Tensor = field(default_factory=lambda: _t(0.0026))
    """``cPMAXSO``. Maximum P concentration in storage organs."""

    kmaxso: torch.Tensor = field(default_factory=lambda: _t(0.0048))
    """``cKMAXSO``. Maximum K concentration in storage organs."""

    lrnr: torch.Tensor = field(default_factory=lambda: _t(0.5))
    """``cLRNR``. Maximum N concentration in **roots** expressed as a
    fraction of the maximum N concentration in leaves [-]."""

    lrpr: torch.Tensor = field(default_factory=lambda: _t(0.5))
    """``cLRPR``. As `lrnr` but for P."""

    lrkr: torch.Tensor = field(default_factory=lambda: _t(0.5))
    """``cLRKR``. As `lrnr` but for K."""

    lsnr: torch.Tensor = field(default_factory=lambda: _t(0.5))
    """``cLSNR``. Maximum N concentration in **stems** as a fraction of
    the maximum N concentration in leaves [-]."""

    lspr: torch.Tensor = field(default_factory=lambda: _t(0.5))
    """``cLSPR``. As `lsnr` but for P."""

    lskr: torch.Tensor = field(default_factory=lambda: _t(0.5))
    """``cLSKR``. As `lsnr` but for K."""

    frnx: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cFRNX``. Optimal N concentration as a fraction of the maximum N
    concentration [-] — controls the N stress index NNI."""

    frpx: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cFRPX``. Optimal P concentration as a fraction of maximum P [-]."""

    frkx: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cFRKX``. Optimal K concentration as a fraction of maximum K [-]."""

    rnflv: torch.Tensor = field(default_factory=lambda: _t(0.004))
    """``cRNFLV``. Residual (non-translocatable) N concentration in
    leaves [g N · g⁻¹ DM]."""

    rnfst: torch.Tensor = field(default_factory=lambda: _t(0.002))
    """``cRNFST``. Residual N concentration in stems."""

    rnfrt: torch.Tensor = field(default_factory=lambda: _t(0.002))
    """``cRNFRT``. Residual N concentration in roots."""

    rpflv: torch.Tensor = field(default_factory=lambda: _t(0.0005))
    """``cRPFLV``. Residual P concentration in leaves."""

    rpfst: torch.Tensor = field(default_factory=lambda: _t(0.0003))
    """``cRPFST``. Residual P concentration in stems."""

    rpfrt: torch.Tensor = field(default_factory=lambda: _t(0.0003))
    """``cRPFRT``. Residual P concentration in roots."""

    rkflv: torch.Tensor = field(default_factory=lambda: _t(0.009))
    """``cRKFLV``. Residual K concentration in leaves."""

    rkfst: torch.Tensor = field(default_factory=lambda: _t(0.005))
    """``cRKFST``. Residual K concentration in stems."""

    rkfrt: torch.Tensor = field(default_factory=lambda: _t(0.005))
    """``cRKFRT``. Residual K concentration in roots."""

    fntrt: torch.Tensor = field(default_factory=lambda: _t(0.15))
    """``cFNTRT``. NPK translocation from **roots** expressed as a
    fraction of the total NPK translocated from leaves and stems [-]."""

    tcnt: torch.Tensor = field(default_factory=lambda: _t(10.0))
    """``cTCNT``. Time constant [d] for N translocation to storage
    organs (first-order kinetics)."""

    tcpt: torch.Tensor = field(default_factory=lambda: _t(10.0))
    """``cTCPT``. Time constant [d] for P translocation to storage
    organs."""

    tckt: torch.Tensor = field(default_factory=lambda: _t(10.0))
    """``cTCKT``. Time constant [d] for K translocation to storage
    organs."""

    nfixf: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cNFIXF``. Fraction of crop N uptake supplied by biological N₂
    fixation [-]; > 0 for legumes."""

    # ------------------------------------------------------------------ #
    # 8. Stress sensitivities (from Lintul5.java)
    # ------------------------------------------------------------------ #

    nlue: torch.Tensor = field(default_factory=lambda: _t(1.1))
    """``cNLUE``. Coefficient of the reduction of RUE due to NPK
    (nutrient) stress."""

    nlai: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cNLAI``. Coefficient of the reduction of LAI growth (juvenile
    phase) due to nutrient stress."""

    nsla: torch.Tensor = field(default_factory=lambda: _t(0.5))
    """``cNSLA``. Coefficient of the reduction of SLA due to nutrient
    stress."""

    npart: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cNPART``. Coefficient of the N-stress effect on leaf biomass
    reduction (re-allocation away from leaves under N deficiency)."""

    # ------------------------------------------------------------------ #
    # 9. DVS thresholds for NPK dynamics (from Lintul5.java)
    # ------------------------------------------------------------------ #

    dvsnt: torch.Tensor = field(default_factory=lambda: _t(0.8))
    """``cDVSNT``. DVS above which N, P and K **translocation** to
    storage organs occurs."""

    dvsnlt: torch.Tensor = field(default_factory=lambda: _t(1.3))
    """``cDVSNLT``. DVS above which crop **uptake** of N, P and K stops."""

    dvsdr: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cDVSDR``. DVS above which death of roots and stems begins."""

    dvsdlt: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cDVSDLT``. DVS above which leaf death (controlled by mean
    temperature, see `rdrltb`) begins."""

    # ------------------------------------------------------------------ #
    # 10. Fertiliser application & recovery (from Lintul5.java)
    # ------------------------------------------------------------------ #

    nrf: torch.Tensor = field(default_factory=lambda: _t(0.7))
    """``cNRF``. Default recovery fraction [0–1] of applied fertiliser N
    (used when no day-resolved `nrftab` is provided)."""

    prf: torch.Tensor = field(default_factory=lambda: _t(0.2))
    """``cPRF``. Default recovery fraction [0–1] of applied fertiliser P."""

    krf: torch.Tensor = field(default_factory=lambda: _t(0.6))
    """``cKRF``. Default recovery fraction [0–1] of applied fertiliser K."""

    nrftab: torch.Tensor | None = None
    """``cNRFTAB``. Optional day-resolved table of N fertiliser recovery
    fractions [-] (shape ``[N, 2]``: x = day, y = fraction)."""

    prftab: torch.Tensor | None = None
    """``cPRFTAB``. Optional day-resolved table of P recovery fractions."""

    krftab: torch.Tensor | None = None
    """``cKRFTAB``. Optional day-resolved table of K recovery fractions."""

    ferntab: torch.Tensor | None = None
    """``cFERNTAB``. Optional table of N fertiliser applications
    [g N · m⁻² · d⁻¹] at given calendar days. Shape ``[N, 2]``."""

    ferptab: torch.Tensor | None = None
    """``cFERPTAB``. Optional table of P fertiliser applications."""

    ferktab: torch.Tensor | None = None
    """``cFERKTAB``. Optional table of K fertiliser applications."""

    scale_factor_fern: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cScaleFactorFERN``. Scale factor on `ferntab` y-values."""

    scale_factor_ferp: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cScaleFactorFERP``. Scale factor on `ferptab` y-values."""

    scale_factor_ferk: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cScaleFactorFERK``. Scale factor on `ferktab` y-values."""

    # ------------------------------------------------------------------ #
    # 11. Run-control flags (from Lintul5.java)
    # ------------------------------------------------------------------ #

    iopt: torch.Tensor = field(default_factory=lambda: _t(1.0))
    """``cIOPT``. Run mode selector:
    ``1`` → optimal (no stresses), ``2`` → water-limited,
    ``3`` → water + N limited, ``4`` → water + N + P + K limited.
    Stored as float so it can be used in masking; cast to int for branching."""

    # ------------------------------------------------------------------ #
    # 12. Heat stress (optional add-on components)
    # ------------------------------------------------------------------ #
    # Crop-specific constants for the optional SIMPLACE heat-stress
    # components, which sit *outside* the core Lintul5 time loop:
    #   * HeatStressOnLeafSenescence — cTempCritical, cFactorAtTempCritical,
    #     cFactorSlope, cDevStageCritical
    #   * HeatStressOnGrain — cTempCritical, cTempLimit, cBeginDevStage,
    #     cEndDevStage
    # Both SIMPLACE components define a constant named ``cTempCritical``;
    # they are disambiguated here with ``leaf_heat_`` / ``grain_heat_``
    # prefixes (mirroring the `Lintul5Model` submodule names). Keeping
    # them in this dataclass lets a per-crop configuration file populate
    # them alongside the core Lintul5 parameters.
    # ------------------------------------------------------------------ #

    leaf_heat_temp_critical: torch.Tensor = field(default_factory=lambda: _t(34.0))
    """``cTempCritical`` of ``HeatStressOnLeafSenescence``. Critical daily
    maximum temperature [°C] above which heat stress accelerates leaf
    senescence."""

    leaf_heat_factor_at_temp_critical: torch.Tensor = field(
        default_factory=lambda: _t(3.0)
    )
    """``cFactorAtTempCritical``. Leaf-senescence multiplier [-] at exactly
    the critical temperature `leaf_heat_temp_critical`."""

    leaf_heat_factor_slope: torch.Tensor = field(default_factory=lambda: _t(0.5))
    """``cFactorSlope``. Increment of the leaf-senescence multiplier per °C
    above `leaf_heat_temp_critical` [°C⁻¹]."""

    leaf_heat_devstage_critical: torch.Tensor = field(
        default_factory=lambda: _t(1.0)
    )
    """``cDevStageCritical``. Development stage [-] above which heat stress
    on leaf senescence applies."""

    grain_heat_temp_critical: torch.Tensor = field(default_factory=lambda: _t(27.0))
    """``cTempCritical`` of ``HeatStressOnGrain``. Critical day-time
    temperature [°C] at which heat stress on grain yield starts."""

    grain_heat_temp_limit: torch.Tensor = field(default_factory=lambda: _t(40.0))
    """``cTempLimit``. Day-time temperature [°C] at which the daily grain
    heat-stress intensity saturates at 1."""

    grain_heat_begin_devstage: torch.Tensor = field(
        default_factory=lambda: _t(0.8)
    )
    """``cBeginDevStage``. Development stage [-] at which the grain
    thermally sensitive window opens."""

    grain_heat_end_devstage: torch.Tensor = field(default_factory=lambda: _t(1.3))
    """``cEndDevStage``. Development stage [-] at which the grain
    thermally sensitive window closes."""

    # ------------------------------------------------------------------ #
    # Init-only inputs (not stored as fields)
    # ------------------------------------------------------------------ #

    crop_name: InitVar = _UNSET
    """Optional crop name selecting a bundled preset from
    ``parameters/crop_data/`` (case- and whitespace-insensitive). When
    omitted, the ``default`` preset is loaded. Pass ``None`` to skip preset
    loading entirely (keep the field values as constructed). Mutually
    exclusive with `config_file`. Not retained as an attribute."""

    config_file: InitVar = None
    """Optional path to a user-provided YAML file (same schema as the
    bundled presets) to load instead of a built-in crop. Mutually exclusive
    with `crop_name`. Not retained as an attribute."""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def __post_init__(
        self, crop_name: Any, config_file: str | Path | None
    ) -> None:
        """Load a crop preset into the parameter fields.

        Resolution:
            * ``config_file`` given → load that YAML file.
            * ``crop_name`` given (a string) → load the matching bundled
              preset via `_builtin_crop_path`.
            * ``crop_name`` omitted and no ``config_file`` → load the
              built-in `_DEFAULT_CROP` preset.
            * ``crop_name=None`` (explicit) → skip loading; retain the field
              values as passed (used internally by `to`).

        Args:
            crop_name: Crop name, ``None`` (skip), or the `_UNSET` sentinel
                (load default).
            config_file: Path to a custom preset YAML, or ``None``.

        Raises:
            ValueError: If both ``crop_name`` and ``config_file`` are given,
                or if ``crop_name`` matches no bundled preset.
            FileNotFoundError: If ``config_file`` does not exist.
        """
        if config_file is not None:
            if crop_name not in (_UNSET, None):
                raise ValueError(
                    "Pass either crop_name or config_file, not both."
                )
            self._apply_preset(_load_preset_file(Path(config_file)))
            return

        if crop_name is None:
            return  # explicit skip — keep field values as constructed

        name = _DEFAULT_CROP if crop_name is _UNSET else crop_name
        self._apply_preset(_load_preset_file(_builtin_crop_path(name)))

    def _apply_preset(self, preset: dict[str, Any]) -> None:
        """Overwrite fields from a parsed preset dict.

        Two preset layouts are supported and may be freely mixed:

        * **Sectioned** (current converter output): a top-level
          ``sections`` mapping whose values are thematic groups
          (``Development``, ``Water Use``, ``Nutrient (N-P-K) Use``,
          ``Root Growth``, ``Stress Response``, …), each carrying its own
          ``scalars`` and/or ``tables`` sub-mapping.
        * **Flat** (legacy / hand-written): top-level ``scalars`` and
          ``tables`` mappings.

        Both forms key on `CropParameters` field names; only the fields
        present are overwritten. Reading both keeps older presets and
        user config files loading unchanged.

        Args:
            preset: Parsed preset mapping in either layout.
        """
        for scalars, tables in _iter_preset_groups(preset):
            for name, value in scalars.items():
                setattr(self, name, _t(float(value)))
            for name, rows in tables.items():
                setattr(
                    self, name, _table([(float(x), float(y)) for x, y in rows])
                )

    def validate(self) -> None:
        """Validate discrete/categorical crop fields.

        Checks the two enumerated run-mode selectors against their
        supported discrete domains (per element when batched):

        * ``idsl`` ∈ {0, 1, 2} — phenology mode (temperature only /
          + day length / + vernalisation).
        * ``iopt`` ∈ {1, 2, 3, 4} — run mode (optimal / water-limited /
          + N-limited / + NPK-limited).
        * ``iairdu`` ∈ {0, 1} — root air-ducts flag (``0`` → non-aquatic,
          ``1`` → aquatic, e.g. rice).

        All three are consumed through hard threshold comparisons
        (``idsl >= 1``/``>= 2``; ``iopt <= 2.5``/``<= 3.5``;
        ``iairdu > 0.5``), so an off-domain value would silently snap to
        the nearest mode.

        Raises:
            ValueError: If ``idsl``, ``iopt`` or ``iairdu`` holds an
                unsupported value.
        """
        _check_discrete("crop_params.idsl", self.idsl, (0.0, 1.0, 2.0))
        _check_discrete("crop_params.iopt", self.iopt, (1.0, 2.0, 3.0, 4.0))
        _check_discrete("crop_params.iairdu", self.iairdu, (0.0, 1.0))

    def to(
        self,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "CropParameters":
        """Cast and/or move all tensor fields to a new dtype/device.

        Args:
            dtype: Target tensor dtype, or ``None`` to leave unchanged.
            device: Target torch device, or ``None`` to leave unchanged.

        Returns:
            A new `CropParameters` with every tensor field moved /
            cast; non-tensor fields (e.g., optional tables set to ``None``)
            are copied through unchanged.
        """
        kwargs: dict[str, Any] = {}
        for f in fields(self):
            t = getattr(self, f.name)
            if isinstance(t, torch.Tensor):
                kwargs[f.name] = t.to(dtype=dtype, device=device)
            else:
                kwargs[f.name] = t
        # crop_name=None skips preset loading so the field values built above
        # are preserved rather than being overwritten by the default preset.
        return CropParameters(crop_name=None, **kwargs)

    @classmethod
    def from_crop_name(
        cls,
        crop_name: str,
        dtype: torch.dtype = torch.float32,
    ) -> "CropParameters":
        """Build a `CropParameters` for a named crop preset.

        Equivalent to ``CropParameters(crop_name=crop_name).to(dtype=dtype)``
        but with explicit dtype control.

        Args:
            crop_name: Crop name (e.g. ``"wheat"``, ``"maize"``), case- and
                whitespace-insensitive. See `available_crops` for the full
                list.
            dtype: Target tensor dtype for all scalar/tabular fields.

        Returns:
            A `CropParameters` populated from the requested preset.

        Raises:
            ValueError: If ``crop_name`` matches no bundled preset.
        """
        return cls(crop_name=crop_name).to(dtype=dtype)


def default_wheat_params(dtype: torch.dtype = torch.float32) -> CropParameters:
    """Return the SIMPLACE Lintul5 wheat-like default parameter set.

    Loads the built-in ``default`` preset (the wheat-like SIMPLACE default),
    equivalent to ``CropParameters(crop_name="default")``.

    Args:
        dtype: Target tensor dtype for all scalar/tabular fields.

    Returns:
        A fresh `CropParameters` with the Lintul5 default parameters
        cast to ``dtype``.
    """
    return CropParameters().to(dtype=dtype)
