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

from dataclasses import dataclass, field, fields
from typing import Any

import torch


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


@dataclass
class CropParameters:
    """Species-specific Lintul5 crop parameters.

    Scalar fields have shape ``[]`` or ``[B]`` (broadcastable against the
    batch dimension). Table fields have shape ``[N, 2]`` or ``[B, N, 2]``.
    Default values reproduce the SIMPLACE Lintul5 defaults (wheat-like).
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

    tsum_branching: torch.Tensor = field(default_factory=lambda: _t(700.0))
    """``cTSUMBranching``. Optional temperature sum [°C·d] from emergence
    to branching (used by some calibrations; ignored otherwise)."""

    tsum_milkripeness: torch.Tensor = field(default_factory=lambda: _t(700.0))
    """``cTSUMMilkripeness``. Optional temperature sum [°C·d] from
    anthesis to milk-ripeness."""

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

    co2: torch.Tensor = field(default_factory=lambda: _t(360.0))
    """``cCO``. Atmospheric CO₂ concentration [ppm] used both as input to
    the RUE CO₂ correction (`cotb`) and the ET₀ CO₂ correction.
    Default is the original Lintul5 reference."""

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

    k: torch.Tensor = field(default_factory=lambda: _t(0.60))
    """Effective canopy light extinction coefficient [-]; scalar fallback
    when `kdiftb` is collapsed to a constant."""

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

    rwrti: torch.Tensor = field(default_factory=lambda: _t(0.0))
    """``cRWRTI``. Initial change in living root biomass
    [g DM m⁻² d⁻¹]."""

    # ------------------------------------------------------------------ #
    # 5. Leaf dynamics & senescence (from Lintul5.java)
    # ------------------------------------------------------------------ #

    sla: torch.Tensor = field(default_factory=lambda: _t(0.0212))
    """Reference specific leaf area [m² leaf · g⁻¹ DM]; scalar fallback
    used when `slatb` is collapsed."""

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
    # 12. Scaffold convenience scalars
    # ------------------------------------------------------------------ #
    # The fields below are simplified scalar surrogates used by the v0.0.1
    # process scaffold (single-value alternatives to the SIMPLACE
    # interpolation tables / fraction-based per-organ derivations above).
    # They are retained so existing process modules keep importing while
    # the table-based equations are progressively wired in.
    # ------------------------------------------------------------------ #

    rootdi: torch.Tensor = field(default_factory=lambda: _t(0.10))
    """Scaffold alias of `rdi` — initial rooting depth [m]."""

    rootdm: torch.Tensor = field(default_factory=lambda: _t(1.20))
    """Scaffold alias of `rdmcr` — maximum rooting depth [m]."""

    rrdmax: torch.Tensor = field(default_factory=lambda: _t(0.012))
    """Scaffold alias of `rri` — maximum daily root-depth growth
    rate [m d⁻¹]."""

    rdrtb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(0.0, 0.0), (1.0, 0.0), (1.5, 0.02), (2.0, 0.05)]
        )
    )
    """Simplified DVS-indexed leaf relative-death-rate table [d⁻¹] used
    by `LeafDynamics` for developmental senescence. Not a direct
    SIMPLACE constant (SIMPLACE separates leaf senescence into
    `rdrltb` (vs T), `rdrrtb` and `rdrstb` (vs DVS))."""

    nmaxlv: torch.Tensor = field(default_factory=lambda: _t(0.050))
    """Scalar max N concentration in leaves [g N · g⁻¹ DM] (scaffold
    surrogate of `nmxlv` table)."""

    nmaxst: torch.Tensor = field(default_factory=lambda: _t(0.020))
    """Scalar max N concentration in stems (≈ ``lsnr × nmaxlv``)."""

    nmaxrt: torch.Tensor = field(default_factory=lambda: _t(0.015))
    """Scalar max N concentration in roots (≈ ``lrnr × nmaxlv``)."""

    pmaxlv: torch.Tensor = field(default_factory=lambda: _t(0.008))
    """Scalar max P concentration in leaves (scaffold surrogate of
    `pmxlv` table)."""

    pmaxst: torch.Tensor = field(default_factory=lambda: _t(0.004))
    """Scalar max P concentration in stems (≈ ``lspr × pmaxlv``)."""

    pmaxrt: torch.Tensor = field(default_factory=lambda: _t(0.003))
    """Scalar max P concentration in roots (≈ ``lrpr × pmaxlv``)."""

    kmaxlv: torch.Tensor = field(default_factory=lambda: _t(0.040))
    """Scalar max K concentration in leaves (scaffold surrogate of
    `kmxlv` table)."""

    kmaxst: torch.Tensor = field(default_factory=lambda: _t(0.025))
    """Scalar max K concentration in stems (≈ ``lskr × kmaxlv``)."""

    kmaxrt: torch.Tensor = field(default_factory=lambda: _t(0.015))
    """Scalar max K concentration in roots (≈ ``lrkr × kmaxlv``)."""

    nresid: torch.Tensor = field(default_factory=lambda: _t(0.004))
    """Aggregated residual N concentration [g N · g⁻¹ DM] used by the
    scaffold (compare per-organ `rnflv`/`rnfst`/`rnfrt`)."""

    presid: torch.Tensor = field(default_factory=lambda: _t(0.001))
    """Aggregated residual P concentration."""

    kresid: torch.Tensor = field(default_factory=lambda: _t(0.005))
    """Aggregated residual K concentration."""

    tmpf_tb: torch.Tensor = field(
        default_factory=lambda: _table(
            [(-10.0, 0.0), (0.0, 0.0), (10.0, 0.6), (20.0, 1.0), (30.0, 1.0), (40.0, 0.0)]
        )
    )
    """Scaffold alias of `tmpftb` (T-response of RUE) used by the
    current `Photosynthesis` module. Will be merged with
    `tmpftb` once the full SIMPLACE RUE chain is wired in."""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

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
        return CropParameters(**kwargs)


def default_wheat_params(dtype: torch.dtype = torch.float32) -> CropParameters:
    """Return the SIMPLACE Lintul5 wheat-like default parameter set.

    Args:
        dtype: Target tensor dtype for all scalar/tabular fields.

    Returns:
        A fresh `CropParameters` with the Lintul5 wheat defaults
        cast to ``dtype``.
    """
    return CropParameters().to(dtype=dtype)
