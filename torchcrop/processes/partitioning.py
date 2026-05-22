"""Biomass allocation to plant organs, with water- and N-stress modification.

References:
    * Biomass partitioning block of ``Lintul5.java``.
    * ``LintulFunctions.SUBPAR`` — modifies the baseline partitioning
      fractions in response to water and nitrogen stress.
    * ``LintulFunctions.RELGR`` — converts the partitioning fractions into per-organ growth rates.

Design:
    The baseline fractions ``FRTWET``, ``FLVT``, ``FSTT``, ``FSOT`` are read from
    the interpolation tables ``frtb``, ``fltb``, ``fstb``, ``fotb`` at the
    current DVS. ``SUBPAR`` then adjusts them depending on which stress is
    limiting:

    * **Water stress more severe** (``TRANRF < NNI``): roots receive a larger
    share via ``FRTMOD = max(1, 1/(TRANRF + 0.5))``, capped at
    ``FRT ≤ 0.6``. Above-ground fractions are unchanged.
    * **N stress more severe** (otherwise): leaf allocation is multiplied by
    ``FLVMOD = exp(-NPART · (1 − NNI))``; the deficit is re-routed to
    stems (``FST ← FSTT + FLVT − FLV``). Roots and storage organs are
    unchanged.

Equations:
    Final organ growth rates follow ``RELGR``:

    $$
    \\begin{aligned}
    G_{\\text{root}} &= G_{\\text{total}} \\cdot F_{rt} \\\\
    G_{\\text{lv}}   &= G_{\\text{total}} \\cdot (1 - F_{rt}) \\cdot F_{lv} \\\\
    G_{\\text{st}}   &= G_{\\text{total}} \\cdot (1 - F_{rt}) \\cdot F_{st} \\\\
    G_{\\text{so}}   &= G_{\\text{total}} \\cdot (1 - F_{rt}) \\cdot F_{so}
    \\end{aligned}
    $$
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.states.model_state import ModelState


class Partitioning(nn.Module):
    """Allocate daily gross biomass production to organ-specific pools.

    The DVS-indexed baseline fractions are first modified for water or N
    stress, then multiplied through by ``gtotal`` and the root/shoot
    split to obtain per-organ growth rates.
    """

    @staticmethod
    def _subpar(
        npart: torch.Tensor,
        tranrf: torch.Tensor,
        nni: torch.Tensor,
        frtwet: torch.Tensor,
        flvt: torch.Tensor,
        fstt: torch.Tensor,
        fsot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the Java ``SUBPAR`` stress modification to partitioning.

        Args:
            npart: N-stress partitioning coefficient ``cNPART``, shape
                broadcastable to ``[B]``.
            tranrf: Water-stress (transpiration reduction) factor in
                ``[0, 1]``, shape ``[B]``.
            nni: Nitrogen Nutrition Index in ``[0, 1]`` (also used here
                as a proxy for the broader NPK index), shape ``[B]``.
            frtwet: Baseline root fraction ``FRTWET`` from ``frtb``,
                shape ``[B]``.
            flvt: Baseline leaf fraction ``FLVT`` from ``fltb``,
                shape ``[B]``.
            fstt: Baseline stem fraction ``FSTT`` from ``fstb``,
                shape ``[B]``.
            fsot: Baseline storage-organ fraction ``FSOT`` from ``fotb``,
                shape ``[B]``.

        Returns:
            Tuple ``(fr, fl, fs, fo)`` of stress-modified fractions, each
            of shape ``[B]``.
        """
        # Water-stress branch (TRANRF < NNI): roots get a larger share.
        frtmod_w = torch.clamp(1.0 / (tranrf + 0.5), min=1.0)
        fr_w = torch.clamp(frtwet * frtmod_w, max=0.6)
        fl_w = flvt
        fs_w = fstt
        fo_w = fsot

        # N-stress branch (TRANRF >= NNI): leaves shrink, stems absorb deficit.
        flvmod_n = torch.exp(-npart * (1.0 - nni))
        fl_n = flvt * flvmod_n
        fs_n = fstt + flvt - fl_n
        fr_n = frtwet
        fo_n = fsot

        water_more_severe = tranrf < nni
        fr = torch.where(water_more_severe, fr_w, fr_n)
        fl = torch.where(water_more_severe, fl_w, fl_n)
        fs = torch.where(water_more_severe, fs_w, fs_n)
        fo = torch.where(water_more_severe, fo_w, fo_n)
        return fr, fl, fs, fo

    def forward(
        self,
        state: ModelState,
        gtotal: torch.Tensor,
        params: CropParameters,
        tranrf: torch.Tensor | None = None,
        nni: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Split gross daily biomass production across organs.

        Args:
            state: Current state (uses ``state.dvs`` for table lookups).
            gtotal: Gross daily biomass production [g DM m⁻² d⁻¹], shape
                ``[B]``.
            params: Crop parameters; uses the partitioning tables ``frtb``,
                ``fltb``, ``fstb``, ``fotb`` and the N-stress
                partitioning coefficient ``npart``.
            tranrf: Water-stress factor in ``[0, 1]``, shape ``[B]``. If
                ``None`` (no stress information available), defaults to
                a tensor of ones, in which case ``SUBPAR`` reduces to
                the N-stress branch with ``FLVMOD = exp(0) = 1`` —
                i.e. baseline fractions are returned unchanged.
            nni: Nitrogen Nutrition Index in ``[0, 1]`` (or NPK index
                proxy), shape ``[B]``. If ``None`` defaults to ones
                (no N stress).

        Returns:
            Dict of ``[B]`` tensors grouped as follows.

            Rate variables (per-organ growth, fed to leaf/root/state updates
            and nutrient demand):

            * ``g_root`` [g DM m⁻² d⁻¹] — Root biomass growth rate
                (``= gtotal * fr``); becomes ``wrt_rate`` after
                subtracting root death in `RootDynamics`.
            * ``g_lv`` [g DM m⁻² d⁻¹] — Leaf growth before senescence
                (``= gtotal * (1 − fr) * fl``); `LeafDynamics`
                converts it into ``wlv_rate`` and ``lai_rate``.
            * ``g_st`` [g DM m⁻² d⁻¹] — Stem growth rate
                (``= gtotal * (1 − fr) * fs``); becomes ``wst_rate``
                directly.
            * ``g_so`` [g DM m⁻² d⁻¹] — Storage-organ growth rate
                (``= gtotal * (1 − fr) * fo``); becomes ``wso_rate``
                and drives final yield.

            Diagnostics:

            * ``fr`` [-] — Below-ground (root) fraction after stress
                modification.
            * ``fl``, ``fs``, ``fo`` [-] — Above-ground fractions to
                leaves, stems, storage organs after stress modification.
        """
        dvs = state.dvs
        frtwet = interpolate(params.frtb, dvs)
        flvt = interpolate(params.fltb, dvs)
        fstt = interpolate(params.fstb, dvs)
        fsot = interpolate(params.fotb, dvs)

        if tranrf is None:
            tranrf = torch.ones_like(dvs)
        if nni is None:
            nni = torch.ones_like(dvs)

        fr, fl, fs, fo = self._subpar(
            npart=params.npart,
            tranrf=tranrf,
            nni=nni,
            frtwet=frtwet,
            flvt=flvt,
            fstt=fstt,
            fsot=fsot,
        )

        # RELGR: gross per-organ growth before subtracting death rates
        # (death rates are handled in LeafDynamics / RootDynamics).
        agrt = gtotal * (1.0 - fr)
        g_root = gtotal * fr
        g_lv = fl * agrt
        g_st = fs * agrt
        g_so = fo * agrt

        return {
            "g_root": g_root,
            "g_lv": g_lv,
            "g_st": g_st,
            "g_so": g_so,
            "fr": fr,
            "fl": fl,
            "fs": fs,
            "fo": fo,
        }
