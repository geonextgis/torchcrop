"""Crop nutrient demand, uptake, translocation and stress indices.

Mechanistic implementation of the Lintul5 NPK block. Every quantity
is a batch-compatible PyTorch tensor of shape ``[B]`` so the module
integrates into the differentiable simulation engine without breaking
the autograd graph or the batch dimension.

The module covers:

* Translocatable pools (``NTRLOC``) — leaf, stem, and root nutrient
  contents above residual concentrations.
* Deficit-based demand (``NDEMND``) — per-organ maximum vs. actual
  concentration, with storage-organ demand filtered by translocation
  time constants.
* Uptake (``NUptake``) — gated by emergence, the DVS uptake window,
  and the water-stress trigger; the ``IOPT`` run-mode switches
  between soil-pool-limited and demand-limited uptake.
* Concentration-based nutrition indices ``NNI``/``PNI``/``KNI``
  combined into ``NPKI = min(NNI, PNI, KNI)``.
* Per-organ partitioning of soil + fixation uptake by demand share
  (``RNUSUB``) and translocation to storage organs by translocatable-
  pool share (``NTRANS``).
* Death-related nutrient losses (``RNLD``) and net per-organ rate
  assembly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchcrop.functions import interpolate
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.parameters.soil_params import SoilParameters
from torchcrop.states.model_state import ModelState


def _safe(t: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Replace near-zero entries with one to make ``t`` safe to divide by."""
    return torch.where(t.abs() > eps, t, torch.ones_like(t))


class NutrientDemand(nn.Module):
    """Daily NPK demand, uptake, translocation, and nutrition indices.

    Implements the deficit-based Lintul5 NPK dynamics:

    1. *Demand* (``NDEMND``): per-organ deficit between the maximum
       (DVS-indexed) and actual organ concentration; storage-organ demand
       is filtered by the time constants ``TCNT``/``TCPT``/``TCKT``.
    2. *Translocatable pools* (``NTRLOC``): leaf/stem/root content above
       the residual concentrations ``RNFLV``/``RNFST``/``RNFRT`` (and
       analogues for P, K), with the root pool bounded by ``FNTRT``.
    3. *Uptake* (``NUptake``): gated by the emergence mask, the DVS
       window ``DVS < DVSNLT`` and the water-stress trigger
       ``TRANRF ≥ 0.01``. The run-mode flag ``IOPT`` selects between
       soil-pool-limited and demand-limited uptake for N (≤2 potential)
       and P/K (≤3 potential).
    4. *Concentration-based stress* (``NNINDX``): the historical
       nutrition indices ``NNI``/``PNI``/``KNI`` derived from
       ``(NFGMR - NRMR) / (NOPTMR - NRMR)``, combined into
       ``NPKI = min(NNI, PNI, KNI)``.
    5. *Per-organ partitioning* (``RNUSUB``) of soil + fixation uptake by
       demand-share, and *translocation* (``NTRANS``) of storage-organ
       supply by translocatable-pool share.
    6. *Net rate assembly* per organ
       ``R{N,P,K}{LV,ST,RT} = uptake − translocation − death-loss``.
    """

    def forward(
        self,
        state: ModelState,
        crop_params: CropParameters,
        soil_params: SoilParameters,
        tranrf: torch.Tensor | None = None,
        dlv: torch.Tensor | None = None,
        drrt: torch.Tensor | None = None,
        drst: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the full NPK rate package for one day.

        Args:
            state: Current `ModelState`. Reads the biomass pools
                (``wlv``, ``wst``, ``wrt``, ``wso``), the per-organ
                nutrient pools (``a{n,p,k}{lv,st,rt,so}``), ``dvs`` and
                ``tsump`` (the latter sets the EMERG mask via
                ``tsump ≥ tsumem``).
            crop_params: Crop parameters; reads the DVS-tables
                ``nmxlv``/``pmxlv``/``kmxlv``, the leaf/stem/root ratios
                ``lsnr``/``lrnr``/``lspr``/``lrpr``/``lskr``/``lrkr``,
                the storage-organ caps ``nmaxso``/``pmaxso``/``kmaxso``,
                the residual concentrations
                ``rnflv``/``rnfst``/``rnfrt`` (and P, K), the
                translocation time constants ``tcnt``/``tcpt``/``tckt``,
                the optimal-fraction multipliers ``frnx``/``frpx``/``frkx``,
                the root-translocation fraction ``fntrt``, the
                N-fixation fraction ``nfixf``, the DVS thresholds
                ``dvsnt`` (translocation start) and ``dvsnlt`` (uptake
                stop), the emergence sum ``tsumem``, and the run-mode
                flag ``iopt``.
            soil_params: Kept in the signature for API symmetry with the
                rest of the process modules; the soil-N supply cap is
                now read from `ModelState` (``nmint``/``pmint``/``kmint``,
                advanced by `SoilNutrients`) so this argument is unused
                here.
            tranrf: Optional water-stress factor in ``[0, 1]``, shape
                ``[B]`` (broadcastable). Drives the soil-uptake cut-off
                ``TRANRF < 0.01``. Defaults to ``1`` (no water stress)
                so the module is callable without water-balance output.
            dlv: Optional daily senesced leaf biomass [g DM m⁻² d⁻¹],
                shape ``[B]``. Feeds ``RNLDLV = RNFLV · DLV`` and its P, K
                analogues. Defaults to zero (death losses suppressed).
            drrt: Optional daily senesced root biomass. Defaults to zero.
            drst: Optional daily senesced stem biomass. Defaults to zero.

        Returns:
            Dict of ``[B]`` tensors grouped as follows.

            Rate variables (consumed by the engine to update the matching
            ``a{n,p,k}{lv,st,rt,so}`` state pools):

            * ``n_lv_rate``, ``n_st_rate``, ``n_rt_rate``,
              ``n_so_rate`` [g N m⁻² d⁻¹] — Net daily change in the
                per-organ N pool: ``uptake − translocation − loss``
                for vegetative organs, ``RNSO`` (translocation in) for
                the storage organ.
            * ``p_lv_rate``, ``p_st_rate``, ``p_rt_rate``,
              ``p_so_rate`` [g P m⁻² d⁻¹] — Per-organ net P rate.
            * ``k_lv_rate``, ``k_st_rate``, ``k_rt_rate``,
              ``k_so_rate`` [g K m⁻² d⁻¹] — Per-organ net K rate.

            Diagnostics:

            * ``nstress`` [-] — ``NPKI = min(NNI, PNI, KNI)``, the
                combined nutrition index in ``[1e-3, 1]`` that
                multiplies ``gtotal`` in `Photosynthesis`.
            * ``nni``, ``pni``, ``kni`` [-] — Per-nutrient indices.
            * ``n_uptake``, ``p_uptake``, ``k_uptake``
                [g X m⁻² d⁻¹] — Whole-plant uptake totals
                (``NUPTR + NFIXTR`` for N).
            * ``n_demand``, ``p_demand``, ``k_demand``
                [g X m⁻² d⁻¹] — Vegetative demand totals
                (``NDEMTO`` etc.).
            * ``nfixtr`` [g N m⁻² d⁻¹] — Biological N fixation flux.
            * ``rnso``, ``rpso``, ``rkso`` [g X m⁻² d⁻¹] —
                Storage-organ supply from translocation.
            * ``nuptr``, ``puptr``, ``kuptr`` [g X m⁻² d⁻¹] —
                Soil-only uptake fluxes (no fixation), fed to
                `SoilNutrients` for the inorganic-pool balance.
            * ``nlimit`` [-] — Nutrient-uptake gate
              ``(DVS < DVSNLT) ∧ (TRANRF ≥ 0.01)``.
            * ``emerg`` [-] — Emergence mask
              ``(tsump ≥ tsumem)``.
        """
        del soil_params  # uptake cap now lives on `ModelState.{nmint,pmint,kmint}`
        cp = crop_params

        # Convenience aliases (all [B]).
        wlv = state.wlv
        wst = state.wst
        wrt = state.wrt
        wso = state.wso
        anlv, anst, anrt, anso = state.anlv, state.anst, state.anrt, state.anso
        aplv, apst, aprt, apso = state.aplv, state.apst, state.aprt, state.apso
        aklv, akst, akrt, akso = state.aklv, state.akst, state.akrt, state.akso
        dvs = state.dvs

        zero = torch.zeros_like(wlv)
        ones = torch.ones_like(wlv)
        if tranrf is None:
            tranrf = ones
        if dlv is None:
            dlv = zero
        if drrt is None:
            drrt = zero
        if drst is None:
            drst = zero

        # ------------- Maximum concentrations (DVS-indexed) ------------
        nmaxlv = interpolate(cp.nmxlv, dvs)
        pmaxlv = interpolate(cp.pmxlv, dvs)
        kmaxlv = interpolate(cp.kmxlv, dvs)
        nmaxst = cp.lsnr * nmaxlv
        nmaxrt = cp.lrnr * nmaxlv
        pmaxst = cp.lspr * pmaxlv
        pmaxrt = cp.lrpr * pmaxlv
        kmaxst = cp.lskr * kmaxlv
        kmaxrt = cp.lrkr * kmaxlv

        # ------------- Translocatable pools (NTRLOC) -------------------
        atnlv = torch.clamp(anlv - wlv * cp.rnflv, min=0.0)
        atnst = torch.clamp(anst - wst * cp.rnfst, min=0.0)
        atnrt = torch.minimum((atnlv + atnst) * cp.fntrt, anrt - wrt * cp.rnfrt)
        atn = atnlv + atnst + atnrt

        atplv = torch.clamp(aplv - wlv * cp.rpflv, min=0.0)
        atpst = torch.clamp(apst - wst * cp.rpfst, min=0.0)
        atprt = torch.minimum((atplv + atpst) * cp.fntrt, aprt - wrt * cp.rpfrt)
        atp = atplv + atpst + atprt

        atklv = torch.clamp(aklv - wlv * cp.rkflv, min=0.0)
        atkst = torch.clamp(akst - wst * cp.rkfst, min=0.0)
        atkrt = torch.minimum((atklv + atkst) * cp.fntrt, akrt - wrt * cp.rkfrt)
        atk = atklv + atkst + atkrt

        # ------------- Demand from deficit (NDEMND) --------------------
        # Storage-organ demand is filtered by the first-order
        # translocation time constants.
        ndeml = torch.clamp(nmaxlv * wlv - anlv, min=0.0)
        ndems = torch.clamp(nmaxst * wst - anst, min=0.0)
        ndemr = torch.clamp(nmaxrt * wrt - anrt, min=0.0)
        ndemso = torch.clamp(cp.nmaxso * wso - anso, min=0.0) / cp.tcnt

        pdeml = torch.clamp(pmaxlv * wlv - aplv, min=0.0)
        pdems = torch.clamp(pmaxst * wst - apst, min=0.0)
        pdemr = torch.clamp(pmaxrt * wrt - aprt, min=0.0)
        pdemso = torch.clamp(cp.pmaxso * wso - apso, min=0.0) / cp.tcpt

        kdeml = torch.clamp(kmaxlv * wlv - aklv, min=0.0)
        kdems = torch.clamp(kmaxst * wst - akst, min=0.0)
        kdemr = torch.clamp(kmaxrt * wrt - akrt, min=0.0)
        kdemso = torch.clamp(cp.kmaxso * wso - akso, min=0.0) / cp.tckt

        ndemto = torch.clamp(ndeml + ndems + ndemr, min=0.0)
        pdemto = torch.clamp(pdeml + pdems + pdemr, min=0.0)
        kdemto = torch.clamp(kdeml + kdems + kdemr, min=0.0)

        # ------------- Storage-organ supply via translocation ----------
        # Translocation activates only above DVSNT.
        translocating = (dvs >= cp.dvsnt).to(wlv.dtype)
        nsupso = translocating * atn / cp.tcnt
        psupso = translocating * atp / cp.tcpt
        ksupso = translocating * atk / cp.tckt

        # ------------- Emergence and NLIMIT gates ----------------------
        # EMERG mask: thermal sum since sowing has exceeded TSUMEM.
        emerg = (state.tsump >= cp.tsumem).to(wlv.dtype)
        # NLIMIT: uptake only during DVS < DVSNLT and only when the
        # soil is wet enough (TRANRF ≥ 0.01).
        within_window = (dvs < cp.dvsnlt).to(wlv.dtype)
        wet_enough = (tranrf >= 0.01).to(wlv.dtype)
        nlimit = within_window * wet_enough

        # ------------- Whole-plant uptake (NUptake, IOPT-aware) --------
        # The soil-uptake cap is the currently available inorganic
        # pool (``NMINT``/``PMINT``/``KMINT``), advanced day-by-day by
        # `SoilNutrients`.
        nmint = state.nmint + zero  # broadcast to [B]
        pmint = state.pmint + zero
        kmint = state.kmint + zero

        # Storage-organ uptake from translocated pool.
        rnso = torch.clamp(torch.minimum(ndemso, nsupso), min=0.0)
        rpso = torch.clamp(torch.minimum(pdemso, psupso), min=0.0)
        rkso = torch.clamp(torch.minimum(kdemso, ksupso), min=0.0)

        # Soil-limited (NPK-limited) uptake formulation.
        nuptr_soil = torch.clamp(
            torch.minimum((1.0 - cp.nfixf) * ndemto, nmint), min=0.0
        ) * nlimit
        puptr_soil = torch.clamp(torch.minimum(pdemto, pmint), min=0.0) * nlimit
        kuptr_soil = torch.clamp(torch.minimum(kdemto, kmint), min=0.0) * nlimit

        # Potential (no-soil-cap) uptake formulation.
        nuptr_pot = torch.clamp((1.0 - cp.nfixf) * ndemto, min=0.0) * nlimit
        puptr_pot = torch.clamp(pdemto, min=0.0) * nlimit
        kuptr_pot = torch.clamp(kdemto, min=0.0) * nlimit

        # Run-mode masks. iopt ∈ {1, 2, 3, 4}; thresholds use 2.5 / 3.5
        # to avoid float equality. Broadcasts whether iopt is 0-d or [B].
        iopt = cp.iopt + zero  # promote to [B]
        n_potential = (iopt <= 2.5).to(wlv.dtype)  # IOPT ≤ 2
        pk_potential = (iopt <= 3.5).to(wlv.dtype)  # IOPT ≤ 3
        nuptr = n_potential * nuptr_pot + (1.0 - n_potential) * nuptr_soil
        puptr = pk_potential * puptr_pot + (1.0 - pk_potential) * puptr_soil
        kuptr = pk_potential * kuptr_pot + (1.0 - pk_potential) * kuptr_soil

        # Biological N₂ fixation — driven by demand, not the soil pool.
        nfixtr = torch.clamp(cp.nfixf * ndemto, min=0.0) * nlimit

        # Apply emergence gate.
        nuptr = nuptr * emerg
        puptr = puptr * emerg
        kuptr = kuptr * emerg
        nfixtr = nfixtr * emerg
        rnso = rnso * emerg
        rpso = rpso * emerg
        rkso = rkso * emerg

        # ------------- Concentration-based nutrition indices -----------
        # Whole-canopy (leaves + stems) mean concentrations, compared
        # to the residual (RMR) and optimal (OPTMR) levels.
        tbgmr = wlv + wst
        s_tbgmr = _safe(tbgmr)

        nrmr = (wlv * cp.rnflv + wst * cp.rnfst) / s_tbgmr
        prmr = (wlv * cp.rpflv + wst * cp.rpfst) / s_tbgmr
        krmr = (wlv * cp.rkflv + wst * cp.rkfst) / s_tbgmr

        noptmr = (cp.frnx * (nmaxlv * wlv + nmaxst * wst)) / s_tbgmr
        poptmr = (cp.frpx * (pmaxlv * wlv + pmaxst * wst)) / s_tbgmr
        koptmr = (cp.frkx * (kmaxlv * wlv + kmaxst * wst)) / s_tbgmr

        s_ndemto = _safe(ndemto)
        s_pdemto = _safe(pdemto)
        s_kdemto = _safe(kdemto)

        nupgmr = anlv + anst
        pupgmr = aplv + apst
        kupgmr = aklv + akst
        nfgmr = (
            nupgmr + (ndeml + ndems) / s_ndemto * (nuptr + nfixtr)
        ) / s_tbgmr
        pfgmr = (
            pupgmr + (pdeml + pdems) / s_pdemto * puptr
        ) / s_tbgmr
        kfgmr = (
            kupgmr + (kdeml + kdems) / s_kdemto * kuptr
        ) / s_tbgmr

        tiny = 1e-3

        def _idx(fgmr: torch.Tensor, rmr: torch.Tensor, optmr: torch.Tensor) -> torch.Tensor:
            return torch.clamp((fgmr - rmr) / _safe(optmr - rmr), tiny, 1.0)

        nni = _idx(nfgmr, nrmr, noptmr)
        pni = _idx(pfgmr, prmr, poptmr)
        kni = _idx(kfgmr, krmr, koptmr)
        npki = torch.minimum(torch.minimum(nni, pni), kni)

        # IOPT overrides.
        nni = torch.where(iopt <= 2.5, ones, nni)
        pni = torch.where(iopt <= 3.5, ones, pni)
        kni = torch.where(iopt <= 3.5, ones, kni)
        npki = torch.where(iopt <= 2.5, ones, npki)
        is_iopt3 = (iopt > 2.5) & (iopt <= 3.5)
        npki = torch.where(is_iopt3, nni, npki)

        # Pre-emergence: no stress.
        emerg_mask = emerg > 0.5
        nni = torch.where(emerg_mask, nni, ones)
        pni = torch.where(emerg_mask, pni, ones)
        kni = torch.where(emerg_mask, kni, ones)
        npki = torch.where(emerg_mask, npki, ones)

        # ------------- Per-organ uptake split (RNUSUB) -----------------
        # Soil + biological N uptake is distributed across leaves,
        # stems and roots by demand share.
        n_in = nuptr + nfixtr
        rnulv = (ndeml / s_ndemto) * n_in
        rnust = (ndems / s_ndemto) * n_in
        rnurt = (ndemr / s_ndemto) * n_in
        rpulv = (pdeml / s_pdemto) * puptr
        rpust = (pdems / s_pdemto) * puptr
        rpurt = (pdemr / s_pdemto) * puptr
        rkulv = (kdeml / s_kdemto) * kuptr
        rkust = (kdems / s_kdemto) * kuptr
        rkurt = (kdemr / s_kdemto) * kuptr

        # ------------- Translocation to storage organs (NTRANS) --------
        # RNSO is partitioned across source organs by translocatable-
        # pool share.
        s_atn = _safe(atn)
        s_atp = _safe(atp)
        s_atk = _safe(atk)
        rntlv = rnso * atnlv / s_atn
        rntst = rnso * atnst / s_atn
        rntrt = rnso * atnrt / s_atn
        rptlv = rpso * atplv / s_atp
        rptst = rpso * atpst / s_atp
        rptrt = rpso * atprt / s_atp
        rktlv = rkso * atklv / s_atk
        rktst = rkso * atkst / s_atk
        rktrt = rkso * atkrt / s_atk

        # ------------- Death losses (RNLD) -----------------------------
        # Residual concentrations multiplied by dead biomass.
        rnldlv = cp.rnflv * dlv
        rnldst = cp.rnfst * drst
        rnldrt = cp.rnfrt * drrt
        rpldlv = cp.rpflv * dlv
        rpldst = cp.rpfst * drst
        rpldrt = cp.rpfrt * drrt
        rkldlv = cp.rkflv * dlv
        rkldst = cp.rkfst * drst
        rkldrt = cp.rkfrt * drrt

        # ------------- Net per-organ rates -----------------------------
        n_lv_rate = rnulv - rntlv - rnldlv
        n_st_rate = rnust - rntst - rnldst
        n_rt_rate = rnurt - rntrt - rnldrt
        n_so_rate = rnso

        p_lv_rate = rpulv - rptlv - rpldlv
        p_st_rate = rpust - rptst - rpldst
        p_rt_rate = rpurt - rptrt - rpldrt
        p_so_rate = rpso

        k_lv_rate = rkulv - rktlv - rkldlv
        k_st_rate = rkust - rktst - rkldst
        k_rt_rate = rkurt - rktrt - rkldrt
        k_so_rate = rkso

        return {
            # Stress factor consumed downstream (Photosynthesis, leaf
            # dynamics, stress combiner): historical concentration-based
            # NPK nutrition index NPKI.
            "nstress": npki,
            "nni": nni,
            "pni": pni,
            "kni": kni,
            # Whole-plant flux diagnostics.
            "n_uptake": nuptr + nfixtr,
            "p_uptake": puptr,
            "k_uptake": kuptr,
            "n_demand": ndemto,
            "p_demand": pdemto,
            "k_demand": kdemto,
            "nfixtr": nfixtr,
            "rnso": rnso,
            "rpso": rpso,
            "rkso": rkso,
            # Soil-only uptake fluxes (no fixation) consumed by
            # `SoilNutrients` to deplete the inorganic pools.
            "nuptr": nuptr,
            "puptr": puptr,
            "kuptr": kuptr,
            # Gates re-used downstream so SoilNutrients runs on the same
            # NLIMIT / EMERG masks as the uptake calculation.
            "nlimit": nlimit,
            "emerg": emerg,
            # Net per-organ NPK rates (consumed by the engine).
            "n_lv_rate": n_lv_rate,
            "n_st_rate": n_st_rate,
            "n_rt_rate": n_rt_rate,
            "n_so_rate": n_so_rate,
            "p_lv_rate": p_lv_rate,
            "p_st_rate": p_st_rate,
            "p_rt_rate": p_rt_rate,
            "p_so_rate": p_so_rate,
            "k_lv_rate": k_lv_rate,
            "k_st_rate": k_st_rate,
            "k_rt_rate": k_rt_rate,
            "k_so_rate": k_so_rate,
        }
