"""Tests for the phenology process module."""

from __future__ import annotations

import torch

from torchcrop.parameters.crop_params import CropParameters
from torchcrop.processes.phenology import Phenology
from torchcrop.states.model_state import ModelState


def _make_state(b: int = 2, dvs: float = 0.5, tsump: float = 200.0) -> ModelState:
    state = ModelState.initial(batch_size=b)
    return state.replace(
        dvs=torch.full_like(state.dvs, dvs),
        tsump=torch.full_like(state.tsump, tsump),
    )


def test_phenology_shapes():
    phen = Phenology()
    state = _make_state()
    davtmp = torch.tensor([15.0, 15.0])
    ddlp = torch.tensor([14.0, 14.0])
    out = phen(state, davtmp, ddlp, CropParameters())
    assert out["dvs_rate"].shape == (2,)
    assert out["tsum_rate"].shape == (2,)


def test_phenology_dvs_gated_pre_emergence():
    phen = Phenology()
    # Before emergence (tsump < tsumem = 110 by default): dvs_rate must be 0
    state = _make_state(dvs=0.0, tsump=0.0)
    davtmp = torch.tensor([15.0, 15.0])
    ddlp = torch.tensor([14.0, 14.0])
    out = phen(state, davtmp, ddlp, CropParameters())
    assert torch.allclose(out["dvs_rate"], torch.zeros(2))


def test_phenology_dvs_nonzero_after_emergence():
    phen = Phenology()
    state = _make_state(dvs=0.5, tsump=500.0)
    davtmp = torch.tensor([15.0, 15.0])
    ddlp = torch.tensor([14.0, 14.0])
    out = phen(state, davtmp, ddlp, CropParameters())
    assert (out["dvs_rate"] > 0).all()


def test_phenology_gated_post_maturity():
    phen = Phenology()
    state = _make_state(dvs=2.0, tsump=3000.0)
    davtmp = torch.tensor([15.0, 15.0])
    ddlp = torch.tensor([14.0, 14.0])
    out = phen(state, davtmp, ddlp, CropParameters())
    assert torch.allclose(out["dvs_rate"], torch.zeros(2))
