"""Tests for the CalibrationManager: paths, bounds, tables, ordering, grads."""

from __future__ import annotations

import pytest
import torch

from torchcrop import Lintul5Model
from torchcrop.calibration import (
    CalibrationManager,
    ConstraintGroup,
    ParameterSpec,
)
from torchcrop.calibration.config import load_calibration_config
from torchcrop.calibration.paths import parse_path
from torchcrop.parameters.crop_params import CropParameters
from torchcrop.utils.io import make_constant_weather


def _model() -> Lintul5Model:
    return Lintul5Model(crop_params=CropParameters().to(dtype=torch.float64)).double()


# --------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------- #


def test_parse_scalar_table_key_and_index():
    c = {"crop": CropParameters(), "soil": None, "site": None}
    scalar = parse_path("crop.tsum1", c)
    assert scalar.field == "tsum1" and not scalar.is_table

    by_key = parse_path("crop.fltb@0.5", c)
    assert by_key.is_table and by_key.field == "fltb"
    # fltb default has a row at DVS 0.5
    assert float(c["crop"].fltb[by_key.row, 0]) == pytest.approx(0.5)

    by_idx = parse_path("crop.fltb[3]", c)
    assert by_idx.row == 3


def test_parse_errors():
    c = {"crop": CropParameters(), "soil": None, "site": None}
    with pytest.raises(ValueError):
        parse_path("crop.nope", c)
    with pytest.raises(ValueError):
        parse_path("bogus.tsum1", c)
    with pytest.raises(ValueError):
        parse_path("crop.tsum1@0.5", c)  # scalar is not a table
    with pytest.raises(ValueError):
        parse_path("crop.fltb@9.9", c)  # no such abscissa


# --------------------------------------------------------------------- #
# Scalar bounds
# --------------------------------------------------------------------- #


def test_scalar_init_seeded_and_bounded():
    model = _model()
    cal = CalibrationManager(
        model, [ParameterSpec("crop.tsum1", bounds=(800.0, 1200.0), init=900.0)]
    )
    # latent seeded so the first materialize reproduces init
    cal.materialize()
    assert float(model.crop_params.tsum1) == pytest.approx(900.0, abs=1e-4)

    # a moderate latent lands strictly inside; an extreme one saturates to the
    # (inclusive) bound but never escapes it
    with torch.no_grad():
        cal.set_flat(torch.tensor([1.5], dtype=torch.float64))
    cal.materialize()
    assert 800.0 < float(model.crop_params.tsum1) < 1200.0
    with torch.no_grad():
        cal.set_flat(torch.tensor([50.0], dtype=torch.float64))
    cal.materialize()
    assert 800.0 <= float(model.crop_params.tsum1) <= 1200.0


def test_only_latents_in_parameters():
    model = _model()
    # make an unrelated crop field a raw nn.Parameter
    model.crop_params.rgrl = torch.nn.Parameter(model.crop_params.rgrl.clone())
    cal = CalibrationManager(model, [ParameterSpec("crop.tsum1", bounds=(800.0, 1200.0))])
    # cal.parameters() must expose exactly the one latent, not the model field
    assert len(list(cal.parameters())) == 1


def test_integer_kind_materializes_integral_with_gradient():
    model = _model()
    cal = CalibrationManager(
        model,
        [ParameterSpec("crop.tsumem", bounds=(80.0, 160.0), kind="integer", init=110.0)],
    )
    vals = cal.materialize()
    v = vals["crop.tsumem"]
    assert float(v) == pytest.approx(round(float(v)))  # integral
    # gradient still reaches the latent (straight-through)
    v.sum().backward()
    grads = [p.grad for p in cal.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


# --------------------------------------------------------------------- #
# Table-entry freeing
# --------------------------------------------------------------------- #


def test_table_entry_freed_others_and_abscissae_fixed():
    model = _model()
    original = model.crop_params.fltb.clone()
    cal = CalibrationManager(model, [ParameterSpec("crop.fltb@0.5", bounds=(0.2, 0.8))])
    with torch.no_grad():
        cal.set_flat(torch.tensor([2.0], dtype=torch.float64))
    cal.materialize()
    new = model.crop_params.fltb
    # abscissa column unchanged
    assert torch.allclose(new[:, 0], original[:, 0])
    # the targeted row's ordinate changed; all others unchanged
    row = parse_path("crop.fltb@0.5", {"crop": model.crop_params, "soil": None, "site": None}).row
    changed = ~torch.isclose(new[:, 1], original[:, 1])
    assert changed.sum().item() == 1 and bool(changed[row])
    assert 0.2 < float(new[row, 1]) < 0.8


def test_table_entry_gradient_flows_through_interpolate():
    from torchcrop.functions.interpolation import interpolate

    model = _model()
    cal = CalibrationManager(model, [ParameterSpec("crop.fltb@0.5", bounds=(0.2, 0.8))])
    cal.materialize()
    y = interpolate(model.crop_params.fltb, torch.tensor([0.5], dtype=torch.float64))
    y.sum().backward()
    g = next(iter(cal.parameters())).grad
    assert g is not None and g.abs().sum() > 0


# --------------------------------------------------------------------- #
# Ordering groups
# --------------------------------------------------------------------- #


def test_descending_group_is_monotone_and_bounded():
    model = _model()
    specs = [
        ParameterSpec("crop.ruetb@0.0", bounds=(2.5, 3.5)),
        ParameterSpec("crop.ruetb@1.3", bounds=(1.5, 3.0)),
        ParameterSpec("crop.ruetb@2.0", bounds=(0.2, 1.5)),
    ]
    group = ConstraintGroup(tuple(s.name for s in specs), order="descending")
    cal = CalibrationManager(model, specs, [group])
    for flat in (torch.tensor([5.0, -3.0, 1.0], dtype=torch.float64),
                 torch.tensor([-4.0, 4.0, -2.0], dtype=torch.float64)):
        with torch.no_grad():
            cal.set_flat(flat)
        vals = cal.materialize()
        a = vals["crop.ruetb@0.0"]
        b = vals["crop.ruetb@1.3"]
        c = vals["crop.ruetb@2.0"]
        assert float(a) >= float(b) >= float(c)
        assert 2.5 <= float(a) <= 3.5
        assert 0.2 <= float(c) <= 1.5


def test_ascending_group_rejects_incompatible_bounds():
    model = _model()
    specs = [
        ParameterSpec("crop.nmaxso", bounds=(0.01, 0.03)),
        ParameterSpec("crop.pmaxso", bounds=(0.005, 0.02)),  # hi DECREASES -> invalid
    ]
    with pytest.raises(ValueError):
        CalibrationManager(
            model, specs, [ConstraintGroup(("crop.nmaxso", "crop.pmaxso"), "ascending")]
        )


# --------------------------------------------------------------------- #
# End-to-end gradient through the simulator
# --------------------------------------------------------------------- #


def test_end_to_end_calibration_gradient():
    weather = make_constant_weather(batch_size=1, n_days=80, dtype=torch.float64)
    model = _model()
    cal = CalibrationManager(
        model,
        specs=[
            ParameterSpec("crop.tsum1", bounds=(700.0, 1100.0), init=900.0),
            ParameterSpec("crop.ruetb@0.0", bounds=(2.0, 3.5)),
        ],
    )
    cal.materialize()
    out = model(weather, start_doy=60)
    out.biomass.sum().backward()
    for p in cal.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert any(p.grad.abs().sum() > 0 for p in cal.parameters())


def test_flatten_set_flat_roundtrip():
    model = _model()
    cal = CalibrationManager(
        model,
        [
            ParameterSpec("crop.tsum1", bounds=(700.0, 1100.0)),
            ParameterSpec("crop.tsum2", bounds=(500.0, 900.0)),
        ],
    )
    flat = torch.tensor([0.3, -0.7], dtype=torch.float64)
    cal.set_flat(flat)
    assert torch.allclose(cal.flatten(), flat, atol=1e-6)


def test_yaml_dict_config_roundtrip():
    config = {
        "parameters": {
            "crop.tsum1": {"bounds": [1000, 2000]},
            "crop.fltb@0.5": {"bounds": [0.3, 0.7]},
            "crop.fltb@0.646": {"bounds": [0.1, 0.5]},
        },
        "constraints": [
            {"members": ["crop.fltb@0.5", "crop.fltb@0.646"], "order": "descending"}
        ],
    }
    specs, groups = load_calibration_config(config)
    assert len(specs) == 3 and len(groups) == 1
    model = _model()
    cal = CalibrationManager(model, specs, groups)
    vals = cal.materialize()
    assert float(vals["crop.fltb@0.5"]) >= float(vals["crop.fltb@0.646"])
