"""Tests for loading crop parameters by name or from a config file."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch
import yaml

from torchcrop.parameters.crop_params import (
    CropParameters,
    available_crops,
    default_wheat_params,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_XML_DIR = _REPO_ROOT / "simplace" / "crop_default"


def test_available_crops_uses_clean_names():
    crops = available_crops()
    assert "wheat" in crops
    assert "maize" in crops
    assert "soybean" in crops
    assert "default" in crops
    # Cryptic SIMPLACE stems are gone.
    assert "wwheat1" not in crops
    assert "mag202" not in crops


def test_load_by_crop_name():
    params = CropParameters(crop_name="wheat")
    assert torch.isclose(params.tsum1, torch.tensor(1050.0))
    assert torch.isclose(params.tsumem, torch.tensor(60.0))
    assert torch.isclose(params.tdwi, torch.tensor(21.0))
    assert params.ruetb.shape == (4, 2)
    assert params.fltb.shape[1] == 2


def test_no_args_loads_default_preset():
    # CropParameters() loads default.yaml (wheat-like SIMPLACE default).
    default = CropParameters()
    wheat = CropParameters(crop_name="wheat")
    assert torch.isclose(default.tsum1, torch.tensor(1050.0))
    assert torch.isclose(default.tsum1, wheat.tsum1)


def test_crop_name_is_case_and_separator_insensitive():
    a = CropParameters(crop_name="maize")
    b = CropParameters(crop_name="MAIZE")
    c = CropParameters(crop_name="oilseed rape")
    d = CropParameters(crop_name="oilseed-rape")
    assert torch.isclose(a.tsum1, b.tsum1)
    assert torch.isclose(c.tsum1, d.tsum1)


def test_from_crop_name_respects_dtype():
    params = CropParameters.from_crop_name("wheat", dtype=torch.float64)
    assert params.tsum1.dtype == torch.float64
    assert params.ruetb.dtype == torch.float64


def test_unknown_crop_name_raises():
    with pytest.raises(ValueError, match="Unknown crop"):
        CropParameters(crop_name="banana")


def test_load_from_config_file(tmp_path):
    cfg = tmp_path / "my_crop.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "crop_name": "my_crop",
                "scalars": {"tsum1": 1234.0, "rgrl": 0.02},
                "tables": {"ruetb": [[0.0, 2.5], [2.0, 0.5]]},
            }
        )
    )
    params = CropParameters(config_file=str(cfg))
    assert torch.isclose(params.tsum1, torch.tensor(1234.0))
    assert torch.isclose(params.rgrl, torch.tensor(0.02))
    assert torch.allclose(params.ruetb, torch.tensor([[0.0, 2.5], [2.0, 0.5]]))


def test_config_file_accepts_path_object(tmp_path):
    cfg = tmp_path / "p.yaml"
    cfg.write_text(yaml.safe_dump({"scalars": {"tsum1": 42.0}}))
    params = CropParameters(config_file=cfg)
    assert torch.isclose(params.tsum1, torch.tensor(42.0))


def test_config_file_omitted_fields_keep_class_default(tmp_path):
    cfg = tmp_path / "p.yaml"
    cfg.write_text(yaml.safe_dump({"scalars": {"tsum1": 42.0}}))
    params = CropParameters(config_file=cfg)
    # laii is a torchcrop-specific field not present in the file.
    assert torch.isclose(params.laii, torch.tensor(0.012))


def test_crop_name_and_config_file_are_mutually_exclusive(tmp_path):
    cfg = tmp_path / "p.yaml"
    cfg.write_text(yaml.safe_dump({"scalars": {"tsum1": 1.0}}))
    with pytest.raises(ValueError, match="not both"):
        CropParameters(crop_name="wheat", config_file=cfg)


def test_missing_config_file_raises():
    with pytest.raises(FileNotFoundError):
        CropParameters(config_file="/no/such/crop.yaml")


def test_to_preserves_values_without_reloading_default():
    # .to() must not re-trigger default-preset loading and clobber values.
    params = CropParameters(crop_name="maize")
    tsum1_before = params.tsum1.item()
    moved = params.to(dtype=torch.float64)
    assert moved.tsum1.dtype == torch.float64
    assert moved.tsum1.item() == pytest.approx(tsum1_before)


def test_preset_matches_source_xml():
    """A loaded preset reproduces the values in its SIMPLACE XML."""
    params = CropParameters(crop_name="maize")
    root = ET.parse(_XML_DIR / "mag202.xml").getroot().find("crop")
    xml = {p.get("id"): p for p in root.findall("parameter")}

    assert torch.isclose(params.tsum1, torch.tensor(float(xml["TSUM1"].text)))
    assert torch.isclose(params.rgrl, torch.tensor(float(xml["RGRLAI"].text)))

    xs = [float(v.text) for v in xml["LeavesPartitioningTableDVS"].findall("value")]
    ys = [
        float(v.text)
        for v in xml["LeavesPartitioningTableFraction"].findall("value")
    ]
    expected = torch.tensor([[x, y] for x, y in zip(xs, ys)])
    assert torch.allclose(params.fltb, expected)


def test_loaded_preset_is_valid():
    CropParameters(crop_name="wheat").validate()  # should not raise


def test_loaded_param_is_optimizable():
    params = CropParameters.from_crop_name("wheat", dtype=torch.float64)
    params.rue = torch.nn.Parameter(params.rue.clone())
    (params.rue * 2).backward()
    assert params.rue.grad is not None


def test_default_wheat_params_still_works():
    params = default_wheat_params(dtype=torch.float64)
    assert params.tsum1.dtype == torch.float64
