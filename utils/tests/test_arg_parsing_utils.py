"""Tests for argument parsing utilities."""

from __future__ import annotations

import sys

import pytest
from file_utils.arg_parsing_utils import check_for_missing_args, parse_args


def test_check_for_missing_args_raises() -> None:
    with pytest.raises(ValueError):
        check_for_missing_args(well_fov=None, patient="NF0001")


def test_check_for_missing_args_passes() -> None:
    check_for_missing_args(well_fov="A01-1", patient="NF0001")


def test_parse_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--well_fov",
            "A01-1",
            "--patient",
            "NF0001",
            "--window_size",
            "3",
            "--clip_limit",
            "0.1",
            "--compartment",
            "Nuclei",
            "--channel",
            "DAPI",
            "--processor_type",
            "CPU",
            "--input_subparent_name",
            "zstack_images",
            "--mask_subparent_name",
            "segmentation_masks",
            "--output_features_subparent_name",
            "feature_data",
        ],
    )
    parsed = parse_args()
    assert parsed["well_fov"] == "A01-1"
    assert parsed["patient"] == "NF0001"
    assert parsed["window_size"] == 3
    assert parsed["clip_limit"] == 0.1
    assert parsed["compartment"] == "Nuclei"
    assert parsed["channel"] == "DAPI"
    assert parsed["processor_type"] == "CPU"
    assert parsed["input_subparent_name"] == "zstack_images"
    assert parsed["mask_subparent_name"] == "segmentation_masks"
    assert parsed["output_features_subparent_name"] == "feature_data"
