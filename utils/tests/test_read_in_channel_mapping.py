"""Tests for channel mapping loader."""

from __future__ import annotations

from pathlib import Path

from file_utils.read_in_channel_mapping import retrieve_channel_mapping


def test_retrieve_channel_mapping(tmp_path: Path) -> None:
    toml_path = tmp_path / "channels.toml"
    toml_path.write_text("[channel_mapping]\nnuclei='405'\ncyto='488'\n")
    mapping = retrieve_channel_mapping(str(toml_path))
    assert mapping["DNA"] == "405"
    assert mapping["ER"] == "488"
    assert mapping["AGP"] == "555"
    assert mapping["Mito"] == "640"
    assert mapping["BF"] == "TRANS"
    assert mapping["Nuclei"] == "nuclei_"
    assert mapping["Cell"] == "cell_"
    assert mapping["Cytoplasm"] == "cytoplasm_"
    assert mapping["Organoid"] == "organoid_"
