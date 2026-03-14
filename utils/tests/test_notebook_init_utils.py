"""Tests for notebook initialization helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from image_analysis_3D.file_utils.notebook_init_utils import init_notebook


def test_init_notebook_finds_git_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    root_dir, in_notebook = init_notebook()
    assert root_dir == tmp_path
    assert in_notebook is False

    def test_init_notebook_in_jupyter_environment(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mocker
    ) -> None:
        (tmp_path / ".git").mkdir()
        monkeypatch.chdir(tmp_path)

        # Mock get_ipython to simulate Jupyter environment
        mock_ipython = mocker.Mock()
        mock_ipython.config = {}
        mocker.patch("builtins.get_ipython", return_value=mock_ipython)

        root_dir, in_notebook = init_notebook()
        assert root_dir == tmp_path
        assert in_notebook is True
