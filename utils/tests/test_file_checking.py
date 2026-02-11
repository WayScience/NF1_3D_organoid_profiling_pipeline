"""Tests for file checking utilities."""

from __future__ import annotations

from pathlib import Path

from file_utils.file_checking import check_number_of_files


def test_check_number_of_files_matches(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    ok, name = check_number_of_files(tmp_path, 2)
    assert ok is True
    assert name is None


def test_check_number_of_files_mismatch(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a")
    ok, name = check_number_of_files(tmp_path, 2)
    assert ok is False
    assert name == tmp_path.name
