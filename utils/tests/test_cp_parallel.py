"""Tests for CellProfiler parallel helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

from file_utils.cp_parallel import results_to_log


def test_results_to_log_writes_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    plate_path = tmp_path / "PlateA"
    plate_path.mkdir()
    command = [
        "cellprofiler",
        "-c",
        "-r",
        "-p",
        "pipe.cppipe",
        "-o",
        plate_path,
        "-i",
        "imgs",
    ]
    results = [
        subprocess.CompletedProcess(
            args=command, returncode=0, stdout=b"", stderr=b"ok"
        )
    ]
    results_to_log(results=results, log_dir=log_dir, run_name="qc")
    log_file = log_dir / f"{plate_path.name}_qc_run.log"
    assert log_file.exists()
