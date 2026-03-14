"""This document provides utility functions for profiling memory and time usage during featurization runs."""

import os
import pathlib
import time
import tracemalloc
from typing import Optional

import pandas as pd
import psutil


def start_profiling() -> tuple[float, float]:
    """
    Start memory and time profiling.

    Parameters
    ----------
    None

    Returns
    -------
    tuple[float, float]
        A ``(start_time, start_mem)`` pair where *start_time* is a Unix
        timestamp and *start_mem* is the current RSS in MB (kept for
        backward-compatibility with the legacy profiler).
    """
    tracemalloc.start()
    start_time = time.time()
    start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    return start_time, start_mem


def stop_profiling(
    start_time: float,
    well_fov: str,
    patient_id: str,
    feature_type: str,
    channel: str,
    compartment: str,
    CPU_GPU: str,
    output_file_dir: pathlib.Path,
    start_mem: Optional[float] = None,
) -> bool:
    """
    Stop profiling, report results, and save them to a parquet file.

    This function stops ``tracemalloc``, computes peak memory usage (via
    ``tracemalloc``) and elapsed wall-clock time, prints a summary, and
    persists the statistics to *output_file_dir* as a Parquet file.

    Parameters
    ----------
    start_time : float
        Unix timestamp returned by :func:`start_profiling`.
    well_fov : str
        Well and field of view for the run.
    patient_id : str
        Patient ID for the run.
    feature_type : str
        Feature type for the run (e.g., ``'intensity'``, ``'shape'``).
    channel : str
        Channel name for the run.
    compartment : str
        Cellular compartment for the run (e.g., ``'nucleus'``,
        ``'cytoplasm'``).
    CPU_GPU : str
        Processing unit used (``'CPU'`` or ``'GPU'``).
    output_file_dir : pathlib.Path
        File path to save the run-statistics Parquet file.
    start_mem : float, optional
        Starting RSS in MB (from :func:`start_profiling`).  Included in
        the output for backward-compatibility but is **not** used for the
        peak-memory calculation.

    Returns
    -------
    bool
        ``True`` if the function ran successfully.
    """
    # --- collect tracemalloc snapshot before stopping ---
    current_mem, peak_mem = tracemalloc.get_traced_memory()  # bytes
    tracemalloc.stop()

    end_time = time.time()
    time_elapsed = end_time - start_time

    # convert from bytes to MB
    current_mem_mb = current_mem / 1024**2
    peak_mem_mb = peak_mem / 1024**2
    end_mem_rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2

    print(f"""
        Memory and time profiling for the run:
        Patient ID: {patient_id}
        Well and FOV: {well_fov}
        Feature type: {feature_type}
        CPU/GPU: {CPU_GPU}
        Peak memory (tracemalloc): {peak_mem_mb:.2f} MB
        Current memory (tracemalloc): {current_mem_mb:.2f} MB
        RSS at end: {end_mem_rss:.2f} MB
        Time elapsed:
        --- {time_elapsed:.2f} seconds ---
        --- {time_elapsed / 60:.2f} minutes ---
        --- {time_elapsed / 3600:.2f} hours ---
    """)

    run_stats = pd.DataFrame(
        {
            "start_time": [start_time],
            "end_time": [end_time],
            "start_mem_rss_mb": [start_mem],
            "end_mem_rss_mb": [end_mem_rss],
            "peak_mem_tracemalloc_mb": [peak_mem_mb],
            "current_mem_tracemalloc_mb": [current_mem_mb],
            "time_taken_seconds": [time_elapsed],
            "gpu": [CPU_GPU],
            "well_fov": [well_fov],
            "patient_id": [patient_id],
            "feature_type": [feature_type],
            "channel": [channel],
            "compartment": [compartment],
        }
    )

    output_file_dir.parent.mkdir(parents=True, exist_ok=True)
    run_stats.to_parquet(output_file_dir)
    return True
