"""This document provides utility functions for profiling memory and time usage during featurization runs."""

import os
import pathlib
import time

import pandas as pd
import psutil


def get_mem_and_time_profiling(
    start_time: float,
    start_mem: float,
    end_time: float,
    end_mem: float,
    well_fov: str,
    patient_id: str,
    feature_type: str,
    channel: str,
    compartment: str,
    CPU_GPU: str,
    output_file_dir: pathlib.Path,
) -> bool:
    """
    Profile memory and time usage for a featurization run and save statistics.

    This function computes memory and time metrics for a featurization job,
    prints the results, and saves them to a parquet file.

    Parameters
    ----------
    start_time : float
        Time when the function started running (Unix timestamp).
    start_mem : float
        Memory usage when the function started running (in MB).
    end_time : float
        Time when the function ended running (Unix timestamp).
    end_mem : float
        Memory usage when the function ended running (in MB).
    well_fov : str
        Well and field of view for the run.
    patient_id : str
        Patient ID for the run.
    feature_type : str
        Feature type for the run (e.g., 'intensity', 'shape').
    channel : str
        Channel name for the run.
    compartment : str
        Cellular compartment for the run (e.g., 'nucleus', 'cytoplasm').
    CPU_GPU : str
        Processing unit used ('CPU' or 'GPU').
    output_file_dir : pathlib.Path
        Directory path to save the run statistics parquet file.

    Returns
    -------
    bool
        True if the function ran successfully, False otherwise.
    """

    end_mem_current = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    end_time_current = time.time()
    time_elapsed = end_time_current - start_time
    mem_used = end_mem_current - start_mem

    print(f"""
        Memory and time profiling for the run:
        Patient ID: {patient_id}
        Well and FOV: {well_fov}
        Feature type: {feature_type}
        CPU/GPU: {CPU_GPU}
        Memory usage: {mem_used:.2f} MB
        Time elapsed:
        --- {time_elapsed:.2f} seconds ---
        --- {time_elapsed / 60:.2f} minutes ---
        --- {time_elapsed / 3600:.2f} hours ---
    """)
    # make a df of the run stats
    run_stats = pd.DataFrame(
        {
            "start_time": [start_time],
            "end_time": [end_time],
            "start_mem": [start_mem],
            "end_mem": [end_mem],
            "time_taken": [(end_time - start_time)],
            "mem_usage": [(end_mem - start_mem)],
            "gpu": [CPU_GPU],
            "well_fov": [well_fov],
            "patient_id": [patient_id],
            "feature_type": [feature_type],
            "channel": [channel],
            "compartment": [compartment],
        }
    )
    # save the run stats to a file
    output_file_dir.parent.mkdir(parents=True, exist_ok=True)
    run_stats.to_parquet(output_file_dir)
    return True
