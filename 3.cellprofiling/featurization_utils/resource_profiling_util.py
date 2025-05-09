import os
import pathlib
import time

import numpy as np
import pandas as pd
import psutil


def get_mem_and_time_profiling(
    start_time: time.time,
    start_mem: psutil.Process(os.getpid()).memory_info().rss / 1024**2,
    end_time: time.time,
    end_mem: psutil.Process(os.getpid()).memory_info().rss / 1024**2,
    well_fov: str,
    patient_id: str,
    feature_type: str,
    CPU_GPU: str,
    output_file_dir: pathlib.Path,
) -> bool:
    """
    Function to get memory and time profiling for the run. This function will

    Parameters
    ----------
    start_time : time.time
        Time when the function started running.\
    start_mem : psutil.Process(os.getpid()).memory_info().rss / 1024**2
        Memory usage when the function started running.
    end_time : time.time
        Time when the function ended running.
    end_mem : psutil.Process(os.getpid()).memory_info().rss / 1024**2
        Memory usage when the function ended running.
    well_fov : str
        Well and field of view for the run.
    patient_id : str
        Patient ID for the run.
    feature_type : str
        Feature type for the run.
    CPU_GPU : str
        Whether the run was done on CPU or GPU.
    output_file_dir : pathlib.Path
        Directory to save the run stats file.

    Returns
    -------
    bool
        True if the function ran successfully, False otherwise.
    """

    end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    end_time = time.time()
    print(f"Memory usage: {end_mem - start_mem:.2f} MB")
    print("Texture time:")
    print("--- %s seconds ---" % (end_time - start_time))
    print("--- %s minutes ---" % ((end_time - start_time) / 60))
    print("--- %s hours ---" % ((end_time - start_time) / 3600))
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
        }
    )
    # save the run stats to a file
    output_file_dir.parent.mkdir(parents=True, exist_ok=True)
    run_stats.to_parquet(output_file_dir)
    return True
