#!/usr/bin/env python
# coding: utf-8

# # 4. Log Parsing
# 
# ## Purpose
# Scan SLURM job log files for all patients and report a pass/fail summary —
# how many pipeline runs completed without a Python traceback.
# 
# This notebook is a **diagnostic utility**, not a processing step. It is run
# interactively after a batch of SLURM jobs to quickly assess overall job health.
# 
# ## Inputs
# - `4.processing_image_based_profiles/logs/patients/*.log`
#   - One log file per SLURM job run
#   - Files named `run_stats` are excluded (they are job scheduler summaries, not stdout logs)
# 
# ## Outputs
# Printed summary: total runs, passed, failed, pass percentage, and the list of failed log filenames.
# 
# ## Notes
# - **Pass heuristic:** a log is counted as passed if it contains no `"Traceback"` string.
#   This only catches clean Python exceptions — jobs killed by SLURM (OOM, timeout)
#   produce no traceback and are silently counted as passed.

# In[ ]:


import os
import pathlib

from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()


# In[2]:


log_directory = pathlib.Path(
    root_dir / "4.processing_image_based_profiles/logs/patients"
).resolve(strict=True)

# Collect all .log files in the directory.
# Files named 'run_stats' are SLURM scheduler summaries, not per-job stdout logs,
# and are excluded from pass/fail counting.
log_files = [f for f in log_directory.iterdir() if f.is_file() and f.suffix == ".log"]

total_runs = 0
passed_list = []
failed_list = []

for log_file in log_files:
    if "run_stats" in log_file.name:
        continue
    total_runs += 1
    with open(log_file, "r") as f:
        content = f.read()
    # Pass heuristic: no Python traceback found in the log.
    # Note: jobs killed by SLURM (OOM, timeout) produce no traceback and
    # are counted as passed — check SLURM accounting for silent failures.
    if "Traceback" in content:
        failed_list.append(log_file.name)
    else:
        passed_list.append(log_file.name)

if total_runs == 0:
    print("No log files found in", log_directory)
else:
    print(f"Total runs:    {total_runs}")
    print(f"Passed runs:   {len(passed_list)}")
    print(f"Failed runs:   {len(failed_list)}")
    print("Pass rate:     {:.2f}%".format((len(passed_list) / total_runs) * 100))
    if failed_list:
        print("\nFailed logs:")
        for name in sorted(failed_list):
            print(f"  {name}")
    if passed_list:
        print("\nPassed logs:")
        for name in sorted(passed_list):
            print(f"  {name}")

