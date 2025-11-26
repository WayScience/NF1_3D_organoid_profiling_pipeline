#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import duckdb
import pandas as pd
from arg_parsing_utils import parse_args
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


log_directory = pathlib.Path(
    root_dir / "4.processing_image_based_profiles/logs/patients"
).resolve(strict=True)
# get a list of all files in the log directory
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
        if "Traceback" in content:
            failed_list.append(log_file.name)
        else:
            passed_list.append(log_file.name)
print(f"Total runs: {total_runs}")
print(f"Passed runs: {len(passed_list)}")
print(f"Failed runs: {len(failed_list)}")
print(
    "Percentage of runs that passed: {:.2f}%".format(
        (len(passed_list) / total_runs) * 100
    )
)
