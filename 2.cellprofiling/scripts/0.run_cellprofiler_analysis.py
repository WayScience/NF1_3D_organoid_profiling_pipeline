#!/usr/bin/env python
# coding: utf-8

# # Perform segmentation and feature extraction for each plate using CellProfiler Parallel

# ## Import libraries

# In[1]:


import argparse
import pathlib
import sys
import time
import tracemalloc

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("../../utils")
from cp_utils import run_cellprofiler

# import run_cellprofiler

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


if not in_notebook:
    print("Running as script")
    # set up arg parser
    parser = argparse.ArgumentParser(description="Segment the nuclei of a tiff image")

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the input directory containing the tiff images",
    )

    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir).resolve(strict=True)
else:
    print("Running in a notebook")
    input_dir = pathlib.Path("../../data/cellprofiler/C4-2/").resolve(strict=True)

print(f"Input directory: {input_dir}")


# ## Set paths and variables

# In[3]:


# set the run type for the parallelization
run_name = "analysis"


path_to_pipeline = pathlib.Path("../pipelines/pipeline.cppipe").resolve(strict=True)

# set main output dir for all plates
output_dir = pathlib.Path(f"../analysis_output/{input_dir.stem}").resolve()
output_dir.mkdir(exist_ok=True, parents=True)
sqlite_name = input_dir.stem


# ## Run analysis pipeline on each plate in parallel
#
# This cell is not finished to completion due to how long it would take. It is ran in the python file instead.

# In[4]:


start = time.time()
# memory profile
tracemalloc.start()


# In[5]:


run_cellprofiler(
    path_to_pipeline=str(path_to_pipeline),
    path_to_input=str(input_dir),
    path_to_output=str(output_dir),
    analysis_run=True,
    hardcode_sqlite_name="temporary",
    sqlite_name=sqlite_name,
    rename_sqlite_file_bool=True,
)


# In[6]:


end = time.time()

# get the memory usage
snapshot = tracemalloc.take_snapshot()
# peak memory usage
top_stats = snapshot.statistics("lineno")

cumulative_mem = 0
peak_mem = 0

for stat in top_stats:
    cumulative_mem += stat.size
    peak_mem = max(peak_mem, stat.size)

print(f"Peak memory usage: {peak_mem / 10**6}MB")
print(f"Cumulative memory usage: {cumulative_mem / 10**6}MB")


# In[7]:


# format the time taken into hours, minutes, seconds
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print(
    "Total time taken: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
)
