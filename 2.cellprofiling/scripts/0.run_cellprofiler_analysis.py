#!/usr/bin/env python
# coding: utf-8

# # Perform segmentation and feature extraction for each plate using CellProfiler Parallel

# ## Import libraries

# In[ ]:


import argparse
import pathlib
import pprint
import sys
import time
import tracemalloc

sys.path.append("../../utils")
from cp_utils import run_cellprofiler

# import run_cellprofiler

# check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[ ]:


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
    input_dir = pathlib.Path("../../data/cellprofiler").resolve(strict=True)

print(f"Input directory: {input_dir}")


# ## Set paths and variables

# In[ ]:


# set the run type for the parallelization
run_name = "analysis"

# set main output dir for all plates
output_dir = pathlib.Path("../analysis_output")
output_dir.mkdir(exist_ok=True, parents=True)

# directory where images are located within folders
images_dir = pathlib.Path("../../data/cellprofiler/raw_z_input/").resolve(strict=True)

path_to_pipeline = pathlib.Path("../pipelines/pipeline.cppipe").resolve(strict=True)

sqlite_name = images_dir.stem


# ## Run analysis pipeline on each plate in parallel
#
# This cell is not finished to completion due to how long it would take. It is ran in the python file instead.

# In[ ]:


start = time.time()
# memory profile
tracemalloc.start()


# In[ ]:


run_cellprofiler(
    path_to_pipeline=str(path_to_pipeline),
    path_to_input=str(input_dir),
    path_to_output=str(output_dir),
    sqlite_name=sqlite_name,
    analysis_run=True,
    rename_sqlite_file_bool=sqlite_name,
)


# In[ ]:


end = time.time()

# get the memory usage
snapshot = tracemalloc.take_snapshot()
# peak memory usage
top_stats = snapshot.statistics("lineno")
print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)
# total memory usage
print("[ Total ]")
pprint.pprint(snapshot.statistics("filename"))


# format the time taken into hours, minutes, seconds
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print(
    "Total time taken: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
)
