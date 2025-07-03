#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import pathlib
import sys

import numpy as np
import tqdm

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
    # check if in a jupyter notebook
try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")

sys.path.append(str(pathlib.Path(f"{root_dir}/utils").resolve()))
from file_checking import check_number_of_files

# In[2]:


if not in_notebook:
    argparser = argparse.ArgumentParser(
        description="set up directories for the analysis of the data"
    )

    argparser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="patient name, e.g. 'P01'",
    )

    args = argparser.parse_args()
    patient = args.patient
else:
    patient = "NF0014"


# In[3]:


# set path to the processed data dir
segmentation_data_dir = pathlib.Path(
    f"{root_dir}/data/{patient}/segmentation_masks/"
).resolve(strict=True)
zstack_dir = pathlib.Path(f"{root_dir}/data/{patient}/zstack_images/").resolve(
    strict=True
)


# In[4]:


# perform checks for each directory
segmentation_data_dirs = list(segmentation_data_dir.glob("*"))
segmentation_data_dirs = [d for d in segmentation_data_dirs if d.is_dir()]
zstack_dirs = list(zstack_dir.glob("*"))
zstack_dirs = [d for d in zstack_dirs if d.is_dir()]

print("Checking segmentation data directories")
for dir in segmentation_data_dirs:
    check_number_of_files(dir, 4)

print("Checking zstack directories")
for dir in zstack_dirs:
    check_number_of_files(dir, 9)
