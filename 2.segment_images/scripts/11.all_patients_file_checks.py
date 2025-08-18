#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import numpy as np
import pandas as pd
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


# In[2]:


patients_file = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(strict=True)
patients = pd.read_csv(patients_file, header=None, names=["patientID"])
patients = patients["patientID"].tolist()


# In[3]:


for patient in patients:
    # get the well fovs
    well_fovs = pathlib.Path(f"{root_dir}/data/{patient}/zstack_images").glob("*")
    well_fovs = [well_fov.name for well_fov in well_fovs if well_fov.is_dir()]
    for well_fov in tqdm.tqdm(well_fovs, desc=f"Processing patient {patient}"):
        # set path to the processed data dir
        zstack_dir = pathlib.Path(
            f"{root_dir}/data/{patient}/zstack_images/{well_fov}"
        ).resolve(strict=True)
        # get the files in the segmentation data dir
        zstack_files = list(zstack_dir.glob("*.tif"))
        # ensure that all files are equal size
        zstack_files = [file.stat().st_size for file in zstack_files]
        if len(set(zstack_files)) != 1:
            print(f"{patient} {well_fov} zstack images has files of different sizes.")
