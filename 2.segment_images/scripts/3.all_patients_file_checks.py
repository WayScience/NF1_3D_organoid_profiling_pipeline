#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pathlib
import sys

import numpy as np
import pandas as pd
import tqdm

cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd
else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break
sys.path.append(str(root_dir / "utils"))
from file_reading import read_zstack_image
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[ ]:


patients_file = pathlib.Path(f"{image_base_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
patients = pd.read_csv(patients_file, header=None, names=["patientID"])
patients = patients["patientID"].tolist()


# In[ ]:


for patient in patients:
    # get the well fovs
    well_fovs = pathlib.Path(f"{image_base_dir}/data/{patient}/zstack_images").glob("*")
    well_fovs = [well_fov.name for well_fov in well_fovs if well_fov.is_dir()]
    for well_fov in tqdm.tqdm(well_fovs, desc=f"Processing patient {patient}"):
        # set path to the processed data dir
        zstack_dir = pathlib.Path(
            f"{image_base_dir}/data/{patient}/zstack_images/{well_fov}"
        ).resolve(strict=True)
        # get the files in the segmentation data dir
        zstack_files = list(zstack_dir.glob("*.tif"))
        # ensure that all files are equal size
        zstack_files = [file.stat().st_size for file in zstack_files]
        if len(set(zstack_files)) != 1:
            print(f"{patient} {well_fov} zstack images has files of different sizes.")
