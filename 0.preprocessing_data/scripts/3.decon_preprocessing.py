#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Prepare inputs and metadata for Huygens deconvolution."""


# In[ ]:

import pathlib
import shutil

import pandas as pd
from image_analysis_3D.file_utils.notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


# get all image_paths
data_path = pathlib.Path(
    "/home/lippincm/mnt/bandicoot/NF1_organoid_data/data/"
).resolve(strict=True)
# get all patient dirs present
patient_dirs = [
    pathlib.Path(f"{p}/zstack_images/")
    for p in data_path.iterdir()
    if p.is_dir()
    and "all_patient_profiles" not in p.name
    and "blocklist" not in p.name
    and "decon" not in p.name
    and "NF0035_T1" in p.name
]
# well_fov_dirs
well_fov_dirs = [f for dir in patient_dirs for f in dir.iterdir() if f.is_dir()]
individual_files_dirs = [
    f for dir in well_fov_dirs for f in dir.iterdir() if f.is_file()
]
print(f"Found {len(patient_dirs)} patient dirs")
individual_files_dirs.sort()
df = pd.DataFrame({"image_path": individual_files_dirs})
df["patient"] = df["image_path"].apply(lambda x: x.parent.parent.parent.name)
df["well"] = df["image_path"].apply(lambda x: x.parent.name)
df["channel"] = df["image_path"].apply(lambda x: x.stem.split("_")[1])
df.head()


# In[3]:


# well_fov_dirs
well_fov_dirs = [f for dir in patient_dirs for f in dir.iterdir() if f.is_dir()]
individual_files_dirs = [
    f for dir in well_fov_dirs for f in dir.iterdir() if f.is_file()
]
print(f"Found {len(patient_dirs)} patient dirs")
individual_files_dirs.sort()
df = pd.DataFrame({"image_path": individual_files_dirs})
df["patient"] = df["image_path"].apply(lambda x: x.parent.parent.parent.name)
df["well"] = df["image_path"].apply(lambda x: x.parent.name)
df["channel"] = df["image_path"].apply(lambda x: x.stem.split("_")[1])
df.head()


# In[4]:


decon_input_path = pathlib.Path(
    "/home/lippincm/Desktop/20TB_A/NF1_processed_organoids/data/decon_inputs"
).resolve()
decon_input_path.mkdir(parents=True, exist_ok=True)
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Copying files"):
    patient = row["patient"]
    well = row["well"]
    channel = row["channel"]
    image_path = row["image_path"]
    dest_dir = pathlib.Path(
        f"{decon_input_path}/{channel}/{patient}_{well}_{channel}.tif"
    )
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(image_path, dest_dir)
