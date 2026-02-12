#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Post-process Huygens deconvolution outputs and organize results."""


# In[ ]:

import os
import pathlib
import shutil

import pandas as pd
from image_analysis_3D.file_utils.notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm

bandicoot_path = pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve()
if bandicoot_path.exists():
    # comment out depending on whose computer you are on
    # bandicoot_path = pathlib.Path("/media/18tbdrive/GFF_organoid_data/")
    raw_image_dir = pathlib.Path(f"{bandicoot_path}/NF1_organoid_data/").resolve()
    output_base_dir = bandicoot_path
else:
    raw_image_dir = pathlib.Path(f"{root_dir}/NF1_organoid_data/").resolve()
    output_base_dir = root_dir


# In[2]:


# get all image_paths
# deconvolution paths
data_path = pathlib.Path(
    "/home/lippincm/Desktop/20TB_A/NF1_processed_organoids/data/decon_inputs/"
).resolve(strict=True)
# get all patient dirs present
channel_dirs = [pathlib.Path(f"{p}") for p in data_path.iterdir() if p.is_dir()]
well_fov_image_paths = [
    f
    for dir in channel_dirs
    for f in dir.iterdir()
    if f.is_file()
    and f.suffix == ".tiff"
    and "NF0035" in str(f)  # filter to only NF0035 for now
]
well_fov_image_paths.sort()
df = pd.DataFrame({"image_path": well_fov_image_paths})
df["file_name"] = df["image_path"].apply(lambda x: x.name)
# df['file_parent'] = df['image_path'].apply(lambda x: x.parent)
df["channel"] = df["file_name"].apply(lambda x: str(x.split("_")[3].split(".tiff")[0]))
df["patient_id"] = df["file_name"].apply(
    lambda x: str(x.split("_")[0]) + "_" + str(x.split("_")[1])
)
df["well_id"] = df["file_name"].apply(lambda x: str(x.split("_")[2]))
df["new_file_name"] = df["well_id"] + "_" + df["channel"] + ".tif"
df.head()


# In[3]:


for idx, row in tqdm(df.iterrows(), total=len(df), desc="Copying files"):
    patient = row["patient_id"]
    well = row["well_id"]
    channel = row["channel"]
    image_path = row["image_path"]
    new_image_path = pathlib.Path(
        f"{raw_image_dir}/data/{patient}/deconvolved_images/{well}/{row['new_file_name']}"
    )
    new_image_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying {image_path} to {new_image_path}")
    shutil.copyfile(image_path, new_image_path)
