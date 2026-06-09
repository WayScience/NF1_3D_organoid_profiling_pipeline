#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import pandas as pd
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from tqdm.auto import tqdm

root_dir, in_notebook = init_notebook()
profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)


# In[2]:


patients_file_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
patients = pd.read_csv(
    patients_file_path,
    header=None,
    names=["patient_id"],
).patient_id.tolist()

load_file_path = pathlib.Path(
    f"{root_dir}/4.processing_image_based_profiles/load_data/load_file.txt"
).resolve()
load_file_path.parent.mkdir(parents=True, exist_ok=True)


# In[3]:


all_parquets_info = []
for patient in tqdm(patients, desc="Patients"):
    patient_dir = profile_base_dir / "data" / patient / "extracted_features"
    if not patient_dir.exists():
        continue
    well_fovs = list(patient_dir.glob("*"))
    for well_fov in tqdm(well_fovs, desc="Well-FOVs", leave=False):
        parquet_paths = list(well_fov.glob("*.parquet"))
        all_parquets_info.extend(parquet_paths)


# In[4]:


df = pd.DataFrame({"parquet_path": all_parquets_info})
df["file_name"] = df["parquet_path"].apply(lambda x: x.parts[-1])
df["patient"] = df["parquet_path"].apply(lambda x: x.parts[-4])
df["well_fov"] = df["parquet_path"].apply(lambda x: x.parts[-2])
# drop rows where the well fov is run_stats
df = df.loc[
    ~df["well_fov"].str.contains("run_stats", case=False, na=False)
].reset_index(drop=True)
df.groupby(["patient", "well_fov"]).size().reset_index(name="num_parquets")


# In[5]:


list_of_compartments = ["Nuclei", "Cytoplasm", "Cell", "Organoid", "Nucleocentric"]


# In[6]:


df["flag_to_delete"] = df["parquet_path"].apply(
    lambda path: (
        not path.parts[-1].startswith(
            (
                "Nuclei",
                "Cytoplasm",
                "Cell",
                "Organoid",
                "Nucleocentric",
            )
        )
    )
)
files_to_remove = df.loc[df["flag_to_delete"] == True, "parquet_path"].tolist()
print(f"Number of files to remove: {len(files_to_remove)}")


# In[7]:


# for file_path in tqdm(files_to_remove, desc="Removing old parquet files"):
#     try:
#         file_path.unlink()
#     except Exception as e:
#         print(f"Error removing file {file_path}: {e}")
