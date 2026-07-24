#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pathlib
import shutil

import pandas as pd
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm
profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)


# In[2]:


DEL_DIR_AFTER_ZIP = False
OVERWRITE = False


# In[ ]:


output_dir_3D = pathlib.Path(f"{root_dir}/data/shippable_dir/profiles_3D/").resolve()
output_dir_2D = pathlib.Path(f"{root_dir}/data/shippable_dir/profiles_2D/").resolve()
output_dir_3D_all_patients = pathlib.Path(
    f"{root_dir}/data/shippable_dir/profiles_3D/all_patients/"
).resolve()
output_dir_2D_all_patients = pathlib.Path(
    f"{root_dir}/data/shippable_dir/profiles_2D/all_patients/"
).resolve()

output_dir_3D_all_patients.mkdir(exist_ok=True, parents=True)
output_dir_2D_all_patients.mkdir(exist_ok=True, parents=True)

patient_ids_file_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve()
patient_ids = pd.read_csv(patient_ids_file_path, header=None)[0].to_list()


# ## Copy over all the combined profiles for all patients in 2D and 3D formats

# In[4]:


all_patient_profiles_dir = pathlib.Path(
    f"{profile_base_dir}/data/all_patient_profiles"
).resolve()
# copy over the parquet files for all patients
all_patient_profiles_files = list(all_patient_profiles_dir.glob("*.parquet"))
# copy over the max_projection and middle_slice directories for all patients
all_patient_profiles_dirs = list(
    [d for d in all_patient_profiles_dir.glob("*") if d.is_dir()]
)
for file in tqdm.tqdm(
    all_patient_profiles_files,
    desc="Copying all patient profiles parquet files",
    total=len(all_patient_profiles_files),
):
    if OVERWRITE or not (output_dir_3D_all_patients / file.name).exists():
        shutil.copy(file, output_dir_3D_all_patients)
for dir in tqdm.tqdm(
    all_patient_profiles_dirs,
    desc="Copying all patient profiles directories",
    total=len(all_patient_profiles_dirs),
):
    if OVERWRITE or not (output_dir_2D_all_patients / dir.name).exists():
        shutil.copytree(dir, output_dir_2D_all_patients / dir.name, dirs_exist_ok=True)


# ## Copy over individually processed patient profiles in 2D and 3D formats

# In[5]:


subpatient_3D_folders_to_copy = [
    "image_based_profiles/4.qc_profiles",
    "image_based_profiles/5.normalized_profiles",
    "image_based_profiles/6.feature_selected_profiles",
    "image_based_profiles/7.aggregated_profiles",
    "image_based_profiles/8.consensus_profiles",
]
subpatient_2D_folders_to_copy = [
    "2D_analysis/4.annotated",
    "2D_analysis/5.normalized",
    "2D_analysis/6.feature_selected",
    "2D_analysis/7.aggregated",
]


# In[ ]:


for patient in patient_ids:
    patient_profile_3D_dir = pathlib.Path(f"{output_dir_3D}/{patient}").resolve()
    patient_profile_2D_dir = pathlib.Path(f"{output_dir_2D}/{patient}").resolve()
    patient_profile_3D_dir.mkdir(exist_ok=True, parents=True)
    patient_profile_2D_dir.mkdir(exist_ok=True, parents=True)

    for subfolder in subpatient_3D_folders_to_copy:
        subfolder_output = subfolder.split("/")[-1]  # Get the last part of the path
        src_dir = pathlib.Path(
            f"{profile_base_dir}/data/{patient}/{subfolder}"
        ).resolve()
        dest_dir = pathlib.Path(
            f"{patient_profile_3D_dir}/{subfolder_output}"
        ).resolve()
        dest_dir.mkdir(exist_ok=True, parents=True)
        print(f"Copying from {src_dir} to {dest_dir}")
        # copy the folder and its contents over
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)

    for subfolder in subpatient_2D_folders_to_copy:
        subfolder_output = subfolder.split("/")[-1]  # Get the last part of the path
        src_dir = pathlib.Path(
            f"{profile_base_dir}/data/{patient}/{subfolder}"
        ).resolve()
        dest_dir = pathlib.Path(
            f"{patient_profile_2D_dir}/{subfolder_output}"
        ).resolve()
        dest_dir.mkdir(exist_ok=True, parents=True)
        print(f"Copying from {src_dir} to {dest_dir}")
        # copy the folder and its contents over
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)


# In[7]:


# zip the shippable_dir for transfer
shippable_dir = pathlib.Path(f"{root_dir}/data/shippable_dir").resolve()
shippable_dir_zip = shippable_dir.with_suffix(".zip")
if shippable_dir_zip.exists():
    shippable_dir_zip.unlink()  # Remove the existing zip file
shutil.make_archive(str(shippable_dir), "zip", str(shippable_dir))
# then delete the copy of the data in the shippable_dir to save space
if DEL_DIR_AFTER_ZIP:
    shutil.rmtree(shippable_dir)
