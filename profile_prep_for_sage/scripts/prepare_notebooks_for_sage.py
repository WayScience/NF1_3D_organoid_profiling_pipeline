#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)
profile_base_dir


# In[2]:


def recursive_remove_empty_dirs(path: pathlib.Path):
    """
    Recursively remove empty directories.

    Parameters
    ----------
    path : pathlib.Path
        The root directory to start removing empty directories from.
    """
    if not path.is_dir():
        return
    for child in path.iterdir():
        recursive_remove_empty_dirs(child)
    if not any(path.iterdir()):
        path.rmdir()


# In[3]:


profiles_dir = pathlib.Path(f"{profile_base_dir}/data/all_patient_profiles").resolve()
# get all patient profile dirs
profile_dirs = [
    d for d in profiles_dir.rglob("*.parquet") if "featurization" not in str(d)
]


# In[4]:


sage_profiles_dir = pathlib.Path(
    "../data_for_sage/Raw Data/bulk quantification/patients_processed_together"
    # see comments below if the spaces in the path annoy you...
    # to match the expected input dir for sage
    # note, the data_for_sage part of the dir does not get synced to synapse
    # this path provided syncs everything that matches ( or not ) a pattern
    # on synapse
    # so we need to make sure the directory structure is correct
    # also, this directory should be temporary and not checked into git
    # so it is in the .gitignore file just in case
    # but is also deleted at the end of this notebook
).resolve()
if sage_profiles_dir.exists():
    shutil.rmtree(sage_profiles_dir)
sage_profiles_dir.mkdir(parents=True, exist_ok=True)


# In[5]:


# get each of the profiles and split them by:
# patient tumor, treatment, dose+units
for profile_file_path in tqdm.tqdm(profile_dirs):
    profile_name = profile_file_path.stem.split("_profiles")[0]
    profile_name = profile_name.replace("fs", "feature_selected")
    profile_name = profile_name.replace("agg", "aggregated")
    df = pd.read_parquet(profile_file_path)
    if "middle" in str(profile_file_path) or "max" in str(profile_file_path):
        df["Metadata_dose_plus_units"] = (
            df["Metadata_dose"].astype(str) + "_" + df["Metadata_dose_unit"].astype(str)
        )
        df.to_parquet(
            f"{sage_profiles_dir}/2D/{profile_name}_2D.parquet",
            partition_cols=[
                "Metadata_patient_tumor",
                "Metadata_treatment",
                "Metadata_dose_plus_units",
            ],
        )
    else:
        df["Metadata_Experiment_dose_plus_units"] = (
            df["Metadata_Experiment_Dose"].astype(str)
            + "_"
            + df["Metadata_Experiment_Unit"].astype(str)
        )
        df.to_parquet(
            f"{sage_profiles_dir}/3D/{profile_name}_3D.parquet",
            partition_cols=[
                "Metadata_Biology_PatientTumor",
                "Metadata_Experiment_Treatment",
                "Metadata_Experiment_dose_plus_units",
            ],
        )


# In[6]:


# output_dirs = [d for d in sage_profiles_dir.glob("**/*") if d.is_dir()]
# # get a list of all output files and dirs
# output_dirs = sorted(
#     [d for d in list(sage_profiles_dir.glob("**/*")) if d.is_dir()],
#     key=lambda x: len(x.parts),
#     reverse=True,
# )
# # get a list of all output files and dirs
# output_dirs = [d for d in sage_profiles_dir.glob("**/*") if d.is_dir()]
list_of_replacement_strings = [
    ("=", "_"),
    ("%", "percent"),
    ("Metadata_treatment___HIVE_DEFAULT_PARTITION__", ""),
    ("Metadata_Biology_PatientTumor_", ""),
    ("Metadata_Experiment_Treatment_", ""),
    ("Metadata_Experiment_dose_plus_units_", ""),
    ("Metadata_patient_tumor_", ""),
    ("Metadata_treatment_", ""),
    ("Metadata_dose_plus_units_", ""),
]
for search_str, replace_str in list_of_replacement_strings:
    output_dirs = sorted(
        [d for d in list(sage_profiles_dir.glob("**/*")) if d.is_dir()],
        key=lambda x: len(x.parts),
        reverse=True,
    )
    for d in output_dirs:
        if search_str in d.name:
            new_name = d.name.replace(search_str, replace_str)
            new_path = d.parent / new_name
            if new_path.exists():
                continue
            try:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                d.rename(new_path)
            except:
                print(f"Directory {new_path} already exists. Skipping rename of {d}.")


# In[7]:


output_files = [f for f in sage_profiles_dir.glob("**/*") if f.is_file()]
# loop through and rename files to contain the proper metadata
for file in output_files:
    if ".parquet/" not in str(file):
        continue
    parent_dir = str(file).split(".parquet/")[0]
    new_file_name = (
        str(file)
        .split(".parquet/")[1]
        .replace("/", "_")
        .replace(f"{str(file.stem)}.", "")
    )
    new_file_path = pathlib.Path(parent_dir) / new_file_name
    new_file_path.parent.mkdir(parents=True, exist_ok=True)
    file.rename(new_file_path)


# In[8]:


# remove the empty dirs from where files used to persist
recursive_remove_empty_dirs(sage_profiles_dir)


# In[9]:


README_path = pathlib.Path("../README.md").resolve()
sage_readme_path = pathlib.Path(f"{sage_profiles_dir}/README.md").resolve()
shutil.copy(README_path, sage_readme_path)


# ## Upload the processed profiles to Synapse for Sage processing

# Tutorial on how to use synapse client: https://python-docs.synapse.org/en/stable/tutorials/python/upload_data_in_bulk/

# In[10]:


# # note, must run synapse config first in terminal to set up .synapseConfig file
# # or set some environment variables
# syn = synapseclient.login()


# In[11]:


# my_project_id = my_project_id = syn.findEntityId(
#     name="A deep learning microscopy framework for NF1 patient-derived organoid drug screening"
# )
# DIRECTORY_FOR_MY_PROJECT = os.path.join(
#     "..", "data_for_sage/"
# )  # tried using pathlib and it throws an error in the generate sync manifest function
# PATH_TO_MANIFEST_FILE = os.path.join(".", "manifest-for-upload.tsv")


# In[12]:


# # generate the manifest file to sync on
# synapseutils.generate_sync_manifest(
#     syn=syn,
#     directory_path=DIRECTORY_FOR_MY_PROJECT,
#     parent_id=my_project_id,
#     manifest_path=PATH_TO_MANIFEST_FILE,
# )


# In[13]:


# # sync the files to synapse
# synapseutils.syncToSynapse(
#     syn=syn, manifestFile=PATH_TO_MANIFEST_FILE, sendMessages=False
# )
