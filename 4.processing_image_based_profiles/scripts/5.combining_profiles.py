#!/usr/bin/env python
# coding: utf-8

# This notebook combines all well fovs for each patient into a single file.
#

# In[1]:


import argparse
import pathlib

import duckdb
import pandas as pd

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

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False


# In[2]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="Patient ID to process, e.g. 'P01'",
    )
    args = argparser.parse_args()
    patient = args.patient

else:
    patient = "NF0018"


# In[3]:


# set paths
profiles_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/0.converted_profiles"
).resolve(strict=True)
# output_paths
sc_merged_output_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/1.combined_profiles/sc.parquet"
).resolve()
organoid_merged_output_path = pathlib.Path(
    f"{root_dir}/data/{patient}/image_based_profiles/1.combined_profiles/organoid.parquet"
).resolve()
organoid_merged_output_path.parent.mkdir(parents=True, exist_ok=True)


# In[4]:


# get all profiles in the directory recursively
profiles = list(profiles_path.glob("**/*.parquet"))
# filter out profiles that are not related
profiles = [x for x in profiles if "related" in str(x)]


# In[5]:


sc_profiles = [str(x) for x in profiles if "sc" in str(x.name)]
organoid_profiles = [str(x) for x in profiles if "organoid" in str(x.name)]


# In[6]:


# concat all sc profiles with duckdb
with duckdb.connect() as conn:
    sc_profile = conn.execute(
        f"SELECT * FROM read_parquet({sc_profiles}, union_by_name=true)"
    ).df()
    organoid_profile = conn.execute(
        f"SELECT * FROM read_parquet({organoid_profiles}, union_by_name=true)"
    ).df()
print(f"Single-cell profiles concatenated. Shape: {sc_profile.shape}")
print(f"Organoid profiles concatenated. Shape: {organoid_profile.shape}")
# drop imageset_1 and image_set_2 columns if they exist
if "image_set_1" in sc_profile.columns:
    sc_profile = sc_profile.drop(columns=["image_set_1"])
if "image_set_2" in sc_profile.columns:
    sc_profile = sc_profile.drop(columns=["image_set_2"])
if "image_set_1" in organoid_profile.columns:
    organoid_profile = organoid_profile.drop(columns=["image_set_1"])
if "image_set_2" in organoid_profile.columns:
    organoid_profile = organoid_profile.drop(columns=["image_set_2"])


# In[7]:


sc_profile.head()


# ## Remvoe all BF channels
#

# In[8]:


print(f"Single-cell profiles shape: {sc_profile.shape}")
list_of_columns_to_drop = [col for col in sc_profile.columns if "BF" in col]
sc_profile = sc_profile.drop(columns=list_of_columns_to_drop)
print(f"Single-cell profiles shape after dropping BF channels: {sc_profile.shape}")

print(f"Organoid profiles shape: {organoid_profile.shape}")
list_of_columns_to_drop = [col for col in organoid_profile.columns if "BF" in col]
organoid_profile = organoid_profile.drop(columns=list_of_columns_to_drop)
print(f"Organoid profiles shape after dropping BF channels: {organoid_profile.shape}")


# In[9]:


sc_profile.to_parquet(sc_merged_output_path, index=False)
organoid_profile.to_parquet(organoid_merged_output_path, index=False)
print(f"Single-cell profiles saved to {sc_merged_output_path}")
print(f"Organoid profiles saved to {organoid_merged_output_path}")
