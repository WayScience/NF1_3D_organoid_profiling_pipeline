#!/usr/bin/env python
# coding: utf-8

# # 5. Combining Profiles
#
# ## Purpose
# Concatenate all per-well-FOV parquet files for a single patient into three
# patient-level combined parquets (SC, organoid, nucleocentric).
#
# This is **step 5 of Stage 4 (image-based profiling)**. It runs once per patient
# and is typically submitted as a per-patient SLURM job.
#
# ## Inputs
# - `data/{patient}/image_based_profiles/1.related_profiles/{well_fov}/`
#   - `sc_profiles_{well_fov}_related.parquet`
#   - `organoid_profiles_{well_fov}_related.parquet`
#   - `nucleocentric_profiles_{well_fov}_related.parquet`
#
# ## Outputs
# Three combined parquets in `data/{patient}/image_based_profiles/2.combined_profiles/`:
#
# | File | Content |
# |---|---|
# | `sc.parquet` | All SC profiles stacked across FOVs |
# | `organoid.parquet` | All organoid profiles stacked across FOVs |
# | `nucleocentric.parquet` | All nucleocentric profiles stacked across FOVs |
#
# ## Notes
# - Concatenation uses DuckDB `union_by_name=true`, which aligns columns by name
#   rather than position. FOVs with missing columns (e.g. empty scaffold tables)
#   will have those columns filled with NULL.
# - Brightfield (BF) channel features are removed after concatenation as they are
#   not part of the fluorescent cell painting panel and are not used in profiling.

# In[1]:


import os
import pathlib

import duckdb
import pandas as pd
from image_analysis_3D.file_utils.arg_parsing_utils import parse_args
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)
profile_base_dir = root_dir


# In[2]:


if not in_notebook:
    args = parse_args()
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    patient = "NF0014_T1"
    image_based_profiles_subparent_name = "image_based_profiles"


# In[3]:


# set paths
profiles_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/1.related_profiles"
).resolve(strict=True)
# output_paths
sc_merged_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/2.combined_profiles/sc.parquet"
).resolve()
organoid_merged_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/2.combined_profiles/organoid.parquet"
).resolve()
nucleocentric_profile_output_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/2.combined_profiles/nucleocentric.parquet"
).resolve()
organoid_merged_output_path.parent.mkdir(parents=True, exist_ok=True)


# In[4]:


# Discover all per-FOV parquet files under 1.related_profiles/.
# The directory structure is 1.related_profiles/{well_fov}/*.parquet,
# so one wildcard level is sufficient.
profiles = list(profiles_path.rglob("*/*.parquet"))


# In[5]:


# Split files by profile type using filename prefix.
# Expected prefixes: 'sc_', 'organoid_', 'nucleocentric_'.
sc_profiles = [str(x) for x in profiles if x.name.startswith("sc_")]
organoid_profiles = [str(x) for x in profiles if x.name.startswith("organoid_")]
nucleocentric_profiles = [
    str(x) for x in profiles if x.name.startswith("nucleocentric_")
]


# In[6]:


for x in nucleocentric_profiles:
    df = pd.read_parquet(x)
    if df.isnull().any().any():
        print(f"Null values found in {x}")
df


# In[7]:


# Concatenate per-FOV parquets for each profile type using DuckDB.
# union_by_name=true aligns columns by name rather than position, so FOVs with
# differing column sets (e.g. empty scaffold tables from notebook 1) are handled
# gracefully — missing columns are filled with NULL rather than causing an error.

with duckdb.connect() as conn:
    sc_profile = conn.execute(
        f"SELECT * FROM read_parquet({sc_profiles}, union_by_name=true)"
    ).df()
    organoid_profile = conn.execute(
        f"SELECT * FROM read_parquet({organoid_profiles}, union_by_name=true)"
    ).df()
    nucleocentric_profile = conn.execute(
        f"SELECT * FROM read_parquet({nucleocentric_profiles}, union_by_name=true)"
    ).df()

print(f"Single-cell profiles concatenated. Shape: {sc_profile.shape}")
print(f"Organoid profiles concatenated. Shape: {organoid_profile.shape}")
print(f"Nucleocentric profiles concatenated. Shape: {nucleocentric_profile.shape}")


# ## Remove all BF channels
#

# In[8]:


# Remove brightfield (BF) channel features from all three profile types.
# BF is a transmitted-light channel not part of the fluorescent cell painting
# panel; its features are may not meaningful for morphological profiling.
# and are interpreted differently
# Note: if no BF columns exist in the data, these drops are no-ops.

bf_cols_sc = [col for col in sc_profile.columns if "BF" in col]
sc_profile = sc_profile.drop(columns=bf_cols_sc)
print(f"SC: dropped {len(bf_cols_sc)} BF columns. Shape: {sc_profile.shape}")

bf_cols_organoid = [col for col in organoid_profile.columns if "BF" in col]
organoid_profile = organoid_profile.drop(columns=bf_cols_organoid)
print(
    f"Organoid: dropped {len(bf_cols_organoid)} BF columns. Shape: {organoid_profile.shape}"
)

bf_cols_nucleocentric = [col for col in nucleocentric_profile.columns if "BF" in col]
nucleocentric_profile = nucleocentric_profile.drop(columns=bf_cols_nucleocentric)
print(
    f"Nucleocentric: dropped {len(bf_cols_nucleocentric)} BF columns. Shape: {nucleocentric_profile.shape}"
)


# In[9]:


sc_profile.to_parquet(sc_merged_output_path, index=False)
organoid_profile.to_parquet(organoid_merged_output_path, index=False)
nucleocentric_profile.to_parquet(nucleocentric_profile_output_path, index=False)


# In[10]:


nucleocentric_profile
