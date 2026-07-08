#!/usr/bin/env python
# coding: utf-8

# # 2. Merge Single-Cell Profiles
#
# ## Purpose
# This notebook reads the per-compartment DuckDB produced by notebook 1 for a single
# well-FOV and merges the Nuclei, Cell, and Cytoplasm tables into a single-cell (SC)
# parquet profile. Organoid and Nucleocentric profiles are passed through and saved as
# separate parquets.
#
# This is **step 2 of Stage 4 (image-based profiling)**. It runs once per well-FOV and
# is typically submitted as a child job via the SLURM scheduler.
#
# ## Inputs
# - `data/{patient}/image_based_profiles/0.converted_profiles/{well_fov}/{well_fov}.duckdb`
#   - Five compartment tables: `Organoid`, `Nuclei`, `Cell`, `Cytoplasm`, `Nucleocentric`
#   - Produced by notebook 1 (`1.merge_feature_parquets.ipynb`)
#
# ## Outputs
# Three parquet files written to `data/{patient}/image_based_profiles/0.converted_profiles/{well_fov}/`:
#
# | File | Content | Rows |
# |---|---|---|
# | `sc_profiles_{well_fov}.parquet` | Merged Nuclei + Cell + Cytoplasm features | One row per object present in all three compartments |
# | `organoid_profiles_{well_fov}.parquet` | Organoid features passed through | One row per segmented organoid |
# | `nucleocentric_profiles_{well_fov}.parquet` | Nucleocentric features passed through | One row per nucleus-centered volume |
#
# ## Notes
# - Only objects present in **all three** of Nuclei, Cell, and Cytoplasm are retained in the SC profile.
#   Objects segmented in only some compartments are dropped.
# - Object IDs are reassigned to a sequential `1..N` range at the end of this notebook.
#   The original segmentation mask IDs are not preserved.

# In[1]:


import os
import pathlib

import duckdb
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
    well_fov = args["well_fov"]
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    patient = "NF0014_T1"
    well_fov = "E6-1"
    image_based_profiles_subparent_name = "image_based_profiles"


# In[3]:


input_sqlite_file = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/0.converted_profiles/{well_fov}/{well_fov}.duckdb"
).resolve(strict=True)
destination_sc_parquet_file = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/0.converted_profiles/{well_fov}/sc_profiles_{well_fov}.parquet"
).resolve()
destination_organoid_parquet_file = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/0.converted_profiles/{well_fov}/organoid_profiles_{well_fov}.parquet"
).resolve()
destination_nucleocentric_parquet_file = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/0.converted_profiles/{well_fov}/nucleocentric_profiles_{well_fov}.parquet"
).resolve()
destination_sc_parquet_file.parent.mkdir(parents=True, exist_ok=True)


# In[4]:


# Load all five compartment tables from the DuckDB produced by notebook 1.
with duckdb.connect(input_sqlite_file) as con:
    tables = con.execute("SHOW TABLES").fetchdf()
    print(tables)
    nuclei_table = con.sql("SELECT * FROM Nuclei").df()
    cells_table = con.sql("SELECT * FROM Cell").df()
    cytoplasm_table = con.sql("SELECT * FROM Cytoplasm").df()
    organoid_table = con.sql("SELECT * FROM Organoid").df()
    nucleocentric_table = con.sql("SELECT * FROM Nucleocentric").df()


# In[5]:


# Retain only objects that were successfully segmented in all three compartments.
# A nucleus without a matched cell/cytoplasm (or vice versa) is not a valid
# single-cell profile and is dropped here.
nuclei_id_set = set(nuclei_table["object_id"].to_list())
cells_id_set = set(cells_table["object_id"].to_list())
cytoplasm_id_set = set(cytoplasm_table["object_id"].to_list())

# find the intersection of the three sets
intersection_set = nuclei_id_set.intersection(cells_id_set, cytoplasm_id_set)

# keep only the rows in the three tables that are in the intersection set
nuclei_table = nuclei_table[nuclei_table["object_id"].isin(intersection_set)]
cells_table = cells_table[cells_table["object_id"].isin(intersection_set)]
cytoplasm_table = cytoplasm_table[cytoplasm_table["object_id"].isin(intersection_set)]


# In[6]:


# Merge the three compartment tables into a single-cell dataframe.
# Because object_ids were already filtered to the intersection in the cell above,
# this LEFT JOIN is effectively an INNER JOIN — no NaN-filled rows will result.
with duckdb.connect() as con:
    con.register("nuclei", nuclei_table)
    con.register("cells", cells_table)
    con.register("cytoplasm", cytoplasm_table)
    # Merge them with SQL
    merged_df = con.execute("""
        SELECT *
        FROM nuclei
        LEFT JOIN cells USING (object_id)
        LEFT JOIN cytoplasm USING (object_id)
    """).df()


# In[7]:


# save the organoid data as parquet
print(f"Final organoid data shape: {organoid_table.shape}")
organoid_table.to_parquet(destination_organoid_parquet_file, index=False)
organoid_table.head()


# In[8]:


print(f"Final merged single cell dataframe shape: {merged_df.shape}")
# save the sc data as parquet
merged_df.to_parquet(destination_sc_parquet_file, index=False)
merged_df.head()


# In[9]:


print(f"Final nucleocentric dataframe shape: {nucleocentric_table.shape}")
# save the nucleocentric data as parquet
nucleocentric_table.to_parquet(destination_nucleocentric_parquet_file, index=False)
nucleocentric_table.head()
