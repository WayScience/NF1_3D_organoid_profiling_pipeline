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
import sys

import duckdb
import numpy as np
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
    well_fov = args["well_fov"]
    patient = args["patient"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]

else:
    # patient = "NF0055_T1"
    # well_fov = "D5-1"
    patient = "NF0014_T1"
    well_fov = "C4-1"
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

# for empty tables:
merged_example_df_path = pathlib.Path(
    f"{root_dir}/4.processing_image_based_profiles/data/DB_structures/single_cell_profile_structure.parquet"
).resolve()
nucleocentric_example_df_path = pathlib.Path(
    f"{root_dir}/4.processing_image_based_profiles/data/DB_structures/nucleocentric_profile_structure.parquet"
).resolve()
organoid_example_df_path = pathlib.Path(
    f"{root_dir}/4.processing_image_based_profiles/data/DB_structures/organoid_profile_structure.parquet"
).resolve()
organoid_example_df_path.parent.mkdir(parents=True, exist_ok=True)


# In[4]:


try:
    # Load all five compartment tables from the DuckDB produced by notebook 1.
    with duckdb.connect(input_sqlite_file) as con:
        tables = con.execute("SHOW TABLES").fetchdf()
        print(tables)
        nuclei_table = con.sql("SELECT * FROM Nuclei").df()
        cells_table = con.sql("SELECT * FROM Cell").df()
        cytoplasm_table = con.sql("SELECT * FROM Cytoplasm").df()
        organoid_table = con.sql("SELECT * FROM Organoid").df()
        nucleocentric_table = con.sql("SELECT * FROM Nucleocentric").df()
except Exception as e:
    if "Catalog" in str(e):
        print(
            f"Error: DuckDB file {input_sqlite_file} does not contain expected tables."
        )
        print("Writing empty DataFrames to output parquet files.")
        merged_df = pd.read_parquet(merged_example_df_path)
        nucleocentric_df = pd.read_parquet(nucleocentric_example_df_path)
        organoid_df = pd.read_parquet(organoid_example_df_path)

        # add the well_fov and objects
        merged_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "well_fov": [well_fov],
                        "object_id": [-1],
                        **{
                            col: [np.nan]
                            for col in merged_df.columns
                            if col not in ["well_fov", "object_id"]
                        },
                    }
                ),
            ]
        )
        nucleocentric_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "well_fov": [well_fov],
                        "object_id": [-1],
                        **{
                            col: [np.nan]
                            for col in nucleocentric_df.columns
                            if col not in ["well_fov", "object_id"]
                        },
                    }
                ),
            ]
        )
        organoid_df = pd.concat(
            [
                pd.DataFrame(
                    {
                        "well_fov": [well_fov],
                        "object_id": [-1],
                        **{
                            col: [np.nan]
                            for col in organoid_df.columns
                            if col not in ["well_fov", "object_id"]
                        },
                    }
                ),
            ]
        )

        merged_df.to_parquet(destination_sc_parquet_file, index=False)
        nucleocentric_df.to_parquet(destination_nucleocentric_parquet_file, index=False)
        organoid_df.to_parquet(destination_organoid_parquet_file, index=False)
    # exit the script after writing the empty DataFrames
    sys.exit(0)


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


# In[10]:


# if patient=NF0014_T1 and well_fov=C4-1, then output zero's out dfs to DB_structures
if patient == "NF0014_T1" and well_fov == "C4-1":
    merged_df = pd.DataFrame(columns=merged_df.columns)
    nucleocentric_df = pd.DataFrame(columns=nucleocentric_table.columns)
    organoid_df = pd.DataFrame(columns=organoid_table.columns)
    merged_df.to_parquet(merged_example_df_path, index=False)
    nucleocentric_df.to_parquet(nucleocentric_example_df_path, index=False)
    organoid_df.to_parquet(organoid_example_df_path, index=False)
