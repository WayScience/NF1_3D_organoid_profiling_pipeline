#!/usr/bin/env python
# coding: utf-8

# # 1. Merge Feature Parquets
#
# ## Purpose
# This notebook merges all per-channel, per-feature-type parquet files for a single well-FOV into a single DuckDB database file, with one table per compartment.
#
# This is **step 1 of Stage 4 (image-based profiling)**. It runs once per well-FOV and is typically submitted as a child job via the SLURM scheduler.
#
# ## Inputs
# - `data/{patient}/extracted_features/{well_fov}/*.parquet`
#   - One parquet per compartment × channel × feature type combination (125–189 files per FOV)
#   - Expected filename format: `{Compartment}_{Channel}_{FeatureType}_{Processor}_features.parquet`
#     - Example: `Nuclei_ER_Granularity_CPU_features.parquet`
#   - Each file contains columns: `object_id`, `image_set`, and feature columns
#
# ## Outputs
# - `data/{patient}/image_based_profiles/0.converted_profiles/{well_fov}/{well_fov}.duckdb`
#   - Five tables, one per compartment:
#
# | Table | Compartment | Description | Feature types |
# |---|---|---|---|
# | `Organoid` | Whole organoid | One row per segmented organoid | Hand-crafted + SAMMed3D |
# | `Nuclei` | Nucleus | One row per segmented nucleus | Hand-crafted + SAMMed3D |
# | `Cell` | Whole cell | One row per segmented cell | Hand-crafted + SAMMed3D |
# | `Cytoplasm` | Cell minus nucleus | One row per cytoplasm region | Hand-crafted + SAMMed3D |
# | `Nucleocentric` | Nucleus-centered crop | One row per nucleus-centered volume | SAMMed3D and CHAMMI75 only |
#
#   - Handcrafted features include AreaSizeShape, Colocalization, Intensity, Granularity, Neighbors, and Texture.
#   - If a compartment has no extracted features for a given well-FOV (e.g. no organoids were detected), an empty scaffold table matching the expected schema is written so downstream scripts don't fail on a missing table.
#
# ## Notes
# - Merges within a compartment use left joins on `object_id` + `image_set`. Objects missing from some channels will have NaN-filled feature columns — this is expected when not all feature types apply to all channels (e.g. colocalization requires two channels).
# - Nucleocentric compartment only supports SAMMed3D and morphem feature types; hand-crafted features are not extracted for nucleocentric volumes.

# In[1]:


import os
import pathlib
from functools import reduce

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
    well_fov = args["well_fov"]
    patient = args["patient"]
    output_features_subparent_name = args["output_features_subparent_name"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]


else:
    well_fov = "C4-2"
    patient = "NF0014_T1"
    output_features_subparent_name = "extracted_features"
    image_based_profiles_subparent_name = "image_based_profiles"


result_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{output_features_subparent_name}/{well_fov}"
).resolve(strict=True)
database_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/{image_based_profiles_subparent_name}/0.converted_profiles/{well_fov}"
).resolve()
database_path.mkdir(parents=True, exist_ok=True)
# create the duckdb database
sqlite_path = database_path / f"{well_fov}.duckdb"
DB_structure_path = pathlib.Path(
    f"{root_dir}/4.processing_image_based_profiles/data/DB_structures/DB_structure_db.duckdb"
).resolve(strict=True)

# get a list of all parquets in the directory recursively
parquet_files = list(result_path.rglob("*.parquet"))
parquet_files.sort()
print(len(parquet_files), "parquet files found")


# In[3]:


# create the nested dictionary to hold the feature types and compartments
feature_types = [
    "AreaSizeShape",
    "Colocalization",
    "Intensity",
    "Granularity",
    "Neighbors",
    "SAMMed3D",
    "Texture",
    "CHAMMI75",
]
compartments = ["Organoid", "Nuclei", "Cell", "Cytoplasm", "Nucleocentric"]


# In[4]:


output_dict = {
    compartment: {
        feature_type: []
        for feature_type in feature_types
        if not (
            compartment == "Nucleocentric"
            and feature_type.lower() not in ["chammi75", "sammed3d"]
        )
    }
    for compartment in compartments
}
output_dict


# In[5]:


# Parse filename metadata for each parquet file.
# Expected format: {Compartment}_{Channel}_{FeatureType}_{Processor}_features.parquet
# Token positions (split on "_"):
#   [0] = compartment  (e.g. "Nuclei", "Organoid")
#   [1] = channel      (e.g. "ER", "DNA"; multi-channel names use hyphens: "ER-Mito")
#   [2] = feature type (e.g. "Granularity", "SAMMed3D")
files = list(result_path.rglob("*.parquet"))
files_df = pd.DataFrame({"file_path": files})
files_df["file_name"] = files_df["file_path"].apply(lambda x: x.name)
files_df["compartment"] = files_df["file_name"].apply(lambda x: x.split("_")[0])
files_df["channel"] = files_df["file_name"].apply(lambda x: x.split("_")[1])
files_df["feature_type"] = files_df["file_name"].apply(
    lambda x: x.split("_")[2].split(".parquet")[0]
)
file_path = files_df.pop("file_path")
files_df.insert(4, "file_path", file_path)

# Validate parsed tokens against expected sets to catch filename convention violations early.
unknown_compartments = set(files_df["compartment"]) - set(compartments)
unknown_feature_types = set(files_df["feature_type"]) - set(feature_types)
if unknown_compartments:
    raise ValueError(f"Unexpected compartment(s) in filenames: {unknown_compartments}")
if unknown_feature_types:
    raise ValueError(
        f"Unexpected feature type(s) in filenames: {unknown_feature_types}"
    )

files_df.head()


# In[6]:


# Phase 1: route each file path into output_dict by compartment x feature type.
for i, row in files_df.iterrows():
    compartment = row["compartment"]
    feature_type = row["feature_type"]
    channel = row["channel"]
    file_path = row["file_path"]
    output_dict[compartment][feature_type].append(file_path)

# Phase 2: for each compartment x feature type group, merge all per-channel parquets
# into one dataframe using left joins on object_id + image_set. Objects missing from
# some channels will have NaN-filled columns — expected when not all feature types
# apply to all channels (e.g. colocalization requires two channels).
final_df_dict = {compartment: {} for compartment in output_dict.keys()}
for compartment in output_dict.keys():
    for feature_type in output_dict[compartment].keys():
        if not output_dict[compartment][feature_type]:
            continue
        final_df_dict[compartment][feature_type] = reduce(
            lambda left, right: pd.merge(
                left,
                right,
                on=["object_id", "image_set"],
                how="left",
            ),
            [pd.read_parquet(file) for file in output_dict[compartment][feature_type]],
        )
# ensure the object_id column is int before merging
for compartment in final_df_dict.keys():
    for feature_type in final_df_dict[compartment].keys():
        final_df_dict[compartment][feature_type]["object_id"] = final_df_dict[
            compartment
        ][feature_type]["object_id"].astype(int)


# In[7]:


# Merge all feature-type dataframes into one dataframe per compartment.
# All compartments (including Nucleocentric) are merged via left joins on
# object_id + image_set across feature types.

compartment_dfs = {}
try:
    for compartment in final_df_dict.keys():
        for df in final_df_dict[compartment].values():
            compartment_dfs[compartment] = reduce(
                lambda left, right: pd.merge(
                    left,
                    right,
                    on=["object_id", "image_set"],
                    how="left",
                ),
                final_df_dict[compartment].values(),
            )
except Exception as e:
    print(f"Error merging dataframes for compartment {compartment}: {e}")
    raise


# In[8]:


# Validate that all single-cell compartments have the same number of objects
# and that object IDs are aligned across Nuclei, Cell, Cytoplasm, and Nucleocentric.
print(
    len(compartment_dfs["Nuclei"]),
    len(compartment_dfs["Cell"]),
    len(compartment_dfs["Cytoplasm"]),
    len(compartment_dfs["Nucleocentric"]),
)
assert (
    len(compartment_dfs["Nuclei"])
    == len(compartment_dfs["Cell"])
    == len(compartment_dfs["Cytoplasm"])
    == len(compartment_dfs["Nucleocentric"])
)
assert (
    compartment_dfs["Nuclei"]["object_id"].equals(compartment_dfs["Cell"]["object_id"])
    and compartment_dfs["Nuclei"]["object_id"].equals(
        compartment_dfs["Cytoplasm"]["object_id"]
    )
    and compartment_dfs["Nuclei"]["object_id"].equals(
        compartment_dfs["Nucleocentric"]["object_id"]
    )
)


# In[9]:


# Load the reference DB schema from a pre-built DuckDB file.
# These empty scaffold tables provide the correct column structure for each compartment.
# Used below as a fallback when a compartment has no extracted features for this
# well-FOV (e.g. no organoids detected), so downstream scripts always find all five tables.
with duckdb.connect(DB_structure_path, read_only=True) as cx:
    organoid_table = cx.execute("SELECT * FROM Organoid").df()
    cell_table = cx.execute("SELECT * FROM Cell").df()
    nuclei_table = cx.execute("SELECT * FROM Nuclei").df()
    cytoplasm_table = cx.execute("SELECT * FROM Cytoplasm").df()
    nucleocentric_table = cx.execute("SELECT * FROM Nucleocentric").df()

dict_of_DB_structures = {
    "Organoid": organoid_table,
    "Cell": cell_table,
    "Nuclei": nuclei_table,
    "Cytoplasm": cytoplasm_table,
    "Nucleocentric": nucleocentric_table,
}


# In[10]:


# Write each compartment dataframe as a table in the output DuckDB.
# If a compartment produced no features (empty df), write the scaffold table instead
# so downstream scripts can always expect all five compartment tables to exist.
with duckdb.connect(sqlite_path, read_only=False) as cx:
    for compartment, df in compartment_dfs.items():
        print(compartment, df.shape)
        write_df = df if not df.empty else dict_of_DB_structures[compartment]
        cx.register("temp_df", write_df)
        cx.execute(f"CREATE OR REPLACE TABLE {compartment} AS SELECT * FROM temp_df")
        cx.unregister("temp_df")
