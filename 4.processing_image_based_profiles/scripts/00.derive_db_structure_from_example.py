#!/usr/bin/env python
# coding: utf-8

# For the purposes of this notebook and those following the "DB_structure" is a blank dataframe that is used to store the results of the profiling pipeline.
# This is used to insert blank dataframes into the final dataframe dictionary for each compartment and feature type if the record is empty so that a final df can be created and merged on the same columns.

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


# In[2]:


if not in_notebook:
    args = parse_args()
    well_fov = args["well_fov"]
    patient = args["patient"]
    output_features_subparent_name = args["output_features_subparent_name"]
    image_based_profiles_subparent_name = args["image_based_profiles_subparent_name"]


else:
    well_fov = "D5-2"
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
# create the sqlite database
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
files_df.head()


# In[6]:


for i, row in files_df.iterrows():
    compartment = row["compartment"]
    feature_type = row["feature_type"]
    channel = row["channel"]
    file_path = row["file_path"]
    output_dict[compartment][feature_type].append(file_path)
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


# In[7]:


# merge the dfs such that each compartment has a single df with all feature types as columns
compartment_dfs = {}
for compartment in final_df_dict.keys():
    for df in final_df_dict[compartment].values():
        if compartment == "Nucleocentric":
            compartment_dfs[compartment] = df
            break
        compartment_dfs[compartment] = reduce(
            lambda left, right: pd.merge(
                left,
                right,
                on=["object_id", "image_set"],
                how="left",
            ),
            final_df_dict[compartment].values(),
        )


# In[8]:


# get the table from the DB_structue
with duckdb.connect(sqlite_path, read_only=False) as cx:
    for compartment, df in compartment_dfs.items():
        if df.empty:
            cx.register("temp_df", dict_of_DB_structues[compartment])
            cx.execute(
                f"CREATE OR REPLACE TABLE {compartment} AS SELECT * FROM temp_df"
            )
            cx.unregister("temp_df")
        else:
            cx.register("temp_df", df)
            cx.execute(
                f"CREATE OR REPLACE TABLE {compartment} AS SELECT * FROM temp_df"
            )
            cx.unregister("temp_df")


# In[9]:


with duckdb.connect(DB_structure_path) as cx:
    for compartment, df in compartment_dfs.items():
        df = df.head(0)
        cx.register("temp_df", df)
        cx.execute(f"CREATE OR REPLACE TABLE {compartment} AS SELECT * FROM temp_df")
        cx.unregister("temp_df")
