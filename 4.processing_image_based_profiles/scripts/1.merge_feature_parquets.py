#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import pathlib
import pprint
import sqlite3
from contextlib import closing
from functools import reduce

import duckdb
import pandas as pd

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

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


# In[ ]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--well_fov",
        type=str,
        required=True,
        help="Well and field of view to process, e.g. 'A01_1'",
    )
    argparser.add_argument(
        "--patient",
        type=str,
        required=True,
        help="Patient ID to process, e.g. 'P01'",
    )
    args = argparser.parse_args()
    well_fov = args.well_fov
    patient = args.patient
else:
    well_fov = "C4-2"
    patient = "NF0014"


result_path = pathlib.Path(
    f"{root_dir}/data/{patient}/extracted_features/{well_fov}"
).resolve(strict=True)
database_path = pathlib.Path(
    f"{root_dir}/data/{patient}/converted_profiles/{well_fov}"
).resolve()
database_path.mkdir(parents=True, exist_ok=True)
# create the sqlite database
sqlite_path = database_path / f"{well_fov}.duckdb"


# get a list of all parquets in the directory recursively
parquet_files = list(result_path.rglob("*.parquet"))
parquet_files.sort()
print(len(parquet_files), "parquet files found")


# In[3]:


# create the nested dictionary to hold the feature types and compartments
feature_types = [
    "AreaSize_Shape",
    "Colocalization",
    "Intensity",
    "Granularity",
    "Neighbor",
    "Texture",
]
compartments = ["Organoid", "Nuclei", "Cell", "Cytoplasm"]

feature_types_dict = {cmp: {ft: [] for ft in feature_types} for cmp in compartments}
# copy the feature types dictionary to another blank dictionary that will hold the parquet files

merged_df_dict = {cmp: {ft: [] for ft in feature_types} for cmp in compartments}


for file in parquet_files:
    for compartment in feature_types_dict.keys():
        for feature_type in feature_types_dict[compartment].keys():
            if compartment in str(file) and feature_type in str(file):
                feature_types_dict[compartment][feature_type].append(file)


# In[4]:


for compartment in feature_types_dict.keys():
    for feature_type in feature_types_dict[compartment].keys():
        if len(feature_types_dict[compartment][feature_type]) > 0:
            for file in feature_types_dict[compartment][feature_type]:
                # check if the file exists
                if not file.exists():
                    if (
                        "neighbor" in file.name.lower()
                        and "nuclei" not in file.name.lower()
                    ):
                        print(f"File {file} does not exist")
                        continue
                # check if the file is a parquet file
                if not file.name.endswith(".parquet"):
                    print(f"File {file} is not a parquet file")
                    continue
                # read the parquet files
                try:
                    df = duckdb.read_parquet(str(file)).to_df()
                except Exception as e:
                    print(
                        f"Error reading {feature_types_dict[compartment][feature_type]}: {e}"
                    )

                # add the dataframe to the dictionary
                merged_df_dict[compartment][feature_type].append(df)
        else:
            if (
                "neighbor" in feature_type.lower()
                and "nuclei" not in compartment.lower()
            ):
                merged_df_dict[compartment][feature_type].append(pd.DataFrame())
            else:
                print(
                    f"No files found for {compartment} {feature_type}. Please check the directory."
                )
                merged_df_dict[compartment][feature_type].append(pd.DataFrame())
                for channel_df in merged_df_dict[compartment][feature_type]:
                    if channel_df.empty:
                        continue
                    # check if the dataframe has the required columns
                    if (
                        "object_id" not in channel_df.columns
                        or "image_set" not in channel_df.columns
                    ):
                        print(
                            f"Dataframe {channel_df} does not have the required columns"
                        )
                        continue
                    # check if the dataframe is empty
                    if channel_df.empty:
                        continue


# In[5]:


feature_types = [
    "AreaSize_Shape",
    "Colocalization",
    "Intensity",
    "Granularity",
    "Neighbor",
    "Texture",
]
compartments = ["Organoid", "Nuclei", "Cell", "Cytoplasm"]

final_df_dict = {
    cmp: {ft: pd.DataFrame() for ft in feature_types} for cmp in compartments
}


# In[6]:


for compartment in merged_df_dict.keys():
    for feature_type in merged_df_dict[compartment].keys():
        for df in merged_df_dict[compartment][feature_type]:
            if df.empty:
                continue
            df.drop(columns=["__index_level_0__"], inplace=True, errors="ignore")
            # if "Texture" not in feature_type:
            final_df_dict[compartment][feature_type] = reduce(
                lambda left, right: pd.merge(
                    left, right, how="left", on=["object_id", "image_set"]
                ),
                merged_df_dict[compartment][feature_type],
            )


# In[7]:


merged_df = pd.DataFrame(
    {
        "object_id": [],
        "image_set": [],
    }
)


# In[8]:


compartment_merged_dict = {
    "Organoid": pd.DataFrame(),
    "Cell": pd.DataFrame(),
    "Nuclei": pd.DataFrame(),
    "Cytoplasm": pd.DataFrame(),
}


# In[9]:


for compartment in final_df_dict.keys():
    print(f"Processing compartment: {compartment}")
    for feature_type in final_df_dict[compartment].keys():
        if compartment != "Nuclei" and feature_type == "Neighbor":
            print(
                f"Skipping {compartment} {feature_type} as it is not applicable for this compartment."
            )
            continue
        if compartment_merged_dict[compartment].empty:
            compartment_merged_dict[compartment] = final_df_dict[compartment][
                feature_type
            ].copy()
        else:
            compartment_merged_dict[compartment] = pd.merge(
                compartment_merged_dict[compartment],
                final_df_dict[compartment][feature_type],
                on=["object_id", "image_set"],
                how="outer",
            )


# In[10]:


with duckdb.connect(sqlite_path) as cx:
    for compartment, df in compartment_merged_dict.items():
        cx.register("temp_df", df)
        cx.execute(f"CREATE OR REPLACE TABLE {compartment} AS SELECT * FROM temp_df")
        cx.unregister("temp_df")

