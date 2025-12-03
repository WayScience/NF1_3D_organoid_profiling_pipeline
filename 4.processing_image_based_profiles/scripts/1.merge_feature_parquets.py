#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
from functools import reduce

import duckdb
import pandas as pd
from arg_parsing_utils import parse_args
from notebook_init_utils import bandicoot_check, init_notebook

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
    well_fov = "C4-2"
    patient = "NF0014_T1"
    output_features_subparent_name = "convolution_22_extracted_features"
    image_based_profiles_subparent_name = "convolution_22_image_based_profiles"


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
    "Neighbor",
    "SAMMed3D",
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


for compartment, df in compartment_merged_dict.items():
    print(compartment, df.shape)


# In[11]:


with duckdb.connect(DB_structure_path) as cx:
    organoid_table = cx.execute("SELECT * FROM Organoid").df()
    cell_table = cx.execute("SELECT * FROM Cell").df()
    nuclei_table = cx.execute("SELECT * FROM Nuclei").df()
    cytoplasm_table = cx.execute("SELECT * FROM Cytoplasm").df()

dict_of_DB_structues = {
    "Organoid": organoid_table,
    "Cell": cell_table,
    "Nuclei": nuclei_table,
    "Cytoplasm": cytoplasm_table,
}


# In[12]:


# get the table from the DB_structue
with duckdb.connect(sqlite_path) as cx:
    for compartment, df in compartment_merged_dict.items():
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
