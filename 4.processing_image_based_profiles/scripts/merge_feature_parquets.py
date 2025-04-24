#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import pprint
import sqlite3

import duckdb
import pandas as pd

# In[2]:


well_fov = "C4-2"

result_path = pathlib.Path(f"../../2.cellprofiling/results/{well_fov}/").resolve(
    strict=True
)
database_path = pathlib.Path(f"../results/converted_profiles/{well_fov}").resolve()
database_path.mkdir(parents=True, exist_ok=True)
# create the sqlite database
sqlite_path = database_path / f"{well_fov}.sqlite"


# get a list of all parquets in the directory
parquet_files = list(result_path.glob("*.parquet"))
parquet_files.sort()


# In[3]:


feature_types_dict = {
    "Organoid": {
        "AreaSize_Shape": [],
        "Colocalization": [],
        "Intensity": [],
        "Granularity": [],
        "Neighbor": [],
        "Texture": [],
    },
    "Cell": {
        "AreaSize_Shape": [],
        "Colocalization": [],
        "Intensity": [],
        "Granularity": [],
        "Neighbor": [],
        "Texture": [],
    },
    "Nuclei": {
        "AreaSize_Shape": [],
        "Colocalization": [],
        "Intensity": [],
        "Granularity": [],
        "Neighbor": [],
        "Texture": [],
    },
    "Cytoplasm": {
        "AreaSize_Shape": [],
        "Colocalization": [],
        "Intensity": [],
        "Granularity": [],
        "Neighbor": [],
        "Texture": [],
    },
}
for file in parquet_files:
    for compartment in feature_types_dict.keys():
        for feature_type in feature_types_dict[compartment].keys():
            if compartment in file.name and feature_type in file.name:
                feature_types_dict[compartment][feature_type].append(file)
pprint.pprint(feature_types_dict)


# In[4]:


# create a record for each compartment
merged_df_dict = {
    "Organoid": {
        "AreaSize_Shape": [],
        "Colocalization": [],
        "Intensity": [],
        "Granularity": [],
        "Neighbor": [],
        "Texture": [],
    },
    "Cell": {
        "AreaSize_Shape": [],
        "Colocalization": [],
        "Intensity": [],
        "Granularity": [],
        "Neighbor": [],
        "Texture": [],
    },
    "Nuclei": {
        "AreaSize_Shape": [],
        "Colocalization": [],
        "Intensity": [],
        "Granularity": [],
        "Neighbor": [],
        "Texture": [],
    },
    "Cytoplasm": {
        "AreaSize_Shape": [],
        "Colocalization": [],
        "Intensity": [],
        "Granularity": [],
        "Neighbor": [],
        "Texture": [],
    },
}

for compartment in feature_types_dict.keys():
    for feature_type in feature_types_dict[compartment].keys():
        if len(feature_types_dict[compartment][feature_type]) > 0:
            for file in feature_types_dict[compartment][feature_type]:
                # check if the file exists
                if not file.exists():
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
                    df = pd.DataFrame()

                # add the dataframe to the dictionary
                merged_df_dict[compartment][feature_type].append(df)
        else:
            print(
                f"No files found for {compartment} {feature_type}. Please check the directory."
            )
            merged_df_dict[compartment][feature_type].append(pd.DataFrame())
        # merge the dataframes
        merged_df_dict[compartment][feature_type] = pd.concat(
            merged_df_dict[compartment][feature_type], ignore_index=True
        )


# In[5]:


conn = sqlite3.connect(sqlite_path)
# merge all the feature types into one dataframe
for compartment in merged_df_dict.keys():
    merged_df = pd.DataFrame(
        {
            "object_id": [],
            "image_set": [],
        }
    )
    for feature_type, feature_type_df in merged_df_dict[compartment].items():
        if len(feature_type_df) > 0:
            merged_df = pd.merge(
                merged_df,
                feature_type_df,
                on=["object_id", "image_set"],
                how="outer",
            )
        else:
            print(f"Dataframe {feature_type} is empty")
            continue

    merged_df.to_sql(
        f"{compartment}",
        conn,
        if_exists="replace",
        index=False,
    )
conn.close()
