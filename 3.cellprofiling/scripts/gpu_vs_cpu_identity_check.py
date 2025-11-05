#!/usr/bin/env python
# coding: utf-8

# This notebook validates that the featurization is equivalent on both CPU and GPU.
# Regardless of the processor used, the output should be the same.

# In[ ]:


import pathlib

import numpy as np
import pandas as pd
import psutil
import skimage

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from featurization_parsable_arguments import parse_featurization_args

# In[ ]:


if not in_notebook:
    arguments_dict = parse_featurization_args()
    patient = arguments_dict["patient"]
    well_fov = arguments_dict["well_fov"]


else:
    well_fov = "C4-2"
    patient = "NF0014_T1"


output_parent_path = pathlib.Path(
    f"{root_dir}/data/{patient}/extracted_features/{well_fov}/"
).resolve(strict=True)


# In[13]:


features_dict = {
    "feature_name": [],
    "feature_processor": [],
    "file_path": [],
}
dict_of_dfs = {}
# get each of the features
feature_files = list(output_parent_path.glob("*parquet"))
feature_files = [f for f in feature_files if f.is_file()]
for file in feature_files:
    if "Area" in file.name and "CPU" in file.name:
        features_dict["feature_name"].append("Area")
        features_dict["feature_processor"].append("CPU")
        features_dict["file_path"].append(file)
        dict_of_dfs["Area_CPU"] = pd.read_parquet(file)
    elif "Area" in file.name and "GPU" in file.name:
        features_dict["feature_name"].append("Area")
        features_dict["feature_processor"].append("GPU")
        features_dict["file_path"].append(file)
        dict_of_dfs["Area_GPU"] = pd.read_parquet(file)
    elif "Coloc" in file.name and "CPU" in file.name:
        features_dict["feature_name"].append("Coloc")
        features_dict["feature_processor"].append("CPU")
        features_dict["file_path"].append(file)
        dict_of_dfs["Coloc_CPU"] = pd.read_parquet(file)
    elif "Coloc" in file.name and "GPU" in file.name:
        features_dict["feature_name"].append("Coloc")
        features_dict["feature_processor"].append("GPU")
        features_dict["file_path"].append(file)
        dict_of_dfs["Coloc_GPU"] = pd.read_parquet(file)
    elif "Intensity" in file.name and "CPU" in file.name:
        features_dict["feature_name"].append("Intensity")
        features_dict["feature_processor"].append("CPU")
        features_dict["file_path"].append(file)
        dict_of_dfs["Intensity_CPU"] = pd.read_parquet(file)
    elif "Intensity" in file.name and "GPU" in file.name:
        features_dict["feature_name"].append("Intensity")
        features_dict["feature_processor"].append("GPU")
        features_dict["file_path"].append(file)
        dict_of_dfs["Intensity_GPU"] = pd.read_parquet(file)
    elif "Gran" in file.name and "CPU" in file.name:
        features_dict["feature_name"].append("Granularity")
        features_dict["feature_processor"].append("CPU")
        features_dict["file_path"].append(file)
        dict_of_dfs["Granularity_CPU"] = pd.read_parquet(file)
    elif "Gran" in file.name and "GPU" in file.name:
        features_dict["feature_name"].append("Granularity")
        features_dict["feature_processor"].append("GPU")
        features_dict["file_path"].append(file)
        dict_of_dfs["Granularity_GPU"] = pd.read_parquet(file)
    elif "Neighbors" in file.name and "CPU" in file.name:
        features_dict["feature_name"].append("Neighbors")
        features_dict["feature_processor"].append("CPU")
        features_dict["file_path"].append(file)
        dict_of_dfs["Neighbors_CPU"] = pd.read_parquet(file)
    elif "Texture" in file.name and "CPU" in file.name:
        features_dict["feature_name"].append("Texture")
        features_dict["feature_processor"].append("CPU")
        features_dict["file_path"].append(file)
        dict_of_dfs["Texture_CPU"] = pd.read_parquet(file)
    else:
        print(f"Unknown feature file: {file.name}")


# In[14]:


import hashlib


def get_file_hash(file_path):
    """Calculate SHA256 hash of a file without loading it as an image."""
    try:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        return f"Error: {e}"


# In[15]:


features_df = pd.DataFrame.from_dict(features_dict)

features_df.rename(columns={"index": "feature_processor"}, inplace=True)
# get the file size in KB
features_df["file_size_KB"] = features_df["file_path"].apply(
    lambda x: x.stat().st_size / (1024)
)
# get the sha256 hash of the file
features_df["sha256"] = features_df["file_path"].apply(get_file_hash)
features_df


# In[16]:


# pivot the dataframe to have one row per feature and processor type
features_df = features_df.pivot(
    index=["feature_name"], columns="feature_processor", values="file_path"
).reset_index()
features_df["CPU_file_size_KB"] = features_df["CPU"].apply(
    lambda x: x.stat().st_size / (1024) if isinstance(x, pathlib.Path) else None
)
features_df["GPU_file_size_KB"] = features_df["GPU"].apply(
    lambda x: x.stat().st_size / (1024) if isinstance(x, pathlib.Path) else None
)
features_df["CPU_sha256"] = features_df["CPU"].apply(get_file_hash)
features_df["GPU_sha256"] = features_df["GPU"].apply(get_file_hash)
features_df.insert(
    1,
    "sha256_match",
    features_df.apply(lambda row: row["CPU_sha256"] == row["GPU_sha256"], axis=1),
)
features_df
