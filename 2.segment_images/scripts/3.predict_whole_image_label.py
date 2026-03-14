#!/usr/bin/env python
# coding: utf-8

# This notebook trains a random forest model to predict the binning label of each image based on the features extracted from the cell profiling pipeline.
# The model is trained on a balanced dataset, where each binning label is represented equally.
# The dataset is split into training, validation, and test sets, and the model's performance is evaluated on the test set.
# The predicted bins are used to adjust the segmentation parameters for each image, which is expected to improve the segmentation quality.

# In[1]:


import argparse
import json
import os
import pathlib
import sys
import time

import joblib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy
import tifffile
from image_analysis_3D.file_utils.arg_parsing_utils import (
    check_for_missing_args,
    parse_args,
)
from image_analysis_3D.file_utils.file_reading import *
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from skimage.filters import sobel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# In[2]:


def read_labels(infile: str) -> dict:
    """
    Description
    ----------
    Read labels from a parquet file.
    Parameters
    ----------
    infile : str
        Path to the input parquet file.
    Returns
    -------
    dict
        Dictionary containing the labels.
    """
    data = pd.read_parquet(infile).to_dict(orient="list")
    return data


# In[3]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[4]:


root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot/NF1_organoid_data")).resolve(),
    root_dir,
)
patient_list_file_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
raw_image_base_dir = pathlib.Path(f"{image_base_dir}/data/").resolve()


# In[5]:


labels_input_file = pathlib.Path("../image_labels/annotations.parquet").resolve(
    strict=True
)
labels_save_file = pathlib.Path(
    "../image_labels/segmentation_classes.parquet"
).resolve()
labels_save_file.parent.mkdir(parents=True, exist_ok=True)
all_features_save_path = pathlib.Path(f"../results/all_features.parquet").resolve(
    strict=True
)
labels = read_labels(labels_input_file)
labels_df = pd.DataFrame(labels)
labels_df["patient"] = labels_df["image_filename"].apply(
    lambda x: (
        "_".join(x.split("_")[0:2]) if not "CQ1" in x else "_".join(x.split("_")[0:3])
    )
)
labels_df["well_fov"] = labels_df["image_filename"].apply(
    lambda x: x.split("_")[2] if not "CQ1" in x else x.split("_")[3]
)
non_feature_cols = [
    "patient",
    "well_fov",
    "image_filename",
    "annotator",
    "label",
    "label_name",
    "timestamp",
]
all_features_df = pd.read_parquet(all_features_save_path)
all_features_df
df = pd.merge(
    all_features_df,
    labels_df,
    on=["patient", "well_fov"],
    how="left",
)
print(df.shape)


# In[6]:


log_reg_model_path = "../models/logistic_regression_model.joblib"
rf_model_path = "../models/random_forest_model.joblib"
log_reg_model = joblib.load(log_reg_model_path)
rf_model = joblib.load(rf_model_path)


# In[7]:


predicted_labels = log_reg_model.predict(df.drop(columns=non_feature_cols))
prediction_df = df.assign(predicted_label=predicted_labels)
prediction_df.drop(
    columns=[col for col in prediction_df.columns if "CHAMI" in col or "SAM" in col],
    inplace=True,
)
prediction_df.drop(columns=["timestamp", "image_filename", "label"], inplace=True)
prediction_df.head()


# In[8]:


output_dict = {"patient": [], "well_fov": [], "label": [], "predicted_or_gt": []}

for index, row in prediction_df.iterrows():
    output_dict["patient"].append(row["patient"])
    output_dict["well_fov"].append(row["well_fov"])
    if pd.isna(row["label_name"]):
        output_dict["label"].append(row["predicted_label"])
        output_dict["predicted_or_gt"].append("predicted")
    else:
        output_dict["label"].append(row["label_name"])
        output_dict["predicted_or_gt"].append("gt")

df = pd.DataFrame(output_dict)
df.to_parquet(labels_save_file, index=False)
print(df.shape)
df.head()


# In[9]:


# get the stats for how many predicted vs gt labels we have and for each patient and class
annotation_stats = (
    df.groupby(["predicted_or_gt"]).size().to_frame(name="count").reset_index()
)


# In[10]:


gt_count = annotation_stats[annotation_stats["predicted_or_gt"] == "gt"][
    "count"
].values[0]
predicted_count = annotation_stats[annotation_stats["predicted_or_gt"] == "predicted"][
    "count"
].values[0]
total_count = gt_count + predicted_count
print(f"GT count: {gt_count}")
print(f"Predicted count: {predicted_count}")
print(f"Total count: {total_count}")
print(f"Proportion of GT labels: {np.round(gt_count / total_count, 3) * 100}%")
