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
from arg_parsing_utils import check_for_missing_args, parse_args
from file_reading import *
from file_reading import read_zstack_image
from notebook_init_utils import bandicoot_check, init_notebook
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


# In[ ]:


labels_save_file = pathlib.Path("../image_labels/annotations.parquet").resolve(
    strict=True
)
all_features_save_path = pathlib.Path(f"../results/all_features.parquet").resolve(
    strict=True
)
labels = read_labels(labels_save_file)
labels_df = pd.DataFrame(labels)
labels_df["patient"] = labels_df["image_filename"].apply(
    lambda x: (
        "_".join(x.split("_")[0:2]) if not "CQ1" in x else "_".join(x.split("_")[0:3])
    )
)
labels_df["well_fov"] = labels_df["image_filename"].apply(
    lambda x: x.split("_")[2] if not "CQ1" in x else x.split("_")[3]
)
labels_df.head()


# In[7]:


all_features_df = pd.read_parquet(all_features_save_path)
all_features_df
df = pd.merge(
    all_features_df,
    labels_df,
    on=["patient", "well_fov"],
    how="right",
)
# drop rows with na
df = df.dropna(subset=["label"])
# drop the failed labels
df = df[df["label_name"] != "fail"]
# drop the blank labels
df = df[df["label_name"] != "blank"]
# drop na
df = df.dropna()


# In[8]:


# holdout a single patient
holdout_patient = "NF0021_T1"
holdout_df = df[df["patient"] == holdout_patient]
# drop the holdout patient from the main df
df = df[df["patient"] != holdout_patient]
# set up data splits
# train: 80%, val: 10%, test: 10%
# stratify by label, patient

train_df, test_df = train_test_split(
    df,
    test_size=0.10,
    random_state=0,
    stratify=df[["label"]],
)
train_df, val_df = train_test_split(
    train_df,
    test_size=0.1111,  # 0.1111 * 0.90 = 0.10 (10% test)
    random_state=0,
    stratify=train_df[["label"]],
)
print(f"Train size: {len(train_df)}")
print(f"Validation size: {len(val_df)}")
print(f"Test size: {len(test_df)}")
print(f"Holdout size: {len(holdout_df)}")
if len(train_df) + len(val_df) + len(test_df) + len(holdout_df) != len(df) + len(
    holdout_df
):
    raise ValueError("Data split sizes do not add up to total dataset size.")


# In[9]:


non_feature_cols = [
    "patient",
    "well_fov",
    "image_filename",
    "annotator",
    "label",
    "label_name",
    "timestamp",
]


# In[11]:


# train a multi-class logistic regression model for the organoid labels

log_reg_model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=0,
)
rf_model = RandomForestClassifier(n_estimators=1000, random_state=0)


log_reg_model.fit(
    train_df.drop(columns=non_feature_cols),
    train_df["label_name"],
)
rf_model.fit(
    train_df.drop(columns=non_feature_cols),
    train_df["label_name"],
)

# save the models
joblib.dump(log_reg_model, "../models/logistic_regression_model.joblib")
joblib.dump(rf_model, "../models/random_forest_model.joblib")


# In[10]:


# log reg preds
train_log_reg_preds = log_reg_model.predict(train_df.drop(columns=non_feature_cols))

val_log_reg_preds = log_reg_model.predict(val_df.drop(columns=non_feature_cols))

test_log_reg_preds = log_reg_model.predict(test_df.drop(columns=non_feature_cols))
holdout_log_reg_preds = log_reg_model.predict(holdout_df.drop(columns=non_feature_cols))

# rf preds
train_rf_preds = rf_model.predict(train_df.drop(columns=non_feature_cols))
val_rf_preds = rf_model.predict(val_df.drop(columns=non_feature_cols))
test_rf_preds = rf_model.predict(test_df.drop(columns=non_feature_cols))
holdout_rf_preds = rf_model.predict(holdout_df.drop(columns=non_feature_cols))


# In[11]:


label_names = {
    1: "globular",
    2: "small",
    3: "dissociated",
    4: "elongated",
    5: "blank",
    6: "cluster",
    7: "fail",
}


# In[12]:


############################################
# Logistic Regression Model Performance
############################################

# plot the confusion matrix for the validation set (row-normalized to percentages)
plt.figure(figsize=(8, 6))
# add a figure title for the whole figure
plt.suptitle("Logistic Regression Model Performance", fontsize=16)
# train - val - test - holdout
############################################

# train
plt.subplot(2, 2, 1)
labels_in_train = np.sort(train_df["label_name"].unique())
cm = confusion_matrix(
    train_df["label_name"], train_log_reg_preds, labels=labels_in_train
)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

im = plt.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
plt.title("Train Confusion Matrix (%)")
cbar = plt.colorbar(im)
cbar.set_label("Percentage (%)")

tick_marks = np.arange(len(labels_in_train))
tick_labels = [label_names.get(l, str(l)) for l in labels_in_train]
plt.xticks(np.arange(len(tick_labels)), tick_labels, rotation=45, ha="right")
plt.yticks(np.arange(len(tick_labels)), tick_labels)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
############################################
# validation
plt.subplot(2, 2, 2)
labels_in_val = np.sort(val_df["label_name"].unique())
cm = confusion_matrix(val_df["label_name"], val_log_reg_preds, labels=labels_in_val)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

im = plt.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
plt.title("Validation Confusion Matrix (%)")
cbar = plt.colorbar(im)
cbar.set_label("Percentage (%)")

tick_marks = np.arange(len(labels_in_val))
tick_labels = [label_names.get(l, str(l)) for l in labels_in_val]
plt.xticks(tick_marks, tick_labels, rotation=45, ha="right")
plt.yticks(tick_marks, tick_labels)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
############################################
# test
plt.subplot(2, 2, 3)
labels_in_test = np.sort(test_df["label_name"].unique())
cm = confusion_matrix(test_df["label_name"], test_log_reg_preds, labels=labels_in_test)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
im = plt.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
plt.title("Test Confusion Matrix (%)")
cbar = plt.colorbar(im)
cbar.set_label("Percentage (%)")
tick_marks = np.arange(len(labels_in_test))
tick_labels = [label_names.get(l, str(l)) for l in labels_in_test]
plt.xticks(tick_marks, tick_labels, rotation=45, ha="right")
plt.yticks(tick_marks, tick_labels)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
############################################
# holdout
plt.subplot(2, 2, 4)
labels_in_holdout = np.sort(holdout_df["label_name"].unique())
cm = confusion_matrix(
    holdout_df["label_name"], holdout_log_reg_preds, labels=labels_in_holdout
)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
im = plt.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
plt.title("Holdout Confusion Matrix (%)")
cbar = plt.colorbar(im)
cbar.set_label("Percentage (%)")
tick_marks = np.arange(len(labels_in_holdout))
tick_labels = [label_names.get(l, str(l)) for l in labels_in_holdout]
plt.xticks(tick_marks, tick_labels, rotation=45, ha="right")
plt.yticks(tick_marks, tick_labels)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()

############################################
# Random Forrest Model Performance
############################################

# plot the confusion matrix for the validation set (row-normalized to percentages)
plt.figure(figsize=(8, 6))
plt.suptitle("Random Forest Model Performance", fontsize=16)
# train - val - test - holdout
############################################

# train
plt.subplot(2, 2, 1)
labels_in_train = np.sort(train_df["label_name"].unique())
cm = confusion_matrix(train_df["label_name"], train_rf_preds, labels=labels_in_train)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

im = plt.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
plt.title("Train Confusion Matrix (%)")
cbar = plt.colorbar(im)
cbar.set_label("Percentage (%)")

tick_marks = np.arange(len(labels_in_train))
tick_labels = [label_names.get(l, str(l)) for l in labels_in_train]
plt.xticks(tick_marks, tick_labels, rotation=45, ha="right")
plt.yticks(tick_marks, tick_labels)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
############################################
# validation
plt.subplot(2, 2, 2)
labels_in_val = np.sort(val_df["label_name"].unique())
cm = confusion_matrix(val_df["label_name"], val_rf_preds, labels=labels_in_val)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

im = plt.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
plt.title("Validation Confusion Matrix (%)")
cbar = plt.colorbar(im)
cbar.set_label("Percentage (%)")

tick_marks = np.arange(len(labels_in_val))
tick_labels = [label_names.get(l, str(l)) for l in labels_in_val]
plt.xticks(tick_marks, tick_labels, rotation=45, ha="right")
plt.yticks(tick_marks, tick_labels)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
############################################
# test
plt.subplot(2, 2, 3)
labels_in_test = np.sort(test_df["label_name"].unique())
cm = confusion_matrix(test_df["label_name"], test_rf_preds, labels=labels_in_test)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
im = plt.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
plt.title("Test Confusion Matrix (%)")
cbar = plt.colorbar(im)
cbar.set_label("Percentage (%)")
tick_marks = np.arange(len(labels_in_test))
tick_labels = [label_names.get(l, str(l)) for l in labels_in_test]
plt.xticks(tick_marks, tick_labels, rotation=45, ha="right")
plt.yticks(tick_marks, tick_labels)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
############################################
# holdout
plt.subplot(2, 2, 4)
labels_in_holdout = np.sort(holdout_df["label_name"].unique())
cm = confusion_matrix(
    holdout_df["label_name"], holdout_rf_preds, labels=labels_in_holdout
)
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
im = plt.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
plt.title("Holdout Confusion Matrix (%)")
cbar = plt.colorbar(im)
cbar.set_label("Percentage (%)")
tick_marks = np.arange(len(labels_in_holdout))
tick_labels = [label_names.get(l, str(l)) for l in labels_in_holdout]
plt.xticks(tick_marks, tick_labels, rotation=45, ha="right")
plt.yticks(tick_marks, tick_labels)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()


# In[13]:


# concatenate the train, val, test df with the predections

train_out = train_df.assign(split="train", log_reg_prediction=train_log_reg_preds)
val_out = val_df.assign(split="val", log_reg_prediction=val_log_reg_preds)
test_out = test_df.assign(split="test", log_reg_prediction=test_log_reg_preds)
holdout_out = holdout_df.assign(
    split="holdout", log_reg_prediction=holdout_log_reg_preds
)

train_out = train_out.assign(split="train", rf_prediction=train_rf_preds)
val_out = val_out.assign(split="val", rf_prediction=val_rf_preds)
test_out = test_out.assign(split="test", rf_prediction=test_rf_preds)
holdout_out = holdout_out.assign(split="holdout", rf_prediction=holdout_rf_preds)

all_preds_df = pd.concat([train_out, val_out, test_out, holdout_out], ignore_index=True)
all_preds_df["image_path"] = all_preds_df.apply(
    lambda row: pathlib.Path(
        f"{raw_image_base_dir}/{row['patient']}/zstack_images/"
        f"{row['well_fov']}/{row['well_fov']}_555.tif"
    ).resolve(),
    axis=1,
)
all_preds_df.head()


# In[14]:


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# calculate the model-wise per split performance metrics
metrics_to_compute = ["accuracy", "precision", "recall", "f1"]
performance_metrics = []
for model in ["log_reg_prediction", "rf_prediction"]:
    for split in ["train", "val", "test", "holdout"]:
        subset_df = all_preds_df[all_preds_df["split"] == split].copy()
        y_true = subset_df["label_name"]
        y_pred = subset_df[model]
        accuracy = np.mean(y_true == y_pred)
        performance_metrics.append(
            {
                "model": model,
                "split": split,
                "accuracy": accuracy,
            }
        )
performance_metrics_df = pd.DataFrame(performance_metrics)
performance_metrics_df


# In[15]:


import seaborn as sns

# plot the accuracy for each split with each model as a different color
# split bar plot
plt.figure(figsize=(8, 6))
plt.title("Model Accuracy by Split")
sns.barplot(data=performance_metrics_df, x="split", y="accuracy", hue="model")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.xlabel("Data Split")
plt.legend(title="Model", loc="upper left")
plt.tight_layout()
plt.show()


# In[16]:


# pick three random images from each predicted class to show
label_names = {
    1: "globular",
    2: "small",
    3: "dissociated",
    4: "elongated",
    5: "blank",
    6: "cluster",
    7: "fail",
}
rng = np.random.default_rng(0)
samples_per_class = 3

sampled_rows = []
for pred_label, group in all_preds_df.groupby("log_reg_prediction"):
    n = min(samples_per_class, len(group))
    sampled_rows.append(group.sample(n=n, random_state=rng.integers(0, 1_000_000)))

sampled_df = pd.concat(sampled_rows, ignore_index=True)

n_classes = len(sampled_df["log_reg_prediction"].unique())
fig, axes = plt.subplots(
    n_classes, samples_per_class, figsize=(4 * samples_per_class, 4 * n_classes)
)
if n_classes == 1:
    axes = np.array([axes])

for row_idx, (pred_label, group) in enumerate(sampled_df.groupby("log_reg_prediction")):
    group = group.reset_index(drop=True)
    for col_idx in range(samples_per_class):
        ax = axes[row_idx, col_idx]
        if col_idx >= len(group):
            ax.axis("off")
            continue
        image_path = group.loc[col_idx, "image_path"]
        img = read_zstack_image(image_path)
        mid = img[img.shape[0] // 2]
        ax.imshow(mid, cmap="inferno")
        ax.axis("off")
        title = label_names.get(pred_label, str(pred_label))
        ax.set_title(title)

plt.tight_layout()
plt.show()
