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
import seaborn as sns
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
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


labels_save_file = pathlib.Path(
    "../image_labels/segmentation_classes.parquet"
).resolve()
annotations_save_file = pathlib.Path("../image_labels/annotations.parquet").resolve()
all_features_save_path = pathlib.Path(f"../results/all_features.parquet").resolve(
    strict=True
)
labels = read_labels(annotations_save_file)
labels_df = pd.DataFrame(labels)
print(labels_df.shape)
labels_df.head()


# In[6]:


all_features_df = pd.read_parquet(all_features_save_path)
all_features_df
df = pd.merge(
    all_features_df,
    labels_df,
    on=["patient", "well_fov"],
    how="right",
)


# In[7]:


# label stats
print("Label distribution:")
print(df["label_name"].value_counts())


# In[8]:


original_shape = df.shape
# drop rows with na
df = df.dropna(subset=["label_name"])
# drop the failed labels
# print the number of failed labels
print(f"Number of failed labels: {df[df['label_name'] == 'fail'].shape[0]}")
df = df[df["label_name"] != "fail"]
# drop the blank labels
print(f"Number of blank labels: {df[df['label_name'] == 'blank'].shape[0]}")
df = df[df["label_name"] != "blank"]
dropped = original_shape[0] - df.shape[0]
print(f"Dropped {dropped} rows with NA values.")
print(f"Remaining shape: {df.shape}")
df.head()


# In[9]:


# holdout a single patient
holdout_patients = ["NF0021_T1", "NF0030_T1"]
holdout_df = df[df["patient"].isin(holdout_patients)]
# drop the holdout patient from the main df
df = df[~df["patient"].isin(holdout_patients)]
# set up data splits
# train: 80%, val: 10%, test: 10%
# stratify by label, patient

train_df, test_df = train_test_split(
    df,
    test_size=0.10,
    random_state=0,
    stratify=df[["label_name"]],
)
train_df, val_df = train_test_split(
    train_df,
    test_size=0.1111,  # 0.1111 * 0.90 = 0.10 (10% test)
    random_state=0,
    stratify=train_df[["label_name"]],
)
print(
    f"Train size: {len(train_df)}; percentage: {np.round(len(train_df) / len(df) * 100, 2)}%"
)
print(
    f"Validation size: {len(val_df)}; percentage: {np.round(len(val_df) / len(df) * 100, 2)}%"
)
print(
    f"Test size: {len(test_df)}; percentage: {np.round(len(test_df) / len(df) * 100, 2)}%"
)
print(
    f"Holdout size: {len(holdout_df)}; percentage: {np.round(len(holdout_df) / len(df) * 100, 2)}%"
)

if len(train_df) + len(val_df) + len(test_df) + len(holdout_df) != len(df) + len(
    holdout_df
):
    raise ValueError("Data split sizes do not add up to total dataset size.")


# In[10]:


non_feature_cols = [
    "patient",
    "well_fov",
    "label_name",
    "image_filename",
    "annotator",
    "label",
    "timestamp",
]


# In[11]:


# fit the train data to the z score normalization


scaler = StandardScaler()
feature_cols = train_df.drop(columns=non_feature_cols).columns

# Fit scaler on training features only
scaler.fit(train_df[feature_cols])

# Transform each split and reconstruct DataFrames
train_scaled = pd.DataFrame(
    scaler.transform(train_df[feature_cols]), columns=feature_cols, index=train_df.index
)
train_df = pd.concat([train_df[non_feature_cols], train_scaled], axis=1)

val_scaled = pd.DataFrame(
    scaler.transform(val_df[feature_cols]), columns=feature_cols, index=val_df.index
)
val_df = pd.concat([val_df[non_feature_cols], val_scaled], axis=1)

test_scaled = pd.DataFrame(
    scaler.transform(test_df[feature_cols]), columns=feature_cols, index=test_df.index
)
test_df = pd.concat([test_df[non_feature_cols], test_scaled], axis=1)

holdout_scaled = pd.DataFrame(
    scaler.transform(holdout_df[feature_cols]),
    columns=feature_cols,
    index=holdout_df.index,
)
holdout_df = pd.concat([holdout_df[non_feature_cols], holdout_scaled], axis=1)


# In[12]:


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


# In[13]:


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


# In[14]:


label_names = {
    1: "globular",
    2: "small",
    3: "dissociated",
    4: "elongated",
    5: "blank",
    6: "cluster",
    7: "fail",
}


# In[15]:


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


# In[16]:


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


# In[17]:


# calculate the model-wise per split performance metrics
performance_metrics = []
for model in ["log_reg_prediction", "rf_prediction"]:
    for split in ["train", "val", "test", "holdout"]:
        subset_df = all_preds_df[all_preds_df["split"] == split].copy()
        y_true = subset_df["label_name"]
        y_pred = subset_df[model]
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        performance_metrics.append(
            {
                "model": model,
                "split": split,
                "balanced_accuracy": balanced_acc,
            }
        )
performance_metrics_df = pd.DataFrame(performance_metrics)
performance_metrics_df


# In[18]:


# plot the accuracy for each split with each model as a different color
# split bar plot
plt.figure(figsize=(8, 6))
plt.title("Model balanced accuracy by Split")
sns.barplot(data=performance_metrics_df, x="split", y="balanced_accuracy", hue="model")
plt.ylim(0, 1)
plt.ylabel("Balanced Accuracy")
plt.xlabel("Data Split")
plt.legend(title="Model", loc="upper right")
plt.tight_layout()
plt.show()


# In[19]:


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


# ## Show wrong predictions

# In[20]:


n_images_per_split = 15
# plot 1 wrong prediction per data split and model
wrong_preds = all_preds_df[
    all_preds_df["label_name"] != all_preds_df["log_reg_prediction"]
].copy()
wrong_preds["model"] = "Logistic Regression"
wrong_rf_preds = all_preds_df[
    all_preds_df["label_name"] != all_preds_df["rf_prediction"]
].copy()
wrong_rf_preds["model"] = "Random Forest"
wrong_preds_combined = pd.concat([wrong_preds, wrong_rf_preds], ignore_index=True)
wrong_preds_combined["split_model"] = (
    wrong_preds_combined["split"] + " - " + wrong_preds_combined["model"]
)
# pick one random wrong prediction per split_model
rng = np.random.default_rng(0)
sampled_wrong_preds = []
for split in wrong_preds_combined["split"].unique():
    group = wrong_preds_combined[wrong_preds_combined["split"] == split]
    sampled_wrong_preds.append(
        group.sample(n=n_images_per_split, random_state=rng.integers(0, 1_000_000))
    )
sampled_wrong_preds_df = pd.concat(sampled_wrong_preds, ignore_index=True)


# In[21]:


# plot in a grid with rows as splits and columns as models
splits = ["train", "val", "test", "holdout"]
n_images_to_visualize = 5
fig, axes = plt.subplots(
    len(splits),
    n_images_to_visualize,  # n  images per data split
    figsize=(8, 16),
)
for row_idx, split in enumerate(splits):
    for col_idx in range(
        n_images_to_visualize
    ):  # n_images_to_visualize images per data split
        ax = axes[row_idx, col_idx]
        sample = sampled_wrong_preds_df[(sampled_wrong_preds_df["split"] == split)]
        label = (
            sample["label_name"].iloc[col_idx]
            if not sample.empty and col_idx < len(sample)
            else "N/A"
        )
        log_reg_model_pred = (
            sample["log_reg_prediction"].iloc[col_idx]
            if not sample.empty and col_idx < len(sample)
            else "N/A"
        )
        rf_model_pred = (
            sample["rf_prediction"].iloc[col_idx]
            if not sample.empty and col_idx < len(sample)
            else "N/A"
        )
        if sample.empty or col_idx >= len(sample):
            ax.axis("off")
            continue
        image_path = sample.iloc[col_idx]["image_path"]
        img = read_zstack_image(image_path)
        mid = img[img.shape[0] // 2]
        ax.imshow(mid, cmap="inferno")
        ax.axis("off")
        title = f"{split}\nTrue: {label}\nLog Reg: {log_reg_model_pred}\nRF: {rf_model_pred}"
        ax.set_title(title)
plt.tight_layout()
plt.show()
