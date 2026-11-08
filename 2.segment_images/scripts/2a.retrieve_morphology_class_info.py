#!/usr/bin/env python
# coding: utf-8

# # Morphology class performance: data prep (Python half)
#
# This is **part 1 of 2** (Python kernel). It reproduces the train/val/test/holdout splits and the trained logistic regression / random forest models from `2.train_whole_FOV_models.ipynb`, predicts on every split, attaches drug/treatment + dose metadata from the plate maps, and writes a per-(split, patient, treatment+dose, model) accuracy summary CSV.
#
# **Part 2, `2b.plot_morphology_class_heatmap.ipynb` (R kernel), reads that CSV and renders it as a single heatmap with ComplexHeatmap** — predicted vs. actual class accuracy as the heatmap color, labeled by split / patient / treatment (+ dose) via row-side color tracks, all drawn with R's built-in color palettes.

# In[1]:


import pathlib

import joblib
import pandas as pd
from image_analysis_3D.file_utils.notebook_init_utils import init_notebook
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# In[2]:


root_dir, in_notebook = init_notebook()


# In[3]:


# inputs produced / used by 2.train_whole_FOV_models.ipynb
annotations_save_file = pathlib.Path("../image_labels/annotations.parquet").resolve(
    strict=True
)
all_features_save_path = pathlib.Path("../results/all_features.parquet").resolve(
    strict=True
)
log_reg_model_path = pathlib.Path("../models/logistic_regression_model.joblib").resolve(
    strict=True
)
rf_model_path = pathlib.Path("../models/random_forest_model.joblib").resolve(
    strict=True
)

# plate map metadata (patient -> plate layout -> well -> drug/treatment)
barcode_platemap_path = pathlib.Path(
    f"{root_dir}/config/platemaps/barcode_platemap.csv"
).resolve(strict=True)
platemap_paths = {
    "platemap1": pathlib.Path(f"{root_dir}/config/platemaps/platemap1.csv").resolve(
        strict=True
    ),
    "platemap2": pathlib.Path(f"{root_dir}/config/platemaps/platemap2.csv").resolve(
        strict=True
    ),
}


# ## Rebuild the same data splits used for training
#
# The code below mirrors `2.train_whole_FOV_models.ipynb` exactly (same `random_state`, same stratification, same holdout patients), so the train/val/test/holdout assignment reproduces deterministically without re-training the models.

# In[4]:


labels_df = pd.read_parquet(annotations_save_file)
all_features_df = pd.read_parquet(all_features_save_path)

df = pd.merge(
    all_features_df,
    labels_df,
    on=["patient", "well_fov"],
    how="right",
)
print(df.shape)
df.head()


# In[5]:


# drop unlabeled / failed / blank images, same as the training notebook
df = df.dropna(subset=["label_name"])
df = df[df["label_name"] != "fail"]
df = df[df["label_name"] != "blank"]
print(df.shape)


# In[6]:


# holdout a set of patients, then split the remainder into train/val/test
holdout_patients = ["NF0021_T1", "NF0030_T1"]
holdout_df = df[df["patient"].isin(holdout_patients)]
df = df[~df["patient"].isin(holdout_patients)]

train_df, test_df = train_test_split(
    df,
    test_size=0.10,
    random_state=0,
    stratify=df[["label_name"]],
)
train_df, val_df = train_test_split(
    train_df,
    test_size=0.1111,  # 0.1111 * 0.90 ~= 0.10 (10% test)
    random_state=0,
    stratify=train_df[["label_name"]],
)

print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")
print(f"Holdout: {len(holdout_df)}")


# In[7]:


non_feature_cols = [
    "patient",
    "well_fov",
    "label_name",
    "image_filename",
    "annotator",
    "label",
    "timestamp",
]

# z-score normalize features, fit on train only (same as the training notebook)
scaler = StandardScaler()
feature_cols = train_df.drop(columns=non_feature_cols).columns
scaler.fit(train_df[feature_cols])


def scale_split(split_df: pd.DataFrame) -> pd.DataFrame:
    scaled = pd.DataFrame(
        scaler.transform(split_df[feature_cols]),
        columns=feature_cols,
        index=split_df.index,
    )
    return pd.concat([split_df[non_feature_cols], scaled], axis=1)


train_df = scale_split(train_df)
val_df = scale_split(val_df)
test_df = scale_split(test_df)
holdout_df = scale_split(holdout_df)


# ## Load the trained models and predict on every split

# In[8]:


log_reg_model = joblib.load(log_reg_model_path)
rf_model = joblib.load(rf_model_path)

splits = {
    "train": train_df,
    "val": val_df,
    "test": test_df,
    "holdout": holdout_df,
}

all_preds = []
for split_name, split_df in splits.items():
    features = split_df.drop(columns=non_feature_cols)
    out = split_df[["patient", "well_fov", "label_name"]].copy()
    out["split"] = split_name
    out["log_reg_prediction"] = log_reg_model.predict(features)
    out["rf_prediction"] = rf_model.predict(features)
    all_preds.append(out)

all_preds_df = pd.concat(all_preds, ignore_index=True)
print(all_preds_df.shape)
all_preds_df.head()


# In[9]:


# sanity check against 2.train_whole_FOV_models.ipynb's reported balanced accuracy
from sklearn.metrics import balanced_accuracy_score

sanity_rows = []
for model_col in ["log_reg_prediction", "rf_prediction"]:
    for split_name in ["train", "val", "test", "holdout"]:
        subset = all_preds_df[all_preds_df["split"] == split_name]
        sanity_rows.append(
            {
                "model": model_col,
                "split": split_name,
                "balanced_accuracy": balanced_accuracy_score(
                    subset["label_name"], subset[model_col]
                ),
            }
        )
pd.DataFrame(sanity_rows)


# ## Attach drug/treatment metadata from the plate maps
#
# Each patient's plate uses one of two plate layouts (`config/platemaps/barcode_platemap.csv` maps patient -> plate layout), and each layout file maps a well to its `Treatment`/`Dose`/`Unit`. `well_fov` values such as `F11-2` encode `well` (`F11`) + field-of-view index (`2`), so the well is recovered by splitting on `"-"`.

# In[10]:


platemap_frames = []
for platemap_number, path in platemap_paths.items():
    pm = pd.read_csv(path)
    pm["platemap_number"] = platemap_number
    platemap_frames.append(
        pm[["platemap_number", "WellPosition", "Treatment", "Dose", "Unit"]]
    )
platemap_df = pd.concat(platemap_frames, ignore_index=True).rename(
    columns={"WellPosition": "well"}
)

barcode_df = pd.read_csv(barcode_platemap_path)

all_preds_df["well"] = all_preds_df["well_fov"].str.split("-").str[0]
all_preds_df = all_preds_df.merge(
    barcode_df, left_on="patient", right_on="patient_tumor_barcode", how="left"
)
all_preds_df = all_preds_df.merge(
    platemap_df, on=["platemap_number", "well"], how="left"
)
all_preds_df["Treatment"] = all_preds_df["Treatment"].fillna("Unknown/Control")

print(all_preds_df["Treatment"].value_counts())
all_preds_df.head()


# ## Export a per-group performance summary for `2b.plot_morphology_class_heatmap.ipynb`
#
# This notebook's job stops at the data: Python/scikit-learn reproduces the splits, loads the trained models, and predicts. The actual visualization — a single heatmap labeled by split / patient / treatment (+ dose), rendered with R's **ComplexHeatmap** — lives entirely in the companion notebook **`2b.plot_morphology_class_heatmap.ipynb`** (R kernel), which reads the CSV written below. One row per (data split, patient, drug/treatment+dose) combination that is actually present in the data, with `accuracy` (predicted == actual class) and `n` (sample count) per group, for both models (long format, one `model` column) so the R notebook can switch models without re-running this one.

# In[11]:


MODEL_COLS = {
    "rf_prediction": "Random Forest",
    "log_reg_prediction": "Logistic Regression",
}

SPLIT_ORDER = ["train", "val", "test", "holdout"]


def build_group_summary(
    df: pd.DataFrame, group_cols: list, model_col: str, model_name: str
) -> pd.DataFrame:
    """One row per group combination: accuracy + sample count (n)."""
    records = []
    for key, g in df.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_cols, key))
        record["model"] = model_name
        record["accuracy"] = float((g["label_name"] == g[model_col]).mean())
        record["n"] = len(g)
        records.append(record)
    return pd.DataFrame(records)


def format_dose(dose, unit) -> str:
    if pd.isna(dose) or pd.isna(unit):
        return ""
    dose_str = str(dose)
    if dose_str.endswith(".0"):
        dose_str = dose_str[:-2]
    return f" ({dose_str}{unit})"


# In[12]:


summary = pd.concat(
    [
        build_group_summary(
            all_preds_df,
            ["split", "patient", "Treatment", "Dose", "Unit"],
            model_col,
            model_name,
        )
        for model_col, model_name in MODEL_COLS.items()
    ],
    ignore_index=True,
)
summary["split"] = pd.Categorical(
    summary["split"], categories=SPLIT_ORDER, ordered=True
)
# the treatment label carries the dose, so e.g. Trametinib 1uM and 10uM are
# distinct, separately-colored entries in the Treatment legend/color bar
summary["treatment_label"] = summary["Treatment"] + summary.apply(
    lambda row: format_dose(row["Dose"], row["Unit"]), axis=1
)
summary = summary.sort_values(
    ["model", "split", "patient", "treatment_label"]
).reset_index(drop=True)

summary_csv_path = pathlib.Path("../results/group_performance_summary.csv").resolve()
summary.to_csv(summary_csv_path, index=False)

print(f"{len(summary)} (model, split, patient, treatment+dose) rows")
print(f"wrote {summary_csv_path}")
summary.head()


# ## Export a predicted-vs-actual confusion summary, irrespective of patient/treatment
#
# The per-group summary above is deliberately split out by patient/treatment for the first heatmap. This second summary answers a different question -- "within each split, how does predicted class map onto actual class overall?" -- so it collapses across patient and treatment entirely, keeping only `(split, model, actual, predicted)` as the group keys. Consumed by `2b.plot_morphology_class_heatmap.ipynb`'s second heatmap (`2b.render_confusion_heatmap.R`).

# In[13]:


confusion_records = []
for model_col, model_name in MODEL_COLS.items():
    counts = (
        all_preds_df.groupby(["split", "label_name", model_col])
        .size()
        .reset_index(name="n")
        .rename(columns={"label_name": "actual", model_col: "predicted"})
    )
    counts["model"] = model_name
    confusion_records.append(counts)

confusion_summary = pd.concat(confusion_records, ignore_index=True)
confusion_summary["split"] = pd.Categorical(
    confusion_summary["split"], categories=SPLIT_ORDER, ordered=True
)
# proportion within each (model, split, actual) group -- i.e. recall of that
# actual class -- so the diagonal cells are on the same [0, 1] scale as the
# "accuracy" cells in the first heatmap and can share its color palette
confusion_summary["proportion"] = confusion_summary["n"] / confusion_summary.groupby(
    ["model", "split", "actual"]
)["n"].transform("sum")
confusion_summary = confusion_summary.sort_values(
    ["model", "split", "actual", "predicted"]
).reset_index(drop=True)

confusion_csv_path = pathlib.Path("../results/confusion_summary.csv").resolve()
confusion_summary.to_csv(confusion_csv_path, index=False)

print(f"{len(confusion_summary)} (model, split, actual, predicted) rows")
print(f"wrote {confusion_csv_path}")
confusion_summary.head()
