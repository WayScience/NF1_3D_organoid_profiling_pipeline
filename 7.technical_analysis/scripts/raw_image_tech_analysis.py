#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import numpy as np
import pandas as pd
import tifffile
import tqdm
from notebook_init_utils import (
    avoid_path_crash_bandicoot,
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[ ]:


def file_corruption_check(image: np.ndarray) -> bool:
    if np.any(np.isnan(image)) or np.any(np.isinf(image)):
        return True
    elif np.max(image) == 0:
        return True
    elif len(image.shape) < 3:
        return True
    else:
        return False


def binarize_instance_masks(instance_mask: np.ndarray) -> np.ndarray:
    instance_mask[instance_mask > 0] = 1
    return instance_mask


def retreive_foreground_background_masks(
    binary_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    foreground_mask = binary_mask > 0
    background_mask = binary_mask == 0
    return foreground_mask, background_mask


def calculate_signal_to_noise_ratio(
    image: np.ndarray, foreground_mask: np.ndarray, background_mask: np.ndarray
) -> float:
    signal = image[foreground_mask]
    noise = image[background_mask]

    mean_signal = np.mean(signal)
    std_noise = np.std(noise)

    if std_noise == 0:
        return float("inf")

    snr = mean_signal / std_noise
    return snr


def michelson_contrast(
    image: np.ndarray, foreground_mask: np.ndarray, background_mask: np.ndarray
) -> float:
    signal = image[foreground_mask]
    background = image[background_mask]

    if signal.size == 0 or background.size == 0:
        return np.nan

    I_max = np.max(signal)
    I_min = np.min(background)

    I_max = float(I_max)
    I_min = float(I_min)
    if (I_max + I_min) == 0:
        return np.nan

    m_contrast = (I_max - I_min) / (I_max + I_min)
    return m_contrast


def calculate_RMS_contrast(
    image: np.ndarray, foreground_mask: np.ndarray, background_mask: np.ndarray
) -> float:
    signal = image[foreground_mask]
    background = image[background_mask]

    mean_signal = np.mean(signal)
    mean_background = np.mean(background)

    rms_contrast = np.sqrt(np.mean((signal - mean_signal) ** 2)) / mean_background
    return rms_contrast


# In[3]:


platemap_file_dir = pathlib.Path(
    f"{image_base_dir}/data/NF0037_T1/platemap/platemap.csv"
).resolve(strict=True)
plate_map_df = pd.read_csv(platemap_file_dir)
results_dir = pathlib.Path(
    "../results/raw_image_quality_metrics/individual_files/"
).resolve()
results_dir.mkdir(parents=True, exist_ok=True)


# In[4]:


# get zstack image_paths
patients = ["NF0037_T1", "NF0037_T1_CQ1"]
file_paths = []
for patient in patients:
    image_dir = pathlib.Path(
        f"{image_base_dir}/data/{patient}/zstack_images/"
    ).resolve()
    image_paths = sorted(list(image_dir.rglob("*")))
    image_paths = [p for p in image_paths if p.is_file()]
    file_paths.extend(image_paths)
print(f"Found {len(file_paths)} zstack image files for patients: {patients}")


# In[5]:


df = pd.DataFrame({"image_path": file_paths})
df["patient"] = df["image_path"].apply(lambda x: x.parent.parent.parent.name)
df["well_fov"] = df["image_path"].apply(lambda x: x.parent.name)
df["channel"] = df["image_path"].apply(lambda x: x.stem.split("_")[-1])

image_path = df.pop("image_path")
df.insert(3, "image_path", image_path)

# filter out rows that contain channel = TRANS
df = df[df["channel"] != "TRANS"].reset_index(drop=True)

# Ensure we pivot patient x well_fov -> one column per channel (values are the image_path)
df = df[["patient", "well_fov", "channel", "image_path"]].copy()
# convert paths to strings (optional)
df["image_path"] = df["image_path"].astype(str)

df = df.pivot_table(
    index=["patient", "well_fov"],
    columns="channel",
    values="image_path",
    aggfunc="first",  # if multiple entries per channel, keep first
).reset_index()

df.columns.name = None
df["nuclei_mask_path"] = df.apply(
    lambda row: pathlib.Path(
        f"{image_base_dir}/data/{row['patient']}/segmentation_masks/{row['well_fov']}/nuclei_mask.tiff"
    ),
    axis=1,
)
df["cell_mask_path"] = df.apply(
    lambda row: pathlib.Path(
        f"{image_base_dir}/data/{row['patient']}/segmentation_masks/{row['well_fov']}/cell_mask.tiff"
    ),
    axis=1,
)
df["organoid_mask_path"] = df.apply(
    lambda row: pathlib.Path(
        f"{image_base_dir}/data/{row['patient']}/segmentation_masks/{row['well_fov']}/organoid_mask.tiff"
    ),
    axis=1,
)
df.head()


# In[6]:


channels = ["405", "488", "555", "640"]
compartments = ["nuclei", "cell", "organoid"]

for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    row_dict = row.to_dict()
    for compartment in compartments:
        for channel in channels:
            # Reconstruct row as a Series-like object
            row = pd.Series(row_dict)

            results_file_dir = pathlib.Path(
                f"{results_dir}/{row['patient']}_{row['well_fov']}_{channel}_{compartment}_image_quality_metrics.parquet"
            ).resolve()

            # skip processing if file already exists - a sort of caching
            if results_file_dir.exists():
                continue

            # loads one mask - binarizes - retrieves foreground and background masks
            if compartment == "nuclei":
                mask = binarize_instance_masks(tifffile.imread(row["nuclei_mask_path"]))
            elif compartment == "cell":
                mask = binarize_instance_masks(tifffile.imread(row["cell_mask_path"]))
            elif compartment == "organoid":
                mask = binarize_instance_masks(
                    tifffile.imread(row["organoid_mask_path"])
                )
            else:
                raise ValueError(f"Unknown compartment: {compartment}")

            foreground_mask, background_mask = retreive_foreground_background_masks(
                binary_mask=mask
            )

            # Load image
            image_path = row[channel]
            image = tifffile.imread(image_path)

            # Calculate metrics
            snr = calculate_signal_to_noise_ratio(
                image, foreground_mask, background_mask
            )
            m_contrast = michelson_contrast(image, foreground_mask, background_mask)
            rms_contrast = calculate_RMS_contrast(
                image, foreground_mask, background_mask
            )

            # Store results
            results_dict = {
                "patient": [row["patient"]],
                "well_fov": [row["well_fov"]],
                "channel": [channel],
                "compartment": [compartment],
                "signal_to_noise_ratio": [snr],
                "michelson_contrast": [m_contrast],
                "RMS_contrast": [rms_contrast],
            }

            result_df = pd.DataFrame(results_dict)
            result_df.to_parquet(results_file_dir, index=False)


# In[7]:


# get a list of all files in the results directory
all_result_files = list(results_dir.rglob("*.parquet"))
df = pd.concat([pd.read_parquet(f) for f in all_result_files], ignore_index=True)
df.head()


# In[8]:


# merge the plate map info into the results
df["well"] = df["well_fov"].str.split("-").str[0]
df_results = df.merge(
    plate_map_df, how="left", left_on="well", right_on="well_position"
)
df_results.sort_values(
    by=["patient", "well_fov", "channel", "compartment"], inplace=True
)
concat_dir = pathlib.Path("../results/raw_image_quality_metrics/").resolve()
df_results.to_parquet(concat_dir / "merged_results.parquet", index=False)
df_results.head()
