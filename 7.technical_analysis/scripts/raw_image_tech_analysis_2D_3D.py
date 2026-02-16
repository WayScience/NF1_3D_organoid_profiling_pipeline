#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib

import numpy as np
import pandas as pd
import tifffile
import tqdm
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


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


def calculate_all_image_metrics(
    image: np.ndarray, foreground_mask: np.ndarray, background_mask: np.ndarray
) -> dict[str, float]:
    # Calculate metrics
    snr = calculate_signal_to_noise_ratio(image, foreground_mask, background_mask)
    m_contrast = michelson_contrast(image, foreground_mask, background_mask)
    rms_contrast = calculate_RMS_contrast(image, foreground_mask, background_mask)
    # return key value pairs of metric name and value
    return {
        "snr": snr,
        "m_contrast": m_contrast,
        "rms_contrast": rms_contrast,
    }


# In[3]:


platemap_file_dir = pathlib.Path(
    f"{image_base_dir}/data/NF0037_T1/platemap/platemap.csv"
).resolve(strict=True)
plate_map_df = pd.read_csv(platemap_file_dir)
results_dir = pathlib.Path(
    "../results/raw_image_quality_metrics/individual_files/"
).resolve()
results_dir.mkdir(parents=True, exist_ok=True)


# In[ ]:


# list of all file paths to analyze
file_paths_output_file = pathlib.Path(
    "../results/list_of_zstack_image_paths.parquet"
).resolve()
file_paths_output_file.parent.mkdir(parents=True, exist_ok=True)

patients = ["NF0037_T1", "NF0037_T1_CQ1"]  # psuedo paired
subparent_dirs = ["zstack_images", "basicpy_zstack_images"]
file_paths = []
for patient in tqdm.tqdm(patients, desc="Collecting image paths", leave=True):
    for subparent_dir in subparent_dirs:
        image_dir = pathlib.Path(
            f"{image_base_dir}/data/{patient}/{subparent_dir}/"
        ).resolve()
        image_paths = sorted(list(image_dir.glob("*")))
        image_paths = [p for p in image_paths if p.is_dir()]
        # actually faster to parse multiple dirs than to
        # recursively glob for all tif files in one dir with many subdirs
        # anecdotally, interesting
        # >30mins for recursive glob vs <2 mins for parsing nested dirs
        for image_path in tqdm.tqdm(
            image_paths,
            desc=f"Collecting image paths for {patient} in {subparent_dir}",
            leave=False,
        ):
            zstack_image_paths = sorted(list(image_path.glob("*.tif")))
            zstack_image_paths = [str(p) for p in zstack_image_paths if p.is_file()]
            file_paths.extend(zstack_image_paths)
print(f"Found {len(file_paths)} zstack image files for patients: {patients}")
file_paths_df = pd.DataFrame({"image_path": file_paths})
file_paths_df.to_parquet(file_paths_output_file, index=False)


# In[ ]:


df = pd.DataFrame({"image_path": file_paths})
df["patient"] = df["image_path"].apply(
    lambda x: pathlib.Path(x).parent.parent.parent.name
)
df["well_fov"] = df["image_path"].apply(lambda x: pathlib.Path(x).parent.name)
df["channel"] = df["image_path"].apply(lambda x: pathlib.Path(x).stem.split("_")[-1])

image_path = df.pop("image_path")
df.insert(3, "image_path", image_path)

# filter out rows that contain channel = TRANS
df = df[df["channel"] != "TRANS"].reset_index(drop=True)

# Ensure we pivot patient x well_fov -> one column per channel (values are the image_path)
df = df[["patient", "well_fov", "channel", "image_path"]].copy()
# convert paths to strings (optional)
df["image_path"] = df["image_path"].astype(str)
df["basicpy_status"] = df["image_path"].apply(
    lambda x: "basicpy" if "basicpy" in str(x) else "raw_image"
)
df = df.pivot_table(
    index=["patient", "well_fov", "basicpy_status"],
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


# In[ ]:


channels = ["405", "488", "555", "640"]
compartments = ["nuclei", "cell", "organoid"]


# loop through the rows
# compartments
# channels to calculate metrics for each channel and compartment combination
for idx, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    row_dict = row.to_dict()
    for compartment in compartments:
        for channel in channels:
            # Reconstruct row as a Series-like object
            row = pd.Series(row_dict)

            # set file paths for results
            results_2D_file_dir = pathlib.Path(
                f"{results_dir}/{row['basicpy_status']}_{row['patient']}_{row['well_fov']}_{channel}_{compartment}_image_quality_metrics_2D_wise.parquet"
            ).resolve()

            results_3D_file_dir = pathlib.Path(
                f"{results_dir}/{row['basicpy_status']}_{row['patient']}_{row['well_fov']}_{channel}_{compartment}_image_quality_metrics_3D_wise.parquet"
            ).resolve()
            try:
                # skip processing if file already exists - a sort of caching
                if results_2D_file_dir.exists() and results_3D_file_dir.exists():
                    continue

                # loads one mask - binarizes - retrieves foreground and background masks
                if compartment == "nuclei":
                    mask = binarize_instance_masks(
                        tifffile.imread(row["nuclei_mask_path"])
                    )
                elif compartment == "cell":
                    mask = binarize_instance_masks(
                        tifffile.imread(row["cell_mask_path"])
                    )
                elif compartment == "organoid":
                    mask = binarize_instance_masks(
                        tifffile.imread(row["organoid_mask_path"])
                    )
                else:
                    raise ValueError(f"Unknown compartment: {compartment}")

                if channel == "640" and patient == "NF0037_T1":
                    # this is temporary until the new masks are generated with the correct channel
                    mask = mask[1:, :, :]

                # Load raw signal image
                image_path = row[channel]
                image = tifffile.imread(image_path)

                ####################################################
                # 3D-wise metrics
                ####################################################
                if results_3D_file_dir.exists():
                    continue
                else:
                    image_metrics_3D_dict = {
                        "patient": [],
                        "well_fov": [],
                        "channel": [],
                        "compartment": [],
                        "signal_to_noise_ratio": [],
                        "michelson_contrast": [],
                        "RMS_contrast": [],
                    }

                    # get the 3D foreground and background masks
                    foreground_mask, background_mask = (
                        retreive_foreground_background_masks(binary_mask=mask)
                    )
                    image_metrics_3D = calculate_all_image_metrics(
                        image=image,
                        foreground_mask=foreground_mask,
                        background_mask=background_mask,
                    )
                    image_metrics_3D_dict["patient"].append(row["patient"])
                    image_metrics_3D_dict["well_fov"].append(row["well_fov"])
                    image_metrics_3D_dict["channel"].append(channel)
                    image_metrics_3D_dict["compartment"].append(compartment)
                    image_metrics_3D_dict["signal_to_noise_ratio"].append(
                        image_metrics_3D["snr"]
                    )
                    image_metrics_3D_dict["michelson_contrast"].append(
                        image_metrics_3D["m_contrast"]
                    )
                    image_metrics_3D_dict["RMS_contrast"].append(
                        image_metrics_3D["rms_contrast"]
                    )
                    image_metrics_3D_df = pd.DataFrame(image_metrics_3D_dict)
                    image_metrics_3D_df.to_parquet(results_3D_file_dir, index=False)

                #####################################################
                # 2D-wise metrics
                #####################################################
                if results_2D_file_dir.exists():
                    continue
                else:
                    for z_slice in range(image.shape[0]):
                        # we had the wrong channel in the first slice of the mito channel
                        # this way we actually align the mask to the image if in mito channel
                        if channel == "640":
                            # this is temporary until the new masks are generated with the correct channel
                            mask_slice = mask[z_slice, :, :].copy()
                        else:
                            mask_slice = mask[z_slice, :, :].copy()
                        image_slice = image[z_slice, :, :].copy()
                        foreground_mask_slice, background_mask_slice = (
                            retreive_foreground_background_masks(binary_mask=mask_slice)
                        )

                        # Calculate metrics
                        image_metrics_2D = calculate_all_image_metrics(
                            image=image_slice,
                            foreground_mask=foreground_mask_slice,
                            background_mask=background_mask_slice,
                        )

                        # Store results
                        results_dict = {
                            "patient": [row["patient"]],
                            "well_fov": [row["well_fov"]],
                            "channel": [channel],
                            "z_slice": [z_slice],
                            "compartment": [compartment],
                            "signal_to_noise_ratio": [image_metrics_2D["snr"]],
                            "michelson_contrast": [image_metrics_2D["m_contrast"]],
                            "RMS_contrast": [image_metrics_2D["rms_contrast"]],
                        }

                    result_df = pd.DataFrame(results_dict)
                    result_df.to_parquet(results_2D_file_dir, index=False)
            except Exception as e:
                print(
                    f"Error processing {row['patient']} {row['well_fov']} {channel} {compartment}: {e}"
                )


# In[ ]:


# get a list of all files in the results directory
all_result_files = list(results_dir.rglob("*.parquet"))
# split the lists into 2D and 3D files based on the filename
result_2D_files = [f for f in all_result_files if "2D" in f.name]
result_3D_files = [f for f in all_result_files if "3D" in f.name]
df_2D = pd.concat([pd.read_parquet(f) for f in result_2D_files], ignore_index=True)
df_3D = pd.concat([pd.read_parquet(f) for f in result_3D_files], ignore_index=True)
df_2D.head()
# df_3D.head()


# In[ ]:


# merge the plate map info into the results
df_2D["well"] = df_2D["well_fov"].str.split("-").str[0]
df_2D_results = df_2D.merge(
    plate_map_df, how="left", left_on="well", right_on="well_position"
)
df_2D_results.sort_values(
    by=["patient", "well_fov", "channel", "compartment"], inplace=True
)
concat_dir = pathlib.Path("../results/raw_image_quality_metrics/").resolve()
df_2D_results.to_parquet(concat_dir / "merged_results_2D.parquet", index=False)

df_3D["well"] = df_3D["well_fov"].str.split("-").str[0]
df_3D_results = df_3D.merge(
    plate_map_df, how="left", left_on="well", right_on="well_position"
)
df_3D_results.sort_values(
    by=["patient", "well_fov", "channel", "compartment"], inplace=True
)
df_3D_results.to_parquet(concat_dir / "merged_results_3D.parquet", index=False)
