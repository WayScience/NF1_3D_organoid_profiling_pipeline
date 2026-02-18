#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import multiprocessing as mp
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


# In[ ]:


if not in_notebook:
    argparse = argparse.ArgumentParser(
        description="Process 2D and 3D image metrics in parallel"
    )
    argparse.add_argument(
        "--n_processes",
        type=int,
        default=18,
        help="Number of processes to use",
    )
    n_processes = argparse.parse_args().n_processes
else:
    # Process tasks in parallel
    n_processes = 18

# catch if the number of process requested exceeds the number of CPUs available and adjust accordingly
if mp.cpu_count() < n_processes:
    n_processes = mp.cpu_count()

print(f"Using {n_processes} processes")


# In[ ]:


def file_corruption_check(image: np.ndarray) -> bool:
    """
    Check if the image is corrupted by looking for NaN, Inf values, zero max value, or less than 3 dimensions.

    Parameters
    ----------
    image : np.ndarray
        The image to check for corruption.
    Returns
    -------
    bool
        True if the image is corrupted, False otherwise.
    """
    if np.any(np.isnan(image)) or np.any(np.isinf(image)):
        return True
    elif np.max(image) == 0:
        return True
    elif len(image.shape) < 3:
        return True
    else:
        return False


def binarize_instance_masks(instance_mask: np.ndarray) -> np.ndarray:
    """
    Convert instance masks to binary masks by setting all non-zero values to 1.
    Parameters
    ----------
    instance_mask : np.ndarray
        The instance mask to convert.
    Returns
    -------
    np.ndarray
        The binary mask.
    """
    instance_mask[instance_mask > 0] = 1
    return instance_mask


def retreive_foreground_background_masks(
    binary_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Given a binary mask, return the foreground and background masks.
    Parameters
    ----------
    binary_mask : np.ndarray
        The binary mask to use for creating foreground and background masks.
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The foreground and background masks.
    """
    foreground_mask = binary_mask > 0
    background_mask = binary_mask == 0
    return foreground_mask, background_mask


def calculate_signal_to_noise_ratio(
    image: np.ndarray, foreground_mask: np.ndarray, background_mask: np.ndarray
) -> float:
    """
    Calculate the signal-to-noise ratio (SNR) for an image.

    Parameters
    ----------
    image : np.ndarray
        The image to calculate SNR for.
    foreground_mask : np.ndarray
        The foreground mask.
    background_mask : np.ndarray
        The background mask.

    Returns
    -------
    float
        The SNR value.
    """
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
    """
    Calculate the Michelson contrast for an image.
    Parameters
    ----------
    image : np.ndarray
        The image to calculate Michelson contrast for.
    foreground_mask : np.ndarray
        The foreground mask.
    background_mask : np.ndarray
        The background mask.

    Returns
    -------
    float
        The Michelson contrast value.
    """
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
    """
    Calculate the RMS contrast for an image.
    Parameters
    ----------
    image : np.ndarray
        The image to calculate RMS contrast for.
    foreground_mask : np.ndarray
        The foreground mask.
    background_mask : np.ndarray
        The background mask.
    Returns
    -------
    float
        The RMS contrast value.
    """
    signal = image[foreground_mask]
    background = image[background_mask]

    mean_signal = np.mean(signal)
    mean_background = np.mean(background)

    rms_contrast = np.sqrt(np.mean((signal - mean_signal) ** 2)) / mean_background
    return rms_contrast


def calculate_all_image_metrics(
    image: np.ndarray, foreground_mask: np.ndarray, background_mask: np.ndarray
) -> dict[str, float]:
    """
    Call all metric calculation functions and return a dictionary of metric names and values.
    Parameters
    ----------
    image : np.ndarray
        The image to calculate metrics for.
    foreground_mask : np.ndarray
        The foreground mask.
    background_mask : np.ndarray
        The background mask.
    Returns
    -------
    dict[str, float]
        A dictionary of metric names and values.
    """
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


# In[ ]:


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


def calculate_slice_by_slice_metrics(
    image: np.ndarray,
    mask: np.ndarray,
    results_2D_file_dir: pathlib.Path,
    kwargs,
) -> bool:
    """
    Calculate image quality metrics for each slice of a 3D image and save results to a parquet file.
    Parameters
    ----------
    image : np.ndarray
        The 3D image to calculate metrics for.
    mask : np.ndarray
        The 3D binary mask to use for calculating metrics.
    results_2D_file_dir : pathlib.Path
        The directory to save the 2D metrics parquet file to.
    kwargs : dict
        A dictionary containing the patient, well_fov, channel, and compartment information for the image.
            patient : str
                The patient identifier for the image.
            well_fov : str
                The well and field of view identifier for the image.
            channel : str
                The channel identifier for the image.
            compartment : str
                The compartment identifier for the image (e.g. nuclei, cell, organoid).
    Returns
    -------
    bool
        True if the metrics were calculated and saved successfully, False otherwise.
    """
    patient, well_fov, channel, compartment = (
        kwargs["patient"],
        kwargs["well_fov"],
        kwargs["channel"],
        kwargs["compartment"],
    )
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
            "patient": [patient],
            "well_fov": [well_fov],
            "channel": [channel],
            "z_slice": [z_slice],
            "compartment": [compartment],
            "signal_to_noise_ratio": [image_metrics_2D["snr"]],
            "michelson_contrast": [image_metrics_2D["m_contrast"]],
            "RMS_contrast": [image_metrics_2D["rms_contrast"]],
        }

    result_df = pd.DataFrame(results_dict)
    result_df.to_parquet(results_2D_file_dir, index=False)
    return True


def calculate_whole_volume_metrics(
    image: np.ndarray,
    mask: np.ndarray,
    results_3D_file_dir: pathlib.Path,
    kwargs,
) -> bool:
    """
    Calculate image quality metrics for a 3D image and save results to a parquet file.
    Parameters
    ----------
    image : np.ndarray
        The 3D image to calculate metrics for.
    mask : np.ndarray
        The 3D binary mask to use for calculating metrics.
    results_3D_file_dir : pathlib.Path
        The directory to save the 3D metrics parquet file to.
    kwargs : dict
        A dictionary containing the patient, well_fov, channel, and compartment information for the image.
            patient : str
                The patient identifier for the image.
            well_fov : str
                The well and field of view identifier for the image.
            channel : str
                The channel identifier for the image.
            compartment : str
                The compartment identifier for the image (e.g. nuclei, cell, organoid).
    Returns
    -------
    bool
        True if the metrics were calculated and saved successfully, False otherwise.
    """
    patient, well_fov, channel, compartment = (
        kwargs["patient"],
        kwargs["well_fov"],
        kwargs["channel"],
        kwargs["compartment"],
    )

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
    foreground_mask, background_mask = retreive_foreground_background_masks(
        binary_mask=mask
    )
    image_metrics_3D = calculate_all_image_metrics(
        image=image,
        foreground_mask=foreground_mask,
        background_mask=background_mask,
    )
    image_metrics_3D_dict["patient"].append(patient)
    image_metrics_3D_dict["well_fov"].append(well_fov)
    image_metrics_3D_dict["channel"].append(channel)
    image_metrics_3D_dict["compartment"].append(compartment)
    image_metrics_3D_dict["signal_to_noise_ratio"].append(image_metrics_3D["snr"])
    image_metrics_3D_dict["michelson_contrast"].append(image_metrics_3D["m_contrast"])
    image_metrics_3D_dict["RMS_contrast"].append(image_metrics_3D["rms_contrast"])
    image_metrics_3D_df = pd.DataFrame(image_metrics_3D_dict)
    image_metrics_3D_df.to_parquet(results_3D_file_dir, index=False)
    return True


# In[ ]:


def process_single_task(task_params):
    """
    Process a single task: one row, compartment, and channel combination.

    Parameters
    ----------
    task_params : tuple
        A tuple containing (row_dict, compartment, channel, results_dir)

    Returns
    -------
    tuple
        A tuple containing (success: bool, message: str) indicating whether the task was successful and any relevant messages (e.g. errors or success info).
    """
    row_dict, compartment, channel, results_dir = task_params

    try:
        row = pd.Series(row_dict)
        patient = row["patient"]
        well_fov = row["well_fov"]

        # set file paths for results
        results_2D_file_dir = pathlib.Path(
            f"{results_dir}/{row['basicpy_status']}_{patient}_{well_fov}_{channel}_{compartment}_image_quality_metrics_2D_wise.parquet"
        ).resolve()

        results_3D_file_dir = pathlib.Path(
            f"{results_dir}/{row['basicpy_status']}_{patient}_{well_fov}_{channel}_{compartment}_image_quality_metrics_3D_wise.parquet"
        ).resolve()

        # skip processing if file already exists - a sort of caching
        if results_2D_file_dir.exists() and results_3D_file_dir.exists():
            return (
                True,
                f"Already processed {patient} {well_fov} {channel} {compartment}",
            )

        # loads one mask - binarizes - retrieves foreground and background masks
        if compartment == "nuclei":
            mask = binarize_instance_masks(tifffile.imread(row["nuclei_mask_path"]))
        elif compartment == "cell":
            mask = binarize_instance_masks(tifffile.imread(row["cell_mask_path"]))
        elif compartment == "organoid":
            mask = binarize_instance_masks(tifffile.imread(row["organoid_mask_path"]))
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
        if not results_3D_file_dir.exists():
            calculate_whole_volume_metrics(
                image=image,
                mask=mask,
                results_3D_file_dir=results_3D_file_dir,
                kwargs={
                    "patient": patient,
                    "well_fov": row["well_fov"],
                    "channel": channel,
                    "compartment": compartment,
                },
            )

        #####################################################
        # 2D-wise metrics
        #####################################################
        if not results_2D_file_dir.exists():
            calculate_slice_by_slice_metrics(
                image=image,
                mask=mask,
                results_2D_file_dir=results_2D_file_dir,
                kwargs={
                    "patient": patient,
                    "well_fov": row["well_fov"],
                    "channel": channel,
                    "compartment": compartment,
                },
            )

        return (
            True,
            f"Successfully processed {patient} {well_fov} {channel} {compartment}",
        )

    except Exception as e:
        return (
            False,
            f"Error processing {row_dict.get('patient', 'unknown')} {row_dict.get('well_fov', 'unknown')} {channel} {compartment}: {str(e)}",
        )


# Prepare all tasks
channels = ["405", "488", "555", "640"]
compartments = ["nuclei", "cell", "organoid"]

tasks = []
for idx, row in df.iterrows():
    row_dict = row.to_dict()
    for compartment in compartments:
        for channel in channels:
            tasks.append((row_dict, compartment, channel, results_dir))

print(f"Total tasks to process: {len(tasks)}")


with mp.Pool(processes=n_processes) as pool:
    results = list(
        tqdm.tqdm(
            pool.imap(process_single_task, tasks),
            total=len(tasks),
            desc="Processing images",
        )
    )

# Print summary
successful = sum(1 for success, _ in results if success)
failed = sum(1 for success, _ in results if not success)
print(
    f"\nProcessing complete: {successful}/{len(results)} tasks successful, {failed} failed"
)

# Print any error messages
errors = [msg for success, msg in results if not success]
if errors:
    print("\nErrors encountered:")
    for error in errors[:10]:  # Print first 10 errors
        print(f"  - {error}")
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more errors")


# In[ ]:


# get a list of all files in the results directory
all_result_files = list(results_dir.rglob("*.parquet"))
# split the lists into 2D and 3D files based on the filename
result_2D_files = [f for f in all_result_files if "2D" in f.name]
result_3D_files = [f for f in all_result_files if "3D" in f.name]
df_2D = pd.concat([pd.read_parquet(f) for f in result_2D_files], ignore_index=True)
df_3D = pd.concat([pd.read_parquet(f) for f in result_3D_files], ignore_index=True)
print(f"Combined 2D metrics dataframe shape: {df_2D.shape}")
print(f"Combined 3D metrics dataframe shape: {df_3D.shape}")
df_2D.head()


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
