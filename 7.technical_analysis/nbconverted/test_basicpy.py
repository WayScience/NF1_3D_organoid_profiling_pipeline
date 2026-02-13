#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import gc
import logging
import os
import sys
from pathlib import Path

os.environ["BASIC_DCT_BACKEND"] = "SCIPY"  # Force BaSiC to CPU, avoids GPU warning
import numpy as np
import tifffile
from basicpy import BaSiC
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

logging.getLogger("jax").setLevel(logging.ERROR)  # suppress JAX logs


# In[2]:


def read_image_stack(image_input: Path | np.ndarray) -> np.ndarray:
    """Load a TIFF file or return a float32 array if already provided.

    Args:
        image_input (Path | np.ndarray): Path to a TIFF file or an already-loaded NumPy array.

    Returns:
        np.ndarray: The image stack as a float32 NumPy array.
    """
    if isinstance(image_input, Path):
        img = tifffile.imread(image_input)
    else:
        img = image_input
    return img.astype("float32")


def apply_basicpy_correction(
    image_input: Path | np.ndarray, out_dir: Path | None = None, stem: str | None = None
) -> np.ndarray:
    """Apply BaSiC illumination correction to an image stack or TIFF file.

    Args:
        image_input (Path | np.ndarray): Path to the TIFF file or a preloaded NumPy array.
        out_dir (Path | None, optional): Directory to save the corrected TIFF file. Defaults to None.
        stem (str | None, optional): Optional filename stem for saving if `image_input` is a NumPy array.

    Returns:
        np.ndarray: The BaSiC-corrected image stack as a float32 NumPy array.
    """
    img = read_image_stack(image_input)

    basic = BaSiC(get_darkfield=True)
    basic.fit(img)
    corrected = basic.transform(img, timelapse=True)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(image_input, Path):
            stem = image_input.stem
        elif stem is None:
            stem = "corrected_image"
        out_path = out_dir / f"{stem}.tif"
        tifffile.imwrite(
            out_path,
            np.clip(np.rint(corrected), 0, 65535).astype(np.uint16),
            imagej=True,
        )

    del img, basic
    gc.collect()
    return corrected


def process_channel_stack(folder: Path, out_well_dir: Path, channel: str) -> None:
    """Process all TIFF files for a given channel in a well/FOV folder.

    Args:
        folder (Path): Path to the well/FOV folder containing TIFF files.
        out_well_dir (Path): Directory to save the corrected TIFFs for this channel.
        channel (str): Channel identifier to process (e.g., "405", "488", "555", "640").
    """
    tifs = sorted(folder.glob(f"*_{channel}*.tif"))
    if not tifs:
        print(f"⚠️ No TIFFs for channel {channel} in {folder.name}")
        return

    for tif in tqdm(tifs, desc=f"{folder.name} | {channel}", leave=False):
        apply_basicpy_correction(tif, out_well_dir)


def process_well_fov(
    folder: Path, out_dir: Path, channels: list[str], num_workers: int = 4
) -> None:
    """Run BaSiC correction for all specified channels in a well/FOV folder using multithreading.

    Args:
        folder (Path): Path to the well/FOV folder containing TIFF files.
        out_dir (Path): Directory where corrected TIFFs will be saved.
        channels (list[str]): List of channel identifiers to process (e.g., ["405", "488", "555", "640"]).
        num_workers (int, optional): Number of worker threads for parallel channel processing. Defaults to 4.
    """
    print(f"Processing {folder.name}...")
    out_well_dir = out_dir / folder.name
    out_well_dir.mkdir(parents=True, exist_ok=True)

    thread_map(
        lambda ch: process_channel_stack(folder, out_well_dir, ch),
        channels,
        max_workers=num_workers,
        desc=f"Channels in {folder.name}",
    )


# In[3]:


# Determine patient ID
if hasattr(sys, "ps1") or not hasattr(sys.modules["__main__"], "__file__"):
    print("Running in an interactive environment. Using hardcoded patient ID.")
    patient = "NF0037_T1"
else:
    parser = argparse.ArgumentParser(description="Process NF1 patient zstack images")
    parser.add_argument(
        "patient", type=str, help="Patient ID to process (e.g., NF0037_T1)"
    )
    patient = parser.parse_args().patient

# Set directory (will need to be updated for other users)
home_dir = Path("/home/jenna/mnt/bandicoot/NF1_organoid_data").resolve(strict=True)

# Resolve patient directory
patient_dir = Path(f"{home_dir}/data/{patient}/zstack_images").resolve()

# Save corrected images to a sibling folder called basicpy_zstack_images
basicpy_output_dir = patient_dir.parent / "basicpy_zstack_images"
basicpy_output_dir.mkdir(parents=True, exist_ok=True)

# List all well/FOV subfolders
well_fov_folders = [p for p in patient_dir.iterdir() if p.is_dir()]
if not well_fov_folders:
    print(f"No well/FOV folders found for patient {patient}")
    sys.exit()


# ## Run BaSiCPy on each channel per well/fov for the patient
#
# This is run in a script, not in the notebook, for better stability, so this cell is not executed.

# In[ ]:


# Main sequential processing for wells/FOVs (use default 4 workers for channels)
channels_to_process = ["405", "488", "555", "640"]

with tqdm(total=len(well_fov_folders), desc="Processing Wells/FOVs") as pbar:
    for folder in well_fov_folders:
        process_well_fov(
            folder=folder,
            out_dir=basicpy_output_dir,
            channels=channels_to_process,
            num_workers=4,  # default number of workers per well
        )
        pbar.update(1)
