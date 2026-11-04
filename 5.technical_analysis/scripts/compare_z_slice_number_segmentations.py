#!/usr/bin/env python
# coding: utf-8

# This notebook compares the z-slice step size for segmentations of the same organoids.
# It samples the higher-resolution (finer z-step) segmentations down so that every z-step
# condition has the same number of z-slices as the coarsest condition tested (1 um), then
# computes the IOU (intersection over union) and volume ratio for each pairwise comparison.
#
# **Naming convention:** masks are loaded from directories named `{patient}_T1-Z-{step}`, where
# `T1` is timepoint 1 (every condition here is a single timepoint) and `{step}` is the z-step size
# in micrometers (um) between consecutive slices in that acquisition/simulation. Because `T1` is
# constant across every comparison in this notebook, the plots below label each condition simply
# by its z-step size (e.g. `1um`, `0.5um`) instead of the full `T1Z{step}` directory tag, and a
# `z_comparison` value like `1um vs 0.5um` means "the 1 um-step mask compared against the
# 0.5 um-step mask, after downsampling both to the same number of slices."
#
# **Outputs:** this notebook is slow to run (it reads every mask for every patient/well-fov/
# compartment). The computed IOU and volume tables are cached to `../processed_data/` as
# parquet files, and every figure is saved to `../figures/zslice_segmentation_comparison/`.
# Set `RECOMPUTE = True` in the next code cell to force the masks to be re-read and the cache
# to be rebuilt; otherwise the notebook (and any other script) can load the cached parquet files
# directly without re-running the mask comparison loop.
#

# In[1]:


import os
import pathlib
from itertools import combinations

# Import dependencies
import pandas as pd
from image_analysis_3D.file_utils.file_reading import read_zstack_image
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)
from image_analysis_3D.file_utils.technical_analysis_segmentation_utils import (
    convert_indexed_mask_to_binary_mask,
    extract_IOU,
)

root_dir, in_notebook = init_notebook()

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)

# set up output locations for cached intermediate results and figures, and whether
# to force re-reading every mask (slow) or load the cached parquet results if present
RECOMPUTE = False

processed_data_dir = root_dir / "5.technical_analysis" / "processed_data"
processed_data_dir.mkdir(parents=True, exist_ok=True)
figures_dir = (
    root_dir / "5.technical_analysis" / "figures" / "zslice_segmentation_comparison"
)
figures_dir.mkdir(parents=True, exist_ok=True)

iou_results_path = processed_data_dir / "zslice_iou_results.parquet"
per_compartment_volume_path = (
    processed_data_dir / "zslice_per_compartment_volume.parquet"
)
cache_exists = iou_results_path.exists() and per_compartment_volume_path.exists()


# In[2]:


# z-slice steps to compare, and the stride needed to down-sample each one
# to match the finest (T1Z1) resolution
z_steps = ["1", "0.5", "0.2", "0.1"]
z_step_strides = {
    "1": 1,
    "0.5": 2,
    "0.2": 5,
    "0.1": 10,
}

# map each patient to the well_fovs that were imaged for it, so that results
# can be aggregated and compared across patients
patient_well_fovs = {
    "NF0037": ["F4-2", "F4-3"],
    "NF0055": ["E4-1-0", "E4-5-0", "G5-2-0", "E4-3-0", "G5-1-0", "G5-5-0"],
}


# In[3]:


# set up output dictionaries for IOU and volume results, keyed by patient so
# that results can be compared across patients
iou_results = {
    "patient": [],
    "well_fov": [],
    "z_comparison": [],
    "IOU": [],
    "compartment": [],
    "volume_ratio": [],
}

per_compartment_volume = {
    "patient": [],
    "well_fov": [],
    "z_slice_step": [],
    "compartment": [],
    "volume": [],
}


# In[4]:


# set eps to avoid division by zero
if RECOMPUTE or not cache_exists:
    for patient, well_fovs in tqdm.tqdm(
        patient_well_fovs.items(), desc="Patients", total=len(patient_well_fovs)
    ):
        for well_fov in well_fovs:
            for compartment in ["nuclei", "cell", "cytoplasm", "organoid"]:
                # load the mask for each z-slice step and sample it down to the
                # same number of z-slices as the finest (T1Z1) step
                mask_dict = {}
                for z_step in z_steps:
                    if z_step == "0.1":
                        segmentation_sub_dir_name = "segmentation_masks_from_0_1um"
                    else:
                        segmentation_sub_dir_name = "segmentation_masks"
                    segmentation_dir = (
                        image_base_dir
                        / "data"
                        / f"{patient}_T1-Z-{z_step}"
                        / segmentation_sub_dir_name
                        / well_fov
                    )
                    mask_file = next(
                        file
                        for file in segmentation_dir.glob("*.tif*")
                        if compartment in str(file)
                    )
                    mask = read_zstack_image(mask_file)
                    stride = z_step_strides[z_step]
                    mask_dict[f"{z_step}um"] = mask[::stride, :, :].copy()

                z_slice_counts = {
                    name: mask.shape[0] for name, mask in mask_dict.items()
                }
                if len(set(z_slice_counts.values())) != 1:
                    raise ValueError(
                        "Sampled masks do not have the same number of z-slices in "
                        f"{patient} {well_fov} for {compartment}: {z_slice_counts}"
                    )

                # pairwise IOU and volume-ratio comparison across z-slice steps
                for (name1, mask1), (name2, mask2) in combinations(
                    mask_dict.items(), 2
                ):
                    iou = extract_IOU(mask1, mask2)
                    volume_ratio = mask1.sum() / mask2.sum() if mask2.sum() != 0 else 0
                    iou_results["patient"].append(patient)
                    iou_results["well_fov"].append(well_fov)
                    iou_results["z_comparison"].append(f"{name1} vs {name2}")
                    iou_results["IOU"].append(iou)
                    iou_results["compartment"].append(compartment)
                    iou_results["volume_ratio"].append(volume_ratio)

                # record binarized volume per z-slice step for this compartment
                for name, mask in mask_dict.items():
                    binary_mask = convert_indexed_mask_to_binary_mask(mask)
                    per_compartment_volume["patient"].append(patient)
                    per_compartment_volume["well_fov"].append(well_fov)
                    per_compartment_volume["z_slice_step"].append(name)
                    per_compartment_volume["compartment"].append(compartment)
                    per_compartment_volume["volume"].append(binary_mask.sum())
else:
    iou_results = None
    per_compartment_volume = None


# In[5]:


if RECOMPUTE or not cache_exists:
    iou_results_df = pd.DataFrame(iou_results)
    per_compartment_volume_df = pd.DataFrame(per_compartment_volume)
    iou_results_df.to_parquet(iou_results_path, index=False)
    per_compartment_volume_df.to_parquet(per_compartment_volume_path, index=False)
else:
    iou_results_df = pd.read_parquet(iou_results_path)
    per_compartment_volume_df = pd.read_parquet(per_compartment_volume_path)

# make compartment a categorical variable
iou_results_df["compartment"] = pd.Categorical(
    iou_results_df["compartment"],
    categories=["organoid", "nuclei", "cytoplasm", "cell"],
    ordered=True,
)


# In[6]:


# normalize per patient and compartment (min-max normalization), since
# absolute volume scales differ across patients
per_compartment_volume_df["normalized_volume"] = per_compartment_volume_df.groupby(
    ["patient", "compartment"]
)["volume"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

# persist the normalized table as its own intermediate file (separate from the raw
# iou/volume cache above) so downstream plotting (e.g. the R/ggplot2 notebook) can load
# it directly instead of recomputing the normalization
per_compartment_volume_df.to_parquet(per_compartment_volume_path, index=False)
iou_results_df.to_parquet(iou_results_path, index=False)
