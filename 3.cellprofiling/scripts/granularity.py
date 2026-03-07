#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import sys
import time

import pandas as pd
import psutil
import tomli
from image_analysis_3D.file_utils.arg_parsing_utils import (
    check_for_missing_args,
    parse_args,
)
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()

from image_analysis_3D.featurization_utils.feature_writing_utils import (
    format_morphology_feature_name,
)
from image_analysis_3D.featurization_utils.granularity_utils import (
    measure_3D_granularity,
)

# from granularity import measure_3D_granularity
from image_analysis_3D.featurization_utils.loading_classes import (
    ImageSetLoader,
    ObjectLoader,
)
from image_analysis_3D.featurization_utils.resource_profiling_util import (
    start_profiling,
    stop_profiling,
)

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


if not in_notebook:
    arguments_dict = parse_args()
    patient = arguments_dict["patient"]
    well_fov = arguments_dict["well_fov"]
    channel = arguments_dict["channel"]
    compartment = arguments_dict["compartment"]
    processor_type = arguments_dict["processor_type"]
    input_subparent_name = arguments_dict["input_subparent_name"]
    mask_subparent_name = arguments_dict["mask_subparent_name"]
    output_features_subparent_name = arguments_dict["output_features_subparent_name"]

else:
    well_fov = "C4-2"
    patient = "NF0014_T1"
    channel = "AGP"
    compartment = "Organoid"
    processor_type = "CPU"
    input_subparent_name = "zstack_images"
    mask_subparent_name = "segmentation_masks"
    output_features_subparent_name = "extracted_features"

image_set_path = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{input_subparent_name}/{well_fov}/"
)
mask_set_path = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{mask_subparent_name}/{well_fov}/"
)
output_parent_path = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{output_features_subparent_name}/{well_fov}/"
)
output_parent_path.mkdir(parents=True, exist_ok=True)
channel_mapping_file_path = pathlib.Path(
    f"{root_dir}/config/channel_mapping.toml"
).resolve(strict=True)


# In[3]:


# read in channel mapping
with open(channel_mapping_file_path, "rb") as f:
    channel_mapping_dict = tomli.load(f)
channel_n_compartment_mapping = channel_mapping_dict["channel_mapping"]


# In[4]:


start_time, start_mem = start_profiling()


# In[5]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    mask_set_path=mask_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_n_compartment_mapping,
    image_set_name=well_fov,
    mask_key_name=[channel_n_compartment_mapping[compartment]],
    raw_image_key_name=[channel_n_compartment_mapping[channel]],
)


# In[6]:


object_loader = ObjectLoader(
    image_set_loader.image_set_dict[channel],
    image_set_loader.image_set_dict[compartment],
    channel,
    compartment,
)
if in_notebook:
    verbose = True
else:
    verbose = False
if processor_type == "CPU":
    object_measurements = measure_3D_granularity(
        object_loader=object_loader,
        radius=10,  # radius of the structuring element for background removal (CellProfiler default)
        granular_spectrum_length=16,  # range of the granular spectrum
        subsample_size=0.25,  # subsample the image for faster processing
        image_sample_size=0.25,  # further subsample for background removal
        mask_threshold=0.9,  # threshold for determining mask after interpolation
        verbose=verbose,
    )
else:
    raise ValueError(
        f"Processor type {processor_type} is not supported. Use 'CPU' only."
    )
final_df = pd.DataFrame(object_measurements)
# get the mean of each value in the array
# melt the dataframe to wide format
final_df = final_df.pivot_table(
    index=["object_id"], columns=["feature"], values=["value"]
)
final_df.columns = final_df.columns.droplevel()
final_df = final_df.reset_index()
# prepend compartment and channel to column names
final_df.rename(
    columns={
        col: format_morphology_feature_name(
            compartment=compartment,
            channel=channel,
            feature_type="Granularity",
            measurement=col,
        )
        if col != "object_id"
        else col
        for col in final_df.columns
    },
    inplace=True,
)
final_df.insert(0, "image_set", image_set_loader.image_set_name)
output_file = pathlib.Path(
    output_parent_path
    / f"Granularity_{compartment}_{channel}_{processor_type}_features.parquet"
)
output_file.parent.mkdir(parents=True, exist_ok=True)
final_df.to_parquet(output_file)
final_df.head()


# In[7]:


stop_profiling(
    start_time=start_time,
    start_mem=start_mem,
    feature_type="Granularity",
    well_fov=well_fov,
    patient_id=patient,
    channel=channel,
    compartment=compartment,
    CPU_GPU=processor_type,
    output_file_dir=pathlib.Path(
        f"{root_dir}/data/{patient}/extracted_features/run_stats/{well_fov}_{channel}_{compartment}_Granularity_{processor_type}.parquet"
    ),
)
