#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import sys
import time

import numpy as np
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

from image_analysis_3D.featurization_utils.area_size_shape_utils import (
    measure_3D_area_size_shape,
)
from image_analysis_3D.featurization_utils.feature_writing_utils import (
    format_morphology_feature_name,
    save_features_as_parquet,
)

# bug in the cucim module but we are using CPU so it does not matter for now
# from image_analysis_3D.featurization_utils.area_size_shape_utils_gpu import measure_3D_area_size_shape_gpu
from image_analysis_3D.featurization_utils.loading_classes import (
    ImageSetLoader,
    ObjectLoader,
)
from image_analysis_3D.featurization_utils.resource_profiling_util import (
    start_profiling,
    stop_profiling,
)
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[2]:


if not in_notebook:
    arguments_dict = parse_args()
    patient = arguments_dict["patient"]
    well_fov = arguments_dict["well_fov"]
    compartment = arguments_dict["compartment"]
    channel = arguments_dict["channel"]
    processor_type = arguments_dict["processor_type"]
    input_subparent_name = arguments_dict["input_subparent_name"]
    mask_subparent_name = arguments_dict["mask_subparent_name"]
    output_features_subparent_name = arguments_dict["output_features_subparent_name"]

else:
    well_fov = "C4-2"
    patient = "NF0014_T1"
    compartment = "Nuclei"
    channel = "NoChannel"
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
)


# In[6]:


object_loader = ObjectLoader(
    image=None,
    label_image=image_set_loader.image_set_dict[compartment],
    channel_name=None,
    compartment_name=compartment,
)

# area, size, shape
if processor_type == "GPU":
    size_shape_dict = measure_3D_area_size_shape_gpu(
        image_set_loader=image_set_loader,
        object_loader=object_loader,
    )
elif processor_type == "CPU":
    size_shape_dict = measure_3D_area_size_shape(
        image_set_loader=image_set_loader,
        object_loader=object_loader,
    )
else:
    raise ValueError(
        f"Processor type {processor_type} is not supported. Use 'CPU' or 'GPU'."
    )


# In[7]:


final_df = pd.DataFrame(size_shape_dict)

# prepend compartment and channel to column names
final_df.rename(
    columns={
        col: format_morphology_feature_name(
            compartment=compartment,
            channel=channel,
            feature_type="AreaSizeShape",
            measurement=col,
        )
        if col != "object_id"
        else col
        for col in final_df.columns
    },
    inplace=True,
)

final_df.insert(1, "image_set", image_set_loader.image_set_name)

save_path = save_features_as_parquet(
    parent_path=output_parent_path,
    df=final_df,
    feature_type="AreaSizeShape",
    channel=channel,
    compartment=compartment,
    cpu_or_gpu=processor_type,
)

final_df.head()


# In[ ]:


stop_profiling(
    start_time=start_time,
    start_mem=start_mem,
    feature_type="AreaSizeShape",
    well_fov=well_fov,
    patient_id=patient,
    channel="NoChannel",
    compartment=compartment,
    CPU_GPU=processor_type,
    output_file_dir=pathlib.Path(
        f"{image_base_dir}/data/{patient}/extracted_features/run_stats/{well_fov}_AreaSizeShape_NoChannel_{compartment}_{processor_type}.parquet"
    ),
)
