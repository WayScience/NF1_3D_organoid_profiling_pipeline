#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import time

import numpy as np
import pandas as pd
import psutil
from notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()

from area_size_shape_utils import measure_3D_area_size_shape
from area_size_shape_utils_gpu import measure_3D_area_size_shape_gpu
from loading_classes import ImageSetLoader, ObjectLoader
from notebook_init_utils import bandicoot_check, init_notebook
from resource_profiling_util import get_mem_and_time_profiling

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
    well_fov = "E11-3"
    patient = "NF0018_T6"
    compartment = "Nuclei"
    channel = "DNA"
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


# In[3]:


channel_n_compartment_mapping = {
    "DNA": "405",
    "AGP": "488",
    "ER": "555",
    "Mito": "640",
    "BF": "TRANS",
    "Nuclei": "nuclei_",
    "Cell": "cell_",
    "Cytoplasm": "cytoplasm_",
    "Organoid": "organoid_",
}


# In[4]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[5]:


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    mask_set_path=mask_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_n_compartment_mapping,
)


# In[6]:


object_loader = ObjectLoader(
    image_set_loader.image_set_dict["DNA"],
    image_set_loader.image_set_dict[compartment],
    "DNA",
    compartment,
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
for col in final_df.columns:
    if col not in ["object_id"]:
        final_df[col] = final_df[col].astype(np.float32)
        final_df.rename(
            columns={col: f"Area.Size.Shape_{compartment}_{col}"},
            inplace=True,
        )

final_df.insert(1, "image_set", image_set_loader.image_set_name)

output_file = pathlib.Path(
    output_parent_path
    / f"AreaSizeShape_{compartment}_{processor_type}_features.parquet"
)
final_df.to_parquet(output_file, index=False)
final_df.head()


# In[8]:


end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
end_time = time.time()
get_mem_and_time_profiling(
    start_mem=start_mem,
    end_mem=end_mem,
    start_time=start_time,
    end_time=end_time,
    feature_type="AreaSizeShape",
    well_fov=well_fov,
    patient_id=patient,
    channel="DNA",
    compartment=compartment,
    CPU_GPU=processor_type,
    output_file_dir=pathlib.Path(
        f"{image_base_dir}/data/{patient}/extracted_features/run_stats/{well_fov}_AreaSizeShape_DNA_{compartment}_{processor_type}.parquet"
    ),
)
