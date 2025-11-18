#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import time

import pandas as pd
import psutil
from arg_parsing_utils import check_for_missing_args, parse_args
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

from loading_classes import ImageSetLoader, ObjectLoader
from resource_profiling_util import get_mem_and_time_profiling
from texture_utils import measure_3D_texture

# In[2]:


if not in_notebook:
    arguments_dict = parse_featurization_args()
    patient = arguments_dict["patient"]
    well_fov = arguments_dict["well_fov"]
    channel = arguments_dict["channel"]
    compartment = arguments_dict["compartment"]
    processor_type = arguments_dict["processor_type"]
else:
    well_fov = "C4-2"
    patient = "NF0014_T1"
    channel = "AGP"
    compartment = "Nuclei"
    processor_type = "CPU"

bandicoot_check

image_set_path = pathlib.Path(f"{root_dir}/data/{patient}/zstack_images/{well_fov}/")
output_parent_path = pathlib.Path(
    f"{root_dir}/data/{patient}/extracted_features/{well_fov}/"
)
output_parent_path.mkdir(parents=True, exist_ok=True)


# In[3]:


channel_mapping = {
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
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)
image_set_loader.image_set_dict.keys()


# In[ ]:


object_loader = ObjectLoader(
    image_set_loader.image_set_dict[channel],
    image_set_loader.image_set_dict[compartment],
    channel,
    compartment,
)
output_texture_dict = measure_3D_texture(
    object_loader=object_loader,
    distance=3,  # distance in pixels 3 is what CP uses
)
final_df = pd.DataFrame(output_texture_dict)

final_df = final_df.pivot(
    index="object_id",
    columns="texture_name",
    values="texture_value",
)
final_df.reset_index(inplace=True)
for col in final_df.columns:
    if col == "object_id":
        continue
    else:
        final_df.rename(
            columns={col: f"Texture_{compartment}_{channel}_{col}"},
            inplace=True,
        )
final_df.insert(0, "image_set", image_set_loader.image_set_name)
final_df.columns.name = None

output_file = pathlib.Path(
    output_parent_path
    / f"Texture_{compartment}_{channel}_{processor_type}_features.parquet"
)
output_file.parent.mkdir(parents=True, exist_ok=True)
final_df.to_parquet(output_file)


# In[8]:


end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
end_time = time.time()
get_mem_and_time_profiling(
    start_mem=start_mem,
    end_mem=end_mem,
    start_time=start_time,
    end_time=end_time,
    feature_type="Texture",
    well_fov=well_fov,
    patient_id=patient,
    channel=channel,
    compartment=compartment,
    CPU_GPU="CPU",
    output_file_dir=pathlib.Path(
        f"{root_dir}/data/{patient}/extracted_features/run_stats/{well_fov}_{channel}_{compartment}_Texture_CPU.parquet"
    ),
)
