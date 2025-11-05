#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pathlib
import time

import pandas as pd
import psutil
from arg_parsing_utils import check_for_missing_args, parse_args
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

from featurization_parsable_arguments import parse_featurization_args
from granularity_utils import measure_3D_granularity

# from granularity import measure_3D_granularity
from loading_classes import ImageSetLoader, ObjectLoader
from resource_profiling_util import get_mem_and_time_profiling

# In[ ]:


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
    channel = "Mito"
    compartment = "Cell"
    processor_type = "CPU"

image_set_path = pathlib.Path(
    f"{root_dir}/data/{patient}/profiling_input_images/{well_fov}/"
)
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


# In[ ]:


object_loader = ObjectLoader(
    image_set_loader.image_set_dict[channel],
    image_set_loader.image_set_dict[compartment],
    channel,
    compartment,
)
if processor_type == "CPU":
    object_measurements = measure_3D_granularity(
        object_loader=object_loader,
        radius=1,  # radius of the sphere to use for granularity measurement in pixels
        granular_spectrum_length=16,  # usually 16 but 2 is used for testing for now
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
for col in final_df.columns:
    if col == "object_id":
        continue
    else:
        final_df.rename(
            columns={col: f"Granularity_{compartment}_{channel}_{col}"},
            inplace=True,
        )
final_df.insert(0, "image_set", image_set_loader.image_set_name)
output_file = pathlib.Path(
    output_parent_path
    / f"Granularity_{compartment}_{channel}_{processor_type}_features.parquet"
)
output_file.parent.mkdir(parents=True, exist_ok=True)
final_df.to_parquet(output_file)


# In[ ]:


end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
end_time = time.time()
get_mem_and_time_profiling(
    start_mem=start_mem,
    end_mem=end_mem,
    start_time=start_time,
    end_time=end_time,
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
