#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import pathlib
import sys
import time

import psutil

sys.path.append("../featurization_utils")
import multiprocessing
import pathlib
from functools import partial
from itertools import product

import pandas as pd
from intensity_utils import measure_3D_intensity_CPU
from loading_classes import ImageSetLoader, ObjectLoader
from resource_profiling_util import get_mem_and_time_profiling

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# In[ ]:


def process_combination(args, image_set_loader):
    """
    Process a single combination of compartment and channel.

    Parameters
    ----------
    args : tuple
        Args that contain the compartment and channel.
        Ordered as (compartment, channel).
        Yes, order matters here.
        channel : str
            The channel name.
        compartment : str
            The compartment name.
    image_set_loader : Class ImageSetLoader
        This contains the image information needed to retreive the objects.

    Returns
    -------
    str
        A string indicating the compartment and channel that was processed.
    """
    compartment, channel = args
    object_loader = ObjectLoader(
        image_set_loader.image_set_dict[channel],
        image_set_loader.image_set_dict[compartment],
        channel,
        compartment,
    )
    output_dict = measure_3D_intensity_CPU(object_loader)
    final_df = pd.DataFrame(output_dict)
    # prepend compartment and channel to column names
    final_df = final_df.pivot(
        index=["object_id"],
        columns="feature_name",
        values="value",
    ).reset_index()
    for col in final_df.columns:
        if col == "object_id":
            continue
        else:
            final_df.rename(
                columns={col: f"Intensity_{compartment}_{channel}_{col}"},
                inplace=True,
            )

    final_df.insert(0, "image_set", image_set_loader.image_set_name)

    output_file = pathlib.Path(
        f"../results/{image_set_loader.image_set_name}/Intensity_{compartment}_{channel}_features.parquet"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_file)

    return f"Processed {compartment} - {channel}"


# In[ ]:


if not in_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--well_fov",
        type=str,
        default="None",
        help="Well and field of view to process, e.g. 'A01_1'",
    )

    args = argparser.parse_args()
    well_fov = args.well_fov
    if well_fov == "None":
        raise ValueError(
            "Please provide a well and field of view to process, e.g. 'A01_1'"
        )

    image_set_path = pathlib.Path(f"../../data/NF0014/cellprofiler/{well_fov}/")
else:
    well_fov = "C4-2"
    image_set_path = pathlib.Path(f"../../data/NF0014/cellprofiler/{well_fov}/")

print(f"Processing {image_set_path}...")


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


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_n_compartment_mapping,
)


# In[ ]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[ ]:


# Generate all combinations of compartments and channels
combinations = list(
    product(image_set_loader.compartments, image_set_loader.image_names)
)
cores = multiprocessing.cpu_count()
print(f"Using {cores} cores for processing.")
# Use multiprocessing to process combinations in parallel
with multiprocessing.Pool(processes=cores) as pool:
    results = list(
        tqdm(
            pool.imap(
                partial(process_combination, image_set_loader=image_set_loader),
                combinations,
            ),
            desc="Processing combinations",
        )
    )

print("Processing complete.")


# In[ ]:


end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
end_time = time.time()
get_mem_and_time_profiling(
    start_mem=start_mem,
    end_mem=end_mem,
    start_time=start_time,
    end_time=end_time,
    process_name="Intensity",
    well_fov=well_fov,
    CPU_GPU="CPU",
)
