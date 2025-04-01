#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pathlib
import sys
import time

sys.path.append("../featurization_utils")
import numpy as np
import pandas as pd
import scipy
import skimage
from loading_classes import ImageSetLoader, ObjectLoader
from neighbors_utils import measure_3D_number_of_neighbors

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False
if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# In[2]:


image_set_path = pathlib.Path("../../data/NF0014/cellprofiler/C4-2/")


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


image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)


# In[ ]:


start_time = time.time()


# In[ ]:


for compartment in tqdm(
    image_set_loader.compartments, desc="Processing compartments", position=0
):
    for channel in tqdm(
        image_set_loader.image_names,
        desc="Processing channels",
        leave=False,
        position=1,
    ):
        object_loader = ObjectLoader(
            image_set_loader.image_set_dict[channel],
            image_set_loader.image_set_dict[compartment],
            channel,
            compartment,
        )
        neighbors_out_dict = measure_3D_number_of_neighbors(
            object_loader=object_loader,
            distance_threshold=10,
            anisotropy_factor=image_set_loader.anisotropy_factor,
        )
        final_df = pd.DataFrame(neighbors_out_dict)
        # prepend compartment and channel to column names
        final_df.columns = [
            f"{compartment}_{channel}_{col}" for col in final_df.columns
        ]
        final_df["image_set"] = image_set_loader.image_set_name

        output_file = pathlib.Path(
            f"../results/{image_set_loader.image_set_name}/Neighbors_{compartment}_{channel}_features.parquet"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_file)


# In[ ]:


print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
