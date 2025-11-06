#!/usr/bin/env python
# coding: utf-8

# This runs all segmentation operations in one place.
# The idea is that this should be faster and easier to envoke as we only have to load the image data once instead of N times (~10).
# Running each individual task as its own script is modular but requires overhead to load the data each time.
# Currently it takes about 15 minutes to complete a single organoid's segmentation for all compartments... (~50,1500,1500) (Z,Y,X) dimensional image.
# Let us see how long this takes!
#
# No we are at ~8 minutes!

# In[1]:


import argparse
import os
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import tifffile
from arg_parsing_utils import check_for_missing_args, parse_args
from file_reading import read_zstack_image
from notebook_init_utils import bandicoot_check, init_notebook
from skimage.filters import sobel

from segmentation_utils import *

# In[2]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[3]:


root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[ ]:


if not in_notebook:
    args = parse_args()
    clip_limit = args["clip_limit"]
    well_fov = args["well_fov"]
    patient = args["patient"]
    input_subparent_name = args["input_subparent_name"]
    mask_subparent_name = args["mask_subparent_name"]
    check_for_missing_args(
        well_fov=well_fov,
        patient=patient,
        clip_limit=clip_limit,
        input_subparent_name=input_subparent_name,
        mask_subparent_name=mask_subparent_name,
    )
else:
    print("Running in a notebook")
    patient = "NF0037_T1-Z-1"
    well_fov = "F4-2"
    window_size = 3
    clip_limit = 0.01
    input_subparent_name = "zstack_images"
    mask_subparent_name = "segmentation_masks"


input_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{input_subparent_name}/{well_fov}"
).resolve(strict=True)
mask_path = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{mask_subparent_name}/{well_fov}"
).resolve()
mask_path.mkdir(exist_ok=True, parents=True)


# In[17]:


print(input_dir)


# In[5]:


return_dict = read_in_channels(find_files_available(input_dir))
nuclei = return_dict["nuclei"]
cyto2 = return_dict["cyto2"]
# run clip_limit here
nuclei = skimage.exposure.equalize_adapthist(
    nuclei, clip_limit=clip_limit, kernel_size=None
)
cyto2 = skimage.exposure.equalize_adapthist(
    cyto2, clip_limit=clip_limit, kernel_size=None
)


# ## Organoid segmentation

# In[ ]:


cyto2_image_shape = cyto2.shape
# commenting out optimization for time being
# uncomment to run optimization
# butterworth_grid_optimization(two_point_five_D_sliding_window, return_plot=False)

filtered_cyto2 = apply_butterworth_filter(
    sliding_window_two_point_five_D(cyto2, window_size=4),  # cyto
    cutoff_frequency_ratio=0.05,
    order=1,
    high_pass=False,
    squared_butterworth=True,
)
model = models.CellposeModel(
    gpu=[True if torch.cuda.is_available() else False][0],
    model_type="cyto3",  # CP3
)
output_dict = {
    "slice": [],
    "labels": [],
    "details": [],
}
for slice in tqdm.tqdm(range(filtered_cyto2.shape[0])):
    labels, details, _ = segment_with_diameter(
        filtered_cyto2[slice],
        model=model,
        diameter=750,
        z_axis=0,
        channels=[1, 0],
        min_diameter=200,  # default 200
        diameter_step=200,  # default 200
    )
    output_dict["slice"].append(slice)
    output_dict["labels"].append(labels)
    output_dict["details"].append(details)

del filtered_cyto2


# In[ ]:


organoid_masks = np.array(
    list(
        decouple_masks(
            reverse_sliding_window_max_projection(
                output_dict,
                window_size=4,
                original_z_slice_count=cyto2_image_shape[0],
            ),
            original_img_shape=cyto2_image_shape,
            distance_threshold=40,
        ).values()
    )
)


# In[ ]:


# generate the coordinates dataframe for reconstruction
coordinates_df = generate_coordinates_for_reconstruction(organoid_masks)
# generate distance pairs dataframe
df = generate_distance_pairs(coordinates_df, x_y_vector_radius_max_constraint=20)
# create and solve graph to get longest paths
longest_paths = solve_graph(graph_creation(df))
# collapse labels based on longest paths and reassign labels in organoid masks
image = reassign_labels(organoid_masks, collapse_labels(coordinates_df, longest_paths))
# refine the organoid masks
organoid_mask = run_post_hoc_refinement(
    mask_image=image,
    sliding_window_context=3,
)
# clean up and send to gc
del image, coordinates_df, df, longest_paths


# ## Nuclei Segmentation

# In[ ]:


nuclei_image_shape = nuclei.shape
#
nuclei_masks = np.array(
    list(  # send to array
        decouple_masks(  # 4. decouple masks
            reverse_sliding_window_max_projection(  # 3. reverse sliding window
                segmentaion_on_two_D(  # 2. segment on 2D
                    sliding_window_two_point_five_D(  # 1. run sliding window
                        image_stack=nuclei, window_size=3
                    )
                ),
                window_size=3,
                original_z_slice_count=nuclei_image_shape[0],
            ),
            original_img_shape=nuclei_image_shape,
            distance_threshold=10,
        ).values()
    )
)


# In[10]:


# generate the coordinates dataframe for reconstruction
coordinates_df = generate_coordinates_for_reconstruction(nuclei_masks)
# generate distance pairs dataframe
df = generate_distance_pairs(coordinates_df, x_y_vector_radius_max_constraint=20)
# generate and solve graph to get longest paths
longest_paths = solve_graph(graph_creation(df))
# collapse labels based on longest paths and reassign labels in nuclei masks
image = reassign_labels(nuclei_masks, collapse_labels(coordinates_df, longest_paths))
# refine the nuclei masks
nuclei_mask = run_post_hoc_refinement(
    mask_image=image,
    sliding_window_context=3,
)
del image, coordinates_df, df, longest_paths


# ## Segment the cells

# In[11]:


cell_mask = segment_cells_with_3D_watershed(
    cyto_signal=cyto2,
    nuclei_mask=nuclei_mask,
)


# ## run the mask reassignment function (post-hoc)
# ### This needs to occur after both nuclei and cell segmentations are done

# In[12]:


cell_df = get_labels_for_post_hoc_reassignment(
    compartment_mask=cell_mask, compartment_name="cell"
)
nuclei_df = get_labels_for_post_hoc_reassignment(
    compartment_mask=nuclei_mask, compartment_name="nuclei"
)


# In[13]:


nuclei_mask, reassigned_nuclei_df = run_post_hoc_mask_reassignment(
    nuclei_mask=nuclei_mask,
    cell_mask=cell_mask,
    nuclei_df=nuclei_df,
    cell_df=cell_df,
    return_dataframe=True,
)


# ## Cytoplasm Segmentation

# In[14]:


cytoplasm_mask = create_cytoplasm_masks(
    nuclei_masks=nuclei_mask,
    cell_masks=cell_mask,
)


# ## Save the segmented masks

# In[15]:


nuclei_mask_output = pathlib.Path(f"{mask_path}/{well_fov}_nuclei_mask.tiff")
cell_mask_output = pathlib.Path(f"{mask_path}/{well_fov}_cell_mask.tiff")
cytoplasm_mask_output = pathlib.Path(f"{mask_path}/{well_fov}_cytoplasm_mask.tiff")
organoid_mask_output = pathlib.Path(f"{mask_path}/{well_fov}_organoid_mask.tiff")
tifffile.imwrite(nuclei_mask_output, nuclei_mask)
tifffile.imwrite(cell_mask_output, cell_mask)
tifffile.imwrite(cytoplasm_mask_output, cytoplasm_mask)
tifffile.imwrite(organoid_mask_output, organoid_mask)


# In[16]:


end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
end_time = time.time()
print(f"""
    Memory and time profiling for the run:\n
    Memory usage: {end_mem - start_mem:.2f} MB\n
    Time:\n
    --- %s seconds --- % {(end_time - start_time)}\n
    --- %s minutes --- % {((end_time - start_time) / 60)}\n
    --- %s hours --- % {((end_time - start_time) / 3600)}
""")
