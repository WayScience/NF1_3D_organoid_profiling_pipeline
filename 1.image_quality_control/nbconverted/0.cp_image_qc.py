#!/usr/bin/env python
# coding: utf-8

# # Run whole image QC pipeline in CellProfiler
# 
# To determine if there are images that of poor quality, we run a CellProfiler pipeline specific to extracting image quality metrics.
# We extract blur and saturation metrics, we can use to identify thresholds for these metrics to separate the good and poor quality images.
# 

# ## Import libraries

# In[1]:


import pathlib
import pprint

import sys

sys.path.append("../utils")
import cp_parallel


# ## Set paths and variables

# In[2]:


# set the run type for the parallelization
run_name = "quality_control"

# set path for pipeline for whole image QC
path_to_pipeline = pathlib.Path("./pipeline/whole_image_qc.cppipe").resolve(strict=True)

# set main output dir for all plates if it doesn't exist
output_dir = pathlib.Path("./qc_results")
output_dir.mkdir(exist_ok=True)

# directory where images are located within folders (parent folder is the plate and the child folders are wells containing images)
images_dir = pathlib.Path("../data/raw_images").resolve(strict=True)

# list for plate names based on folders to use to create dictionary
plate_names = []
# iterate through 0.download_data and append plate names from folder names that contain image data from that plate
for file_path in images_dir.iterdir():
    plate_names.append(str(file_path.stem.split("_")[0]))

print("There are a total of", len(plate_names), "plates. The names of the plates are:")
for plate in plate_names:
    print(plate)


# ## Generate dictionary with plate info to run CellProfiler

# In[3]:


# create plate info dictionary with all parts of the CellProfiler CLI command
plate_info_dictionary = {
    name: {
        "path_to_images": pathlib.Path(list(images_dir.rglob(f"{name}_raw_images"))[0]).resolve(
            strict=True
        ),
        "path_to_output": pathlib.Path(f"{output_dir}/{name}_qc_results"),
        "path_to_pipeline": path_to_pipeline,
    }
    for name in plate_names
}

# view the dictionary to assess that all info is added correctly
pprint.pprint(plate_info_dictionary, indent=4)


# ## Run QC pipeline in CellProfiler

# In[ ]:


cp_parallel.run_cellprofiler_parallel(
    plate_info_dictionary=plate_info_dictionary, run_name=run_name
)

