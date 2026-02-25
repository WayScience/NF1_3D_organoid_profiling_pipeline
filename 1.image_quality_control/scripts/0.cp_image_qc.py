#!/usr/bin/env python
# coding: utf-8

# # Run whole image QC pipeline in CellProfiler
#
# To determine if there are images that of poor quality, we run a CellProfiler pipeline specific to extracting image quality metrics.
# We extract blur and saturation metrics, we can use to identify thresholds for these metrics to separate the good and poor quality images.
#

# ## Import libraries

# In[ ]:


import os
import pathlib
import pprint

import cp_parallel
from image_analysis_3D.file_utils.notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()


# ## Set paths and variables

# In[ ]:


bandicoot_path = pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve()
if bandicoot_path.exists():
    # comment out depending on whose computer you are on
    # mike's computer
    image_base_dir = pathlib.Path("~/mnt/bandicoot").resolve()
else:
    image_base_dir = root_dir


# In[6]:


# set the run type for the parallelization
run_name = "quality_control"

# set path for pipeline for whole image QC
path_to_pipeline = pathlib.Path(
    f"{root_dir}/1.image_quality_control/pipeline/whole_image_qc.cppipe"
).resolve(strict=True)

# set main output dir for all plates if it doesn't exist
output_dir = pathlib.Path(f"{root_dir}/1.image_quality_control/pipeline/qc_results")
output_dir.mkdir(exist_ok=True)

# directory where images are located within folders (parent folder is the plate and the child folders are wells containing images)
images_dir = pathlib.Path(f"{image_base_dir}/data").resolve(strict=True)

# filter plate names to include only those starting with 'NF' or 'SARCO'
plate_names = []
for file_path in images_dir.iterdir():
    plate_name = str(file_path.stem)

    if plate_name.startswith("NF") or plate_name.startswith("SARCO"):
        print("Plate name:", plate_name)
        plate_names.append(plate_name)

print("There are a total of", len(plate_names), "plates. The names of the plates are:")
for plate in plate_names:
    print(plate)


# ## Generate dictionary with plate info to run CellProfiler

# In[3]:


# filter plate names to include only those that haven't been processed (output directory doesn't exist or is empty)
unprocessed_plate_names = [
    name
    for name in plate_names
    if not (output_dir / f"{name}_qc_results").exists()
    or not any((output_dir / f"{name}_qc_results").iterdir())
]

# create plate info dictionary with all parts of the CellProfiler CLI command
plate_info_dictionary = {
    name: {
        "path_to_images": pathlib.Path(
            list(images_dir.rglob(f"{name}/raw_images"))[0]
        ).resolve(strict=True),
        "path_to_output": pathlib.Path(f"{output_dir}/{name}_qc_results"),
        "path_to_pipeline": path_to_pipeline,
    }
    for name in unprocessed_plate_names
}

# view the dictionary to assess that all info is added correctly
pprint.pprint(plate_info_dictionary, indent=4)


# ## Run QC pipeline in CellProfiler

# In[ ]:


cp_parallel.run_cellprofiler_parallel(
    plate_info_dictionary=plate_info_dictionary, run_name=run_name
)
