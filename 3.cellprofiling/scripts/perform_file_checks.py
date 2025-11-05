#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import itertools
import pathlib

import numpy as np
import pandas as pd
from arg_parsing_utils import check_for_missing_args, parse_args
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path("~/mnt/bandicoot/NF1_organoid_data").resolve(), root_dir
)

from file_checking import check_number_of_files
from loading_classes import ImageSetLoader

# In[2]:


patient = "NF0014_T1"
well_fov = "C2-1"
# set path to the processed data dir

image_set_path = pathlib.Path(
    f"{profile_base_dir}/data/{patient}/profiling_input_images/{well_fov}/"  # just to get channels structure
)
patient_id_file_path = pathlib.Path(f"{profile_base_dir}/data/patient_IDs.txt").resolve(
    strict=True
)
rerun_combinations_path = pathlib.Path(
    f"{profile_base_dir}/3.cellprofiling/load_data/rerun_combinations.txt"
).resolve()
rerun_combinations_path.parent.mkdir(parents=True, exist_ok=True)
patient_ids = pd.read_csv(
    patient_id_file_path, header=None, names=["patient_id"]
).patient_id.tolist()

patient_ids = [patient]  # for testing


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
image_set_loader = ImageSetLoader(
    image_set_path=image_set_path,
    anisotropy_spacing=(1, 0.1, 0.1),
    channel_mapping=channel_mapping,
)

channels = image_set_loader.image_names
compartments = image_set_loader.compartments
channel_combinations = list(itertools.combinations(channels, 2))


# For each well fov there should be the following number of files:
# Of course this depends on if both CPU and GPU versions are run, but the CPU version is always run.
# | Feature Type | No. Compartments | No. Channels | No. Processors | Total No. Files |
# |--------------|------------------|---------------|----------------|-----------------|
# | AreaSizeShape | 4 | 1 | 2 | 8 |
# | Colocalization | 4 | 10 | 2 | 80 |
# | Granularity | 4 | 5 | 1 | 20 |
# | Intensity | 4 | 5 | 2 | 40 |
# | Neighbors | 1 | 1 | 1 | 1 |
# | Texture | 4 | 5 | 1 | 20 |
#
# Total no. files per well fov = 169
#
# ### OR
# For CPU only:
# For each well fov there should be the following number of files:
# | Feature Type | No. Compartments | No. Channels | No. Processors | Total No. Files |
# |--------------|------------------|---------------|----------------|-----------------|
# | AreaSizeShape | 4 | 1 | 1 | 4 |
# | Colocalization | 4 | 10 | 1 | 40 |
# | Granularity | 4 | 5 | 1 | 20 |
# | Intensity | 4 | 5 | 1 | 20 |
# | Neighbors | 1 | 1 | 1 | 1 |
# | Texture | 4 | 5 | 1 | 20 |
#
# Total no. files per well fov = 105
#
#

# In[4]:


feature_types = [
    "AreaSizeShape",
    "Colocalization",
    "Granularity",
    "Intensity",
    "Neighbors",
    "Texture",
]


# In[5]:


processor_types = [
    "CPU",
    # "GPU"
]


# In[6]:


feature_list = []
# construct the file space

# area, size, shape
for compartment in compartments:
    for processor_type in processor_types:
        feature_list.append(f"AreaSizeShape_{compartment}_{processor_type}_features")
# colocalization
for channel in channel_combinations:
    for compartment in compartments:
        for processor_type in processor_types:
            feature_list.append(
                f"Colocalization_{compartment}_{channel[0]}.{channel[1]}_{processor_type}_features"
            )
# granularity
for channel in channels:
    for compartment in compartments:
        feature_list.append(f"Granularity_{compartment}_{channel}_CPU_features")
# intensity
for channel in channels:
    for compartment in compartments:
        for processor_type in processor_types:
            feature_list.append(
                f"Intensity_{compartment}_{channel}_{processor_type}_features"
            )
# neighbors
feature_list.append("Neighbors_Nuclei_DNA_CPU_features")
# texture
for channel in channels:
    for compartment in compartments:
        feature_list.append(f"Texture_{compartment}_{channel}_CPU_features")
len(feature_list)  # should be 105 or 169 depending on CPU vs CPU and GPU


# In[7]:


featurization_rerun_dict = {
    "patient": [],
    "well_fov": [],
    "feature": [],
    "compartment": [],
    "channel": [],
    "processor_type": [],
}


# In[8]:


total_files = 0
files_present = 0
for patient in patient_ids:
    well_fovs = pathlib.Path(
        f"{profile_base_dir}/data/{patient}/zstack_images/"
    ).resolve()

    # perform checks for each directory
    featurization_data_dirs = list(well_fovs.glob("*"))

    for dir in featurization_data_dirs:
        if dir.name != "run_stats":
            dir = pathlib.Path(
                f"{profile_base_dir}/data/{patient}/extracted_features/{dir.name}"
            ).resolve(strict=True)
            total_files += len(feature_list)
            if not check_number_of_files(dir, len(feature_list)):
                # find the missing files
                # cross reference the files in the directory
                # with the expected feature list
                existing_files = [f.stem for f in dir.glob("*") if f.is_file()]

                files_present += len(existing_files)
                missing_files = set(feature_list) - set(existing_files)
                assert len(missing_files) >= 0, "There should be no missing files"
                assert len(missing_files) <= len(feature_list), (
                    f"There should be at most {len(feature_list)} missing files"
                )
                print((len(missing_files) + len(existing_files)), len(feature_list))
                assert len(missing_files) + len(existing_files) == len(feature_list), (
                    f"There should be exactly {len(feature_list)} files in the directory"
                )
                if missing_files:
                    for missing_file in missing_files:
                        if missing_file.split("_")[0] == "Colocalization":
                            featurization_rerun_dict["channel"].append(
                                missing_file.split("_")[2].split(".")[0]
                                + "."
                                + missing_file.split("_")[2].split(".")[1]
                            )
                            featurization_rerun_dict["processor_type"].append(
                                missing_file.split("_")[3]
                            )
                        elif missing_file.split("_")[0] == "AreaSizeShape":
                            featurization_rerun_dict["channel"].append(
                                "DNA"
                            )  # AreaSizeShape is always DNA
                            featurization_rerun_dict["processor_type"].append(
                                missing_file.split("_")[2]
                            )
                        else:
                            featurization_rerun_dict["channel"].append(
                                missing_file.split("_")[2]
                            )
                            featurization_rerun_dict["processor_type"].append(
                                missing_file.split("_")[3]
                            )
                        featurization_rerun_dict["patient"].append(patient)
                        featurization_rerun_dict["well_fov"].append(dir.name)
                        featurization_rerun_dict["feature"].append(
                            missing_file.split("_")[0]
                        )
                        featurization_rerun_dict["compartment"].append(
                            missing_file.split("_")[1]
                        )
            else:
                files_present += len([f.stem for f in dir.glob("*") if f.is_file()])


# In[9]:


print(f"Total files expected: {total_files}")
print(f"Total files present: {files_present}")
print(f"Only {total_files - files_present} files are missing.")
if total_files == 0:
    print("No files were expected, so percent present is undefined.")
else:
    print(
        "Percent of files present:", np.round(files_present / total_files * 100, 2), "%"
    )


# In[10]:


df = pd.DataFrame(featurization_rerun_dict)
df.to_csv(rerun_combinations_path, sep="\t", index=False)
df.head()


# In[11]:


df.groupby(["patient"]).count()
