#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import os
import pathlib

import numpy as np
import pandas as pd
import tomli
from image_analysis_3D.featurization_utils.loading_classes import ImageSetLoader
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()


# In[2]:


bandicoot_mount_path = pathlib.Path(os.path.expanduser("~/mnt/bandicoot"))
bandicoot_mount_path = bandicoot_check(bandicoot_mount_path, root_dir)


# In[3]:


patient_id_file = pathlib.Path(f"{bandicoot_mount_path}/data/patient_IDs.txt").resolve(
    strict=True
)
patients = pd.read_csv(
    patient_id_file, header=None, names=["patient_id"]
).patient_id.tolist()
patients = ["NF0014_T1"]
load_combinations_path = pathlib.Path(
    f"{root_dir}/3.cellprofiling/load_data/load_combinations.txt"
)
load_combinations_path.parent.mkdir(parents=True, exist_ok=True)
channel_mapping_file_path = pathlib.Path(
    f"{root_dir}/config/channel_mapping.toml"
).resolve(strict=True)


# In[4]:


features = [
    "AreaSizeShape",
    "Colocalization",
    "Granularity",
    "Intensity",
    "Neighbors",
    "SAMMed3D",
    "Texture",
]


# In[5]:


# read in channel mapping
with open(channel_mapping_file_path, "rb") as f:
    channel_mapping_dict = tomli.load(f)
channel_n_compartment_mapping = channel_mapping_dict["channel_mapping"]


# In[6]:


channels = ["DNA", "ER", "Mito", "AGP"]
compartments = ["Organoid", "Nuclei", "Cytoplasm", "Cell"]


# In[7]:


output_dict = {
    "patient": [],
    "well_fov": [],
    "feature": [],
    "compartment": [],
    "channel": [],
    "processor_type": [],
    "subdir_input": [],
    "subdir_mask": [],
    "subdir_output": [],
}


# In[8]:


# get all channel combinations
channel_combinations = list(itertools.combinations(channels, 2))


# In[9]:


for patient in patients:
    # get the well_fov for each patient
    patient_well_fovs = list(
        pathlib.Path(f"{bandicoot_mount_path}/data/{patient}/zstack_images/").glob("*")
    )
    # for well_fov in patient_well_fovs:
    #     well_fov = well_fov.name
    for well_fov in ["C2-1", "C2-2", "C4-1", "C4-2", "D5-2", "E5-2"]:
        for feature in features:
            if feature == "Neighbors":
                output_dict["patient"].append(patient)
                output_dict["well_fov"].append(well_fov)
                output_dict["feature"].append("Neighbors")
                output_dict["compartment"].append("Nuclei")
                output_dict["channel"].append("NoChannel")
                output_dict["processor_type"].append("CPU")
                output_dict["subdir_input"].append("zstack_images")
                output_dict["subdir_mask"].append("segmentation_masks")
                output_dict["subdir_output"].append("extracted_features")
            for compartment in compartments:
                if feature == "AreaSizeShape":
                    output_dict["patient"].append(patient)
                    output_dict["well_fov"].append(well_fov)
                    output_dict["feature"].append("AreaSizeShape")
                    output_dict["compartment"].append(compartment)
                    output_dict["channel"].append("NoChannel")
                    output_dict["processor_type"].append("CPU")
                    output_dict["subdir_input"].append("zstack_images")
                    output_dict["subdir_mask"].append("segmentation_masks")
                    output_dict["subdir_output"].append("extracted_features")
                elif feature == "Colocalization":
                    for channel in channel_combinations:
                        output_dict["patient"].append(patient)
                        output_dict["well_fov"].append(well_fov)
                        output_dict["feature"].append("Colocalization")
                        output_dict["compartment"].append(compartment)
                        output_dict["channel"].append(channel[0] + "-" + channel[1])
                        output_dict["processor_type"].append("CPU")
                        output_dict["subdir_input"].append("zstack_images")
                        output_dict["subdir_mask"].append("segmentation_masks")
                        output_dict["subdir_output"].append("extracted_features")
                else:
                    for channel in channels:
                        if (
                            feature != "Neighbors"
                            and feature != "AreaSizeShape"
                            and feature != "Colocalization"
                        ):
                            if feature == "Granularity":
                                output_dict["patient"].append(patient)
                                output_dict["well_fov"].append(well_fov)
                                output_dict["feature"].append(feature)
                                output_dict["compartment"].append(compartment)
                                output_dict["channel"].append(channel)
                                output_dict["processor_type"].append("CPU")
                                output_dict["subdir_input"].append("zstack_images")
                                output_dict["subdir_mask"].append("segmentation_masks")
                                output_dict["subdir_output"].append(
                                    "extracted_features"
                                )
                            elif feature == "Intensity":
                                output_dict["patient"].append(patient)
                                output_dict["well_fov"].append(well_fov)
                                output_dict["feature"].append(feature)
                                output_dict["compartment"].append(compartment)
                                output_dict["channel"].append(channel)
                                output_dict["processor_type"].append("CPU")
                                output_dict["subdir_input"].append("zstack_images")
                                output_dict["subdir_mask"].append("segmentation_masks")
                                output_dict["subdir_output"].append(
                                    "extracted_features"
                                )
                            elif feature == "Texture":
                                output_dict["patient"].append(patient)
                                output_dict["well_fov"].append(well_fov)
                                output_dict["feature"].append(feature)
                                output_dict["compartment"].append(compartment)
                                output_dict["channel"].append(channel)
                                output_dict["processor_type"].append("CPU")
                                output_dict["subdir_input"].append("zstack_images")
                                output_dict["subdir_mask"].append("segmentation_masks")
                                output_dict["subdir_output"].append(
                                    "extracted_features"
                                )
                            elif feature == "SAMMed3D":
                                output_dict["patient"].append(patient)
                                output_dict["well_fov"].append(well_fov)
                                output_dict["feature"].append(feature)
                                output_dict["compartment"].append(compartment)
                                output_dict["channel"].append(channel)
                                output_dict["processor_type"].append("GPU")
                                output_dict["subdir_input"].append("zstack_images")
                                output_dict["subdir_mask"].append("segmentation_masks")
                                output_dict["subdir_output"].append(
                                    "extracted_features"
                                )
                            else:
                                raise ValueError(f"Unknown feature: {feature}")
        for channel in channels:
            for nulceo_centric_feature in ["SAMMed3D", "CHAMMI75"]:
                output_dict["patient"].append(patient)
                output_dict["well_fov"].append(well_fov)
                output_dict["feature"].append(nulceo_centric_feature)
                output_dict["compartment"].append("Nucleocentric")
                output_dict["channel"].append(channel)
                output_dict["processor_type"].append("GPU")
                output_dict["subdir_input"].append("zstack_images")
                output_dict["subdir_mask"].append("segmentation_masks")
                output_dict["subdir_output"].append("extracted_features")


# For each well fov there should be the following number of files:
# Of course this depends on if both CPU and GPU versions are run, but the CPU version is always run.
#
# No BF here!
#
# | Feature Type | No. Compartments | No. Channels | Total No. Files |
# |--------------|------------------|---------------|-----------------|
# | AreaSizeShape | 4 | 1 | 4 |
# | Colocalization | 4 | 6 | 24 |
# | Granularity | 4 | 4 | 16 |
# | Intensity | 4 | 4 | 16 |
# | Neighbors | 1 | 1 | 1 |
# | SAMMed3D | 4 | 4 | 16 |
# | Texture | 4 | 4 | 16 |
# | Nucleocentric + SAMMed3D | 0 | 4 | 4 |
# | Nulceocentric + CHAMMI75 | 0 | 4 | 4 |
#
# Total no. files per well fov = 101
#
#

# In[10]:


df = pd.DataFrame(output_dict)
print(
    f"Total combinations: {df.shape[0]}"
)  # 582 when counting nucleocentric features as grouped


# In[11]:


# reorder columns
df = df[
    [
        "patient",
        "well_fov",
        "compartment",
        "channel",
        "feature",
        "processor_type",
        "subdir_input",
        "subdir_mask",
        "subdir_output",
    ]
]


# In[12]:


df["feature_file_path"] = df.apply(
    lambda row: (
        bandicoot_mount_path
        / "data"
        / row["patient"]
        / row["subdir_output"]
        / row["well_fov"]
        / f"{row['compartment']}_{row['channel']}_{row['feature']}_{row['processor_type']}_features.parquet"
    ),
    axis=1,
)
df["feature_file_path_exists"] = df["feature_file_path"].apply(
    lambda x: pathlib.Path(x).exists()
)
# if wellfov for nulceocentric exists for both chammi75 and sammed3d
# we drop the sammed3d entry since we compute both in the same run
nucleocentric_df = df[df["compartment"] == "Nucleocentric"]
nucleocentric_df = nucleocentric_df[
    nucleocentric_df["feature"].isin(["SAMMed3D", "CHAMMI75"])
]
nucleocentric_df = nucleocentric_df.sort_values(
    by=["patient", "well_fov", "channel", "feature"]
)
# find where there are two entries for the same patient, well_fov, channel
# but different feature (SAMMed3D and CHAMMI75)
# remove the SAMMed3D entry in those cases
nucleocentric_df = (
    nucleocentric_df.groupby(["patient", "well_fov", "channel"])
    .filter(lambda x: x["feature"].nunique() == 2)
    .groupby(["patient", "well_fov", "channel"])
    .apply(lambda x: x[x["feature"] != "CHAMMI75"])
    .reset_index()
    .drop(columns=["level_3"])
)

# drop all nucleocentric entries from the original df and add back the filtered nucleocentric_df
df = df[df["compartment"] != "Nucleocentric"]
df = pd.concat([df, nucleocentric_df], ignore_index=True)
# filter by feature files that do not exist
original_number_of_feature_files = df.shape[0]
df = df[~df["feature_file_path_exists"]]

df.drop(columns=["feature_file_path", "feature_file_path_exists"], inplace=True)
# sort by patient, well_fov, feature, compartment, channel, processor_type
df.sort_values(
    by=["feature", "patient", "well_fov", "compartment", "channel", "processor_type"],
    inplace=True,
)

print(
    f"{original_number_of_feature_files - df.shape[0]}/{original_number_of_feature_files}: {((original_number_of_feature_files - df.shape[0]) / original_number_of_feature_files) * 100:.2f}% of combinations have feature files that exist."
)


# In[13]:


# write to a txt file with each row as a combination
# each column is a feature of the combination
df.to_csv(load_combinations_path, sep="\t", index=False)


# In[14]:


df.head()


# In[15]:


df.groupby(["patient"]).count()
