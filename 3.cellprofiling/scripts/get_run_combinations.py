#!/usr/bin/env python
# coding: utf-8

# # Get run combinations
#
# This notebook keeps the original behavior while reducing repetitive row-building code via a small helper function.

# In[15]:


import itertools
import multiprocessing
import os
import pathlib

import pandas as pd
import tomli
from image_analysis_3D.file_utils.notebook_init_utils import (
    bandicoot_check,
    init_notebook,
)

root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.auto as tqdm
else:
    import tqdm
bandicoot_mount_path = pathlib.Path(os.path.expanduser("~/mnt/bandicoot"))
bandicoot_mount_path = bandicoot_check(bandicoot_mount_path, root_dir)


# In[16]:


patient_id_file = pathlib.Path(f"{bandicoot_mount_path}/data/patient_IDs.txt").resolve(
    strict=True
)
patients = pd.read_csv(
    patient_id_file, header=None, names=["patient_id"]
).patient_id.tolist()

load_combinations_path = pathlib.Path(
    f"{root_dir}/3.cellprofiling/load_data/load_combinations.txt"
)
load_combinations_path.parent.mkdir(parents=True, exist_ok=True)
blocked_toml_path = pathlib.Path(f"{root_dir}/config/blocked_fovs/blocked_fovs.toml")
channel_mapping_file_path = pathlib.Path(
    f"{root_dir}/config/channel_mapping.toml"
).resolve(strict=True)
with open(channel_mapping_file_path, "rb") as f:
    channel_mapping_dict = tomli.load(f)
channel_n_compartment_mapping = channel_mapping_dict["channel_mapping"]

with open(blocked_toml_path, "rb") as f:
    blocked_fovs_dict = tomli.load(f)
patient_well_fov_block_df = pd.DataFrame(blocked_fovs_dict["blocked_patient_well_fovs"])
features = [
    "AreaSizeShape",
    "Colocalization",
    "Granularity",
    "Intensity",
    "Neighbors",
    "SAMMed3D",
    "Texture",
]
channels = ["DNA", "ER", "Mito", "AGP"]
compartments = ["Organoid", "Nuclei", "Cytoplasm", "Cell"]
channel_combinations = list(itertools.combinations(channels, 2))

# use the per-patient zstack_images directories to determine well_fovs dynamically
input_subdir_name = "zstack_images"


# In[17]:


rows = []
DEFAULT_SUBDIR_INPUT = "zstack_images"
DEFAULT_SUBDIR_MASK = "segmentation_masks"
DEFAULT_SUBDIR_OUTPUT = "extracted_features"


def add_row(
    patient: str,
    well_fov: str,
    feature: str,
    compartment: str,
    channel: str,
    processor_type: str,
    subdir_input: str = DEFAULT_SUBDIR_INPUT,
    subdir_mask: str = DEFAULT_SUBDIR_MASK,
    subdir_output: str = DEFAULT_SUBDIR_OUTPUT,
):
    """
    This helper function adds a row to the record.

    Parameters
    ----------
    patient : str
        The patient ID.
    well_fov : str
        The well and field of view identifier.
    feature : str
        The feature name.
    compartment : str
        The compartment name.
    channel : str
        The channel name.
    processor_type : str
        The processor type (e.g., "CPU", "GPU").
    subdir_input : _type_, optional
        The input subdirectory path, by default DEFAULT_SUBDIR_INPUT
    subdir_mask : _type_, optional
        The mask subdirectory path, by default DEFAULT_SUBDIR_MASK
    subdir_output : _type_, optional
        The output subdirectory path, by default DEFAULT_SUBDIR_OUTPUT
    """
    rows.append(
        {
            "patient": patient,
            "well_fov": well_fov,
            "feature": feature,
            "compartment": compartment,
            "channel": channel,
            "processor_type": processor_type,
            "subdir_input": subdir_input,
            "subdir_mask": subdir_mask,
            "subdir_output": subdir_output,
        }
    )


# In[18]:


patients = [
    "NF0014_T1",
    # "NF0014_T2",
    # "NF0016_T1",
    # "NF0018_T6",
    # "NF0021_T1",
    # "NF0030_T1",
]


# In[19]:


for patient in patients:
    patient_well_fovs = sorted(
        [
            path.name
            for path in (
                bandicoot_mount_path / "data" / patient / input_subdir_name
            ).glob("*")
            if path.is_dir()
        ]
    )
    if not patient_well_fovs:
        print(f"No well_fov directories found for patient {patient}; skipping.")
        continue

    for well_fov in patient_well_fovs:
        # check if the patient and well_fov combination is in the block list, and skip if so
        if (
            (patient_well_fov_block_df["patient"] == patient)
            & (patient_well_fov_block_df["well_fov"] == well_fov)
        ).any():
            print(
                f"Skipping blocked combination: patient {patient}, well_fov {well_fov}"
            )
            continue
        for feature in features:
            if feature == "Neighbors":
                add_row(
                    patient=patient,
                    well_fov=well_fov,
                    feature="Neighbors",
                    compartment="Nuclei",
                    channel="NoChannel",
                    processor_type="CPU",
                )
                continue

            for compartment in compartments:
                if feature == "AreaSizeShape":
                    add_row(
                        patient=patient,
                        well_fov=well_fov,
                        feature="AreaSizeShape",
                        compartment=compartment,
                        channel="NoChannel",
                        processor_type="CPU",
                    )
                elif feature == "Colocalization":
                    for ch1, ch2 in channel_combinations:
                        add_row(
                            patient=patient,
                            well_fov=well_fov,
                            feature="Colocalization",
                            compartment=compartment,
                            channel=f"{ch1}-{ch2}",
                            processor_type="CPU",
                        )
                else:
                    for channel in channels:
                        processor = "GPU" if feature == "SAMMed3D" else "CPU"
                        add_row(
                            patient=patient,
                            well_fov=well_fov,
                            feature=feature,
                            compartment=compartment,
                            channel=channel,
                            processor_type=processor,
                        )

        for channel in channels:
            for nucleocentric_feature in ["SAMMed3D", "CHAMMI75"]:
                add_row(
                    patient=patient,
                    well_fov=well_fov,
                    feature=nucleocentric_feature,
                    compartment="Nucleocentric",
                    channel=channel,
                    processor_type="GPU",
                )

df = pd.DataFrame(rows)
print(f"Total combinations: {df.shape[0]}")


# In[20]:


# Build paths with vectorized string ops
df["feature_file_path"] = (
    root_dir.as_posix()
    + "/data/"
    + df["patient"]
    + "/"
    + df["subdir_output"]
    + "/"
    + df["well_fov"]
    + "/"
    + df["compartment"]
    + "_"
    + df["channel"]
    + "_"
    + df["feature"]
    + "_"
    + df["processor_type"]
    + "_features.parquet"
)

# Faster existence check: scan each output directory once, then use set membership
existing_feature_files = set()
candidate_dir_df = df[["patient", "subdir_output", "well_fov"]].drop_duplicates()

for patient, subdir_output, well_fov in tqdm.tqdm(
    candidate_dir_df.itertuples(index=False, name=None),
    total=len(candidate_dir_df),
    desc="Scanning feature directories",
):
    feature_dir = root_dir / "data" / patient / subdir_output / well_fov
    if feature_dir.exists():
        existing_feature_files.update(
            p.as_posix() for p in feature_dir.glob("*_features.parquet") if p.is_file()
        )

df["feature_file_path_exists"] = df["feature_file_path"].isin(existing_feature_files)


# In[21]:


# If Nucleocentric has both CHAMMI75 and SAMMed3D, keep only CHAMMI75 entry
nucleocentric_df = (
    df[df["compartment"] == "Nucleocentric"]
    .loc[lambda d: d["feature"].isin(["SAMMed3D", "CHAMMI75"])]
    .sort_values(by=["patient", "well_fov", "channel", "feature"])
)

# Identify groups where both features are present
has_both_features = (
    nucleocentric_df.groupby(["patient", "well_fov", "channel"])["feature"]
    .transform("nunique")
    .eq(2)
)

# # Keep only CHAMMI75 rows from groups containing both features
# nucleocentric_df = nucleocentric_df[
#     has_both_features & nucleocentric_df["feature"].eq("CHAMMI75")
# ]

df = df[df["compartment"] != "Nucleocentric"]
df = pd.concat([df, nucleocentric_df], ignore_index=True)
all_df = df.copy()

original_number_of_feature_files = df.shape[0]
df = df[~df["feature_file_path_exists"]]
df.drop(columns=["feature_file_path", "feature_file_path_exists"], inplace=True)
df.sort_values(
    by=[
        # "feature",
        "patient",
        "well_fov",
        "compartment",
        "channel",
        "processor_type",
    ],
    inplace=True,
)
df.reset_index(drop=True, inplace=True)

print(
    f"{original_number_of_feature_files - df.shape[0]}/{original_number_of_feature_files}: {((original_number_of_feature_files - df.shape[0]) / original_number_of_feature_files) * 100:.2f}% of combinations have feature files that exist."
)


# In[22]:


df = df.loc[df["feature"] == "AreaSizeShape"]


# In[23]:


df.to_csv(load_combinations_path, sep="\t", index=False)
df.head()


# In[24]:


df.groupby(["feature"]).size().to_frame(name="count").reset_index()


# In[25]:


df.groupby(["patient", "feature"]).size().to_frame(name="count").reset_index().head(50)


# In[26]:


# Find patient well_fovs with complete feature-file coverage
complete_feature_count = all_df.groupby(["patient", "well_fov"], as_index=False).agg(
    feature_file_path_exists_count=("feature_file_path_exists", "sum"),
    expected_feature_count=("feature_file_path_exists", "size"),
)

complete_feature_count["completion_status"] = (
    complete_feature_count["feature_file_path_exists_count"]
    .eq(complete_feature_count["expected_feature_count"])
    .map({True: "Complete", False: "Incomplete"})
)

complete_feature_count.groupby(["patient", "completion_status"]).size().to_frame(
    name="count"
).reset_index()

complete_feature_count


# In[27]:


complete_feature_count.loc[complete_feature_count["completion_status"] == "Complete"]


# In[28]:


complete_feature_count.loc[complete_feature_count["completion_status"] == "Incomplete"]
