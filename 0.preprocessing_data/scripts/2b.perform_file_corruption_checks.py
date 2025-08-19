#!/usr/bin/env python
# coding: utf-8

# # Check the files for corrupted files, or files that were not copied over correctly.

# ## Import libraries

# In[ ]:


import pathlib
import pprint

import numpy as np
import tifffile as tiff

# check if in a jupyter notebook

try:
    cfg = get_ipython().config
    in_notebook = True
except NameError:
    in_notebook = False

    # Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[ ]:

# Create max projections per well-fov-patient to evaluate the size of the files (small = corrupted)
def max_z_projection(patient: str, well_fov: str) -> None:
    """
    Create a maximum intensity projection of the z-stack images for a given patient and well_fov.
    Args:
        patient (str): The patient identifier.
        well_fov (str): The well and field of view identifier, formatted as "well_fov".
    Returns:
        None: The function saves the maximum intensity projection images to the specified output directory.
    Raises:
        FileNotFoundError: If the specified raw images directory does not exist.
        ValueError: If no images are found for the specified channels.
    """
    # Image extensions that we are looking to copy
    image_extensions = {".tif", ".tiff"}
    channel_names = ["405", "488", "555", "640", "TRANS"]
    # make a dictionary that contains a list for each channel name, storing both filepath and filename
    channel_images = {
        channel_name: {"filename": [], "filepath": []} for channel_name in channel_names
    }
    raw_images_path = pathlib.Path(f"{root_dir}/data/{patient}/raw_images").resolve(
        strict=True
    )
    zstack_output_path = pathlib.Path(
        f"{root_dir}/data/{patient}/zstack_images"
    ).resolve(strict=True)
    well_fov_dir = raw_images_path / well_fov
    channel_images = {
        channel_name: {"filename": [], "filepath": []} for channel_name in channel_names
    }
    for filename in well_fov_dir.glob("*"):
        if filename.suffix in image_extensions:
            for channel_name in channel_names:
                if channel_name in filename.name:
                    channel_images[channel_name]["filepath"].append(filename)

    for channel_name in tqdm.tqdm(
        channel_names, desc=f"Processing channels in {well_fov_dir.name}", leave=False
    ):
        channel_images[channel_name]["filepath"] = sorted(
            channel_images[channel_name]["filepath"]
        )
        if not channel_images[channel_name]["filepath"]:
            print(
                f"No files found for channel {channel_name} in {well_dir}. Skipping..."
            )
            continue

        images_to_stack = np.array(
            [
                tiff.imread(filepath)
                for filepath in channel_images[channel_name]["filepath"]
            ]
        )
        filepath = channel_images[channel_name]["filepath"][0]
        well = str(filepath.parent).split("/")[-1]
        output_path = zstack_output_path / f"{well}" / f"{well}_{channel_name}.tif"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        print(
            f"Saving max projection for {channel_name} channel in {well_fov} to {output_path}"
        )
        tiff.imwrite(output_path, images_to_stack)


# ## Set input and output directories

# In[ ]:


list_of_patients = [  # will be in a separate file in the future
    "NF0014",
    "NF0016",
    "NF0018",
    "NF0021",
    "NF0030",
    "NF0040",
    "SARCO219",
    "SARCO361",
]


# In[ ]:


patient_input_dict = {}
for patient in list_of_patients:
    patient_input_dict[patient] = {
        "raw_images": pathlib.Path(f"{root_dir}/data/{patient}/raw_images").resolve(),
        "zstack_output": pathlib.Path(
            f"{root_dir}/data/{patient}/zstack_images"
        ).resolve(),
    }
pprint.pprint(patient_input_dict)


# ## Create list of the well-site folders
# Create a list of the well-site folders in the stack directory.
# Then loop through each well-site folder and create a list of the channel images.
# Then find (if any) corrupted files in the channel images.
# This is done by checking if the size of the channel images for a given well-fov is the same as the size of the channel images for the other well-fovs.
# If the size is different, then the file is corrupted.

# In[ ]:


patient_well_fovs_to_fix = []
# loop through patients, well_fovs, and each channel
# outer loop through patients
for patient in tqdm.tqdm(patient_input_dict.keys(), desc="Processing patients"):
    input_dirs = [
        x for x in patient_input_dict[patient]["zstack_output"].iterdir() if x.is_dir()
    ]
    input_dirs.sort()

    for well_dir in tqdm.tqdm(
        input_dirs, desc=f"Processing wells for patient {patient}", leave=False
    ):
        # get the list of files in the well directory
        files = list(well_dir.iterdir())
        files.sort()
        # make sure there are 5 files in the well directory
        if len(files) != 5:
            patient_well_fovs_to_fix.append(f"{patient} {well_dir.name}")
        # check the file size of each file
        file_sizes = [file.stat().st_size for file in files]

        # check if the file sizes are all the same
        if len(set(file_sizes)) != 1:
            patient_well_fovs_to_fix.append(f"{patient} {well_dir.name}")
print(
    f"""Need to check and fix a total of {len(patient_well_fovs_to_fix)} patient well_fovs:"""
)
pprint.pprint(patient_well_fovs_to_fix)


# ## With the list of corrupted files, recreate the z-stack images
# This is the point where the z-stack images are created from the individual z-slice images for each FOV per well.

# In[ ]:


for patient_well_fov in patient_well_fovs_to_fix:
    patient, well_fov = patient_well_fov.split()
    max_z_projection(patient, well_fov)
