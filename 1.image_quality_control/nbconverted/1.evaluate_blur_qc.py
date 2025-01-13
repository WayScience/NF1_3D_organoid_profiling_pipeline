#!/usr/bin/env python
# coding: utf-8

# # Whole image quality control metric evaluation - Blur
# 
# In this notebook, we will use the outputted QC metrics per image (every z-slice per channel) to start working on developing thresholds using z-score to flag images during CellProfiler processing.
# We are loading in the results from the preliminary data (across three patients) to attempt to develop generalizable thresholds.
# This data is 3D, so we are decide if it make sense to remove a whole organoid based on if one z-slice fails.
# 
# ## Blurry image detection
# 
# For detecting poor quality images based on blur, we use the feature `PowerLogLogSlope`, where more negative values indicate blurry images.
# We first create distribution plots per plates and per channel to evaluate if the distributions across channels are different.
# We will use this to determine if we process the data with all channels combined or separately.
# 
# We will use a method called `coSMicQC`, which takes a feature of interest and detect outliers based on z-scoring and how far from the mean that outliers will be.

# ## Import libraries

# In[1]:


import pathlib
import pandas as pd
import cosmicqc
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import re


# ## Set paths and variables

# In[2]:


# Set the threshold for identifying outliers with z-scoring for all metrics (# of standard deviations away from mean)
threshold_z = 2

# Directory for figures to be outputted
figure_dir = pathlib.Path("../qc_figures")
figure_dir.mkdir(exist_ok=True)

# Directory containing the QC results
qc_results_dir = pathlib.Path("../qc_results")

# Find all Image.csv files for all plates using glob
image_csv_paths = qc_results_dir.glob("*/Image.csv")

# Path to the template pipeline file to update with proper thresholds for flagging
pipeline_path = pathlib.Path("../pipeline/template_flag_pipeline.cppipe")


# ## Load in QC results per plate and combine

# In[3]:


# Define prefixes for columns to select
prefixes = (
    "Metadata",
    "FileName",
    "PathName",
    "ImageQuality_PowerLogLogSlope",
)

# Load and concatenate the data for all plates
qc_dfs = []
for path in image_csv_paths:
    # Load only the required columns by filtering columns with specified prefixes
    plate_df = pd.read_csv(path, usecols=lambda col: col.startswith(prefixes))
    qc_dfs.append(plate_df)

# Concatenate all plate data into a single dataframe
concat_qc_df = pd.concat(qc_dfs, ignore_index=True)

print(concat_qc_df.shape)
concat_qc_df.head(2)


# ## Generate blur density distribution plot

# In[4]:


# Step 1: Select only the columns containing "PowerLogLogSlope" and keep Metadata_Plate
relevant_columns = [col for col in concat_qc_df.columns if "PowerLogLogSlope" in col]
relevant_columns.insert(0, "Metadata_Plate")

# Filter the dataframe
filtered_df = concat_qc_df[relevant_columns]

# Step 2: Reshape the dataframe into a long format
long_df = filtered_df.melt(
    id_vars=["Metadata_Plate"], var_name="Channel", value_name="PowerLogLogSlope"
)

# Step 3: Clean up the channel names
long_df["Channel"] = long_df["Channel"].str.replace(
    "ImageQuality_PowerLogLogSlope_", ""
)

# Drop rows with missing or non-numeric values
long_df = long_df.dropna(subset=["PowerLogLogSlope"])

# Define the custom color palette
custom_palette = {
    "DNA": "blue",
    "ER": "green",
    "AGP": "red",
    "Mito": "purple",
    "Brightfield": "orange",
}

# Step 4: Create the facet grid
g = sns.displot(
    data=long_df,
    x="PowerLogLogSlope",
    hue="Channel",
    palette=custom_palette,  # Use the custom palette
    col="Metadata_Plate",
    kind="kde",
    fill=True,
    alpha=0.5,
    col_wrap=3,  # Adjust for 3 columns per row
)

# Customize the plot
g.set_titles("Plate: {col_name}")
g.set_axis_labels("PowerLogLogSlope", "Density")
g.tight_layout()

# Show the plot
plt.show()


# Given the plot, we can conclude that the distributions across channels are very different.
# We will process each channel independently and determine thresholds per channel.
# The distributions across plate look similar, so we will not be determining per plate.

# ## Detect blur in DNA channel

# In[5]:


# Identify metadata columns (columns that do not start with 'ImageQuality')
metadata_columns = [
    col for col in concat_qc_df.columns if not col.startswith("ImageQuality")
]

# Find large nuclei outliers for the current plate
blur_DNA_outliers = cosmicqc.find_outliers(
    df=concat_qc_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_DNA": -2,
    },
)

pd.DataFrame(blur_DNA_outliers)


# In[6]:


# Combine PathName and FileName columns to construct full paths for DNA
blur_DNA_outliers["Full_Path_DNA"] = (
    blur_DNA_outliers["PathName_DNA"] + "/" + blur_DNA_outliers["FileName_DNA"]
)

# Create a figure to display images
plt.figure(figsize=(15, 5))

# Loop through the first 3 rows of the blur_DNA_outliers dataframe and display each image
for idx, row in enumerate(blur_DNA_outliers.itertuples(), start=1):
    if idx > 3:  # Only display the first 3 images
        break
    image_path = row.Full_Path_DNA
    # Format the metadata title based on your desired structure
    metadata_title = f"{row.Metadata_Plate}_{row.Metadata_Well}-{int(row.Metadata_Site)}_{row.Metadata_Zslice}"

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        continue
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Convert from BGR to RGB for proper display

    # Add the image to the plot
    plt.subplot(1, 3, idx)  # Use idx for subplot placement
    plt.imshow(image)
    plt.title(metadata_title)  # Set the formatted metadata as the title
    plt.axis("off")

# Show the plot
plt.tight_layout()
plt.show()


# ## Detect blur in Mito channel

# In[7]:


# Identify metadata columns (columns that do not start with 'ImageQuality')
metadata_columns = [
    col for col in concat_qc_df.columns if not col.startswith("ImageQuality")
]

# Find large nuclei outliers for the current plate
blur_Mito_outliers = cosmicqc.find_outliers(
    df=concat_qc_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_Mito": -3,
    },
)

pd.DataFrame(blur_Mito_outliers)

blur_Mito_outliers = blur_Mito_outliers.sort_values(
    by="ImageQuality_PowerLogLogSlope_Mito", ascending=True
)

blur_Mito_outliers.head()


# In[8]:


# Combine PathName and FileName columns to construct full paths for Mito
blur_Mito_outliers["Full_Path_Mito"] = (
    blur_Mito_outliers["PathName_Mito"] + "/" + blur_Mito_outliers["FileName_Mito"]
)

# Group by Plate, Well, and Site to ensure uniqueness
unique_groups = blur_Mito_outliers.groupby(
    ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]
)
print(len(unique_groups))

# Randomly sample one row per group
unique_samples = unique_groups.apply(lambda group: group.sample(n=1, random_state=0))

# Reset the index for convenience
unique_samples = unique_samples.reset_index(drop=True)

# Further randomly select 3 unique images
if len(unique_samples) < 3:
    print("Not enough unique Plate-Well-Site combinations for the requested images.")
else:
    selected_images = unique_samples.sample(n=3, random_state=0)

# Create a figure to display images
plt.figure(figsize=(15, 5))

# Loop through the selected image paths and display each image
for idx, row in enumerate(selected_images.itertuples(), start=1):
    image_path = row.Full_Path_Mito
    # Format the metadata title based on your desired structure
    metadata_title = f"{row.Metadata_Plate}_{row.Metadata_Well}-{int(row.Metadata_Site)}_{row.Metadata_Zslice}"

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        continue
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Convert from BGR to RGB for proper display

    # Add the image to the plot
    plt.subplot(1, 3, idx)  # Use idx for subplot placement
    plt.imshow(image)
    plt.title(metadata_title)  # Set the formatted metadata as the title
    plt.axis("off")

# Show the plot
plt.tight_layout()
plt.show()


# ## Detect blur in ER channel

# In[9]:


# Identify metadata columns (columns that do not start with 'ImageQuality')
metadata_columns = [
    col for col in concat_qc_df.columns if not col.startswith("ImageQuality")
]

# Find large nuclei outliers for the current plate
blur_er_outliers = cosmicqc.find_outliers(
    df=concat_qc_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_ER": -2,
    },
)

pd.DataFrame(blur_er_outliers).head()


# In[10]:


# Combine PathName and FileName columns to construct full paths
blur_er_outliers["Full_Path_ER"] = (
    blur_er_outliers["PathName_ER"] + "/" + blur_er_outliers["FileName_ER"]
)

# Group by Plate, Well, and Site to ensure uniqueness
unique_groups = blur_er_outliers.groupby(
    ["Metadata_Plate", "Metadata_Well", "Metadata_Site"],
)
print(len(unique_groups))

# Randomly sample one row per group
unique_samples = unique_groups.apply(lambda group: group.sample(n=1, random_state=0))

# Reset the index for convenience
unique_samples = unique_samples.reset_index(drop=True)

# Further randomly select 3 unique images
if len(unique_samples) < 3:
    print("Not enough unique Plate-Well-Site combinations for the requested images.")
else:
    selected_images = unique_samples.sample(n=3, random_state=0)

# Create a figure to display images
plt.figure(figsize=(15, 5))

# Loop through the selected image paths and display each image
for idx, row in enumerate(
    selected_images.itertuples(), start=1
):  # Enumerate for subplot indexing
    image_path = row.Full_Path_ER
    metadata_title = f"{row.Metadata_Plate}_{row.Metadata_Well}-{int(row.Metadata_Site)}_{row.Metadata_Zslice}"

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        continue
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Convert from BGR to RGB for proper display

    # Add the image to the plot
    plt.subplot(1, 3, idx)  # Use idx for subplot placement
    plt.imshow(image)
    plt.title(metadata_title)
    plt.axis("off")

# Show the plot
plt.tight_layout()
plt.show()


# ## Detect blur in AGP channel

# In[11]:


# Identify metadata columns (columns that do not start with 'ImageQuality')
metadata_columns = [
    col for col in concat_qc_df.columns if not col.startswith("ImageQuality")
]

# Find large nuclei outliers for the current plate
blur_agp_outliers = cosmicqc.find_outliers(
    df=concat_qc_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_AGP": -2.25,
    },
)

pd.DataFrame(blur_agp_outliers).head()


# In[12]:


# Combine PathName and FileName columns to construct full paths
blur_agp_outliers["Full_Path_AGP"] = (
    blur_agp_outliers["PathName_AGP"] + "/" + blur_agp_outliers["FileName_AGP"]
)

# Group by Plate, Well, and Site to ensure uniqueness
unique_groups = blur_agp_outliers.groupby(
    ["Metadata_Plate", "Metadata_Well", "Metadata_Site"],
)
print(len(unique_groups))

# Randomly sample one row per group
unique_samples = unique_groups.apply(lambda group: group.sample(n=1, random_state=0))

# Reset the index for convenience
unique_samples = unique_samples.reset_index(drop=True)

# Further randomly select 3 unique images
if len(unique_samples) < 3:
    print("Not enough unique Plate-Well-Site combinations for the requested images.")
else:
    selected_images = unique_samples.sample(n=3, random_state=0)

# Create a figure to display images
plt.figure(figsize=(15, 5))

# Loop through the selected image paths and display each image
for idx, row in enumerate(
    selected_images.itertuples(), start=1
):  # Enumerate for subplot indexing
    image_path = row.Full_Path_AGP
    metadata_title = f"{row.Metadata_Plate}_{row.Metadata_Well}-{int(row.Metadata_Site)}_{row.Metadata_Zslice}"

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        continue
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Convert from BGR to RGB for proper display

    # Add the image to the plot
    plt.subplot(1, 3, idx)  # Use idx for subplot placement
    plt.imshow(image)
    plt.title(metadata_title)
    plt.axis("off")

# Show the plot
plt.tight_layout()
plt.show()


# ## Detect blur in Brightfield channel

# In[13]:


# Identify metadata columns (columns that do not start with 'ImageQuality')
metadata_columns = [
    col for col in concat_qc_df.columns if not col.startswith("ImageQuality")
]

# Find large nuclei outliers for the current plate
blur_brightfield_outliers = cosmicqc.find_outliers(
    df=concat_qc_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_Brightfield": -3,
    },
)

pd.DataFrame(blur_brightfield_outliers).head()


# In[14]:


# Combine PathName and FileName columns to construct full paths
blur_brightfield_outliers["Full_Path_Brightfield"] = (
    blur_brightfield_outliers["PathName_Brightfield"] + "/" + blur_brightfield_outliers["FileName_Brightfield"]
)

# Group by Plate, Well, and Site to ensure uniqueness
unique_groups = blur_brightfield_outliers.groupby(
    ["Metadata_Plate", "Metadata_Well", "Metadata_Site"],
)
print(len(unique_groups))

# Randomly sample one row per group
unique_samples = unique_groups.apply(lambda group: group.sample(n=1, random_state=0))

# Reset the index for convenience
unique_samples = unique_samples.reset_index(drop=True)

# Further randomly select 3 unique images
if len(unique_samples) < 3:
    print("Not enough unique Plate-Well-Site combinations for the requested images.")
else:
    selected_images = unique_samples.sample(n=3, random_state=0)

# Create a figure to display images
plt.figure(figsize=(15, 5))

# Loop through the selected image paths and display each image
for idx, row in enumerate(
    selected_images.itertuples(), start=1
):  # Enumerate for subplot indexing
    image_path = row.Full_Path_Brightfield
    metadata_title = f"{row.Metadata_Plate}_{row.Metadata_Well}-{int(row.Metadata_Site)}_{row.Metadata_Zslice}"

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image at {image_path}")
        continue
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Convert from BGR to RGB for proper display

    # Add the image to the plot
    plt.subplot(1, 3, idx)  # Use idx for subplot placement
    plt.imshow(image)
    plt.title(metadata_title)
    plt.axis("off")

# Show the plot
plt.tight_layout()
plt.show()


# ## Create parquet file with each plate/well/site combos and boolean for pass/fail blur per channel

# In[15]:


# Combine all blur outliers dataframes into a single dataframe
blur_outliers = pd.concat(
    [blur_DNA_outliers, blur_Mito_outliers, blur_agp_outliers, blur_brightfield_outliers, blur_er_outliers],
    keys=['DNA', 'Mito', 'AGP', 'Brightfield', 'ER'],
    names=['Channel']
).reset_index(level='Channel')

# Create a new dataframe with unique combinations of Metadata_Plate, Metadata_Well, and Metadata_Site
unique_combos = concat_qc_df[['Metadata_Plate', 'Metadata_Well', 'Metadata_Site']].drop_duplicates()

# Initialize columns for each channel with False
for channel in ['DNA', 'Mito', 'AGP', 'Brightfield', 'ER']:
    unique_combos[f'Blurry_{channel}'] = False

# Flag the combos for blur detection
for channel in ['DNA', 'Mito', 'AGP', 'Brightfield', 'ER']:
    blur_combos = blur_outliers[blur_outliers['Channel'] == channel][['Metadata_Plate', 'Metadata_Well', 'Metadata_Site']].drop_duplicates()
    unique_combos.loc[
        unique_combos.set_index(['Metadata_Plate', 'Metadata_Well', 'Metadata_Site']).index.isin(blur_combos.set_index(['Metadata_Plate', 'Metadata_Well', 'Metadata_Site']).index),
        f'Blurry_{channel}'
    ] = True

# Reset the index on the unique combos dataframe
unique_combos = unique_combos.reset_index(drop=True)

# Save the unique_combos dataframe to a parquet file
unique_combos.to_parquet(qc_results_dir / "all_plates_qc_results.parquet")

# Print the number of rows with at least one Blurry column set to True
num_blurry_rows = unique_combos.loc[:, 'Blurry_DNA':'Blurry_ER'].any(axis=1).sum()
print(f"Number of organoids detected as poor quality due to blur (in any channel): {num_blurry_rows}")

# Calculate and print the percentage of organoids detected as containing blur
percentage_blurry = (num_blurry_rows / len(unique_combos)) * 100
print(f"Percentage of organoids detected as poor quality due to blur: {percentage_blurry:.2f}%")

# Display the resulting dataframe
print(unique_combos.shape)
unique_combos.head()

