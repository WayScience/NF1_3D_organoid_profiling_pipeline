#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import time
import urllib.request
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns

# import tifffile
import torch
import torch.nn as nn
import umap
from file_reading import *
from notebook_init_utils import bandicoot_check, init_notebook
from sammed3d_featurizer import call_whole_image_sammed3d_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import transforms as v2
from transformers import AutoModel

# In[2]:


def generate_umap_and_pca(
    feature_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply standard scalar and fit umap + PCA to the data

    Parameters
    ----------
    feature_df : pd.DataFrame
        The dataframe containing the features extracted from the images
    metadata_df : pd.DataFrame
        The dataframe containing the metadata associated with the images

        **Note please ensure that the index of the metadata_df matches the index of the feature_df
    Returns
    -------
    pd.DataFrame
        The metadata dataframe with added UMAP and PCA embeddings
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_df)

    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = umap_model.fit_transform(features_scaled)
    metadata_df["umap_1"] = umap_embeddings[:, 0]
    metadata_df["umap_2"] = umap_embeddings[:, 1]

    pca_model = PCA(n_components=2)
    pca_embeddings = pca_model.fit_transform(features_scaled)
    metadata_df["pca_1"] = pca_embeddings[:, 0]
    metadata_df["pca_2"] = pca_embeddings[:, 1]

    return metadata_df


# In[3]:


# Noise Injector transformation
class SaturationNoiseInjector(nn.Module):
    def __init__(self, low=200, high=255):
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel = x[0].clone()
        noise = torch.empty_like(channel).uniform_(self.low, self.high)
        mask = (channel == 255).float()
        noise_masked = noise * mask
        channel[channel == 255] = 0
        channel = channel + noise_masked
        x[0] = channel
        return x


# Self Normalize transformation
class PerImageNormalize(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,
            affine=False,
            track_running_stats=False,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.instance_norm(x)
        if x.shape[0] == 1:
            x = x.squeeze(0)
        return x


def featurize_2D_image_w_chami75(
    image_tensor: torch.Tensor, model: torch.nn.Module, device: torch.device
):
    # Bag of Channels (BoC) - process each channel independently
    with torch.no_grad():
        batch_feat = []
        image_tensor = image_tensor.to(device)

        for c in range(image_tensor.shape[1]):
            # Extract single channel: (N, C, H, W) -> (N, 1, H, W)
            # where:
            # N is batch size (1 in this case),
            # C is number of channels,
            # H and W are Y and X dimensions
            single_channel = image_tensor[:, c, :, :].unsqueeze(1)

            # Apply transforms
            single_channel = transform(single_channel.squeeze(1)).unsqueeze(1)

            # Extract features
            output = model.forward_features(single_channel)
            feat_temp = output["x_norm_clstoken"].cpu().detach().numpy()
            batch_feat.append(feat_temp)
    return batch_feat[0]


# load models
sam3dmed_checkpoint_url = (
    "https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"
)
sam3dmed_checkpoint_path = pathlib.Path("../models/sam-med3d-turbo.pth").resolve()
if not sam3dmed_checkpoint_path.exists():
    sam3dmed_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(sam3dmed_checkpoint_url, str(sam3dmed_checkpoint_path))

# Load model
device = "cuda"
model = AutoModel.from_pretrained("CaicedoLab/MorphEm", trust_remote_code=True)
model.to(device).eval()
# Define transforms
transform = v2.Compose(
    [
        SaturationNoiseInjector(),
        PerImageNormalize(),
        v2.Resize(size=(224, 224), antialias=True),
    ]
)


# In[4]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[5]:


root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm
image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[6]:


patients_file_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve()
patients = pd.read_csv(patients_file_path, header=None)[0].tolist()


# In[ ]:


warnings.filterwarnings(
    "ignore",
    message="input's size at dim=1 does not match num_features.*",
    category=UserWarning,
)
# loop through patients, then well fovs, get the image list
for patient_id in tqdm.tqdm(patients, desc="Processing patients", leave=True):
    input_dir = pathlib.Path(
        f"{image_base_dir}/data/{patient_id}/zstack_images/"
    ).resolve(strict=True)
    well_fovs = input_dir.glob("*")
    well_fovs = [f.stem for f in well_fovs if f.is_dir()]
    for well_fov_dir in tqdm.tqdm(
        well_fovs, desc=f"Processing well FOVs for patient {patient_id}", leave=False
    ):
        well_fov = pathlib.Path(well_fov_dir).name
        well_fov_dir = pathlib.Path(f"{input_dir}/{well_fov}").resolve(strict=True)
        # save path
        feature_save_path = pathlib.Path(
            f"{image_base_dir}/data/{patient_id}/whole_image_features/{well_fov}_whole_image_features.parquet"
        ).resolve()
        feature_save_path.parent.mkdir(exist_ok=True, parents=True)
        # if feature_save_path.exists():
        #     continue

        # get all well fovs for this patient
        images_to_process = {
            "patient": [],
            "well_fov": [],
            "2D_image": [],
            "3D_image": [],
            "channel": [],
        }
        images_to_load = [
            x for x in pathlib.Path(well_fov_dir).glob("*") if x.is_file()
        ]
        for image_file in images_to_load:
            try:
                image = read_zstack_image(image_file)
            except Exception as e:
                print(f"Error reading image {image_file}: {e}, skipping...")
                continue
            # load the middle slice
            mid_slice = image.shape[0] // 2
            image_mid = image[mid_slice, :, :]
            images_to_process["patient"].append(patient_id)
            images_to_process["well_fov"].append(well_fov)
            images_to_process["2D_image"].append(image_mid)
            images_to_process["3D_image"].append(image)
            images_to_process["channel"].append(f"{image_file.stem.split('_')[1]}")

        # Convert list of 2D images (Y, X) to tensor (B, C, Y, X)
        # where B is batch size (number of images),
        # C is number of channels (1 in this case),
        # Y and X are spatial dimensions
        # Stack images and add channel dimension
        if len(images_to_process["2D_image"]) == 0:
            print(
                f"No images found for patient {patient_id}, well FOV {well_fov}, skipping..."
            )
            continue
        images = torch.stack(
            [
                torch.tensor(img, dtype=torch.float32)
                for img in images_to_process["2D_image"]
            ]
        )
        try:
            volumes = torch.stack(
                [
                    torch.tensor(vol, dtype=torch.float32)
                    for vol in images_to_process["3D_image"]
                ]
            )
            # images is now (B, Y, X), add channel dimension -> (B, 1, Y, X)
            images = images.unsqueeze(1)
            # Replicate channel 3 times to get (B, 3, Y, X)
            images = images.repeat(1, 3, 1, 1)

            feature_dict = {
                "patient": [],
                "well_fov": [],
                "feature_name": [],
                "feature_value": [],
            }
            # featurize with SAM-Med3D and CHAMI75

            for image_index in range(volumes.shape[0]):
                channel_id = images_to_process["channel"][image_index]

                image = volumes[image_index].cpu().numpy()
                output_dict = call_whole_image_sammed3d_pipeline(
                    image=image,
                    SAMMed3D_model_path=str(sam3dmed_checkpoint_path),
                    feature_type="cls",
                )
                feature_dict["patient"].extend(
                    [f"{patient_id}"] * len(output_dict["feature_name"])
                )
                feature_dict["well_fov"].extend(
                    [f"{well_fov}"] * len(output_dict["feature_name"])
                )
                feature_dict["feature_name"].extend(
                    f"{channel_id}_{feature_name}"
                    for feature_name in output_dict["feature_name"]
                )
                feature_dict["feature_value"].extend(output_dict["value"])

            batch_feat = featurize_2D_image_w_chami75(images, model, device)
            for f_idx in range(batch_feat.shape[1]):
                feature_name = f"{channel_id}_CHAMI75_feature_{f_idx}"
                feature_value = batch_feat[image_index, f_idx]
                feature_dict["patient"].extend([f"{patient_id}"])
                feature_dict["well_fov"].extend([f"{well_fov}"])
                feature_dict["feature_name"].append(feature_name)
                feature_dict["feature_value"].append(feature_value)

            df = pd.DataFrame(feature_dict)
            df = (
                df.pivot_table(
                    index=["patient", "well_fov"],
                    columns="feature_name",
                    values="feature_value",
                )
                .reset_index()
                .rename_axis(None, axis=1)
            )
            df.to_parquet(
                feature_save_path,
                index=False,
            )
        except Exception as e:
            print(
                f"Error processing patient {patient_id}, well FOV {well_fov}: {e}, skipping..."
            )
            continue


# In[ ]:


end_time = time.time()
# get starting memory (cpu)
end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2

print(f"Time taken: {end_time - start_time} seconds")
print(f"Memory used: {end_mem - start_mem} MB")


# In[ ]:


patient_ids_file_path = pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve()
patient_ids = pd.read_csv(patient_ids_file_path, header=None)[0].tolist()


# In[ ]:


list_of_file_paths = []
for patient in patient_ids:
    whole_image_features_path = pathlib.Path(
        f"{image_base_dir}/data/{patient}/whole_image_features"
    ).resolve()
    if not whole_image_features_path.exists():
        continue

    list_of_file_paths.extend(list(whole_image_features_path.glob("*.parquet")))
print(f"Found {len(list_of_file_paths)} files to combine.")


# In[ ]:


# save the dfs
chami_features_save_path = pathlib.Path(f"../results/chami_features.parquet").resolve()
chami_features_save_path.parent.mkdir(exist_ok=True, parents=True)
sammed_features_save_path = pathlib.Path(
    f"../results/sammed_features.parquet"
).resolve()
all_features_save_path = pathlib.Path(f"../results/all_features.parquet").resolve()


# In[ ]:


if (
    chami_features_save_path.exists()
    and sammed_features_save_path.exists()
    and all_features_save_path.exists()
):
    df = pd.read_parquet(all_features_save_path)
    chami_df = pd.read_parquet(chami_features_save_path)
    sammed_df = pd.read_parquet(sammed_features_save_path)
else:
    df = pd.concat(
        [pd.read_parquet(file_path) for file_path in list_of_file_paths],
        ignore_index=True,
    )
    metadata_df = df[["patient", "well_fov"]]
    chami_df = df[[x for x in df.columns if "chami" in x.lower()]]
    chami_df = pd.concat([metadata_df, chami_df], axis=1)
    sammed_df = df[[x for x in df.columns if "sammed" in x.lower()]]
    sammed_df = pd.concat([metadata_df, sammed_df], axis=1)

    chami_df.to_parquet(chami_features_save_path)
    sammed_df.to_parquet(sammed_features_save_path)
    df.to_parquet(all_features_save_path)


# In[ ]:


feature_df = df.drop(columns=["patient", "well_fov"])
metadata_df = df[["patient", "well_fov"]]
sammed_features = df[[x for x in df.columns if "sammed" in x.lower()]]
chami_features = df[[x for x in df.columns if "chami" in x.lower()]]


all_features_projection_df = generate_umap_and_pca(feature_df, metadata_df.copy())
sammed_features_projection_df = generate_umap_and_pca(
    sammed_features, metadata_df.copy()
)
chami_features_projection_df = generate_umap_and_pca(chami_features, metadata_df.copy())


# In[ ]:


# plot PCA and UMAP for both features
# row 1: PCA, row 2: UMAP
# column 1: all features, column 2: sammed features, column 3: chami features
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
sns.scatterplot(
    data=all_features_projection_df,
    x="pca_1",
    y="pca_2",
    hue="patient",
    ax=axes[0, 0],
    palette="tab10",
)
axes[0, 0].set_title("PCA - All Features")
sns.scatterplot(
    data=sammed_features_projection_df,
    x="pca_1",
    y="pca_2",
    hue="patient",
    ax=axes[0, 1],
    palette="tab10",
)
axes[0, 1].set_title("PCA - Sammed Features")
sns.scatterplot(
    data=chami_features_projection_df,
    x="pca_1",
    y="pca_2",
    hue="patient",
    ax=axes[0, 2],
    palette="tab10",
)
axes[0, 2].set_title("PCA - Chami Features")
sns.scatterplot(
    data=all_features_projection_df,
    x="umap_1",
    y="umap_2",
    hue="patient",
    ax=axes[1, 0],
    palette="tab10",
)
axes[1, 0].set_title("UMAP - All Features")
sns.scatterplot(
    data=sammed_features_projection_df,
    x="umap_1",
    y="umap_2",
    hue="patient",
    ax=axes[1, 1],
    palette="tab10",
)
axes[1, 1].set_title("UMAP - Sammed Features")
sns.scatterplot(
    data=chami_features_projection_df,
    x="umap_1",
    y="umap_2",
    hue="patient",
    ax=axes[1, 2],
    palette="tab10",
)
axes[1, 2].set_title("UMAP - Chami Features")
plt.tight_layout()
plt.show()


# ## Drop the CQ1 data

# In[ ]:


df = df.loc[df["patient"] != "NF0037_T1_CQ1"]
feature_df = df.drop(columns=["patient", "well_fov"])
metadata_df = df[["patient", "well_fov"]]
sammed_features = df[[x for x in df.columns if "sammed" in x.lower()]]
chami_features = df[[x for x in df.columns if "chami" in x.lower()]]


all_features_projection_df = generate_umap_and_pca(feature_df, metadata_df.copy())
sammed_features_projection_df = generate_umap_and_pca(
    sammed_features, metadata_df.copy()
)
chami_features_projection_df = generate_umap_and_pca(chami_features, metadata_df.copy())


# In[ ]:


# plot PCA and UMAP for both features
# row 1: PCA, row 2: UMAP
# column 1: all features, column 2: sammed features, column 3: chami features
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
sns.scatterplot(
    data=all_features_projection_df,
    x="pca_1",
    y="pca_2",
    hue="patient",
    ax=axes[0, 0],
    palette="tab10",
)
axes[0, 0].set_title("PCA - All Features")
sns.scatterplot(
    data=sammed_features_projection_df,
    x="pca_1",
    y="pca_2",
    hue="patient",
    ax=axes[0, 1],
    palette="tab10",
)
axes[0, 1].set_title("PCA - Sammed Features")
sns.scatterplot(
    data=chami_features_projection_df,
    x="pca_1",
    y="pca_2",
    hue="patient",
    ax=axes[0, 2],
    palette="tab10",
)
axes[0, 2].set_title("PCA - Chami Features")
sns.scatterplot(
    data=all_features_projection_df,
    x="umap_1",
    y="umap_2",
    hue="patient",
    ax=axes[1, 0],
    palette="tab10",
)
axes[1, 0].set_title("UMAP - All Features")
sns.scatterplot(
    data=sammed_features_projection_df,
    x="umap_1",
    y="umap_2",
    hue="patient",
    ax=axes[1, 1],
    palette="tab10",
)
axes[1, 1].set_title("UMAP - Sammed Features")
sns.scatterplot(
    data=chami_features_projection_df,
    x="umap_1",
    y="umap_2",
    hue="patient",
    ax=axes[1, 2],
    palette="tab10",
)
axes[1, 2].set_title("UMAP - Chami Features")
plt.tight_layout()
plt.show()
