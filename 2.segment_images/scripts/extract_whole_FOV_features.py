#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import os
import pathlib
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import psutil
import tifffile
import torch
import torch.nn as nn
from arg_parsing_utils import check_for_missing_args, parse_args
from file_reading import *
from notebook_init_utils import bandicoot_check, init_notebook
from sammed3d_featurizer import call_whole_image_sammed3d_pipeline
from torchvision import transforms as v2
from transformers import AutoModel

# In[2]:


start_time = time.time()
# get starting memory (cpu)
start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2


# In[3]:


root_dir, in_notebook = init_notebook()

image_base_dir = bandicoot_check(
    pathlib.Path(os.path.expanduser("~/mnt/bandicoot")).resolve(), root_dir
)


# In[4]:


if not in_notebook:
    args = parse_args()
    well_fov = args["well_fov"]
    patient = args["patient"]
    input_subparent_name = args["input_subparent_name"]
    check_for_missing_args(
        well_fov=well_fov,
        patient=patient,
        input_subparent_name=input_subparent_name,
    )
else:
    print("Running in a notebook")
    patient = "NF0014_T1"
    well_fov = "C4-2"
    input_subparent_name = "zstack_images"


input_dir = pathlib.Path(
    f"{image_base_dir}/data/{patient}/{input_subparent_name}/{well_fov}"
).resolve(strict=True)
# save path
feature_save_path = pathlib.Path(
    f"{image_base_dir}/data/{patient}/whole_image_features/{well_fov}_whole_image_features.parquet"
).resolve()
feature_save_path.parent.mkdir(exist_ok=True, parents=True)


# In[5]:


if feature_save_path.exists():
    print(f"Features already exist at {feature_save_path}, skipping...")
    sys.exit(0)


# In[ ]:


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


# In[ ]:


# get all well fovs for this patient

images_to_process = {"patient": [], "well_fov": [], "image": [], "channel": []}

images_to_load = [x for x in input_dir.glob("*.tif")]
for image_file in images_to_load:
    image = read_zstack_image(image_file)
    # load the middle slice
    mid_slice = image.shape[0] // 2
    image_mid = image[mid_slice, :, :]
    images_to_process["patient"].append(patient)
    images_to_process["well_fov"].append(well_fov)
    images_to_process["image"].append(image_mid)
    images_to_process["channel"].append(f"{image_file.stem.split('_')[1]}")

# Convert list of 2D images (H, W) to tensor (B, C, H, W)
# where B is batch size (number of images),
# C is number of channels (1 in this case),
# H and W are Y and X dimensions
# Stack images and add channel dimension
images = torch.stack(
    [torch.tensor(img, dtype=torch.float32) for img in images_to_process["image"]]
)
# images is now (B, H, W), add channel dimension -> (B, 1, H, W)
images = images.unsqueeze(1)
# Replicate channel 3 times to get (B, 3, H, W)
images = images.repeat(1, 3, 1, 1)


# In[ ]:


feature_dict = {
    "patient": [],
    "well_fov": [],
    "feature_name": [],
    "feature_value": [],
}

for image_index in range(images.shape[0]):
    channel_id = images_to_process["channel"][image_index]

    image = images[image_index].cpu().numpy()
    output_dict = call_whole_image_sammed3d_pipeline(
        image=image,
        SAMMed3D_model_path=str(sam3dmed_checkpoint_path),
        feature_type="cls",
    )
    feature_dict["patient"].extend([f"{patient}"] * len(output_dict["feature_name"]))
    feature_dict["well_fov"].extend([f"{well_fov}"] * len(output_dict["feature_name"]))
    feature_dict["feature_name"].extend(
        f"{channel_id}_{feature_name}" for feature_name in output_dict["feature_name"]
    )
    feature_dict["feature_value"].extend(output_dict["value"])

    batch_feat = featurize_2D_image_w_chami75(images, model, device)
    for f_idx in range(batch_feat.shape[1]):
        feature_name = f"{channel_id}_CHAMI75_feature_{f_idx}"
        feature_value = batch_feat[image_index, f_idx]
        feature_dict["patient"].extend([f"{patient}"])
        feature_dict["well_fov"].extend([f"{well_fov}"])
        feature_dict["feature_name"].append(feature_name)
        feature_dict["feature_value"].append(feature_value)

df = pd.DataFrame(feature_dict)


# In[ ]:


df = (
    df.pivot_table(
        index=["patient", "well_fov"], columns="feature_name", values="feature_value"
    )
    .reset_index()
    .rename_axis(None, axis=1)
)
df.head()


# In[ ]:


df.to_parquet(
    feature_save_path,
    index=False,
)


# In[ ]:


end_time = time.time()
# get starting memory (cpu)
end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2

print(f"Time taken: {end_time - start_time} seconds")
print(f"Memory used: {end_mem - start_mem} MB")
