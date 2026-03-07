"""
This utils file has module that utilize CHAMMI-75's featurization model.
This used a self-supervised deep-learning model
that uses a Vision Transformer (ViT) architecture
"""

import numpy
import torch
import torch.nn as nn
from torchvision import transforms as v2
from transformers import AutoModel


# get the model
def get_chammi75_model(device):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("CaicedoLab/MorphEm", trust_remote_code=True)
    model.to(device).eval()

    return model


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
    """Normalize each image independently using InstanceNorm2d."""

    def __init__(self, eps=1e-7):
        """Parameters
        ----------
        eps : float, optional
            A small value added to the denominator for numerical stability. Default is 1e-7
        """
        super().__init__()
        self.eps = eps
        self.instance_norm = nn.InstanceNorm2d(
            num_features=1,
            affine=False,
            track_running_stats=False,
            eps=self.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass on the network

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C, H, W) where N is batch size, C is number of channels, H and W are height and width.

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as input.

        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.instance_norm(x)
        if x.shape[0] == 1:
            x = x.squeeze(0)
        return x


def featurize_2D_image_w_chammi75(
    image_tensor: torch.Tensor, model: torch.nn.Module, device: torch.device
):
    # Define transforms
    transform = v2.Compose(
        [
            SaturationNoiseInjector(),
            PerImageNormalize(),
            v2.Resize(size=(224, 224), antialias=True),
        ]
    )
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
    return batch_feat


def call_chammi75_featurization_pipeline(
    cropped_image: numpy.ndarray, model: torch.nn.Module
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = torch.from_numpy(cropped_image).float().unsqueeze(0)  # Add batch dimension
    # images is now (B, Y, X), add channel dimension -> (B, 1, Y, X)
    images = images.unsqueeze(1)
    # Replicate channel 3 times to get (B, 3, Y, X)
    images = images.repeat(1, 3, 1, 1)
    batch_feat = featurize_2D_image_w_chammi75(images, model, device)
    return batch_feat[0]
