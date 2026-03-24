"""
This utils file has module that utilize CHAMMI-75's featurization model.
This used a self-supervised deep-learning model
that uses a Vision Transformer (ViT) architecture
"""

from __future__ import annotations

import numpy
import torch
import torch.nn as nn
from torchvision import transforms as v2
from transformers import AutoModel


# get the model
def get_chammi75_model(device: str | None) -> torch.nn.Module:
    """Load the CHAMMI-75 (MorphEm) model from Hugging Face.

    Parameters
    ----------
    device : str or None
        The device to load the model on (``'cuda'`` or ``'cpu'``). If
        ``None``, CUDA is used when available, otherwise CPU.

    Returns
    -------
    torch.nn.Module
        The CHAMMI-75 (MorphEm) model in evaluation mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("CaicedoLab/MorphEm", trust_remote_code=True)
    model.to(device).eval()

    return model


# Noise Injector transformation
class SaturationNoiseInjector(nn.Module):
    """Inject uniform random noise into saturated pixels of an image tensor.
    There are three channels to the image where image 2 and 3 are duplicates
    of the first channel.
    We have three channels to fit the ViT architecture which expects three-channel input.
    This transformation replaces saturated pixels (value == 255) in the first
    channel of an image with uniform random noise sampled from
    ``[low, high]``. It is applied as a pre-processing step before
    passing the image to the CHAMMI-75 model.
    """

    def __init__(self, low: int = 200, high: int = 255) -> None:
        """Initialize the noise injector.

        Parameters
        ----------
        low : int, optional
            Lower bound of the uniform noise distribution, by default 200.
        high : int, optional
            Upper bound of the uniform noise distribution, by default 255.
        """
        super().__init__()
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply saturation-noise injection to the first channel.

        Parameters
        ----------
        x : torch.Tensor
            Image tensor of shape ``(C, H, W)`` where saturated pixels in the
            first channel (index 0) have value 255.

        Returns
        -------
        torch.Tensor
            Tensor with the same shape as ``x`` where saturated pixels in
            first channel have been replaced by uniform random noise.
        """
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

    def __init__(self, eps: float = 1e-7) -> None:
        """Initialize with a numerical stability epsilon.

        Parameters
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
) -> list[numpy.ndarray]:
    """Extract CHAMMI-75 CLS-token features from a multi-channel 2D image.

    The function processes each channel of the input image independently (Bag-of-Channels
    strategy). In step 1, the function resizes the image tensor to 224×224. In step 2, the function injects random noise
    into saturated pixels. In step 3, the function normalizes each image. In step 4, the function passes the stacked image into the
    Vision Transformer encoder. Lastly, in step 5, the function outputs the ``x_norm_clstoken``
    per channel.

    Parameters
    ----------
    image_tensor : torch.Tensor
        Batch of images with shape ``(N, C, H, W)`` where *N* is the batch
        size, *C* is the number of channels, and *H*, *W* are the spatial
        dimensions.
    model : torch.nn.Module
        The loaded CHAMMI-75 (MorphEm) model (see :func:`get_chammi75_model`).
    device : torch.device
        Device on which to run inference (``'cuda'`` or ``'cpu'``).

    Returns
    -------
    list of numpy.ndarray
        A list of length *C* where each element is a ``(N, 384)`` array
        containing the CLS-token embedding for that channel.
    """
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
    cropped_image: numpy.ndarray,
    model: torch.nn.Module,
    device: str | torch.device = "cpu",
) -> numpy.ndarray:
    """Run the CHAMMI-75 featurization pipeline on a single cropped 2D image.

    Converts the input NumPy array to a three-channel PyTorch tensor (by
    replicating the single channel) and extracts CLS-token features from the
    first channel. Because the ViT architecture expects three-channel input
    but we feed it a single fluorescence channel, the channel is replicated
    three times yet only the first copy's features are returned.

    Parameters
    ----------
    cropped_image : numpy.ndarray
        A 2D single-channel image array of shape ``(H, W)`` containing the
        cropped object region.
    model : torch.nn.Module
        The loaded CHAMMI-75 model (see :func:`get_chammi75_model`).

    Returns
    -------
    numpy.ndarray
        A ``(1, 384)`` array of CLS-token embeddings for the input image.
    """
    images = torch.from_numpy(cropped_image).float().unsqueeze(0)  # Add batch dimension
    # images is now (B, Y, X), add channel dimension -> (B, 1, Y, X)
    images = images.unsqueeze(1)
    # Replicate channel 3 times to get (B, 3, Y, X)
    images = images.repeat(1, 3, 1, 1)
    batch_feat = featurize_2D_image_w_chammi75(images, model, device)
    # return the first element
    # the other elements are duplications as we copied the channel into 3
    # the ViT framework is designed for 3-channel input,
    # so we replicated the single channel into 3 channels
    # we only want to feature once
    # loop loop this per channel image input
    # and not using channels here.
    # hence the CHAMMI name
    # channel adaptive -
    return batch_feat[0]
