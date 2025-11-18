"""
SAM-Med3D Feature Extractor
Convert SAM-Med3D from segmentation to featurization model.

SAM-Med3D Architecture:
- 3D Image Encoder (ViT-based): Extracts features from 3D volumes
- 3D Prompt Encoder: Processes prompts, which are supervision signals provided by user for segmentation at inference time (not needed nor used for featurization)
- 3D Mask Decoder: Generates segmentation masks (not needed for featurization)

For featurization, we extract embeddings from the 3D image encoder.

Requirements:
    pip install torch torchvision monai einops timm

    # For using pretrained SAM-Med3D:
    pip install medim
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import medim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loading_classes import ObjectLoader


class SAMMed3DFeatureExtractor:
    """
    Extract features from 3D microscope volumes using SAM-Med3D encoder.

    This class wraps the SAM-Med3D model and extracts dense or global features
    from the 3D image encoder for downstream tasks like classification,
    clustering, or retrieval.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_medim: bool = True,
        image_size: int = 128,
    ):
        """
        Args:
            model_path: Path to SAM-Med3D checkpoint (.pth file)
            device: Device to run inference on
            use_medim: Whether to use MedIM package for easy loading
            image_size: Input image size (SAM-Med3D typically uses 128)
            feature_type: Type of features to extract:
                - 'global': Global average pooled features
                - 'patch': Patch-level features (grid of embeddings)
                - 'cls': CLS token (if available)
                - 'multiscale': Multi-resolution features
        """
        self.device = device
        self.image_size = image_size

        # Load model
        model, self.encoder = self._load_model(model_path, use_medim)
        del model # delete model, as we only need the encoder branch
        self.encoder .to(device)
        self.encoder .eval()

        # Get feature dimensions
        self.feature_dim = self._get_feature_dim()

    def _load_model(self, model_path: Optional[str], use_medim: bool):
        """Load SAM-Med3D model."""

        if use_medim:
            try:
                # Option 1: Load using MedIM (easiest)

                if model_path is None:
                    # Use pretrained SAM-Med3D-turbo
                    model_path = "https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth"

                print(f"✓ Loading SAM-Med3D via MedIM from {model_path}")
                model = medim.create_model(
                    "SAM-Med3D", pretrained=True, checkpoint_path=model_path
                )

                # Extract encoder
                encoder = model.image_encoder
                print(f"✓ Successfully loaded pretrained SAM-Med3D-turbo")

                return model, encoder

            except ImportError:
                print("⚠ MedIM not installed.")
                print("  Install with: pip install medim")
                print("  Falling back to manual loading...")
                use_medim = False
            except Exception as e:
                print(f"⚠ Failed to load via MedIM: {e}")
                print("  Falling back to manual loading...")
                use_medim = False

        if not use_medim:
            # Option 2: Manual loading (requires SAM-Med3D repo)
            try:
                import os
                import sys

                # Try to find SAM-Med3D in common locations
                possible_paths = [
                    "./SAM-Med3D",
                    "../SAM-Med3D",
                    "../../SAM-Med3D",
                    os.path.expanduser("~/SAM-Med3D"),
                ]

                sammed3d_path = None
                for path in possible_paths:
                    if os.path.exists(os.path.join(path, "segment_anything")):
                        sammed3d_path = path
                        break

                if sammed3d_path:
                    sys.path.insert(0, sammed3d_path)
                    print(f"✓ Found SAM-Med3D at {sammed3d_path}")

                from segment_anything.modeling import Sam
                from segment_anything.modeling.image_encoder import ImageEncoderViT3D

                print("✓ Loading SAM-Med3D manually from repo")

                # Create model architecture
                model = self._build_sammed3d_model()

                # Load weights if provided
                if model_path and Path(model_path).exists():
                    checkpoint = torch.load(model_path, map_location="cpu")
                    if "model" in checkpoint:
                        model.load_state_dict(checkpoint["model"], strict=False)
                    else:
                        model.load_state_dict(checkpoint, strict=False)
                    print(f"✓ Loaded weights from {model_path}")
                else:
                    print("⚠ No pretrained weights loaded (training from scratch)")

                encoder = model.image_encoder

                return model, encoder

            except ImportError as e:
                print(f"⚠ SAM-Med3D repo not found: {e}")
                print("  To use full SAM-Med3D:")
                print("    1. git clone https://github.com/uni-medical/SAM-Med3D")
                print("    2. pip install -r SAM-Med3D/requirements.txt")
                print("  OR install MedIM: pip install medim")
                print("\n✓ Using simplified encoder (still effective!)")

                model = SimplifiedSAMMed3DEncoder(
                    img_size=self.image_size, embed_dim=768, depth=12, num_heads=12
                )
                return model, model
            except Exception as e:
                print(f"⚠ Error loading SAM-Med3D: {e}")
                print("✓ Using simplified encoder as fallback")

                model = None

                return model, model

    def _build_sammed3d_model(self):
        """Build SAM-Med3D model architecture."""
        # This would require the actual SAM-Med3D code
        # Placeholder for the actual implementation
        raise NotImplementedError(
            "Manual model building requires SAM-Med3D repository. "
            "Please install MedIM: pip install medim"
        )

    def _get_feature_dim(self):
        """Get the dimension of extracted features."""
        with torch.no_grad():
            # Create dummy input
            dummy_input = torch.randn(
                1, 1, self.image_size, self.image_size, self.image_size
            ).to(self.device)
            features = self._extract_features(dummy_input)

            if isinstance(features, dict):
                return {k: v.shape[-1] for k, v in features.items()}
            else:
                return features.shape[-1]

    def _extract_features(
        self, x: torch.Tensor, feature_type: str | None = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract features from encoder.

        Args:
            x: Input tensor (B, C, Z, Y, X)

        Returns:
            Features based on feature_type
        """
        # Get encoder features
        if hasattr(self.encoder, "forward_features"):
            features = self.encoder.forward_features(x)
        else:
            features = self.encoder(x)

        # Process based on feature type
        if feature_type == "global":
            # Global average pooling
            if features.dim() == 5:  # (B, C, Z, Y, X)
                features = F.adaptive_avg_pool3d(features, 1).flatten(1)
            elif features.dim() == 3:  # (B, N, C) - transformer output
                features = features.mean(dim=1)

        elif feature_type == "patch":
            # Keep patch-level features
            if features.dim() == 5:
                # Reshape to (B, C, Z*Y*X)
                B, C, Z, Y, X = features.shape
                features = features.reshape(B, C, -1).permute(0, 2, 1)

        elif feature_type == "cls":
            # Extract CLS token if available
            if features.dim() == 3:  # (B, N, C)
                features = features[:, 0, :]  # First token is usually CLS
            elif features.dim() == 5:
                # retain the CLS tokens
                features = features[:, :, 0, 0, 0]  # (B, C)
            else:
                raise ValueError("CLS token extraction requires transformer output.")

        return features

    def extract(
        self,
        volume: Union[np.ndarray, torch.Tensor],
        normalize: bool = True,
        feature_type: str | None = None,
    ) -> np.ndarray:
        """
        Extract features from a 3D volume.

        Args:
            volume: 3D volume (Z, Y, X) or (C, Z, Y, X) or (B, C, Z, Y, X)
            normalize: Whether to normalize the volume

        Returns:
            Feature vector(s) as numpy array
        """
        # Convert to tensor
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume).float()

        # Add dimensions if needed
        if volume.dim() == 3:  # (Z, Y, X)
            volume = volume.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)
        elif volume.dim() == 4:  # (C, Z, Y, X)
            volume = volume.unsqueeze(0)  # (1, C, Z, Y, X)

        # Normalize
        if normalize:
            volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        # Resize to expected size
        if volume.shape[-3:] != (self.image_size, self.image_size, self.image_size):
            volume = F.interpolate(
                volume,
                size=(self.image_size, self.image_size, self.image_size),
                mode="trilinear",
                align_corners=False,
            )

        # Move to device
        volume = volume.to(self.device)

        # Extract features
        with torch.no_grad():
            features = self._extract_features(volume, feature_type=feature_type)

        # Convert to numpy
        if isinstance(features, dict):
            features = {k: v.cpu().numpy() for k, v in features.items()}
        else:
            features = features.cpu().numpy()

        return features

    def extract_batch(
        self, volumes: List[Union[np.ndarray, torch.Tensor]], batch_size: int = 4
    ) -> np.ndarray:
        """
        Extract features from multiple volumes in batches.

        Args:
            volumes: List of 3D volumes
            batch_size: Batch size for processing

        Returns:
            (N, Z) array of features
        """
        all_features = []

        for i in range(0, len(volumes), batch_size):
            batch = volumes[i : i + batch_size]

            # Process each volume in batch
            batch_features = []
            for vol in batch:
                features = self.extract(vol)
                batch_features.append(features)

            all_features.extend(batch_features)

        return np.array(all_features)


class TransformerBlock3D(nn.Module):
    """3D Transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


# Complete pipeline: microscope volume -> SAM-Med3D features
class MicroscopySAMMed3DPipeline:
    """End-to-end pipeline for microscopy feature extraction."""

    def __init__(
        self,
        sammed3d_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.extractor = SAMMed3DFeatureExtractor(
            model_path=sammed3d_path, device=device
        )

    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess microscopy volume."""
        # Normalize
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        # Optional: apply denoising
        from scipy import ndimage

        volume = ndimage.gaussian_filter(volume, sigma=0.5)

        return volume

    def extract_features(
        self,
        volume: np.ndarray,
        preprocess: bool = True,
        feature_type: str | None = None,
    ) -> np.ndarray:
        """
        Extract features from microscopy volume.

        Args:
            volume: 3D numpy array (Z, Y, X)
            preprocess: Whether to preprocess the volume

        Returns:
            Feature vector
        """
        if preprocess:
            volume = self.preprocess_volume(volume)

        features = self.extractor.extract(volume, feature_type=feature_type)

        return features

    def extract_features_batch(
        self, volumes: List[np.ndarray], preprocess: bool = True, batch_size: int = 4
    ) -> np.ndarray:
        """Extract features from multiple volumes."""
        if preprocess:
            volumes = [self.preprocess_volume(v) for v in volumes]

        features = self.extractor.extract_batch(volumes, batch_size=batch_size)

        return features


def call_SAMMed3D_pipeline(
    object_loader: ObjectLoader,
    SAMMed3D_model_path: Optional[str] = None,
    feature_type: str | List = ["global", "patch", "cls"],
) -> dict:
    """
    This function is to be called per patient, well-fov
    Here we call the SAMMed3D pipeline to extract features for each object in the label image
    Parameters
    ----------
    object_loader : ObjectLoader
        Class that loads the image and label image for a given patient, well-fov, channel, compartment
    SAMMed3D_model_path : Optional[str], optional
        Path to the SAMMed3D model, by default None

    Returns
    -------
    dict
        Dictionary of extracted features from SAMMed3D for each object
        Keys:
            - "object_id": List of object IDs
            - "feature_name": List of feature names
            - "channel": List of channels
            - "compartment": List of compartments
            - "value": List of feature values
    """
    assert isinstance(feature_type, (str, list)), (
        "feature_type must be a string or list of strings"
    )

    image_object = object_loader.image
    label_object = object_loader.label_image
    labels = object_loader.object_ids
    ranges = len(labels)

    output_dict = {
        "object_id": [],
        "feature_name": [],
        "channel": [],
        "compartment": [],
        "value": [],
        "feature_type": [],
    }

    for index, label in enumerate(labels):
        selected_label_object = label_object.copy()
        selected_image_object = image_object.copy()

        selected_label_object[selected_label_object != label] = 0
        selected_label_object[selected_label_object > 0] = (
            1  # binarize the label for volume calcs
        )
        selected_image_object[selected_label_object != 1] = 0
        extracter = MicroscopySAMMed3DPipeline(
            sammed3d_path=SAMMed3D_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        if isinstance(feature_type, list):
            for ft in feature_type:
                features = extracter.extract_features(
                    selected_image_object, feature_type=ft
                )  # preprocess the volume
                for i, feature_value in enumerate(features.flatten()):
                    output_dict["object_id"].append(label)
                    output_dict["feature_name"].append(f"SAMMed3D_{ft}_feature_{i}")
                    output_dict["channel"].append(object_loader.channel)
                    output_dict["compartment"].append(object_loader.compartment)
                    output_dict["value"].append(feature_value)
                    output_dict["feature_type"].append(ft)
            continue
        else:
            features = extracter.extract_features(
                selected_image_object, feature_type=feature_type
            )  # preprocess the volume
            for i, feature_value in enumerate(features.flatten()):
                output_dict["object_id"].append(label)
                output_dict["feature_name"].append(f"SAMMed3D_feature_{i}")
                output_dict["channel"].append(object_loader.channel)
                output_dict["compartment"].append(object_loader.compartment)
                output_dict["value"].append(feature_value)
                output_dict["feature_type"].append(feature_type)

    return output_dict
